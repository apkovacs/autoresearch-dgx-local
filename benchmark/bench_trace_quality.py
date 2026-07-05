#!/usr/bin/env python3
"""
Level 4: Trace quality — how suited is a model to the original agent design?

Parses Claude Code stream-json transcripts (logs/transcripts/*.jsonl) and the
orchestrator event log (logs/events.jsonl) to measure agentic overhead: how
far the model's execution trace deviates from the ideal experiment loop.

The ideal loop is ~7 tool calls per experiment:
    Edit -> git add -> git commit -> run_experiment.sh -> grep results
    -> git rev-parse -> log_result.sh

Metrics per session (one transcript = one session):
  - tool_calls, bash/edit/read/write breakdown
  - tool calls per completed experiment (vs. IDEAL_CALLS_PER_EXPERIMENT)
  - permission denials (tool_result errors mentioning permission/approval)
  - tool errors overall
  - repeated identical tool calls (same tool + same input)
  - redundant reads (re-reading a file already read this session)
  - instruction-friction indicators: direct `python train.py`, shell
    redirection, &&-chained git, backgrounded commands, run.log polling
  - thinking volume (chars) and output tokens (from usage metadata)

Session-level context from events.jsonl (if present): restarts, sessions
with zero new experiments (degenerate loops), context compactions.

Usage:
    python benchmark/bench_trace_quality.py \
        --transcripts logs/transcripts \
        --events logs/events.jsonl \
        --label "deepseek-v4-flash-dwarf/minimal" \
        --output benchmark/results/trace_quality.tsv

No GPU and no Ollama needed — this is pure log analysis and can run on the
host or in the benchmark container.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

IDEAL_CALLS_PER_EXPERIMENT = 7

# Friction / instruction-adherence patterns applied to Bash commands.
# In guarded mode these are rule violations; in minimal mode they are
# behavioral indicators (the model chose a path the environment punishes).
FRICTION_PATTERNS = {
    "direct_train": re.compile(r"\bpython3?\s+train\.py\b"),
    "redirect": re.compile(r"(?<![>&|])>{1,2}\s"),
    "chained_git": re.compile(r"\bgit\b[^;|&]*&&[^;|&]*\bgit\b"),
    "background": re.compile(r"&\s*$"),
    "poll_runlog": re.compile(r"\b(tail|cat|less|head)\b[^|;&]*\brun\.log\b"),
}

DENIAL_PATTERN = re.compile(
    r"permission|denied|not allowed|requires approval|haven't granted", re.IGNORECASE
)

TSV_COLUMNS = [
    "session",
    "label",
    "tool_calls",
    "bash_calls",
    "edit_calls",
    "read_calls",
    "write_calls",
    "other_calls",
    "experiments",
    "calls_per_exp",
    "denials",
    "tool_errors",
    "repeated_calls",
    "redundant_reads",
    "direct_train",
    "redirect",
    "chained_git",
    "background",
    "poll_runlog",
    "thinking_kchars",
    "output_tokens",
]


def analyze_transcript(path):
    """Parse one stream-json transcript and return a metrics dict."""
    m = {
        "tool_calls": 0,
        "bash_calls": 0,
        "edit_calls": 0,
        "read_calls": 0,
        "write_calls": 0,
        "other_calls": 0,
        "denials": 0,
        "tool_errors": 0,
        "repeated_calls": 0,
        "redundant_reads": 0,
        "thinking_chars": 0,
        "output_tokens": 0,
        "log_result_calls": 0,
    }
    for key in FRICTION_PATTERNS:
        m[key] = 0

    seen_calls = Counter()
    files_read = Counter()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "assistant":
                msg = event.get("message", {})
                usage = msg.get("usage") or {}
                m["output_tokens"] += usage.get("output_tokens", 0) or 0

                for block in msg.get("content", []):
                    btype = block.get("type", "")

                    if btype == "thinking":
                        m["thinking_chars"] += len(block.get("thinking", ""))

                    elif btype == "tool_use":
                        name = block.get("name", "?")
                        inp = block.get("input", {})
                        m["tool_calls"] += 1

                        sig = (name, json.dumps(inp, sort_keys=True))
                        seen_calls[sig] += 1
                        if seen_calls[sig] > 1:
                            m["repeated_calls"] += 1

                        if name == "Bash":
                            m["bash_calls"] += 1
                            cmd = inp.get("command", "")
                            for key, pattern in FRICTION_PATTERNS.items():
                                if pattern.search(cmd):
                                    m[key] += 1
                            if "log_result.sh" in cmd:
                                m["log_result_calls"] += 1
                        elif name == "Edit":
                            m["edit_calls"] += 1
                        elif name == "Read":
                            m["read_calls"] += 1
                            fp = inp.get("file_path", "")
                            files_read[fp] += 1
                            if files_read[fp] > 1:
                                m["redundant_reads"] += 1
                        elif name == "Write":
                            m["write_calls"] += 1
                        else:
                            m["other_calls"] += 1

            elif etype == "user":
                msg = event.get("message", {})
                content_blocks = msg.get("content", [])
                if not isinstance(content_blocks, list):
                    continue
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    if not block.get("is_error"):
                        continue
                    m["tool_errors"] += 1
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content if isinstance(b, dict)
                        )
                    if isinstance(content, str) and DENIAL_PATTERN.search(content):
                        m["denials"] += 1

    return m


def count_experiments_from_results(results_path):
    """Rows in results.tsv, excluding header."""
    p = Path(results_path)
    if not p.exists():
        return None
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


def analyze_events(events_path):
    """Summarize orchestrator events: restarts, degenerate sessions, compactions."""
    p = Path(events_path)
    summary = {"restarts": 0, "zero_progress_sessions": 0, "context_compactions": 0}
    if not p.exists():
        return summary
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("event", "")
            if etype == "agent_restart":
                summary["restarts"] += 1
            elif etype == "context_compaction":
                summary["context_compactions"] += 1
            elif etype == "agent_exit" and event.get("new_experiments") == 0:
                summary["zero_progress_sessions"] += 1
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--transcripts",
        default="logs/transcripts",
        help="Directory of stream-json transcript .jsonl files (or a single file)",
    )
    parser.add_argument(
        "--events", default="logs/events.jsonl", help="Orchestrator event log"
    )
    parser.add_argument(
        "--results",
        default="results.tsv",
        help="results.tsv for total experiment count (used when a session's "
        "log_result.sh calls are not visible in the transcript)",
    )
    parser.add_argument(
        "--label",
        default="unlabeled",
        help="Model/mode label recorded with each row, e.g. 'qwen3.6:27b/guarded'",
    )
    parser.add_argument(
        "--output",
        default="benchmark/results/trace_quality.tsv",
        help="Output TSV (appended, header written if new)",
    )
    args = parser.parse_args()

    tpath = Path(args.transcripts)
    if tpath.is_file():
        transcripts = [tpath]
    else:
        transcripts = sorted(tpath.glob("*.jsonl"))
    if not transcripts:
        print(f"No transcripts found in {args.transcripts}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    rows = []
    totals = Counter()
    for t in transcripts:
        m = analyze_transcript(t)
        # Per-session experiment count: log_result.sh calls seen in this
        # transcript are the most direct signal of completed experiments.
        experiments = m["log_result_calls"]
        calls_per_exp = m["tool_calls"] / experiments if experiments else 0.0

        row = {
            "session": t.stem,
            "label": args.label,
            "experiments": experiments,
            "calls_per_exp": f"{calls_per_exp:.1f}",
            "thinking_kchars": f"{m['thinking_chars'] / 1000:.1f}",
            **{
                k: m[k]
                for k in TSV_COLUMNS
                if k in m and k not in ("thinking_kchars",)
            },
        }
        rows.append(row)
        for k, v in m.items():
            totals[k] += v

    with open(out_path, "a") as f:
        if write_header:
            f.write("\t".join(TSV_COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in TSV_COLUMNS) + "\n")

    events = analyze_events(args.events)
    total_experiments = totals["log_result_calls"]
    results_experiments = count_experiments_from_results(args.results)

    # --- Summary ---
    print(f"\n=== Trace Quality: {args.label} ===")
    print(f"  Sessions analyzed:        {len(transcripts)}")
    print(f"  Experiments (transcript): {total_experiments}")
    if results_experiments is not None:
        print(f"  Experiments (results.tsv): {results_experiments}")
    print(f"  Total tool calls:         {totals['tool_calls']}")
    if total_experiments:
        cpe = totals["tool_calls"] / total_experiments
        overhead = (cpe / IDEAL_CALLS_PER_EXPERIMENT - 1) * 100
        print(
            f"  Tool calls / experiment:  {cpe:.1f} "
            f"(ideal ~{IDEAL_CALLS_PER_EXPERIMENT}, overhead {overhead:+.0f}%)"
        )
    print(f"  Permission denials:       {totals['denials']}")
    print(f"  Tool errors:              {totals['tool_errors']}")
    print(f"  Repeated identical calls: {totals['repeated_calls']}")
    print(f"  Redundant file reads:     {totals['redundant_reads']}")
    print(f"  Friction indicators:")
    for key in FRICTION_PATTERNS:
        print(f"    {key:<14} {totals[key]}")
    print(f"  Thinking volume:          {totals['thinking_chars'] / 1000:.0f}K chars")
    print(f"  Output tokens:            {totals['output_tokens']}")
    print(f"  Orchestrator events:")
    print(f"    restarts               {events['restarts']}")
    print(f"    zero-progress sessions {events['zero_progress_sessions']}")
    print(f"    context compactions    {events['context_compactions']}")
    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
