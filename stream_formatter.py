"""
Real-time formatter for Claude Code stream-json output.

Reads stream-json from stdin, writes raw JSON to a transcript file,
and prints human-readable output to stdout. Designed to sit between
Claude Code and the terminal so you see agent activity and training
progress without losing the full transcript.

Usage:
    claude -p --verbose --output-format stream-json "..." \
        2>&1 | python3 stream_formatter.py logs/transcripts/agent.jsonl
"""

import json
import re
import sys

# ANSI colors
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Patterns that indicate training progress output
TRAINING_PATTERNS = [
    re.compile(r"step\s+\d+"),
    re.compile(r"val_bpb"),
    re.compile(r"tokens/sec"),
    re.compile(r"MFU"),
    re.compile(r"loss[\s=:]"),
    re.compile(r"training time"),
    re.compile(r"compil", re.IGNORECASE),
]


def is_training_output(text):
    return any(p.search(text) for p in TRAINING_PATTERNS)


def format_event(event):
    """Format a stream-json event for terminal display."""
    etype = event.get("type", "")

    if etype == "assistant":
        msg = event.get("message", {})
        for block in msg.get("content", []):
            btype = block.get("type", "")

            if btype == "thinking":
                text = block.get("thinking", "")
                if text:
                    # Show first 200 chars of thinking
                    preview = text[:200].replace("\n", " ")
                    if len(text) > 200:
                        preview += "..."
                    print(f"{DIM}{CYAN}[thinking]{RESET} {DIM}{preview}{RESET}")

            elif btype == "text":
                text = block.get("text", "")
                if text:
                    print(f"{GREEN}{BOLD}[agent]{RESET} {text[:300]}")

            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                if name == "Bash":
                    cmd = inp.get("command", "")
                    print(f"{YELLOW}[bash]{RESET} {cmd[:150]}")
                elif name in ("Read", "Edit", "Write"):
                    print(f"{YELLOW}[{name.lower()}]{RESET} {inp.get('file_path', '')}")
                else:
                    print(f"{YELLOW}[tool: {name}]{RESET}")

    elif etype == "result":
        # Tool results — show training-relevant output in full
        content = event.get("result", "")
        if isinstance(content, str) and content:
            lines = content.split("\n")
            for line in lines:
                if is_training_output(line):
                    print(f"  {line}")
                elif line.strip():
                    # Show first few non-training lines dimmed
                    pass  # suppress noise, training output is the priority

        # Also check for nested content structures
        if isinstance(content, dict):
            stdout = content.get("stdout", "")
            if stdout:
                for line in stdout.split("\n"):
                    if is_training_output(line):
                        print(f"  {line}")

    elif etype == "system":
        msg = event.get("message", "")
        if msg:
            print(f"{BLUE}[system]{RESET} {msg[:200]}")


def main():
    transcript_path = sys.argv[1] if len(sys.argv) >= 2 else None
    transcript = open(transcript_path, "w") if transcript_path else None

    try:
        for line in sys.stdin:
            # Write raw JSON to transcript file if provided
            if transcript:
                transcript.write(line)
                transcript.flush()

            # Parse and format for terminal
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
                format_event(event)
            except json.JSONDecodeError:
                # Non-JSON output (stderr, etc.) — print directly
                print(stripped)
    finally:
        if transcript:
            transcript.close()

    sys.stdout.flush()


if __name__ == "__main__":
    main()
