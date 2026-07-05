#!/usr/bin/env python3
"""
Benchmark Dashboard Generator

Reads benchmark results TSVs and generates a self-contained HTML dashboard
with interactive charts comparing models and harnesses.

Usage:
    python benchmark/bench_dashboard.py
    python benchmark/bench_dashboard.py --output benchmark/results/dashboard.html
    python benchmark/bench_dashboard.py --results-dir benchmark/results/

Opens the generated dashboard in the default browser (pass --no-open to skip).
"""

import argparse
import csv
import json
import os
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_tsv(path: Path) -> list[dict]:
    """Read a TSV file into a list of dicts."""
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)


def aggregate_edit_quality(rows: list[dict]) -> dict:
    """Aggregate Level 1 results by model."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    result = {}
    for model, trials in by_model.items():
        n = len(trials)
        result[model] = {
            "trials": n,
            "success_rate": sum(1 for t in trials if t["success"] == "True") / n * 100,
            "json_valid": sum(1 for t in trials if t["json_valid"] == "True") / n * 100,
            "schema_valid": sum(1 for t in trials if t["schema_valid"] == "True") / n * 100,
            "edits_apply": sum(1 for t in trials if t["edits_apply"] == "True") / n * 100,
            "syntax_ok": sum(1 for t in trials if t["syntax_ok"] == "True") / n * 100,
            "is_meaningful": sum(1 for t in trials if t["is_meaningful"] == "True") / n * 100,
            "mean_time": sum(float(t["time_s"]) for t in trials) / n,
            "mean_tokens": sum(int(t.get("tokens_generated", 0)) for t in trials) / n,
        }
    return result


def aggregate_harness(rows: list[dict]) -> dict:
    """Aggregate Level 2 results by adapter."""
    by_adapter = defaultdict(list)
    for r in rows:
        by_adapter[r["adapter"]].append(r)

    result = {}
    for adapter, trials in by_adapter.items():
        n = len(trials)
        times = sorted(float(t["time_s"]) for t in trials)
        result[adapter] = {
            "model": trials[0].get("model", "unknown"),
            "trials": n,
            "success_rate": sum(1 for t in trials if t["success"] == "True") / n * 100,
            "syntax_ok": sum(1 for t in trials if t["syntax_ok"] == "True") / n * 100,
            "is_meaningful": sum(1 for t in trials if t["is_meaningful"] == "True") / n * 100,
            "mean_time": sum(times) / n,
            "p50_time": times[n // 2] if n > 0 else 0,
            "p95_time": times[int(n * 0.95)] if n > 0 else 0,
        }
    return result


def aggregate_e2e(rows: list[dict]) -> list[dict]:
    """Process Level 3 results (already one row per run)."""
    processed = []
    for r in rows:
        processed.append({
            "harness": r.get("harness", "unknown"),
            "model": r.get("model", "unknown"),
            "budget": int(r.get("budget", 0)),
            "total_experiments": int(r.get("total_experiments", 0)),
            "kept": int(r.get("kept", 0)),
            "discarded": int(r.get("discarded", 0)),
            "crashed": int(r.get("crashed", 0)),
            "baseline_bpb": float(r.get("baseline_bpb", 0)),
            "best_bpb": float(r.get("best_bpb", 0)),
            "improvement": float(r.get("improvement", 0)),
            "wall_time_m": round(float(r.get("wall_time_s", 0)) / 60, 1),
        })
    return processed


def aggregate_trace(rows: list[dict]) -> dict:
    """Aggregate Level 4 (trace quality) results by label (model/mode)."""
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    def total(trials, key):
        return sum(int(t.get(key, 0) or 0) for t in trials)

    result = {}
    for label, trials in by_label.items():
        experiments = total(trials, "experiments")
        tool_calls = total(trials, "tool_calls")
        friction = sum(
            total(trials, k)
            for k in ("direct_train", "redirect", "chained_git", "background", "poll_runlog")
        )
        result[label] = {
            "sessions": len(trials),
            "experiments": experiments,
            "tool_calls": tool_calls,
            "calls_per_exp": tool_calls / experiments if experiments else 0.0,
            "denials": total(trials, "denials"),
            "tool_errors": total(trials, "tool_errors"),
            "repeated_calls": total(trials, "repeated_calls"),
            "redundant_reads": total(trials, "redundant_reads"),
            "friction": friction,
            "output_tokens": total(trials, "output_tokens"),
        }
    return result


def generate_html(edit_quality: dict, harness: dict, e2e: list[dict],
                  trace: dict, results_dir: Path) -> str:
    """Generate a self-contained HTML dashboard."""

    # Prepare chart data
    eq_models = list(edit_quality.keys())
    eq_success = [edit_quality[m]["success_rate"] for m in eq_models]
    eq_json = [edit_quality[m]["json_valid"] for m in eq_models]
    eq_schema = [edit_quality[m]["schema_valid"] for m in eq_models]
    eq_apply = [edit_quality[m]["edits_apply"] for m in eq_models]
    eq_syntax = [edit_quality[m]["syntax_ok"] for m in eq_models]
    eq_times = [round(edit_quality[m]["mean_time"], 1) for m in eq_models]

    h_adapters = list(harness.keys())
    h_success = [harness[a]["success_rate"] for a in h_adapters]
    h_times = [round(harness[a]["mean_time"], 1) for a in h_adapters]
    h_p50 = [round(harness[a]["p50_time"], 1) for a in h_adapters]
    h_p95 = [round(harness[a]["p95_time"], 1) for a in h_adapters]

    # Count result files
    tsv_files = list(results_dir.glob("*.tsv"))
    total_rows = 0
    for f in tsv_files:
        total_rows += len(read_tsv(f))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autoresearch Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f1117; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 8px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
           gap: 24px; margin-bottom: 32px; }}
  .card {{ background: #1a1d27; border-radius: 12px; padding: 24px;
           border: 1px solid #2a2d37; }}
  .card h2 {{ font-size: 1.1rem; margin-bottom: 16px; color: #fff; }}
  .card h3 {{ font-size: 0.85rem; color: #888; margin-bottom: 12px;
              text-transform: uppercase; letter-spacing: 0.05em; }}
  .stats {{ display: flex; gap: 32px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat {{ text-align: center; }}
  .stat .value {{ font-size: 2rem; font-weight: 700; color: #6366f1; }}
  .stat .label {{ font-size: 0.75rem; color: #888; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #2a2d37;
       color: #888; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1f222c; }}
  tr:hover {{ background: #1f222c; }}
  .good {{ color: #22c55e; }}
  .warn {{ color: #eab308; }}
  .bad {{ color: #ef4444; }}
  .chart-container {{ position: relative; height: 280px; }}
  .empty {{ color: #555; font-style: italic; padding: 40px; text-align: center; }}
  .footer {{ text-align: center; color: #555; font-size: 0.75rem; margin-top: 40px; }}
</style>
</head>
<body>

<h1>Autoresearch Benchmark Dashboard</h1>
<p class="subtitle">Generated {generated} &mdash; {len(tsv_files)} result files, {total_rows} total data points</p>

<!-- Summary Stats -->
<div class="stats">
  <div class="stat">
    <div class="value">{len(eq_models)}</div>
    <div class="label">Models Tested</div>
  </div>
  <div class="stat">
    <div class="value">{len(h_adapters)}</div>
    <div class="label">Harnesses Compared</div>
  </div>
  <div class="stat">
    <div class="value">{len(e2e)}</div>
    <div class="label">E2E Runs</div>
  </div>
  <div class="stat">
    <div class="value">{total_rows}</div>
    <div class="label">Total Trials</div>
  </div>
</div>

<div class="grid">

<!-- Level 1: Edit Quality -->
<div class="card">
  <h2>Level 1: Edit Quality</h2>
  <h3>Success rate by model (% of trials producing valid, applicable, syntactically correct edits)</h3>
  {"<div class='empty'>No edit quality results yet. Run: bash benchmark/run-bench.sh edit-quality</div>" if not eq_models else f"<div class='chart-container'><canvas id='eqChart'></canvas></div>"}
</div>

<div class="card">
  <h2>Edit Quality Breakdown</h2>
  <h3>Pipeline stage pass rates</h3>
  {"<div class='empty'>No data</div>" if not eq_models else f'''
  <table>
    <tr><th>Model</th><th>Trials</th><th>JSON</th><th>Schema</th><th>Apply</th><th>Syntax</th><th>Success</th><th>Time</th></tr>
    {"".join(f"""<tr>
      <td>{m}</td>
      <td>{edit_quality[m]['trials']}</td>
      <td class='{"good" if edit_quality[m]["json_valid"]>=90 else "warn" if edit_quality[m]["json_valid"]>=50 else "bad"}'>{edit_quality[m]['json_valid']:.0f}%</td>
      <td class='{"good" if edit_quality[m]["schema_valid"]>=90 else "warn" if edit_quality[m]["schema_valid"]>=50 else "bad"}'>{edit_quality[m]['schema_valid']:.0f}%</td>
      <td class='{"good" if edit_quality[m]["edits_apply"]>=90 else "warn" if edit_quality[m]["edits_apply"]>=50 else "bad"}'>{edit_quality[m]['edits_apply']:.0f}%</td>
      <td class='{"good" if edit_quality[m]["syntax_ok"]>=90 else "warn" if edit_quality[m]["syntax_ok"]>=50 else "bad"}'>{edit_quality[m]['syntax_ok']:.0f}%</td>
      <td class='{"good" if edit_quality[m]["success_rate"]>=70 else "warn" if edit_quality[m]["success_rate"]>=40 else "bad"}'><strong>{edit_quality[m]['success_rate']:.0f}%</strong></td>
      <td>{edit_quality[m]['mean_time']:.1f}s</td>
    </tr>""" for m in eq_models)}
  </table>'''}
</div>

<!-- Level 2: Harness Comparison -->
<div class="card">
  <h2>Level 2: Harness Comparison</h2>
  <h3>Success rate &amp; latency by harness</h3>
  {"<div class='empty'>No harness comparison results yet. Run: bash benchmark/run-bench.sh harness</div>" if not h_adapters else f"<div class='chart-container'><canvas id='harnessChart'></canvas></div>"}
</div>

<div class="card">
  <h2>Harness Latency Profile</h2>
  <h3>Time per edit attempt (seconds)</h3>
  {"<div class='empty'>No data</div>" if not h_adapters else f'''
  <table>
    <tr><th>Harness</th><th>Model</th><th>Trials</th><th>Success</th><th>Mean</th><th>P50</th><th>P95</th></tr>
    {"".join(f"""<tr>
      <td>{a}</td>
      <td>{harness[a]['model']}</td>
      <td>{harness[a]['trials']}</td>
      <td class='{"good" if harness[a]["success_rate"]>=70 else "warn" if harness[a]["success_rate"]>=40 else "bad"}'><strong>{harness[a]['success_rate']:.0f}%</strong></td>
      <td>{harness[a]['mean_time']:.1f}s</td>
      <td>{harness[a]['p50_time']:.1f}s</td>
      <td>{harness[a]['p95_time']:.1f}s</td>
    </tr>""" for a in h_adapters)}
  </table>'''}
</div>

<!-- Level 3: End-to-End -->
<div class="card" style="grid-column: 1 / -1;">
  <h2>Level 3: End-to-End Results</h2>
  <h3>Full experiment loop on GPU</h3>
  {"<div class='empty'>No end-to-end results yet. Run: BENCH_GPU=1 bash benchmark/run-bench.sh e2e</div>" if not e2e else f'''
  <table>
    <tr><th>Harness</th><th>Model</th><th>Budget</th><th>Completed</th><th>Kept</th><th>Crashed</th><th>Baseline BPB</th><th>Best BPB</th><th>Improvement</th><th>Wall Time</th></tr>
    {"".join(f"""<tr>
      <td>{r['harness']}</td>
      <td>{r['model']}</td>
      <td>{r['budget']}</td>
      <td>{r['total_experiments']}</td>
      <td class='good'>{r['kept']}</td>
      <td class='{"bad" if r["crashed"]>0 else ""}'>{r['crashed']}</td>
      <td>{r['baseline_bpb']:.6f}</td>
      <td><strong>{r['best_bpb']:.6f}</strong></td>
      <td class='{"good" if r["improvement"]>0 else "bad"}'>{r['improvement']:+.6f}</td>
      <td>{r['wall_time_m']:.1f}m</td>
    </tr>""" for r in e2e)}
  </table>'''}
</div>

<!-- Level 4: Trace Quality -->
<div class="card" style="grid-column: 1 / -1;">
  <h2>Level 4: Trace Quality</h2>
  <h3>Agentic overhead per model/mode &mdash; ideal is ~7 tool calls per experiment, zero denials, zero friction</h3>
  {"<div class='empty'>No trace quality results yet. Run: bash benchmark/run-bench.sh trace --label model/mode</div>" if not trace else f'''
  <table>
    <tr><th>Model / Mode</th><th>Sessions</th><th>Experiments</th><th>Tool Calls</th><th>Calls/Exp</th><th>Denials</th><th>Errors</th><th>Repeats</th><th>Re-reads</th><th>Friction</th><th>Output Tokens</th></tr>
    {"".join(f"""<tr>
      <td>{label}</td>
      <td>{trace[label]['sessions']}</td>
      <td>{trace[label]['experiments']}</td>
      <td>{trace[label]['tool_calls']}</td>
      <td class='{"good" if 0 < trace[label]["calls_per_exp"] <= 10 else "warn" if trace[label]["calls_per_exp"] <= 15 else "bad"}'><strong>{trace[label]['calls_per_exp']:.1f}</strong></td>
      <td class='{"good" if trace[label]["denials"] == 0 else "bad"}'>{trace[label]['denials']}</td>
      <td class='{"good" if trace[label]["tool_errors"] == 0 else "warn"}'>{trace[label]['tool_errors']}</td>
      <td>{trace[label]['repeated_calls']}</td>
      <td>{trace[label]['redundant_reads']}</td>
      <td class='{"good" if trace[label]["friction"] == 0 else "warn" if trace[label]["friction"] <= 3 else "bad"}'>{trace[label]['friction']}</td>
      <td>{trace[label]['output_tokens']}</td>
    </tr>""" for label in trace)}
  </table>'''}
</div>

</div>

<div class="footer">
  Autoresearch Benchmark Suite &mdash; <a href="https://github.com/apkovacs/autoresearch-dgx-local" style="color:#6366f1;">apkovacs/autoresearch-dgx-local</a>
</div>

<script>
// Chart.js defaults for dark theme
Chart.defaults.color = '#888';
Chart.defaults.borderColor = '#2a2d37';

// Level 1: Edit Quality chart
{f'''
const eqCtx = document.getElementById('eqChart');
if (eqCtx) {{
  new Chart(eqCtx, {{
    type: 'bar',
    data: {{
      labels: {json.dumps(eq_models)},
      datasets: [
        {{ label: 'Success', data: {json.dumps(eq_success)}, backgroundColor: '#6366f1' }},
        {{ label: 'JSON Valid', data: {json.dumps(eq_json)}, backgroundColor: '#22c55e44', borderColor: '#22c55e', borderWidth: 1 }},
        {{ label: 'Edits Apply', data: {json.dumps(eq_apply)}, backgroundColor: '#eab30844', borderColor: '#eab308', borderWidth: 1 }},
        {{ label: 'Syntax OK', data: {json.dumps(eq_syntax)}, backgroundColor: '#3b82f644', borderColor: '#3b82f6', borderWidth: 1 }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{ y: {{ beginAtZero: true, max: 100, ticks: {{ callback: v => v + '%' }} }} }},
      plugins: {{ legend: {{ position: 'bottom' }} }}
    }}
  }});
}}
''' if eq_models else ''}

// Level 2: Harness comparison chart
{f'''
const hCtx = document.getElementById('harnessChart');
if (hCtx) {{
  new Chart(hCtx, {{
    type: 'bar',
    data: {{
      labels: {json.dumps(h_adapters)},
      datasets: [
        {{ label: 'Success %', data: {json.dumps(h_success)}, backgroundColor: '#6366f1', yAxisID: 'y' }},
        {{ label: 'Mean Time (s)', data: {json.dumps(h_times)}, backgroundColor: '#f97316', yAxisID: 'y1' }},
        {{ label: 'P95 Time (s)', data: {json.dumps(h_p95)}, backgroundColor: '#f9731644', borderColor: '#f97316', borderWidth: 1, yAxisID: 'y1' }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      scales: {{
        y: {{ beginAtZero: true, max: 100, position: 'left', ticks: {{ callback: v => v + '%' }} }},
        y1: {{ beginAtZero: true, position: 'right', grid: {{ drawOnChartArea: false }}, ticks: {{ callback: v => v + 's' }} }}
      }},
      plugins: {{ legend: {{ position: 'bottom' }} }}
    }}
  }});
}}
''' if h_adapters else ''}
</script>

</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard from benchmark results"
    )
    parser.add_argument("--results-dir",
                        default=str(REPO_ROOT / "benchmark/results"),
                        help="Directory containing result TSV files")
    parser.add_argument("--output",
                        default=str(REPO_ROOT / "benchmark/results/dashboard.html"),
                        help="Output HTML file path")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open the dashboard in a browser")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    # Read all available results
    edit_quality_rows = read_tsv(results_dir / "edit_quality.tsv")
    harness_rows = read_tsv(results_dir / "harness_comparison.tsv")
    e2e_rows = read_tsv(results_dir / "e2e.tsv")
    trace_rows = read_tsv(results_dir / "trace_quality.tsv")

    # Aggregate
    edit_quality = aggregate_edit_quality(edit_quality_rows)
    harness = aggregate_harness(harness_rows)
    e2e = aggregate_e2e(e2e_rows)
    trace = aggregate_trace(trace_rows)

    has_data = edit_quality or harness or e2e or trace
    if not has_data:
        print("No benchmark results found. Run some benchmarks first:")
        print("  bash benchmark/run-bench.sh edit-quality --models qwen3.6:27b --trials 5")
        print("  bash benchmark/run-bench.sh harness --adapters ollama_raw --trials 5")

    # Generate
    html = generate_html(edit_quality, harness, e2e, trace, results_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Dashboard written to {output_path}")

    if not args.no_open:
        try:
            webbrowser.open(f"file://{output_path.resolve()}")
            print("Opened in browser.")
        except Exception:
            print(f"Open manually: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
