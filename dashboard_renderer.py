"""
Dashboard renderer for the autoresearch game monitor.

Reads the event log (JSONL) and prints a formatted dashboard to stdout:
  - Current state (mode, round, status)
  - Leaderboard table
  - Recent events

Usage:
    python3 dashboard_renderer.py logs/events.jsonl
"""

import json
import sys


def render(event_log_path):
    events = []
    try:
        with open(event_log_path) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print("  No event log found.")
        return

    if not events:
        print("  No events yet.")
        return

    # --- Current state ---
    current_round = 0
    mode = "?"
    for e in events:
        if e["event"] == "orchestrator_start":
            mode = e.get("mode", "?")
        if e["event"] == "round_start":
            current_round = e.get("round", 0)

    last = events[-1]
    activity = last["event"]
    branch = last.get("branch", "")

    if activity == "branch_progress":
        status = f"Running: {branch} ({last['completed']}/{last['target']} experiments, {last['elapsed_s']:.0f}s)"
    elif activity == "branch_start":
        status = f"Starting: {branch} (target: {last['target_experiments']} experiments)"
    elif activity == "branch_end":
        status = f"Finished: {branch} ({last['completed']} experiments in {last['duration_s']:.0f}s)"
    elif activity == "round_end":
        status = f"Round {last['round']} complete ({last['duration_s']:.0f}s)"
    elif activity == "migration":
        status = f"Migration: {last['source']} -> {last['dest']}"
    elif activity == "adoption":
        status = f"Adoption: {last['loser']} adopts from {last['winner']}"
    else:
        status = activity

    print(f"  Mode:     {mode}")
    print(f"  Round:    {current_round}")
    print(f"  Status:   {status}")
    print()

    # --- Leaderboard ---
    lb_events = [e for e in events if e["event"] == "leaderboard"]
    if lb_events:
        lb = lb_events[-1]
        entries = lb.get("entries", {})
        best_branch = lb.get("global_best_branch")
        best_bpb = lb.get("global_best_bpb")

        print(f"  {'Branch':<16} {'Best BPB':>10} {'Experiments':>12} {'Kept':>6} {'Imp/Exp':>10}")
        print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*6} {'-'*10}")
        for name in sorted(entries.keys()):
            entry = entries[name]
            bpb = entry.get("best_val_bpb")
            bpb_s = f"{bpb:.6f}" if bpb else "N/A"
            marker = " *" if name == best_branch else ""
            imp = entry.get("improvement_rate", 0)
            imp_s = f"{imp:.6f}" if imp else "—"
            print(
                f"  {name:<16} {bpb_s:>10} {entry['total_experiments']:>12} "
                f"{entry['kept_experiments']:>6} {imp_s:>10}{marker}"
            )

        if best_bpb:
            print()
            print(f"  Global best: {best_bpb:.6f} ({best_branch})")
        print()

    # --- Recent events ---
    recent = [e for e in events[-20:] if e["event"] != "branch_progress"][-8:]
    if recent:
        print("  Recent events:")
        for e in recent:
            ts = e["ts"][11:19]
            evt = e["event"]
            if evt == "branch_start":
                print(f"    {ts}  {e.get('branch',''):<12} started (target: {e['target_experiments']})")
            elif evt == "branch_end":
                print(f"    {ts}  {e['branch']:<12} finished ({e['completed']} exp, {e['duration_s']:.0f}s)")
            elif evt == "round_start":
                print(f"    {ts}  Round {e['round']} started")
            elif evt == "round_end":
                print(f"    {ts}  Round {e['round']} ended ({e['duration_s']:.0f}s)")
            elif evt == "migration":
                print(f"    {ts}  Migration: {e['source']} -> {e['dest']} [{e['params']}]")
            elif evt == "adoption":
                print(f"    {ts}  Adoption: {e['loser']} <- {e['winner']} [{e['params']}]")
            elif evt == "error":
                print(f"    {ts}  ERROR: {e.get('message','')}")
            else:
                print(f"    {ts}  {evt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dashboard_renderer.py <event_log.jsonl>")
        sys.exit(1)
    render(sys.argv[1])
