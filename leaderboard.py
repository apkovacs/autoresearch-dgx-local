"""
Cross-branch results tracking and leaderboard management.

Parses per-branch results.tsv files and maintains a global leaderboard
across all research branches in the game config state.
"""

import os

RESULTS_DIR = "results"
HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\ttimestamp"


def parse_results_tsv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                rows.append({
                    "commit": parts[0],
                    "val_bpb": float(parts[1]) if parts[1] != "0.000000" else None,
                    "memory_gb": float(parts[2]),
                    "status": parts[3],
                    "description": parts[4] if len(parts) > 4 else "",
                    "timestamp": parts[5] if len(parts) > 5 else "",
                })
    return rows


def write_results_header(path):
    with open(path, "w") as f:
        f.write(HEADER + "\n")


def get_best_bpb(results):
    valid = [r["val_bpb"] for r in results if r["val_bpb"] is not None and r["status"] == "keep"]
    return min(valid) if valid else None


def get_improvement_rate(results, last_n=None):
    kept = [r for r in results if r["val_bpb"] is not None and r["status"] == "keep"]
    if last_n is not None:
        kept = kept[-last_n:]
    if len(kept) < 2:
        return 0.0
    first_bpb = kept[0]["val_bpb"]
    last_bpb = kept[-1]["val_bpb"]
    return (first_bpb - last_bpb) / len(kept)


def update_leaderboard(config, branch_name):
    path = os.path.join(RESULTS_DIR, f"{branch_name}.tsv")
    results = parse_results_tsv(path)
    best = get_best_bpb(results)
    total = len(results)
    kept = sum(1 for r in results if r["status"] == "keep")

    if "leaderboard" not in config["state"]:
        config["state"]["leaderboard"] = {}

    config["state"]["leaderboard"][branch_name] = {
        "best_val_bpb": best,
        "total_experiments": total,
        "kept_experiments": kept,
        "improvement_rate": get_improvement_rate(results, last_n=5),
    }


def get_global_best(config):
    best_branch = None
    best_bpb = None
    for name, entry in config["state"].get("leaderboard", {}).items():
        bpb = entry.get("best_val_bpb")
        if bpb is not None and (best_bpb is None or bpb < best_bpb):
            best_bpb = bpb
            best_branch = name
    return best_branch, best_bpb


def print_leaderboard(config):
    lb = config["state"].get("leaderboard", {})
    if not lb:
        print("Leaderboard: (no results yet)")
        return

    global_branch, global_best = get_global_best(config)
    print(f"\n{'='*70}")
    print(f"  LEADERBOARD — Round {config['state'].get('current_round', 0)}")
    print(f"{'='*70}")
    print(f"  {'Branch':<16} {'Best BPB':>10} {'Experiments':>12} {'Kept':>6} {'Imp/Exp':>10} {'':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*6} {'-'*10} {'-'*6}")

    for name in sorted(lb.keys()):
        entry = lb[name]
        bpb = entry.get("best_val_bpb")
        bpb_str = f"{bpb:.6f}" if bpb is not None else "N/A"
        marker = " *" if name == global_branch else ""
        imp = entry.get("improvement_rate", 0)
        imp_str = f"{imp:.6f}" if imp != 0 else "—"
        print(f"  {name:<16} {bpb_str:>10} {entry['total_experiments']:>12} {entry['kept_experiments']:>6} {imp_str:>10} {marker:>6}")

    if global_best is not None:
        print(f"\n  Global best: {global_best:.6f} ({global_branch})")
    print(f"{'='*70}\n")
