"""
Meta-research orchestrator for autoresearch-dgx.

Manages multiple research branches using game-theory-inspired strategies.
Supports four modes: base, island, bandit, coopetition.

Usage:
    python orchestrator.py                          # uses game_config.yaml
    python orchestrator.py --config my_config.yaml  # custom config
    python orchestrator.py --mode island            # override mode
"""

import argparse
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import yaml

from hyperparams import extract_hyperparams, apply_hyperparams, diff_hyperparams
from leaderboard import (
    parse_results_tsv, write_results_header, get_best_bpb,
    get_improvement_rate, update_leaderboard, print_leaderboard, get_global_best,
    RESULTS_DIR,
)
from event_log import (
    log_orchestrator_start, log_round_start, log_round_end,
    log_branch_start, log_branch_progress, log_branch_end,
    log_migration, log_adoption, log_meta_agent, log_leaderboard, log_error,
    LOGS_DIR,
)

CONFIG_PATH = "game_config.yaml"
BOUNDED_TEMPLATE = "branch_templates/bounded.md"
AGENT_PROMPT_FILE = "_round_prompt.md"
TRANSCRIPTS_DIR = os.path.join(LOGS_DIR, "transcripts")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config, path=CONFIG_PATH):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Branch management
# ---------------------------------------------------------------------------

def branch_name(mode, name, tag):
    return f"autoresearch/{mode}/{tag}/{name}"


def branch_exists(name):
    result = subprocess.run(["git", "rev-parse", "--verify", name],
                            capture_output=True, text=True)
    return result.returncode == 0


def setup_branches(config):
    mode = config["mode"]
    tag = config["tag"]
    base = config["base_branch"]
    branches = get_branch_list(config)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for b in branches:
        bname = branch_name(mode, b["name"], tag)
        if not branch_exists(bname):
            print(f"  Creating branch: {bname} (from {base})")
            subprocess.run(["git", "checkout", base], check=True,
                           capture_output=True, text=True)
            subprocess.run(["git", "checkout", "-b", bname], check=True,
                           capture_output=True, text=True)
        else:
            print(f"  Branch exists: {bname}")


def get_branch_list(config):
    mode = config["mode"]
    if mode == "island":
        return config["island"]["branches"]
    elif mode == "bandit":
        return [{"name": a["name"], "focus": a["strategy"]}
                for a in config["bandit"]["arms"]]
    elif mode == "coopetition":
        return config["coopetition"]["branches"]
    return []


# ---------------------------------------------------------------------------
# Results swapping
# ---------------------------------------------------------------------------

def swap_results_out(branch_name_short):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists("results.tsv"):
        shutil.copy("results.tsv", os.path.join(RESULTS_DIR, f"{branch_name_short}.tsv"))


def swap_results_in(branch_name_short):
    src = os.path.join(RESULTS_DIR, f"{branch_name_short}.tsv")
    if os.path.exists(src):
        shutil.copy(src, "results.tsv")
    else:
        write_results_header("results.tsv")


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

def count_results_lines(path="results.tsv"):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)


def render_prompt(max_experiments, focus_description):
    with open(BOUNDED_TEMPLATE) as f:
        template = f.read()
    return template.replace("{max_experiments}", str(max_experiments)) \
                   .replace("{focus_description}", focus_description)


def run_branch_round(config, branch_name_short, num_experiments, focus):
    mode = config["mode"]
    tag = config["tag"]
    round_num = config["state"]["current_round"] + 1
    bname = branch_name(mode, branch_name_short, tag)
    timeout = config.get("round_timeout_minutes", 60) * 60

    print(f"\n--- Round: {branch_name_short} ({num_experiments} experiments) ---")
    print(f"  Branch: {bname}")
    print(f"  Focus: {focus}")

    log_branch_start(branch_name_short, round_num, num_experiments, focus)

    subprocess.run(["git", "checkout", bname], check=True,
                   capture_output=True, text=True)

    prompt = render_prompt(num_experiments, focus)
    with open(AGENT_PROMPT_FILE, "w") as f:
        f.write(prompt)

    initial_count = count_results_lines()
    target_count = initial_count + num_experiments

    # Capture agent transcript to a per-round JSON stream file
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    transcript_path = os.path.join(
        TRANSCRIPTS_DIR,
        f"r{round_num:03d}_{branch_name_short}.jsonl"
    )
    transcript_file = open(transcript_path, "w")

    cmd = f'claude -p --output-format stream-json "$(cat {AGENT_PROMPT_FILE})"'
    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid,
                            stdout=transcript_file, stderr=subprocess.STDOUT)

    start = time.time()
    try:
        while True:
            time.sleep(30)
            current = count_results_lines()
            elapsed = time.time() - start
            completed = current - initial_count
            print(f"  [{elapsed/60:.0f}m] Experiments: {completed}/{num_experiments}")
            log_branch_progress(branch_name_short, round_num, completed,
                                num_experiments, elapsed)

            if current >= target_count:
                print(f"  Round complete ({completed} experiments)")
                break
            if elapsed > timeout:
                print(f"  Round timeout ({timeout/60:.0f}m)")
                log_error("branch_timeout", branch=branch_name_short,
                          round=round_num, elapsed_s=elapsed)
                break
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        transcript_file.close()

    if os.path.exists(AGENT_PROMPT_FILE):
        os.remove(AGENT_PROMPT_FILE)

    final_count = count_results_lines()
    completed = final_count - initial_count
    duration = time.time() - start
    print(f"  Completed: {completed} experiments")
    print(f"  Transcript: {transcript_path}")
    log_branch_end(branch_name_short, round_num, completed, duration)


# ---------------------------------------------------------------------------
# Strategy: Island Model
# ---------------------------------------------------------------------------

def compute_island_schedule(config, round_num):
    island = config["island"]
    branches = island["branches"]
    lb = config["state"].get("leaderboard", {})

    if not island.get("adaptive_allocation") or round_num <= 1 or not lb:
        return [(b["name"], b["budget"], b.get("focus", "general")) for b in branches]

    deltas = {}
    for b in branches:
        entry = lb.get(b["name"], {})
        deltas[b["name"]] = max(0, entry.get("improvement_rate", 0))

    total_delta = sum(deltas.values())
    total_budget = sum(b["budget"] for b in branches)
    mean_budget = total_budget / len(branches)
    min_budget = max(1, int(island.get("min_budget_ratio", 0.2) * mean_budget))

    schedule = []
    for b in branches:
        if total_delta > 0:
            share = deltas[b["name"]] / total_delta
            budget = max(min_budget, int(share * total_budget))
        else:
            budget = b["budget"]
        schedule.append((b["name"], budget, b.get("focus", "general")))

    return schedule


def run_island_migration(config):
    island = config["island"]
    branches = island["branches"]
    tag = config["tag"]
    lb = config["state"].get("leaderboard", {})
    migration_rate = island.get("migration_rate", 0.3)
    migration_select = island.get("migration_select", "best_delta")

    rates = {b["name"]: lb.get(b["name"], {}).get("improvement_rate", 0)
             for b in branches}

    sorted_branches = sorted(rates.keys(), key=lambda x: rates[x], reverse=True)
    if len(sorted_branches) < 2:
        return

    source = sorted_branches[0]
    if rates[source] <= 0:
        print("  Migration: no branch improved, skipping")
        return

    source_branch = branch_name("island", source, tag)
    subprocess.run(["git", "checkout", source_branch], check=True,
                   capture_output=True, text=True)
    source_params = extract_hyperparams("train.py")

    for dest in sorted_branches[1:]:
        dest_branch = branch_name("island", dest, tag)
        subprocess.run(["git", "checkout", dest_branch], check=True,
                       capture_output=True, text=True)
        dest_params = extract_hyperparams("train.py")

        diffs = diff_hyperparams(dest_params, source_params)
        if not diffs:
            continue

        n_migrate = max(1, math.ceil(migration_rate * len(diffs)))

        if migration_select == "best_delta":
            to_migrate = diffs[:n_migrate]
        else:
            import random
            to_migrate = random.sample(diffs, min(n_migrate, len(diffs)))

        migrate_params = {d[0]: source_params[d[0]] for d in to_migrate if d[0] in source_params}
        if not migrate_params:
            continue

        apply_hyperparams("train.py", migrate_params)
        param_names = ", ".join(migrate_params.keys())
        subprocess.run(["git", "add", "train.py"], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"[migration] adopted {param_names} from {source}"],
            check=True, capture_output=True, text=True
        )
        print(f"  Migration: {source} → {dest}: {param_names}")
        log_migration(source, dest, migrate_params.keys())


# ---------------------------------------------------------------------------
# Strategy: Multi-Armed Bandit (UCB1)
# ---------------------------------------------------------------------------

def compute_bandit_schedule(config, round_num):
    bandit = config["bandit"]
    arms = bandit["arms"]
    warmup = bandit.get("warmup_rounds", 2)
    exp_per_round = bandit.get("experiments_per_round", 3)
    C = bandit.get("exploration_constant", 1.414)
    lb = config["state"].get("leaderboard", {})
    history = config["state"].get("round_history", [])

    arm_plays = {a["name"]: 0 for a in arms}
    arm_rewards = {a["name"]: 0.0 for a in arms}
    for entry in history:
        name = entry.get("branch")
        if name in arm_plays:
            arm_plays[name] += 1
            arm_rewards[name] += max(0, entry.get("best_delta", 0))

    total_plays = sum(arm_plays.values())

    if total_plays < warmup * len(arms):
        idx = total_plays % len(arms)
        selected = arms[idx]
    else:
        best_ucb = -float('inf')
        selected = arms[0]
        for a in arms:
            n = max(1, arm_plays[a["name"]])
            mean = arm_rewards[a["name"]] / n
            ucb = mean + C * math.sqrt(math.log(total_plays) / n)
            if ucb > best_ucb:
                best_ucb = ucb
                selected = a

    return [(selected["name"], exp_per_round, selected.get("strategy", "general"))]


# ---------------------------------------------------------------------------
# Strategy: Iterated Coopetition
# ---------------------------------------------------------------------------

def compute_coopetition_schedule(config, round_num):
    coop = config["coopetition"]
    branches = coop["branches"]
    exp = coop.get("experiments_per_round", 4)
    return [(b["name"], exp, b.get("focus", "general exploration")) for b in branches]


def run_coopetition_adoption(config):
    coop = config["coopetition"]
    branches = coop["branches"]
    tag = config["tag"]
    lb = config["state"].get("leaderboard", {})
    adoption_count = coop.get("adoption_count", 2)
    adoption_select = coop.get("adoption_select", "best_delta")

    if len(branches) != 2:
        print("  Coopetition requires exactly 2 branches")
        return

    a_name, b_name = branches[0]["name"], branches[1]["name"]
    a_rate = lb.get(a_name, {}).get("improvement_rate", 0)
    b_rate = lb.get(b_name, {}).get("improvement_rate", 0)

    if a_rate == b_rate == 0:
        print("  Adoption: no improvement from either branch, skipping")
        return

    if a_rate >= b_rate:
        winner, loser = a_name, b_name
    else:
        winner, loser = b_name, a_name

    print(f"  Adoption: winner={winner}, loser={loser}")

    winner_branch = branch_name("coopetition", winner, tag)
    subprocess.run(["git", "checkout", winner_branch], check=True,
                   capture_output=True, text=True)
    winner_params = extract_hyperparams("train.py")

    loser_branch = branch_name("coopetition", loser, tag)
    subprocess.run(["git", "checkout", loser_branch], check=True,
                   capture_output=True, text=True)
    loser_params = extract_hyperparams("train.py")

    diffs = diff_hyperparams(loser_params, winner_params)
    if not diffs:
        print("  Adoption: no parameter differences found")
        return

    if adoption_select == "best_delta":
        to_adopt = diffs[:adoption_count]
    elif adoption_select == "worst_own":
        to_adopt = list(reversed(diffs))[:adoption_count]
    else:
        import random
        to_adopt = random.sample(diffs, min(adoption_count, len(diffs)))

    adopt_params = {d[0]: winner_params[d[0]] for d in to_adopt if d[0] in winner_params}
    if not adopt_params:
        return

    apply_hyperparams("train.py", adopt_params)
    param_names = ", ".join(adopt_params.keys())
    subprocess.run(["git", "add", "train.py"], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"[adoption] {loser} adopted {param_names} from {winner}"],
        check=True, capture_output=True, text=True
    )
    print(f"  Adoption: {loser} adopted {param_names} from {winner}")
    log_adoption(winner, loser, adopt_params.keys())


# ---------------------------------------------------------------------------
# Meta-agent (Layer 3)
# ---------------------------------------------------------------------------

def meta_agent_step(config):
    print("\n=== Meta-Agent Step ===")
    lb = config["state"].get("leaderboard", {})
    history = config["state"].get("round_history", [])

    prompt = f"""You are a meta-research agent reviewing the performance of a multi-branch
research optimization system. Your job is to tune the game parameters to improve
overall research performance.

Current game configuration:
  Mode: {config['mode']}
  Round: {config['state']['current_round']}

Leaderboard:
"""
    for name, entry in lb.items():
        bpb = entry.get("best_val_bpb", "N/A")
        rate = entry.get("improvement_rate", 0)
        prompt += f"  {name}: best_bpb={bpb}, improvement_rate={rate:.6f}\n"

    prompt += f"""
Recent round history (last 5):
"""
    for entry in history[-5:]:
        prompt += f"  Round {entry.get('round')}: {entry.get('branch')} — delta={entry.get('best_delta', 0):.6f}\n"

    mode = config["mode"]
    if mode == "island":
        prompt += f"""
Current island parameters:
  migration_rate: {config['island']['migration_rate']}
  adaptive_allocation: {config['island']['adaptive_allocation']}
  min_budget_ratio: {config['island']['min_budget_ratio']}

You may adjust: migration_rate (0.0-1.0), min_budget_ratio (0.0-1.0).
Respond with YAML like:
  migration_rate: 0.4
  min_budget_ratio: 0.3
Or respond with "no changes" if the current settings are working well.
"""
    elif mode == "bandit":
        prompt += f"""
Current bandit parameters:
  exploration_constant: {config['bandit']['exploration_constant']}

You may adjust: exploration_constant (0.1-3.0).
Respond with YAML like:
  exploration_constant: 1.8
Or respond with "no changes" if the current settings are working well.
"""

    print(f"  Sending prompt to meta-agent ({len(prompt)} chars)")

    try:
        result = subprocess.run(
            ["claude", "--print", prompt],
            capture_output=True, text=True, timeout=120
        )
        response = result.stdout.strip()
        print(f"  Meta-agent response: {response[:200]}")

        if "no changes" in response.lower():
            print("  Meta-agent: no changes recommended")
            return

        try:
            updates = yaml.safe_load(response)
            if isinstance(updates, dict):
                mode_config = config.get(mode, {})
                applied = {}
                for key, value in updates.items():
                    if key in mode_config:
                        print(f"  Updating {mode}.{key}: {mode_config[key]} → {value}")
                        mode_config[key] = value
                        applied[key] = value
                if applied:
                    log_meta_agent(applied)
        except yaml.YAMLError:
            print("  Meta-agent: could not parse response as YAML, skipping")

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Meta-agent error: {e}")


# ---------------------------------------------------------------------------
# Base mode
# ---------------------------------------------------------------------------

def run_base_mode(config):
    print("=== Base Mode: Original autoresearch loop ===")
    print("Delegating to run-dgx-agent.sh...")
    os.execvp("bash", ["bash", "run-dgx-agent.sh"])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def compute_schedule(config, round_num):
    mode = config["mode"]
    if mode == "island":
        return compute_island_schedule(config, round_num)
    elif mode == "bandit":
        return compute_bandit_schedule(config, round_num)
    elif mode == "coopetition":
        return compute_coopetition_schedule(config, round_num)
    return []


def main():
    parser = argparse.ArgumentParser(description="autoresearch meta-research orchestrator")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to game config YAML")
    parser.add_argument("--mode", choices=["base", "island", "bandit", "coopetition"],
                        help="Override mode in config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
        save_config(config, args.config)

    mode = config["mode"]
    print(f"=== autoresearch-dgx Orchestrator ===")
    print(f"  Mode: {mode}")
    print(f"  Tag: {config['tag']}")
    print(f"  Config: {args.config}")

    if mode == "base":
        run_base_mode(config)
        return

    log_orchestrator_start(mode, config["tag"], args.config)

    print(f"\nSetting up branches...")
    setup_branches(config)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nStarting game loop (Ctrl+C to stop)...")
    print(f"  Event log:    logs/events.jsonl")
    print(f"  Transcripts:  logs/transcripts/")

    try:
        while True:
            round_num = config["state"]["current_round"] + 1
            print(f"\n{'='*70}")
            print(f"  ROUND {round_num}")
            print(f"{'='*70}")

            schedule = compute_schedule(config, round_num)
            round_start_time = time.time()
            log_round_start(round_num, schedule)

            prev_bpbs = {}
            for b in get_branch_list(config):
                entry = config["state"].get("leaderboard", {}).get(b["name"], {})
                prev_bpbs[b["name"]] = entry.get("best_val_bpb")

            for branch_short, num_experiments, focus in schedule:
                swap_results_in(branch_short)
                run_branch_round(config, branch_short, num_experiments, focus)
                swap_results_out(branch_short)
                update_leaderboard(config, branch_short)

                curr_bpb = config["state"]["leaderboard"].get(branch_short, {}).get("best_val_bpb")
                prev_bpb = prev_bpbs.get(branch_short)
                delta = (prev_bpb - curr_bpb) if (prev_bpb and curr_bpb) else 0

                config["state"].setdefault("round_history", []).append({
                    "round": round_num,
                    "branch": branch_short,
                    "experiments": num_experiments,
                    "best_delta": delta,
                })

            if mode == "island":
                print("\n--- Migration Phase ---")
                run_island_migration(config)
            elif mode == "coopetition":
                print("\n--- Adoption Phase ---")
                run_coopetition_adoption(config)

            if (config.get("meta_agent", {}).get("enabled") and
                    round_num % config["meta_agent"].get("interval_rounds", 5) == 0):
                meta_agent_step(config)

            config["state"]["current_round"] = round_num
            save_config(config)
            print_leaderboard(config)

            log_round_end(round_num, time.time() - round_start_time)
            gb, gbpb = get_global_best(config)
            log_leaderboard(config["state"].get("leaderboard", {}), gb, gbpb)

    except KeyboardInterrupt:
        print("\n\nOrchestrator stopped by user.")
        save_config(config)
        print_leaderboard(config)
        global_branch, global_best = get_global_best(config)
        if global_best:
            print(f"Best result: {global_best:.6f} on branch {global_branch}")


if __name__ == "__main__":
    main()
