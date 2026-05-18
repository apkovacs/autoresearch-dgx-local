"""
Structured event logging for the orchestrator.

Writes JSON-lines to logs/events.jsonl. Each line is a self-contained event
with a timestamp, event type, and payload. Designed to be tailed in real time
or parsed after the fact.

Event types:
    orchestrator_start   Game loop launched
    round_start          New round beginning
    round_end            Round complete
    branch_start         Agent starting on a branch
    branch_progress      Periodic experiment count update
    branch_end           Agent finished on a branch
    migration            Island model migration event
    adoption             Coopetition adoption event
    meta_agent           Meta-agent parameter update
    leaderboard          Leaderboard snapshot after a round
    error                Something went wrong
"""

import json
import os
import time
from datetime import datetime, timezone

LOGS_DIR = "logs"
EVENT_LOG = os.path.join(LOGS_DIR, "events.jsonl")


def _ensure_log_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def log_event(event_type, **kwargs):
    """Append a structured event to the event log."""
    _ensure_log_dir()
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.monotonic(),
        "event": event_type,
        **kwargs,
    }
    with open(EVENT_LOG, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def log_orchestrator_start(mode, tag, config_path):
    log_event("orchestrator_start", mode=mode, tag=tag, config=config_path)


def log_round_start(round_num, schedule):
    branches = [{"branch": b, "experiments": n, "focus": f} for b, n, f in schedule]
    log_event("round_start", round=round_num, schedule=branches)


def log_round_end(round_num, duration_s):
    log_event("round_end", round=round_num, duration_s=round(duration_s, 1))


def log_branch_start(branch, round_num, num_experiments, focus):
    log_event("branch_start", branch=branch, round=round_num,
              target_experiments=num_experiments, focus=focus)


def log_branch_progress(branch, round_num, completed, target, elapsed_s):
    log_event("branch_progress", branch=branch, round=round_num,
              completed=completed, target=target, elapsed_s=round(elapsed_s, 1))


def log_branch_end(branch, round_num, completed, duration_s):
    log_event("branch_end", branch=branch, round=round_num,
              completed=completed, duration_s=round(duration_s, 1))


def log_migration(source, dest, params):
    log_event("migration", source=source, dest=dest, params=list(params))


def log_adoption(winner, loser, params):
    log_event("adoption", winner=winner, loser=loser, params=list(params))


def log_meta_agent(updates):
    log_event("meta_agent", updates=updates)


def log_leaderboard(leaderboard, global_best_branch, global_best_bpb):
    log_event("leaderboard", entries=leaderboard,
              global_best_branch=global_best_branch,
              global_best_bpb=global_best_bpb)


def log_error(message, **context):
    log_event("error", message=message, **context)
