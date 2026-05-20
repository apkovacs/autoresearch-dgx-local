# autoresearch — bounded round

This is an experiment to have the LLM do its own research.

## Research Focus

Your research focus for this round: {focus_description}

Prioritize experiments in this area. You may still explore adjacent ideas if they seem promising, but most of your experiments should target your assigned focus.

## Setup

The branch and results.tsv are already set up by the orchestrator. Read these files for context:
- `README.md` — repository context.
- `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
- `train.py` — the file you modify. Model architecture, optimizer, training loop.

Data is already prepared at `/cache/autoresearch` (also symlinked to `~/.cache/autoresearch/`). Do NOT run prepare.py or verify data — it is ready.

Review `results.tsv` for previous experiment results on this branch.

Once oriented, begin experimenting immediately.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: If results.tsv has no data rows yet, your first run should establish the baseline by running the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_bpb	memory_gb	status	description	timestamp
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried
6. timestamp — added automatically by `log_result.sh`

## The experiment loop

Run exactly **{max_experiments} experiments** (counting rows you add to results.tsv). After completing {max_experiments} experiments, print "ROUND_COMPLETE" and stop.

**Safety nets**: `bash run_experiment.sh` automatically syntax-checks `train.py` before running. If there is a syntax error, it exits immediately with code 2. If you break `train.py` and can't fix it, run `bash revert_train.sh` to restore the last working version.

For each experiment, follow these steps EXACTLY:

1. Edit `train.py` with your experimental idea (use the Edit tool, NOT python3 -c)
2. `git add train.py && git commit -m "description of change"`
3. Run: `bash run_experiment.sh`
4. If exit code is 2 (syntax error): run `bash revert_train.sh`, then fix your edit and retry
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it, run `bash revert_train.sh` and move on.
7. **MANDATORY — Log the result to results.tsv** using the wrapper script:
   `bash log_result.sh COMMIT VAL_BPB MEM_GB STATUS DESCRIPTION`
   Example: `bash log_result.sh a1b2c3d 1.879972 7.6 keep baseline`
   Do NOT use `>>` redirection — it is blocked by the sandbox. Do NOT skip this step.
   (Do not commit results.tsv — leave it untracked by git)
8. If val_bpb IMPROVED (lower): keep the commit, move on
9. If val_bpb is equal or worse: **run `git reset --hard HEAD~1`** to revert. Do NOT manually undo code changes.

**CRITICAL**: You MUST update results.tsv after EVERY experiment. You MUST use `git reset --hard HEAD~1` to revert failed experiments. If train.py is broken, run `bash revert_train.sh`.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Do NOT ask the human anything.** You are autonomous for this round. Complete your {max_experiments} experiments, then print "ROUND_COMPLETE" and stop.
