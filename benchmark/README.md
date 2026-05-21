# Autoresearch Agent Benchmark

Three-level evaluation suite for comparing models and harnesses on the autoresearch experiment loop.

## Levels

### Level 1: Edit Quality (no GPU)

Tests whether a model can produce valid, applicable, syntactically correct edits to train.py via the raw Ollama API.

```bash
python benchmark/bench_edit_quality.py \
    --models qwen3.6:27b gemma4:26b qwen2.5-coder:14b \
    --trials 10
```

**Measures:** JSON validity, schema compliance, edit applicability, syntax correctness, meaningfulness.

### Level 2: Harness Comparison (mostly no GPU)

Compares different agent frameworks on the same task with the same model.

```bash
python benchmark/bench_harness.py \
    --adapters ollama_raw aider claude_code openhands \
    --model qwen3.6:27b \
    --trials 20
```

**Adapters:**
| Adapter | Framework | Install |
|---|---|---|
| `ollama_raw` | Raw Ollama API (hypothesis generator) | Built-in |
| `aider` | Aider CLI | `pip install aider-chat` |
| `claude_code` | Claude Code | `npm install -g @anthropic-ai/claude-code` |
| `openhands` | OpenHands (Docker) | `docker pull ghcr.io/openhands/openhands` |

Unavailable adapters are automatically skipped.

### Level 3: End-to-End (requires GPU)

Full experiment loop: launches the hypothesis generator or full agent, runs N experiments, measures val_bpb improvement.

```bash
python benchmark/bench_e2e.py \
    --harness hyp \
    --model qwen3.6:27b \
    --budget 20
```

**Harnesses:** `hyp` (run-dgx-local.sh) or `agent` (run-dgx-agent.sh).

## Prerequisites

- Ollama running locally (`ollama serve`)
- Models pulled (`ollama pull qwen3.6:27b`)
- For Level 3: DGX Spark or equivalent GPU + Docker

## Results

Results are written to `benchmark/results/` as TSV files (gitignored).

## Fixtures

`benchmark/fixtures/` contains frozen test data:
- `train_baseline.py` — snapshot of train.py for reproducible benchmarks
- `results_empty.tsv` — header-only (tests first-run behavior)
- `results_5_experiments.tsv` — 5 experiments (tests mid-run behavior)
