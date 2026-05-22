# Autoresearch Agent Benchmark

Three-level evaluation suite for comparing models and harnesses on the autoresearch experiment loop. Runs inside a Docker container with all adapter dependencies pre-installed.

## Quick Start

```bash
# Build the benchmark image (first time only)
docker build -t autoresearch-bench benchmark/

# Level 1: Edit quality (no GPU needed)
bash benchmark/run-bench.sh edit-quality --models qwen3.6:27b gemma4:26b --trials 10

# Level 2: Harness comparison
bash benchmark/run-bench.sh harness --adapters ollama_raw aider --model qwen3.6:27b --trials 10

# Level 3: End-to-end (needs GPU)
BENCH_GPU=1 bash benchmark/run-bench.sh e2e --harness hyp --model qwen3.6:27b --budget 10

# Interactive shell inside the container
bash benchmark/run-bench.sh shell
```

## Prerequisites

- Docker installed
- For Level 3: GPU with Docker GPU support
- Model weights cached on the host (`~/.ollama/models`) for fast startup (optional — models are pulled automatically if missing)

## Levels

### Level 1: Edit Quality (no GPU)

Tests whether a model can produce valid, applicable, syntactically correct edits to train.py via the raw Ollama API (hypothesis generator pattern).

```bash
bash benchmark/run-bench.sh edit-quality \
    --models qwen3.6:27b gemma4:26b qwen2.5-coder:14b \
    --trials 10
```

**Measures:** JSON validity, schema compliance, edit applicability, syntax correctness, meaningfulness.

### Level 2: Harness Comparison (mostly no GPU)

Compares different agent frameworks on the same task with the same model.

```bash
bash benchmark/run-bench.sh harness \
    --adapters ollama_raw aider claude_code openhands \
    --model qwen3.6:27b \
    --trials 20
```

**Adapters (all pre-installed in the container):**

| Adapter | Framework | Approach |
|---|---|---|
| `ollama_raw` | Raw Ollama API | Hypothesis generator — structured JSON, no agent framework |
| `aider` | Aider CLI | Git-native pair programmer with Ollama support |
| `claude_code` | Claude Code | Full agent with Edit/Bash/Read tool use |
| `openhands` | OpenHands | Autonomous agent in Docker sandbox |

Unavailable adapters are automatically skipped.

### Level 3: End-to-End (requires GPU)

Full experiment loop: launches the hypothesis generator or full agent, runs N experiments, measures val_bpb improvement.

```bash
BENCH_GPU=1 bash benchmark/run-bench.sh e2e \
    --harness hyp \
    --model qwen3.6:27b \
    --budget 20
```

**Harnesses:** `hyp` (run-dgx-local.sh) or `agent` (run-dgx-agent.sh).

## Dashboard

Generate an HTML dashboard from benchmark results:

```bash
bash benchmark/run-bench.sh dashboard
```

Opens `benchmark/results/dashboard.html` — a self-contained page with Chart.js visualizations:
- **Level 1:** Grouped bar chart of success rates by model (JSON, schema, apply, syntax)
- **Level 2:** Dual-axis chart comparing harness success rate and latency
- **Level 3:** Table with baseline/best BPB, improvement percentage, wall time

Use `--no-open` to generate without opening in a browser.

## Docker Details

The benchmark Dockerfile (`benchmark/Dockerfile`) bundles:
- Python 3.12
- Ollama (runs inside the container — no host installation needed)
- Aider (`pip install aider-chat`)
- Claude Code (`npm install -g @anthropic-ai/claude-code`)
- OpenHands CLI
- Docker client (for OpenHands adapter, via socket mount)

The entrypoint automatically starts Ollama, waits for readiness, and pulls any models specified via `OLLAMA_MODEL`, `--model`, or `--models` flags before running the benchmark command.

Host model weights are mounted at `~/.ollama/models` for fast startup — if a model is already cached, the pull is a no-op.

```bash
# Manual docker run (equivalent to run-bench.sh):
docker run --rm -it \
    -v $(pwd):/workspace \
    -v ~/.ollama/models:/root/.ollama/models \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e OLLAMA_MODEL=qwen3.6:27b \
    autoresearch-bench \
    python benchmark/bench_edit_quality.py --models qwen3.6:27b --trials 5
```

## Results

Results are written to `benchmark/results/` as TSV files (gitignored). The results directory is mounted from the host, so results persist across container runs.

## Fixtures

`benchmark/fixtures/` contains frozen test data for reproducible benchmarks:
- `train_baseline.py` — snapshot of train.py at a known state
- `results_empty.tsv` — header-only (tests first-run / baseline behavior)
- `results_5_experiments.tsv` — 5 experiments (tests mid-run strategy selection)
