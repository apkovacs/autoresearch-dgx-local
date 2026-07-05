# DGX Spark Detailed Setup Guide

## Prerequisites

### Hardware
- NVIDIA DGX Spark (GB10 Blackwell GPU, 128 GB unified memory, ARM64 Grace CPU)

### Software
- Docker Engine 24.0+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- Git

### Verify NVIDIA Container Toolkit

```bash
# Check installation
nvidia-ctk --version

# Test GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If the test fails, install the toolkit:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Persistent Shard Storage

Training data shards (~50 MB each, ~6500 total available) are downloaded from HuggingFace. By default, 10 shards are downloaded on first run. These are persisted on the host filesystem via Docker volume mounts.

### How It Works

The `prepare.py` script stores shards at the path set by `AUTORESEARCH_CACHE_DIR`. The Docker launch scripts mount a host directory to this path:

```
Host:      ~/.cache/autoresearch/     (configurable via SHARD_CACHE_DIR)
Container: /cache/autoresearch/       (set via AUTORESEARCH_CACHE_DIR env var)
```

On container restart, existing shards are detected and download is skipped.

### Custom Shard Location

```bash
SHARD_CACHE_DIR=/data/autoresearch bash run-dgx.sh
```

### Reusing Shards from an Existing Container

If you have a running or stopped container that already downloaded shards (e.g., from the original repo or the David-Barnes fork):

```bash
# Find the container
docker ps -a --format '{{.Names}}'

# Copy shards from the container to your host
docker cp <container-name>:/root/.cache/autoresearch $HOME/.cache/autoresearch

# Now run-dgx.sh will find them automatically
bash run-dgx.sh
```

If the shards are at a different path inside the container:

```bash
# Check where they are
docker exec <container-name> find / -name "shard_*.parquet" -type f 2>/dev/null | head -5

# Copy the parent directory
docker cp <container-name>:<path-to-data-dir> $HOME/.cache/autoresearch/data
```

### Verify Persistence

```bash
# Run once to download shards
bash run-dgx.sh --test

# Check host directory
ls -la ~/.cache/autoresearch/data/
# Should show shard_*.parquet files

# Run again — should say "all shards already downloaded"
bash run-dgx.sh
```

---

## Local LLM Agent Setup

The autonomous experiment loop runs entirely on-device using:
- **Ollama**: Serves the local LLM with GPU acceleration
- **Claude Code**: Agent framework that drives the experiment loop
- **program.md**: Instructions the agent follows to modify, train, evaluate, and iterate

### How It Works

Claude Code connects to Ollama via environment variables that redirect its API calls to the local server:

```
ANTHROPIC_BASE_URL=http://localhost:11434
ANTHROPIC_AUTH_TOKEN=ollama
ANTHROPIC_API_KEY=ollama
```

This approach is based on [this guide](https://medium.com/@luongnv89/run-claude-code-on-local-cloud-models-in-5-minutes-ollama-openrouter-llama-cpp-6dfeaee03cda).

### Model Selection Guide

| Model | Ollama Tag | Size (Q4) | Memory Use | Reliability | Notes |
|---|---|---|---|---|---|
| **Qwen3.6 27B** | `qwen3.6:27b` | ~18 GB | Default | High | Strong code reasoning, fewest edit failures |
| Gemma 4 26B | `gemma4:26b` | ~18 GB | | High | Strong general + code; occasional edit imprecision |
| Gemma 4 E4B | `gemma4:e4b` | ~10 GB | | Medium | More memory headroom; more edit retries needed |
| Gemma 4 E2B | `gemma4:e2b` | ~7 GB | | Low | Lightweight; struggles with multi-step loops |
| Qwen 2.5 Coder 14B | `qwen2.5-coder:14b` | ~8 GB | | Medium | Precise edits but weaker experiment reasoning |
| Qwen3 8B | `qwen3:8b` | ~5 GB | | Low | Lightweight; may not sustain the experiment loop |

Any Ollama-compatible model works — just set `OLLAMA_MODEL` to its tag.

### Choosing Your Agent Mode

Three modes span the spectrum from maximum scaffolding to Karpathy's original design:

| Scenario | Recommended mode | Command | Dependencies |
|---|---|---|---|
| Highly capable local models (e.g. DeepSeek V4 Flash) | Minimal agent | `run-dgx-agent.sh --mode minimal` | Claude Code + Ollama |
| Frontier API (Claude, GPT-4) | Minimal agent | `run-dgx-agent.sh --mode minimal` | Claude Code + Ollama |
| Large local (27B+, reliable tool use) | Guarded agent | `run-dgx-agent.sh` | Claude Code + Ollama |
| Local models with reliability issues | Hypothesis generator | `run-dgx-local.sh` | Ollama only |

**Minimal agent mode** (`run-dgx-agent.sh --mode minimal`): Karpathy's original design — program.md drives everything, the model runs with all permissions granted, no output token cap, and a facts-only CLAUDE.md. The infrastructure that's invisible to the model (session restarts, deterministic logging, `run_experiment.sh` heartbeat) stays. Use the trace-quality benchmark (`bash benchmark/run-bench.sh trace`) to verify a model is capable enough for this mode before committing GPU time.

**Guarded agent mode** (`run-dgx-agent.sh`, default): Same full agent loop, but with behavioral guardrails for smaller local models: an output token cap that breaks repetitive thinking loops, an action-first CLAUDE.md with explicit rules, and a narrow permission allowlist.

**Hypothesis generator mode** (`run-dgx-local.sh`): The LLM only proposes edits as structured JSON. Everything else (git, training, logging, keep/revert) is handled deterministically by the script. No Claude Code or Node.js needed. Higher reliability — no permission denials, no tool confusion, no repetitive thinking loops. Lower flexibility — the agent can't inspect training logs or do multi-step reasoning.

```bash
# Minimal agent mode (original design, for highly capable models)
bash run-dgx-agent.sh --mode minimal

# Guarded agent mode (default, for capable local models)
bash run-dgx-agent.sh

# Hypothesis generator mode (recommended for smaller local models)
bash run-dgx-local.sh
bash run-dgx-local.sh --max-experiments 100
```

### Custom GGUF Models (e.g. DeepSeek V4 Flash)

Community quants not published in the Ollama library can be imported from a local GGUF file with `OLLAMA_GGUF`:

```bash
# DeepSeek V4 Flash Dwarf Star quant (~81GB, ~2.3 bits/param) in minimal mode
OLLAMA_GGUF=~/models/deepseek-v4-flash-dwarf.gguf \
OLLAMA_MODEL=deepseek-v4-flash-dwarf \
OLLAMA_NUM_CTX=32768 \
bash run-dgx-agent.sh --mode minimal
```

The file is mounted read-only into the container and imported via `ollama create` (a one-time cost — the model store is a persistent volume, so later runs detect the existing model and skip the import). `OLLAMA_NUM_CTX` sets the context window (default 32768; Ollama's own default of 4096 is far too small for the agent loop).

Note the memory math: an ~81GB model plus training leaves little headroom on 128GB unified memory. Keep `OLLAMA_KEEP_ALIVE=0` (the default) so the model unloads during training runs.

### Memory Budget

With 128 GB unified memory:

| Component | Memory | Notes |
|---|---|---|
| Training (DEPTH=4, batch=8) | ~1 GB | Model + optimizer + activations |
| LLM model (varies) | 5–18 GB | Depends on model choice |
| Ollama server | ~1 GB | Runtime overhead |
| System / OS | ~2–4 GB | |
| **Available headroom** | **104–120 GB** | Comfortable margin |

By default, `OLLAMA_KEEP_ALIVE=0` causes Ollama to unload the model from GPU memory immediately after each agent reasoning turn. During the 5-minute training run, the ~18 GB used by the LLM is freed for PyTorch. The model reloads (~10–30s) when the agent needs to reason again. Set `OLLAMA_KEEP_ALIVE=30s` to keep the model warm for quick back-to-back reasoning, at the cost of less training headroom.

If you increase training parameters (larger DEPTH or DEVICE_BATCH_SIZE), the memory freed by model unloading becomes especially valuable.

### Model Weight Persistence

Ollama model weights are persisted via volume mount:

```
Host:      ~/.ollama/models/          (configurable via OLLAMA_MODELS)
Container: /root/.ollama/models/
```

First run downloads the model weights (~16–18 GB for 27B models). Subsequent runs use the cached weights.

### Using a Different Model

```bash
# One-time: different model
OLLAMA_MODEL=gemma4:26b bash run-dgx-agent.sh

# Permanent: set in shell profile
echo 'export OLLAMA_MODEL=gemma4:26b' >> ~/.bashrc
source ~/.bashrc
bash run-dgx-agent.sh
```

### Using Untested Models

Any model in the [Ollama library](https://ollama.com/library) should work:

```bash
OLLAMA_MODEL=deepseek-coder-v2:16b bash run-dgx-agent.sh
OLLAMA_MODEL=codestral:22b bash run-dgx-agent.sh
OLLAMA_MODEL=llama3.1:70b bash run-dgx-agent.sh  # if you have the memory
```

The model needs to handle code modification and instruction following. Models under ~8B parameters may struggle with the complexity of the experiment loop.

### Local Model Limitations and Mitigations

Running the experiment loop with locally-hosted models (rather than frontier API models) surfaces predictable failure modes. The launcher scripts include mitigations for all of these, but understanding them helps when choosing a model or debugging agent behavior.

**Edit tool accuracy.** Smaller models frequently get the `old_string` wrong when using the Edit tool — introducing extra whitespace, misquoting comments, or changing punctuation slightly. This causes "String to replace not found" errors and wastes context on retries. The `run_experiment.sh` wrapper syntax-checks `train.py` before training so a bad edit never wastes a 5-minute training run, and `revert_train.sh` provides one-command recovery.

**Context window exhaustion.** Local models have limited context windows (typically 32K-128K tokens). Context fills fast when the agent: reads verbose git log output (~40K tokens), reads `run.log` via `tail` (60-73KB due to `\r` carriage returns in training progress), or retries failed edits repeatedly. Mitigations: git log is restricted to `--oneline` in the permission list; `run_experiment.sh` converts `\r` to `\n` in `run.log` after training; the context compaction watchdog restarts the agent every N experiments for a fresh window.

**Premature exit.** Smaller models sometimes describe their plan then stop (exit code 0) instead of executing it. The auto-restart loop treats any exit as premature (since the agent is instructed to never stop) and relaunches with a forward-looking resume prompt that injects only the best `val_bpb` as a target to beat.

**Anxious polling.** Without periodic output, the agent may assume a long-running command has completed and start polling `run.log` for results. `run_experiment.sh` prints a heartbeat every 30 seconds showing training progress percentage, keeping the agent informed that training is still running.

**Whitespace and formatting errors.** Local models occasionally introduce literal tabs, trailing whitespace, or other formatting issues when editing Python code. The syntax-check step in `run_experiment.sh` catches these immediately (exit code 2), and the CLAUDE.md instructions direct the agent to use `revert_train.sh` to recover.

**Repetitive thinking loops.** Some models (especially Gemma 4 26B) can enter a degenerate loop where the model's internal reasoning repeats the same sentence thousands of times, consuming the entire output budget without emitting a tool call. The launcher creates a capped model alias with `num_predict 16384` to break these loops. If a model consistently produces zero experiments across multiple restarts, the zero-progress detector warns and suggests switching models.

**Going in circles.** When an edit fails, smaller models sometimes retry the exact same approach repeatedly. The context compaction watchdog helps by restarting the agent with a clean context after a configurable number of experiments, breaking the cycle.

**Recommended models for reliability:** Qwen3.6 27B and Gemma 4 26B are the most reliable for the experiment loop. Models under 14B parameters show noticeably more edit failures and premature exits. Code-specialized models (Qwen 2.5 Coder 14B) can be more precise with edits but may be weaker at experiment reasoning.

---

## Parameter Tuning

The default parameters are conservative. If training runs stably, you can try increasing them:

| Parameter | Default | Try | Effect |
|---|---|---|---|
| `DEPTH` | 4 | 6 or 8 | Larger model, potentially better val_bpb |
| `DEVICE_BATCH_SIZE` | 8 | 16 or 32 | More tokens per step, faster training |
| `TOTAL_BATCH_SIZE` | 2^16 | 2^17 or 2^18 | More grad accum, smoother updates |

**Important**: `TOTAL_BATCH_SIZE` must be divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN` (= `DEVICE_BATCH_SIZE * 2048`).

Monitor memory usage with `bash monitor-dgx.sh`. If you see memory pressure, reduce parameters before OOM freezes the system.

---

### Pre-built Docker Image

Building a custom image bakes in all dependencies (Python packages, Ollama, Claude Code, Node.js) so container launches take seconds instead of minutes:

```bash
docker build -t autoresearch-dgx-local .
DOCKER_IMAGE=autoresearch-dgx-local bash run-dgx-agent.sh
DOCKER_IMAGE=autoresearch-dgx-local bash run-dgx-game.sh --mode island
```

The launcher scripts automatically detect the pre-built image and skip installation steps.

---

## Running the Agent Workflow

The agent follows `program.md`, which instructs it to:

1. Read `train.py` and identify hyperparameters to tune
2. Make a change and commit it
3. Run training (`bash run_experiment.sh`)
4. Evaluate the result (`val_bpb`)
5. Log results (`bash log_result.sh`)
6. Keep or discard the change based on improvement (git reset for discards)
7. Repeat indefinitely until manually stopped

The agent logs results to `results.tsv` with columns: commit, val_bpb, memory_gb, status, description, and timestamp. Three wrapper scripts handle sandbox restrictions and provide safety nets:
- `run_experiment.sh` — syntax-checks `train.py`, runs training with heartbeat progress output, cleans `run.log`, and backs up the last working version
- `log_result.sh` — appends results to `results.tsv` with UTC timestamp
- `revert_train.sh` — restores `train.py` from the last successful training run

The agent auto-restarts on any exit (configurable with `--max-restarts N` or `--no-restart`). A context compaction watchdog restarts the agent every 30 experiments (configurable with `--experiments-per-session N`) for a fresh context window.

Stop the agent with `Ctrl+C` or `docker stop autoresearch-dgx-local-agent`. All committed experiments are preserved in git history.

---

## Monitoring

All monitoring runs from a separate terminal on the host while the agent is running:

```bash
bash monitor-game.sh                    # live dashboard with leaderboard
bash monitor-game.sh --status           # one-shot snapshot
bash monitor-game.sh --transcript       # agent activity + training progress
bash monitor-game.sh --events           # orchestrator event stream
bash monitor-game.sh --transcript-raw   # raw stream-json output
```

The `--status` mode shows a quick snapshot of experiments completed, GPU utilization, git state, and whether the agent is actively running. The `--transcript` mode shows agent thinking, tool calls, and live training step/loss progress during the 5-minute training runs.
