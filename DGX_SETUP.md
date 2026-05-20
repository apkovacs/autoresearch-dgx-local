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

| Model | Ollama Tag | Size (Q4) | Memory Use | Strengths |
|---|---|---|---|---|
| **Qwen3.6 27B** | `qwen3.6:27b` | ~18 GB | Default | Strong code reasoning, instruction following |
| Gemma 4 26B | `gemma4:26b` | ~18 GB | | Strong general + code capability |
| Gemma 4 E4B | `gemma4:e4b` | ~10 GB | | Good capability with more memory headroom |
| Gemma 4 E2B | `gemma4:e2b` | ~7 GB | | Lightweight edge model |
| Qwen 2.5 Coder 14B | `qwen2.5-coder:14b` | ~8 GB | | Purpose-built for code modification |
| Qwen3 8B | `qwen3:8b` | ~5 GB | | Lightweight, fast inference |

Any Ollama-compatible model works — just set `OLLAMA_MODEL` to its tag.

### Memory Budget

With 128 GB unified memory:

| Component | Memory | Notes |
|---|---|---|
| Training (DEPTH=4, batch=8) | ~1 GB | Model + optimizer + activations |
| LLM model (varies) | 5–18 GB | Depends on model choice |
| Ollama server | ~1 GB | Runtime overhead |
| System / OS | ~2–4 GB | |
| **Available headroom** | **104–120 GB** | Comfortable margin |

If you increase training parameters (larger DEPTH or DEVICE_BATCH_SIZE), choose a smaller LLM to maintain headroom.

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
docker build -t autoresearch-dgx .
DOCKER_IMAGE=autoresearch-dgx bash run-dgx-agent.sh
DOCKER_IMAGE=autoresearch-dgx bash run-dgx-game.sh --mode island
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

The agent logs results to `results.tsv` with columns: commit, val_bpb, memory_gb, status, and description. Two wrapper scripts handle sandbox restrictions:
- `run_experiment.sh` — runs training and captures output to `run.log`
- `log_result.sh` — appends results to `results.tsv`

Stop the agent with `Ctrl+C` or `docker stop autoresearch-dgx-agent`. All committed experiments are preserved in git history.

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
