# DGX Spark Quick Start

## Prerequisites

- NVIDIA DGX Spark with Docker and NVIDIA Container Toolkit installed
- ~2 GB disk space for training shards (first run downloads ~10 shards)
- ~18 GB disk space for LLM model weights (agent mode only)

## Training Only

Run a single training experiment:

```bash
git clone https://github.com/apkovacs/autoresearch-dgx-local.git
cd autoresearch
git checkout dgx-spark
bash run-dgx.sh
```

The first run downloads training shards (~5 min) and trains a tokenizer. Subsequent runs skip this step thanks to persistent shard storage.

Monitor in another terminal:

```bash
bash monitor-dgx.sh
```

## Full Autonomous Agent

Run the autonomous experiment loop with a local LLM (no external APIs):

```bash
git clone https://github.com/apkovacs/autoresearch-dgx-local.git
cd autoresearch
git checkout dgx-spark
bash run-dgx-agent.sh
```

This starts Ollama with Qwen3.6 27B (default), installs Claude Code, and begins the experiment loop defined in `program.md`.

### Use a Different Model

```bash
OLLAMA_MODEL=gemma4:26b bash run-dgx-agent.sh     # Gemma 4 26B
OLLAMA_MODEL=gemma4:e4b bash run-dgx-agent.sh     # Gemma 4 E4B (more memory headroom)
OLLAMA_MODEL=qwen2.5-coder:14b bash run-dgx-agent.sh  # Code-specialized
```

Run `bash run-dgx-agent.sh --help` to see all tested models.

### Pre-built Docker Image (Faster Startup)

Build once to skip dependency installation on every launch:

```bash
docker build -t autoresearch-dgx-local .
DOCKER_IMAGE=autoresearch-dgx-local bash run-dgx-agent.sh
```

## Multi-Branch Game Strategies

Run multiple competing/cooperating research branches:

```bash
# Island Model — 3 branches with migration
bash run-dgx-game.sh --mode island

# Multi-Armed Bandit — UCB1 arm selection
bash run-dgx-game.sh --mode bandit

# Iterated Coopetition — 2 branches, forced adoption
bash run-dgx-game.sh --mode coopetition
```

Configure branch count, migration rates, and other parameters in `game_config.yaml`. See [Game Strategies](GAME_STRATEGIES.md) for details.

## Verify Setup Without Training

```bash
bash run-dgx.sh --test
```

This checks Docker, GPU access, PyTorch, and the cache directory without running any training.

## Monitoring

```bash
bash monitor-game.sh                    # live dashboard (leaderboard, recent events)
bash monitor-game.sh --status           # one-shot snapshot (experiments, GPU, git)
bash monitor-game.sh --transcript       # agent thinking + tool calls + training progress
bash monitor-game.sh --events           # orchestrator event stream
bash monitor-game.sh --transcript-raw   # raw stream-json
```

Stop a running container from another terminal:

```bash
docker stop autoresearch-dgx-local-agent      # or autoresearch-dgx-local-game
```

## What to Expect

- **Startup**: ~2 min for Docker pull + dependency install (first run), ~10 sec with pre-built image
- **Shard download**: ~5 min for 10 shards (first run only — cached after)
- **Compilation**: First ~10 training steps are `torch.compile` warmup (slow, expected)
- **Steady state**: Training log shows step number, loss, tokens/sec, MFU%, and remaining time
- **Completion**: After 5 minutes of training time, final `val_bpb` is printed

## Next Steps

- [Game Strategies](GAME_STRATEGIES.md) — Multi-branch strategy details and tuning
- [Detailed Setup](DGX_SETUP.md) — Persistent storage, model selection, parameter tuning
- [Troubleshooting](DGX_TROUBLESHOOTING.md) — If something goes wrong
- [Overview](DGX_SPARK_README.md) — What changed from the original and why
