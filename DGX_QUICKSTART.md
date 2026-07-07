# DGX Spark Quick Start

## Prerequisites

- NVIDIA DGX Spark with Docker and NVIDIA Container Toolkit installed
- ~2 GB disk space for training shards (first run downloads ~10 shards)
- ~18 GB disk space for LLM model weights (agent mode only)

## Training Only

Run a single training experiment:

```bash
git clone https://github.com/apkovacs/autoresearch-dgx-local.git
cd autoresearch-dgx-local
bash run-dgx.sh
```

The first run downloads training shards (~5 min) and trains a tokenizer. Subsequent runs skip this step thanks to persistent shard storage.

Monitor in another terminal:

```bash
bash monitor-dgx.sh
```

## Hypothesis Generator (Recommended for Local Models)

Deterministic experiment loop — the LLM only proposes edits, everything else is automated:

```bash
git clone https://github.com/apkovacs/autoresearch-dgx-local.git
cd autoresearch-dgx-local
git checkout dgx-spark
bash run-dgx-local.sh                              # default: Qwen3.6 27B
OLLAMA_MODEL=gemma4:26b bash run-dgx-local.sh      # alternative model
bash run-dgx-local.sh --max-experiments 100         # set experiment budget
```

No Claude Code or Node.js needed — just Ollama and Python. Higher reliability than the full agent mode: no permission denials, no tool confusion, no repetitive thinking loops.

The backend doesn't have to be Ollama — any OpenAI-compatible server works (llama-server, vLLM, ds4), which unlocks engine features Ollama lacks, like speculative decoding:

```bash
# Point the loop at a llama-server running on the host
INFERENCE_BACKEND=openai \
INFERENCE_URL=http://host.docker.internal:8080/v1 \
OLLAMA_MODEL=deepseek-v4-flash-dwarf \
bash run-dgx-local.sh
```

See [DGX_SETUP.md](DGX_SETUP.md#alternative-inference-backends-llama-server-vllm-ds4) for details.

## Full Autonomous Agent

Run the full agent loop with Claude Code (best for frontier API models or highly capable local models):

```bash
bash run-dgx-agent.sh
```

This starts Ollama with Qwen3.6 27B (default), installs Claude Code, and begins the experiment loop defined in `program.md`. The default (guarded) mode adds behavioral guardrails for local models; for highly capable models, minimal mode runs Karpathy's original design with no guardrails:

```bash
bash run-dgx-agent.sh --mode minimal
```

### Use a Different Model

```bash
OLLAMA_MODEL=gemma4:26b bash run-dgx-agent.sh     # Gemma 4 26B
OLLAMA_MODEL=gemma4:e4b bash run-dgx-agent.sh     # Gemma 4 E4B (more memory headroom)
OLLAMA_MODEL=qwen2.5-coder:14b bash run-dgx-agent.sh  # Code-specialized

# Import a local GGUF that isn't in the Ollama library (e.g. DeepSeek V4 Flash Dwarf Star)
OLLAMA_GGUF=~/models/deepseek-v4-flash-dwarf.gguf OLLAMA_MODEL=deepseek-v4-flash-dwarf \
    bash run-dgx-agent.sh --mode minimal
```

Run `bash run-dgx-agent.sh --help` to see all tested models.

## Frontier-Scale Local: DeepSeek V4 Flash (Dwarf Star)

The Dwarf Star selective quantization compresses DeepSeek V4 Flash (284B-parameter MoE, 13B active) from 568 GB to ~81 GB — small enough to fit the Spark's 128 GB unified memory. This is the target model for **minimal mode**: it was designed for agentic workflows with strict tool calling, so it should run the original autoresearch design without the guardrails smaller local models need.

Because the quant is a community GGUF release (not in the Ollama library), the launchers import it from a local file:

```bash
# 1. Download the Dwarf Star GGUF (~81 GB) to the Spark, e.g. ~/models/

# 2. Launch in minimal mode — imported into Ollama automatically on first run
OLLAMA_GGUF=~/models/deepseek-v4-flash-dwarf.gguf \
OLLAMA_MODEL=deepseek-v4-flash-dwarf \
bash run-dgx-agent.sh --mode minimal
```

Notes:
- First launch hashes the 81 GB file into the Ollama model store (several minutes); later runs skip it.
- `OLLAMA_NUM_CTX` sets the context window (default 32768). Expect ~15–30 tok/s decode.
- Keep `OLLAMA_KEEP_ALIVE=0` (the default) — the model must unload during training runs, as 81 GB plus PyTorch would exhaust unified memory.
- After a session, run `bash benchmark/run-bench.sh trace --label "deepseek-v4-flash-dwarf/minimal"` to measure how cleanly the model executed the loop compared to guarded-mode models.

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
