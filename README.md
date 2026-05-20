# autoresearch-dgx-local

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for the NVIDIA DGX Spark, with persistent storage, local LLM agent support, and a game-theory-inspired meta-research orchestration framework.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

## What's different from upstream

This fork targets the **NVIDIA DGX Spark** (GB10 Blackwell GPU, 128 GB unified LPDDR5X, ARM64 Grace CPU) and adds three major capabilities on top of the original autoresearch:

1. **DGX Spark adaptation** -- SDPA fallback for flash-attn3, reduced hyperparameters, Docker containerization with OOM protection and persistent shard storage
2. **Local LLM agent** -- Ollama + Claude Code running entirely on-device, configurable model (default: Qwen3.6 27B)
3. **Meta-research orchestration** -- Game-theory-inspired strategies (island model, multi-armed bandit, iterated coopetition) that manage multiple competing/cooperating research branches on a single GPU

Inspired in part by [David-Barnes-Data-Imaginations/autoresearch-DGX-Spark](https://github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark).

## Architecture

```
Layer 3: Meta-Agent (optional)  -->  tunes game parameters
Layer 2: Orchestrator           -->  manages branches, strategies, migration
Layer 1: Research Agent         -->  modifies train.py, runs experiments
Layer 0: Training               -->  PyTorch training loop (train.py)
```

Each layer is independently useful. You can run just the training loop, add the research agent, or enable the full orchestrator with multiple branches and optional meta-agent.

## Quick start

### Training only

```bash
git clone https://github.com/apkovacs/autoresearch-dgx-local.git
cd autoresearch-dgx-local
git checkout dgx-spark
bash run-dgx.sh
```

### Autonomous agent (single branch)

```bash
bash run-dgx-agent.sh
```

This starts Ollama with Qwen3.6 27B, installs Claude Code, and begins the experiment loop defined in `program.md`.

### Multi-branch game strategies

```bash
# Island Model -- 3 branches with migration
bash run-dgx-game.sh --mode island

# Multi-Armed Bandit -- UCB1 arm selection
bash run-dgx-game.sh --mode bandit

# Iterated Coopetition -- 2 branches, forced adoption
bash run-dgx-game.sh --mode coopetition
```

See [DGX_QUICKSTART.md](DGX_QUICKSTART.md) for full setup instructions and [GAME_STRATEGIES.md](GAME_STRATEGIES.md) for strategy details.

## How it works

The core idea is unchanged from upstream: give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You are programming the `program.md` Markdown files that provide context to the AI agents, not touching the Python files directly.

The metric is **val_bpb** (validation bits per byte) -- lower is better, and vocab-size-independent so architectural changes are fairly compared.

### Game strategies

The orchestrator adds a meta-research layer that manages multiple research branches:

| Mode | Branches | Mechanism | Best for |
|---|---|---|---|
| **Base** | 1 | Original autoresearch loop | Baseline, simple experiments |
| **Island** | 3+ | Independent evolution with periodic migration | Broad exploration across optimization axes |
| **Bandit** | 2+ | UCB1 selects which strategy gets compute | Discovering which direction yields most improvement |
| **Coopetition** | 2 | Head-to-head with forced adoption by loser | Focused comparison with knowledge transfer |

### Local LLM support

The agent runs entirely on-device using Ollama. Tested models:

| Model | VRAM headroom | Notes |
|---|---|---|
| Qwen3.6 27B (default) | Moderate | Strong reasoning, good default |
| Gemma 4 27B | Moderate | Competitive alternative |
| Gemma 4 12B | High | More memory for training |
| Qwen 2.5 Coder 14B | High | Code-specialized |

Switch models with:
```bash
OLLAMA_MODEL=gemma4:27b bash run-dgx-agent.sh
```

## DGX Spark adaptations

Key changes from the upstream H100 configuration:

- **SDPA fallback** for flash-attn3 (no ARM64 builds available)
- **Reduced hyperparameters**: `DEPTH=4`, `TOTAL_BATCH_SIZE=2^16`, `DEVICE_BATCH_SIZE=8`
- **MFU constant**: 209 TFLOPS BF16 (GB10 Blackwell, replacing H100's 989 TFLOPS)
- **OOM protection**: `--oom-score-adj 1000` prevents GPU memory pressure from freezing the host (unified memory architecture)
- **Persistent storage**: Training shards and Ollama model weights survive container restarts via Docker volume mounts

## Project structure

```
train.py                    -- model, optimizer, training loop (agent modifies this)
prepare.py                  -- constants, data prep + runtime utilities
program.md                  -- agent instructions (human modifies this)
pyproject.toml              -- dependencies

run-dgx.sh                 -- Docker launcher: training only
run-dgx-agent.sh           -- Docker launcher: autonomous agent (single branch)
run-dgx-game.sh            -- Docker launcher: meta-research orchestrator
monitor-dgx.sh             -- real-time container monitoring
monitor-game.sh            -- live game dashboard, event stream, transcript viewer

orchestrator.py             -- game engine: schedules branches, migration, adoption
hyperparams.py              -- hyperparameter extraction/injection for cross-branch migration
leaderboard.py              -- cross-branch results tracking
event_log.py                -- structured event logging (JSON lines)
game_config.yaml            -- game configuration (mode, branches, strategy params)
branch_templates/           -- prompt templates for bounded experiment rounds
logs/                       -- event log + per-round agent transcripts (gitignored)

DGX_SPARK_README.md         -- overview of DGX Spark changes
DGX_QUICKSTART.md           -- quick start guide
DGX_SETUP.md                -- detailed setup, persistent storage, model selection
DGX_TROUBLESHOOTING.md      -- common issues and fixes
GAME_STRATEGIES.md          -- strategy documentation with tuning tips
```

## Design choices

Inherited from upstream:
- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes of wall clock time, making experiments directly comparable regardless of what the agent changes.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages.

Added in this fork:
- **Docker-first.** All execution happens inside containers with pinned images (`nvcr.io/nvidia/pytorch:25.12-py3`), ensuring reproducibility.
- **Persistent by default.** Training shards and model weights are cached on the host, surviving container restarts.
- **Strategies are deterministic.** The orchestrator (branch scheduling, migration, adoption) is plain Python with no LLM calls. Only the research agent and optional meta-agent use the LLM.
- **Git branches as isolation.** Each research branch is a real git branch, making it easy to inspect, compare, and cherry-pick results.

## Documentation

| Document | Description |
|---|---|
| [DGX_QUICKSTART.md](DGX_QUICKSTART.md) | Get running in under 5 minutes |
| [DGX_SETUP.md](DGX_SETUP.md) | Persistent storage, model selection, parameter tuning |
| [DGX_TROUBLESHOOTING.md](DGX_TROUBLESHOOTING.md) | Common issues and fixes |
| [DGX_SPARK_README.md](DGX_SPARK_README.md) | What changed from upstream and why |
| [GAME_STRATEGIES.md](GAME_STRATEGIES.md) | Strategy details, math, and tuning tips |

## Attribution

- Original project: [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy
- DGX Spark inspiration: [David-Barnes-Data-Imaginations/autoresearch-DGX-Spark](https://github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark) by David Barnes

## Notable forks (upstream)

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
