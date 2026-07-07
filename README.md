# autoresearch-dgx-local

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that has grown into a **platform for studying how AI models of every capability class can drive autonomous ML research on a single NVIDIA DGX Spark** — from 2-bit quantized frontier-scale models running Karpathy's original design, down to small local models wrapped in deterministic scaffolding, with a benchmark suite that objectively measures where any given model falls on that spectrum.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

## What this repo became

The original autoresearch gives an AI agent a small but real LLM training setup and lets it experiment autonomously: modify `train.py`, train for 5 minutes, keep or discard based on `val_bpb`, repeat. That core loop is unchanged here.

What grew around it, over months of running local models on a DGX Spark and watching exactly how they fail, is everything needed to run that loop with **any** model:

1. **DGX Spark adaptation** — SDPA fallback for flash-attn3, reduced hyperparameters, Docker containerization with OOM protection and persistent shard storage
2. **Three agent modes spanning the capability spectrum** — because a 7B local model and a 284B frontier-scale model need completely different amounts of scaffolding
3. **Custom GGUF support** — run community quants like DeepSeek V4 Flash "Dwarf Star" (284B MoE compressed to ~81 GB) that aren't in the Ollama library
4. **Pluggable inference backends** — the hypothesis generator and benchmark suite speak to Ollama or any OpenAI-compatible server (llama-server, vLLM, ds4), opening the door to engine features Ollama lacks, like speculative decoding
5. **A four-level benchmark suite** — objectively measure a model's edit quality, harness fit, end-to-end results, and agentic overhead before burning GPU-days on it
6. **Meta-research orchestration** — game-theory-inspired strategies (island model, multi-armed bandit, iterated coopetition) that manage multiple competing/cooperating research branches on one GPU

Inspired in part by [David-Barnes-Data-Imaginations/autoresearch-DGX-Spark](https://github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark).

## The central idea: scaffolding should match capability

Most of what goes wrong when a local model drives the research loop is not the science — it's the mechanics. Edit-tool mismatches, permission denials, repetitive thinking loops, tool confusion. Watching those failures led to a spectrum of modes where the scaffolding shrinks as the model gets stronger:

| Mode | Command | The model's job | Best for |
|---|---|---|---|
| **Hypothesis generator** | `bash run-dgx-local.sh` | Propose one edit as structured JSON; a deterministic script does everything else | Smaller local models, maximum reliability |
| **Guarded agent** | `bash run-dgx-agent.sh` | Drive the full loop via Claude Code, with guardrails: output token cap, action-first prompt, narrow permissions | Capable local models (27B+) |
| **Minimal agent** | `bash run-dgx-agent.sh --mode minimal` | Karpathy's original design — program.md drives everything, no caps, all permissions | Frontier API models and highly capable local models (e.g. DeepSeek V4 Flash) |

The infrastructure the model never sees — session restarts with durable results.tsv memory, syntax validation before training, heartbeat output, structured event logging — is shared by all three modes. Only the behavioral scaffolding changes.

The benchmark suite (below) exists to answer the obvious question: *which mode does my model qualify for?*

## Quick start

### Training only

```bash
git clone https://github.com/apkovacs/autoresearch-dgx-local.git
cd autoresearch-dgx-local
git checkout dgx-spark
bash run-dgx.sh
```

### Autonomous agent

```bash
# Hypothesis generator (most reliable with local models)
bash run-dgx-local.sh

# Guarded agent (default full-agent mode)
bash run-dgx-agent.sh

# Minimal agent — the original design, for highly capable models
bash run-dgx-agent.sh --mode minimal
```

### Frontier-scale local: DeepSeek V4 Flash (Dwarf Star)

The Dwarf Star selective quantization compresses DeepSeek V4 Flash (284B MoE, 13B active) from 568 GB to ~81 GB — small enough for the Spark's 128 GB unified memory. It's distributed as a GGUF outside the Ollama library, so the launchers accept a local file:

```bash
OLLAMA_GGUF=~/models/deepseek-v4-flash-dwarf.gguf \
OLLAMA_MODEL=deepseek-v4-flash-dwarf \
bash run-dgx-agent.sh --mode minimal
```

The file is imported once into the persistent Ollama model store; later runs skip the import. See [DGX_SETUP.md](DGX_SETUP.md#custom-gguf-models-eg-deepseek-v4-flash) for context-window settings and memory-headroom notes.

The hypothesis generator can also bypass Ollama entirely and use any OpenAI-compatible server — useful for engines with speculative decoding:

```bash
INFERENCE_BACKEND=openai INFERENCE_URL=http://host.docker.internal:8080/v1 \
OLLAMA_MODEL=deepseek-v4-flash-dwarf bash run-dgx-local.sh
```

### Multi-branch game strategies

```bash
bash run-dgx-game.sh --mode island        # 3 branches with migration
bash run-dgx-game.sh --mode bandit        # UCB1 arm selection
bash run-dgx-game.sh --mode coopetition   # 2 branches, forced adoption
```

See [DGX_QUICKSTART.md](DGX_QUICKSTART.md) for full setup and [GAME_STRATEGIES.md](GAME_STRATEGIES.md) for strategy details.

## Benchmark suite

A containerized, four-level evaluation suite (`benchmark/`) for comparing models and harnesses on the autoresearch task itself — Ollama runs inside the container, results render to an HTML dashboard:

| Level | Command | Measures | GPU |
|---|---|---|---|
| 1. Edit quality | `bash benchmark/run-bench.sh edit-quality` | JSON validity, schema compliance, edit applicability, syntax | No |
| 2. Harness comparison | `bash benchmark/run-bench.sh harness` | Same task across raw Ollama API, Aider, Claude Code, OpenHands | Mostly no |
| 3. End-to-end | `BENCH_GPU=1 bash benchmark/run-bench.sh e2e` | val_bpb improvement over a full experiment budget | Yes |
| 4. Trace quality | `bash benchmark/run-bench.sh trace` | Agentic overhead from real transcripts: tool calls per experiment, permission denials, friction, degenerate loops | No |

Level 4 is the qualifying exam for minimal mode: run any agent session, analyze its traces, and see whether the model executes the loop cleanly enough to shed the guardrails. `bash benchmark/run-bench.sh dashboard` renders everything to a self-contained HTML page.

See [benchmark/README.md](benchmark/README.md).

## Architecture

```
Layer 3: Meta-Agent (optional)  -->  tunes game parameters
Layer 2: Orchestrator           -->  manages branches, strategies, migration
Layer 1: Research Agent         -->  modifies train.py, runs experiments
                                     (hypothesis generator | guarded | minimal)
Layer 0: Training               -->  PyTorch training loop (train.py)

Sidecar: Benchmark suite        -->  measures which Layer-1 mode a model qualifies for
```

Each layer is independently useful. You can run just the training loop, add a research agent in any mode, or enable the full orchestrator with multiple branches.

## How it works

The core idea is unchanged from upstream: give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You are programming the `program.md` Markdown files that provide context to the AI agents, not touching the Python files directly.

The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

### Game strategies

The orchestrator adds a meta-research layer that manages multiple research branches:

| Mode | Branches | Mechanism | Best for |
|---|---|---|---|
| **Base** | 1 | Original autoresearch loop | Baseline, simple experiments |
| **Island** | 3+ | Independent evolution with periodic migration | Broad exploration across optimization axes |
| **Bandit** | 2+ | UCB1 selects which strategy gets compute | Discovering which direction yields most improvement |
| **Coopetition** | 2 | Head-to-head with forced adoption by loser | Focused comparison with knowledge transfer |

### Local LLM support

Everything runs on-device via Ollama. Tested library models:

| Model | VRAM headroom | Reliability | Notes |
|---|---|---|---|
| Qwen3.6 27B (default) | Moderate | High | Strong reasoning, fewest edit failures |
| Gemma 4 26B | Moderate | Medium | Competitive but prone to repetitive thinking loops in agent mode |
| Gemma 4 12B | High | Medium | More memory headroom; more edit retries |
| Qwen 2.5 Coder 14B | High | Medium | Precise edits but weaker experiment reasoning |
| DeepSeek V4 Flash (Dwarf Star GGUF) | Tight (~81 GB) | Under evaluation | Frontier-scale MoE; target for minimal mode |

Local-model failure modes and their mitigations (output token caps, action-first prompts, zero-progress detection, session restarts) are documented in [DGX_SETUP.md](DGX_SETUP.md#local-model-limitations-and-mitigations) — or sidestep them entirely with the hypothesis generator mode.

## DGX Spark adaptations

Key changes from the upstream H100 configuration:

- **SDPA fallback** for flash-attn3 (no ARM64 builds available)
- **Reduced hyperparameters**: `DEPTH=4`, `TOTAL_BATCH_SIZE=2^16`, `DEVICE_BATCH_SIZE=8`
- **MFU constant**: 209 TFLOPS BF16 (GB10 Blackwell, replacing H100's 989 TFLOPS)
- **OOM protection**: `--oom-score-adj 1000` prevents GPU memory pressure from freezing the host (unified memory architecture)
- **GPU memory sharing**: `OLLAMA_KEEP_ALIVE=0` unloads the LLM during training, freeing memory for PyTorch — essential when the LLM itself is 81 GB
- **Persistent storage**: Training shards, Ollama model weights, and imported GGUFs survive container restarts via Docker volume mounts

## Project structure

```
train.py                    -- model, optimizer, training loop (agent modifies this)
prepare.py                  -- constants, data prep + runtime utilities
program.md                  -- agent instructions (human modifies this)
pyproject.toml              -- dependencies

run-dgx.sh                  -- Docker launcher: training only
run-dgx-local.sh            -- Docker launcher: hypothesis generator mode
run-dgx-agent.sh            -- Docker launcher: full agent (guarded or minimal mode)
run-dgx-game.sh             -- Docker launcher: meta-research orchestrator
monitor-dgx.sh              -- real-time container monitoring
monitor-game.sh             -- live game dashboard, event stream, transcript viewer

hypothesis_generator.py     -- propose/apply edit engine (backends: Ollama, OpenAI-compatible)
stream_formatter.py         -- live formatting + transcript capture of agent output
orchestrator.py             -- game engine: schedules branches, migration, adoption
hyperparams.py              -- hyperparameter extraction/injection for cross-branch migration
leaderboard.py              -- cross-branch results tracking
event_log.py                -- structured event logging (JSON lines)
game_config.yaml            -- game configuration (mode, branches, strategy params)
branch_templates/           -- prompt templates for bounded experiment rounds
logs/                       -- event log + per-round agent transcripts (gitignored)

benchmark/                  -- containerized 4-level model/harness evaluation suite
  run-bench.sh              -- benchmark launcher (Ollama runs inside the container)
  bench_edit_quality.py     -- Level 1: edit quality
  bench_harness.py          -- Level 2: harness comparison (Ollama/Aider/Claude Code/OpenHands)
  bench_e2e.py              -- Level 3: end-to-end val_bpb
  bench_trace_quality.py    -- Level 4: agentic overhead from transcripts
  bench_dashboard.py        -- self-contained HTML results dashboard

DGX_SPARK_README.md         -- overview of DGX Spark changes
DGX_QUICKSTART.md           -- quick start guide
DGX_SETUP.md                -- detailed setup, model selection, agent modes, custom GGUFs
DGX_TROUBLESHOOTING.md      -- common issues and fixes
GAME_STRATEGIES.md          -- strategy documentation with tuning tips
```

## Design choices

Inherited from upstream:
- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes of wall clock time, making experiments directly comparable regardless of what the agent changes.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages.

Added in this fork:
- **Scaffolding matches capability.** Behavioral guardrails are opt-out, not baked in — a capable model can run the original unguarded design, and the trace-quality benchmark tells you when that's safe.
- **Measure before you commit GPU time.** The benchmark suite turns "is this model good enough?" from a vibe into four levels of numbers.
- **Docker-first.** All execution happens inside containers with pinned images, ensuring reproducibility — including the benchmark suite itself.
- **Persistent by default.** Training shards, model weights, and imported GGUFs are cached on the host, surviving container restarts.
- **Strategies are deterministic.** The orchestrator (branch scheduling, migration, adoption) is plain Python with no LLM calls. Only the research agent and optional meta-agent use the LLM.
- **Git branches as isolation.** Each research branch is a real git branch, making it easy to inspect, compare, and cherry-pick results.

## Documentation

| Document | Description |
|---|---|
| [DGX_QUICKSTART.md](DGX_QUICKSTART.md) | Get running in under 5 minutes |
| [DGX_SETUP.md](DGX_SETUP.md) | Persistent storage, model selection, agent modes, custom GGUFs |
| [DGX_TROUBLESHOOTING.md](DGX_TROUBLESHOOTING.md) | Common issues and fixes |
| [DGX_SPARK_README.md](DGX_SPARK_README.md) | What changed from upstream and why |
| [GAME_STRATEGIES.md](GAME_STRATEGIES.md) | Strategy details, math, and tuning tips |
| [benchmark/README.md](benchmark/README.md) | The four-level benchmark suite and dashboard |

## Attribution

- Original project: [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy
- DGX Spark inspiration: [David-Barnes-Data-Imaginations/autoresearch-DGX-Spark](https://github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark) by David Barnes
- Dwarf Star quantization of DeepSeek V4 Flash: community GGUF release (see [DGX_SETUP.md](DGX_SETUP.md#custom-gguf-models-eg-deepseek-v4-flash))

## Notable forks (upstream)

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
