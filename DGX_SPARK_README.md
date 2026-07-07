# autoresearch — DGX Spark Adaptation

This branch adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to run on the NVIDIA DGX Spark desktop workstation — and extends it with three agent modes (hypothesis generator, guarded, minimal), custom GGUF model support, and a benchmark suite for measuring which mode a model qualifies for. See the [main README](README.md) for the full picture; this document focuses on the hardware adaptation.

## What Changed and Why

The DGX Spark has fundamentally different hardware than the H100 the original repo targets:

| | H100 | DGX Spark (GB10) |
|---|---|---|
| GPU | H100 (80 GB HBM3) | GB10 Blackwell (128 GB unified LPDDR5X) |
| CPU | x86_64 | ARM64 (Grace) |
| CUDA SM | 9.0 (Hopper) | 12.0 (Blackwell) |
| Memory model | Dedicated GPU VRAM | CPU+GPU shared pool |

These differences require changes in three areas:

### Attention Kernel

flash-attn3 has no ARM64 builds. We replace it with PyTorch's built-in `scaled_dot_product_attention` (SDPA). This means the sliding window pattern (`SSSL`) is accepted but the window sizes are effectively ignored — all attention is full-context. The model still trains and converges; you just lose the memory savings of short windows.

### Training Parameters

| Parameter | Original (H100) | DGX Spark | Why |
|---|---|---|---|
| `DEPTH` | 8 | 4 | Halves model to ~10M params for memory safety |
| `DEVICE_BATCH_SIZE` | 128 | 8 | 16x reduction for unified memory |
| `TOTAL_BATCH_SIZE` | 2^19 (524K) | 2^16 (65K) | Matches smaller batch; 4 grad accum steps |
| MFU reference | H100 989.5 TFLOPS | GB10 ~209 TFLOPS | Approximate; makes MFU% meaningful |

### Infrastructure

- Docker containerization with unified-memory-safe flags (`--oom-score-adj`, `--shm-size 64gb`)
- GPU memory time-sharing: `OLLAMA_KEEP_ALIVE=0` unloads the LLM during training, freeing ~18 GB for PyTorch
- Persistent training shard storage via Docker volume mounts
- Optional local LLM agent (Ollama + Claude Code) for autonomous experimentation
- Pre-built Docker image (`Dockerfile`) for fast container launches
- Observability stack: live dashboard, transcript viewer, status snapshots
- OOM detection with clean exit (unified memory OOM can freeze the whole machine)

## Persistent Shard Storage

Training data shards (~50 MB each) download from HuggingFace on first run. The Docker configuration persists these to the host filesystem so they survive container restarts:

```
Host: ~/.cache/autoresearch/  →  Container: /cache/autoresearch/
```

Set `SHARD_CACHE_DIR` to change the host path. See [DGX_SETUP.md](DGX_SETUP.md) for details.

## Local LLM Agent

Three modes span the capability spectrum — the scaffolding shrinks as the model gets stronger:

```bash
# Hypothesis generator — LLM proposes edits as JSON, script handles everything else
bash run-dgx-local.sh                           # default: Qwen3.6 27B
bash run-dgx-local.sh --max-experiments 100     # set experiment budget

# Guarded agent — full Claude Code loop with guardrails for local models
bash run-dgx-agent.sh                           # default: Qwen3.6 27B
OLLAMA_MODEL=gemma4:26b bash run-dgx-agent.sh   # alternative model

# Minimal agent — Karpathy's original design, for highly capable models
bash run-dgx-agent.sh --mode minimal
```

Frontier-scale community quants that aren't in the Ollama library (e.g. DeepSeek V4 Flash "Dwarf Star", 284B MoE compressed to ~81 GB) can be imported from a local GGUF file:

```bash
OLLAMA_GGUF=~/models/deepseek-v4-flash-dwarf.gguf \
OLLAMA_MODEL=deepseek-v4-flash-dwarf \
bash run-dgx-agent.sh --mode minimal
```

Use the trace-quality benchmark (`bash benchmark/run-bench.sh trace`) to verify a model executes the loop cleanly enough for minimal mode.

The hypothesis generator mode is backend-agnostic: set `INFERENCE_BACKEND=openai` and `INFERENCE_URL` to use any OpenAI-compatible server (llama-server, vLLM, ds4) instead of in-container Ollama — the path to engine features like speculative decoding. See [DGX_SETUP.md](DGX_SETUP.md#alternative-inference-backends-llama-server-vllm-ds4).

For faster startup, build a pre-built image that bakes in all dependencies:

```bash
docker build -t autoresearch-dgx-local .
DOCKER_IMAGE=autoresearch-dgx-local bash run-dgx-agent.sh
```

Monitor the agent from another terminal:

```bash
bash monitor-game.sh --status       # snapshot: experiments, GPU, git state
bash monitor-game.sh --transcript   # live agent activity + training progress
```

See [DGX_SETUP.md](DGX_SETUP.md) for the full model selection guide.

## Known Limitations

- **No sliding window attention**: SDPA ignores `window_size`. All layers use full-context attention.
- **Approximate MFU**: The GB10 BF16 peak FLOPS (~209 TFLOPS) is approximate. Use `val_bpb` as the authoritative metric.
- **Single GPU only**: The DGX Spark has one GPU. Multi-GPU features are not applicable.
- **Local model agent quality**: Locally-hosted models (26-27B) are less reliable than frontier API models at code editing and long-horizon instruction following. The launcher scripts include extensive mitigations (syntax validation, auto-restart, context compaction, heartbeat output). See [DGX_SETUP.md](DGX_SETUP.md#local-model-limitations-and-mitigations) for details.

## Attribution

- **Original project**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy
- **DGX Spark adaptation inspiration**: [David-Barnes-Data-Imaginations/autoresearch-DGX-Spark](https://github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark) — SDPA fallback approach, Docker configuration patterns, hyperparameter reduction strategy
- **Claude Code local model setup**: Based on the approach described in [this guide](https://medium.com/@luongnv89/run-claude-code-on-local-cloud-models-in-5-minutes-ollama-openrouter-llama-cpp-6dfeaee03cda)

## Quick Links

- [Quick Start](DGX_QUICKSTART.md) — Get running in 3 steps
- [Detailed Setup](DGX_SETUP.md) — Full configuration guide
- [Troubleshooting](DGX_TROUBLESHOOTING.md) — Common issues and fixes
