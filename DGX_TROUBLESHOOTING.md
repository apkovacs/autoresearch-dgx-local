# DGX Spark Troubleshooting

## System Freeze / Unresponsive Machine

**Symptom**: The DGX Spark becomes completely unresponsive during training. No SSH, no keyboard input.

**Cause**: Unified memory OOM. Unlike dedicated GPU VRAM (which throws a clean CUDA error), unified memory exhaustion can lock up the entire system because the GPU and CPU share the same memory pool.

**Fix**:
- Hard reboot the machine (hold power button)
- After restart, reduce `DEVICE_BATCH_SIZE` in `train.py` (try halving it)
- If running the agent, switch to a smaller LLM: `OLLAMA_MODEL=gemma4:e4b bash run-dgx-agent.sh`
- The `--oom-score-adj 1000` Docker flag helps the kernel kill the container before the system freezes, but it's not guaranteed

**Prevention**: The training loop includes OOM detection that exits cleanly with a helpful message. If you see this message, reduce batch size before restarting.

---

## Slow First Training Steps

**Symptom**: The first ~10 training steps take 10–30x longer than later steps.

**Cause**: `torch.compile` JIT compilation. This is expected behavior — PyTorch compiles optimized CUDA kernels on the first pass.

**Fix**: No fix needed. The training timer excludes the first 10 steps automatically. Steady-state speed is what matters.

---

## CUDA Architecture Error

**Symptom**: Error message mentioning CUDA architecture, SM version, or unsupported GPU.

**Cause**: PyTorch wasn't told to target the GB10's Blackwell architecture (SM 12.0).

**Fix**: The `run-dgx.sh` and `run-dgx-agent.sh` scripts set `TORCH_CUDA_ARCH_LIST=12.0` automatically. If running manually:

```bash
export TORCH_CUDA_ARCH_LIST=12.0
```

---

## Shard Download Failures

**Symptom**: Download hangs, times out, or reports failures.

**Cause**: Network issues reaching HuggingFace.

**Fix**:
- The downloader retries up to 5 times with exponential backoff
- If behind a proxy: `export HTTP_PROXY=http://proxy:port HTTPS_PROXY=http://proxy:port`
- Download fewer shards for testing: edit the `prepare.py` call in `run-dgx.sh` to use `--num-shards 2`
- Manual download: `wget https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_00000.parquet -O ~/.cache/autoresearch/data/shard_00000.parquet`

---

## "No Training Shards Found"

**Symptom**: Error at startup saying no parquet files were found.

**Cause**: The `AUTORESEARCH_CACHE_DIR` doesn't point to where shards are stored, or the Docker volume mount is misconfigured.

**Fix**:
- Check that shards exist on the host: `ls ~/.cache/autoresearch/data/shard_*.parquet`
- If using a custom path: `SHARD_CACHE_DIR=/your/path bash run-dgx.sh`
- Verify the volume mount inside the container: `docker exec <container> ls /cache/autoresearch/data/`
- If shards are in a different container, copy them out: `docker cp <old-container>:/root/.cache/autoresearch ~/.cache/autoresearch`

---

## Container Can't See GPU

**Symptom**: `torch.cuda.is_available()` returns False, or `nvidia-smi` fails inside the container.

**Cause**: NVIDIA Container Toolkit not installed or not configured.

**Fix**:

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# If it fails, install the toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Docker Permission Denied

**Symptom**: `Got permission denied while trying to connect to the Docker daemon`

**Fix**:

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker ps
```

---

## Ollama Not Starting (Agent Mode)

**Symptom**: Agent script hangs at "Starting Ollama server" or reports connection refused.

**Cause**: Ollama failed to start inside the container.

**Fix**:
- Check if port 11434 is already in use on the host: `lsof -i :11434`
- If Ollama is running on the host, stop it first: `systemctl stop ollama` or `killall ollama`
- Check container logs: `docker logs autoresearch-dgx-local-agent`

---

## Claude Code Can't Connect to Ollama

**Symptom**: Claude Code starts but errors with connection or authentication failures.

**Cause**: Environment variables not set correctly.

**Fix**: Verify inside the container:

```bash
echo $ANTHROPIC_BASE_URL    # should be http://localhost:11434
echo $ANTHROPIC_AUTH_TOKEN   # should be "ollama"
echo $ANTHROPIC_API_KEY      # should be "ollama"
```

Test Ollama directly:

```bash
curl http://localhost:11434/api/tags
```

---

## Training Loss Explodes or NaN

**Symptom**: Training prints "FAIL" and exits, or loss values are very high / NaN.

**Cause**: Learning rates may be too high for the current model configuration, or numerical instability.

**Fix**:
- The fast-fail check exits when loss > 100 or is NaN
- Try reducing learning rates: lower `MATRIX_LR`, `EMBEDDING_LR` in `train.py`
- Ensure `DEPTH` and other parameters produce a valid configuration

---

## Container Name Conflict

**Symptom**: `docker: Error response from daemon: Conflict. The container name is already in use.`

**Fix**:

```bash
# Remove the old container
docker rm autoresearch-dgx-local          # for run-dgx.sh
docker rm autoresearch-dgx-local-agent    # for run-dgx-agent.sh

# Or force-remove if still running
docker rm -f autoresearch-dgx-local
```

---

## Git Permission Error After Container Run

**Symptom**: `error: insufficient permission for adding an object to repository database .git/objects` when pulling or committing on the host.

**Cause**: The Docker container runs as root and creates `.git/objects` files owned by root. The host user can't write to them.

**Fix**:

```bash
sudo chown -R $(whoami) .git
```

The launcher scripts include an EXIT trap that restores `.git` ownership automatically on clean shutdown. This issue only occurs if the container is hard-killed (`docker kill`) before the trap fires.

---

## Agent Stops After Permission Denial

**Symptom**: The agent runs one experiment and then the session ends. Transcript shows `"permission_denials"` and `"stop_reason":"end_turn"`.

**Cause**: The Claude Code sandbox blocks bash output redirection (`>` and `>>`). If the agent tries `printf ... >> results.tsv`, the sandbox blocks it and smaller models treat this as a fatal error.

**Fix**: The launcher scripts generate wrapper scripts (`run_experiment.sh`, `log_result.sh`) that handle redirects internally. Make sure you're running the latest version of the launcher scripts. The `CLAUDE.md` instructions tell the agent to use these wrappers instead of direct redirection.

---

## Ollama Model Pull Fails

**Symptom**: `Error: pull model manifest: file does not exist` when pulling a model.

**Cause**: The model tag may have been renamed or removed from the Ollama registry.

**Fix**: Check available tags at [ollama.com/library](https://ollama.com/library) and use an updated tag. For example, `gemma4:12b` was replaced by `gemma4:e4b`.

```bash
OLLAMA_MODEL=gemma4:e4b bash run-dgx-agent.sh
```
