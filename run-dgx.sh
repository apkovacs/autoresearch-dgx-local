#!/usr/bin/env bash
# run-dgx.sh — Launch autoresearch training on NVIDIA DGX Spark
#
# Docker configuration inspired by:
#   github.com/David-Barnes-Data-Imaginations/autoresearch-DGX-Spark
#
# Usage:
#   bash run-dgx.sh              # interactive mode (default)
#   bash run-dgx.sh -d           # detached mode
#   bash run-dgx.sh --test       # test mode (check setup, don't train)
#
# Environment variables:
#   SHARD_CACHE_DIR   Host path for persistent training shards (default: ~/.cache/autoresearch)
#   DOCKER_IMAGE      Base Docker image (default: nvcr.io/nvidia/pytorch:25.12-py3)
#   SHM_SIZE          Shared memory size (default: 64gb)

set -euo pipefail

# --- Defaults ---
SHARD_CACHE_DIR="${SHARD_CACHE_DIR:-$HOME/.cache/autoresearch}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
SHM_SIZE="${SHM_SIZE:-64gb}"
CONTAINER_NAME="autoresearch-dgx"
MODE="interactive"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--detached) MODE="detached"; shift ;;
        --test) MODE="test"; shift ;;
        -h|--help)
            echo "Usage: bash run-dgx.sh [-d|--detached] [--test] [-h|--help]"
            echo ""
            echo "Options:"
            echo "  -d, --detached   Run container in background"
            echo "  --test           Verify setup without training"
            echo "  -h, --help       Show this help"
            echo ""
            echo "Environment variables:"
            echo "  SHARD_CACHE_DIR  Host path for persistent shards (default: ~/.cache/autoresearch)"
            echo "  DOCKER_IMAGE     Docker image (default: nvcr.io/nvidia/pytorch:25.12-py3)"
            echo "  SHM_SIZE         Shared memory (default: 64gb)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Pre-flight checks ---
echo "=== DGX Spark Autoresearch Launcher ==="
echo ""

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed. Install Docker first."
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon is not running. Start Docker first."
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
    echo "WARNING: NVIDIA Container Toolkit may not be installed or GPU not accessible."
    echo "Install with: sudo apt-get install -y nvidia-container-toolkit"
    echo "Continuing anyway..."
fi

# --- Create persistent directories ---
mkdir -p "$SHARD_CACHE_DIR"

echo "Configuration:"
echo "  Docker image:     $DOCKER_IMAGE"
echo "  Shard cache:      $SHARD_CACHE_DIR"
echo "  Shared memory:    $SHM_SIZE"
echo "  Mode:             $MODE"
echo "  Container name:   $CONTAINER_NAME"
echo ""

# --- Build docker run command ---
DOCKER_ARGS=(
    --rm
    --gpus all
    --ipc=host
    --shm-size "$SHM_SIZE"
    --oom-score-adj 1000
    --ulimit memlock=-1
    --ulimit stack=67108864
    --name "$CONTAINER_NAME"
    -v "$(pwd)":/workspace
    -v "$SHARD_CACHE_DIR":/cache/autoresearch
    -e AUTORESEARCH_CACHE_DIR=/cache/autoresearch
    -e NCCL_P2P_DISABLE=1
    -e TORCH_CUDA_ARCH_LIST=12.0
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e HF_HUB_DISABLE_PROGRESS_BARS=1
    -w /workspace
)

case "$MODE" in
    interactive)
        echo "Starting interactive training container..."
        docker run -it "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" \
            bash -c "pip install -q rustbpe huggingface_hub tiktoken pyarrow requests && python prepare.py --num-shards 10 && python train.py"
        ;;
    detached)
        echo "Starting detached training container..."
        docker run -d "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" \
            bash -c "pip install -q rustbpe huggingface_hub tiktoken pyarrow requests && python prepare.py --num-shards 10 && python train.py"
        echo "Container started. Monitor with: bash monitor-dgx.sh"
        ;;
    test)
        echo "Running setup test..."
        docker run -it "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" \
            bash -c "
                echo '=== GPU Info ===' && nvidia-smi
                echo ''
                echo '=== Python/PyTorch ===' && python -c 'import torch; print(f\"PyTorch {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}\"); print(f\"Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB\" if torch.cuda.is_available() else \"\")'
                echo ''
                echo '=== Cache Directory ===' && echo \$AUTORESEARCH_CACHE_DIR && ls -la /cache/autoresearch/ 2>/dev/null || echo '(empty - shards will be downloaded on first run)'
                echo ''
                echo '=== Test PASSED ==='
            "
        ;;
esac
