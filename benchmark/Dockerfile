# Dockerfile — Benchmark suite for autoresearch agent harness evaluation
#
# Bundles all adapter dependencies (Aider, Claude Code, OpenHands CLI)
# for reproducible, portable benchmarking across models and harnesses.
#
# Build:
#   docker build -t autoresearch-bench benchmark/
#
# Run (Level 1 — edit quality, no GPU):
#   docker run --rm -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock \
#       --network host autoresearch-bench \
#       python benchmark/bench_edit_quality.py --models qwen3.6:27b --trials 10
#
# Run (Level 2 — harness comparison):
#   docker run --rm -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock \
#       --network host autoresearch-bench \
#       python benchmark/bench_harness.py --adapters ollama_raw aider --model qwen3.6:27b
#
# Run (Level 3 — end-to-end, needs GPU):
#   docker run --rm --gpus all -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock \
#       --network host autoresearch-bench \
#       python benchmark/bench_e2e.py --harness hyp --model qwen3.6:27b --budget 10
#
# Notes:
#   - --network host: allows the container to reach Ollama on localhost:11434
#   - -v /var/run/docker.sock: required for the OpenHands adapter (Docker-in-Docker)
#   - Ollama must be running on the host (or reachable at the default URL)

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- System dependencies ---
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git \
        curl \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
# Core: benchmark scripts + hypothesis_generator.py
RUN pip install -q \
    requests

# Aider adapter
RUN pip install -q \
    aider-chat

# --- Node.js + Claude Code adapter ---
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y -qq --no-install-recommends nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    rm -rf /var/lib/apt/lists/*

# --- OpenHands CLI ---
# Pre-pull is optional; the adapter will pull on first use via Docker socket.
# Install the lightweight CLI wrapper if available.
RUN pip install -q openhands-ai 2>/dev/null || true

# --- Git config ---
RUN git config --global user.email "bench@autoresearch.local" && \
    git config --global user.name "AutoResearch Bench" && \
    git config --global --add safe.directory /workspace && \
    git config --global init.defaultBranch main

WORKDIR /workspace

# Default: show help
CMD ["python", "benchmark/bench_edit_quality.py", "--help"]
