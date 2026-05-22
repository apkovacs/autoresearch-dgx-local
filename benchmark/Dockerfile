# Dockerfile — Benchmark suite for autoresearch agent harness evaluation
#
# Bundles Ollama + all adapter dependencies (Aider, Claude Code, OpenHands CLI)
# for self-contained, reproducible benchmarking. Ollama runs inside the
# container — no host Ollama installation needed.
#
# Build:
#   docker build -t autoresearch-bench benchmark/
#
# Run (Level 1 — edit quality, no GPU):
#   bash benchmark/run-bench.sh edit-quality --models qwen3.6:27b --trials 10
#
# Run (Level 2 — harness comparison):
#   bash benchmark/run-bench.sh harness --adapters ollama_raw aider --model qwen3.6:27b
#
# Run (Level 3 — end-to-end, needs GPU):
#   BENCH_GPU=1 bash benchmark/run-bench.sh e2e --harness hyp --budget 10
#
# Notes:
#   - Ollama runs inside the container; mount ~/.ollama/models to reuse cached weights
#   - -v /var/run/docker.sock: required for the OpenHands adapter (Docker-in-Docker)
#   - The entrypoint starts Ollama, pulls the requested model, then runs the benchmark

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
        procps \
    && rm -rf /var/lib/apt/lists/*

# --- Ollama (runs inside container) ---
RUN curl -fsSL https://ollama.com/install.sh | sh

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
RUN pip install -q openhands-ai 2>/dev/null || true

# --- Git config ---
RUN git config --global user.email "bench@autoresearch.local" && \
    git config --global user.name "AutoResearch Bench" && \
    git config --global --add safe.directory /workspace && \
    git config --global init.defaultBranch main

# --- Entrypoint ---
# Starts Ollama, waits for readiness, pulls the model, then runs the command.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "benchmark/bench_edit_quality.py", "--help"]
