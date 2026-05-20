# Dockerfile — Pre-built autoresearch environment for DGX Spark
#
# Bakes in all dependencies (Python packages, Ollama, Claude Code, Node.js)
# so container launches take seconds instead of minutes.
#
# Build:
#   docker build -t autoresearch-dgx .
#
# Then launch with:
#   DOCKER_IMAGE=autoresearch-dgx bash run-dgx-agent.sh
#   DOCKER_IMAGE=autoresearch-dgx bash run-dgx-game.sh --mode island

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# --- System dependencies ---
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        zstd \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
# These are the packages needed by prepare.py, train.py, and the orchestrator.
# torch is already in the base NVIDIA image.
RUN pip install --no-cache-dir -q \
    rustbpe \
    huggingface_hub \
    tiktoken \
    pyarrow \
    requests \
    pyyaml

# --- Node.js + Claude Code ---
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y -qq nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    rm -rf /var/lib/apt/lists/*

# --- Ollama ---
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- Git config ---
# Pre-configure git identity so the agent never needs to touch git config
RUN git config --global user.email "agent@autoresearch.local" && \
    git config --global user.name "AutoResearch Agent" && \
    git config --global --add safe.directory /workspace

WORKDIR /workspace
