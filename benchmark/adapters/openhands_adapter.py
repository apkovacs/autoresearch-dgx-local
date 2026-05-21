"""OpenHands adapter for benchmark harness.

OpenHands (https://github.com/OpenHands/OpenHands) is an autonomous AI
development agent that runs in a Docker sandbox. This adapter uses its
headless CLI mode to propose and apply edits to train.py.

Requires: docker pull ghcr.io/openhands/openhands
"""

import difflib
import json
import os
import py_compile
import shutil
import subprocess
import time
from pathlib import Path

from .base import EditResult, HarnessAdapter


class OpenHandsAdapter(HarnessAdapter):
    name = "openhands"

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 image: str = "ghcr.io/openhands/openhands:latest"):
        self.ollama_url = ollama_url
        self.image = image

    def propose_and_apply(self, workdir: Path, model: str,
                          results_tsv: Path | None = None) -> EditResult:
        train_py = workdir / "train.py"
        original = train_py.read_text()

        best_bpb = "unknown"
        if results_tsv and results_tsv.exists():
            lines = results_tsv.read_text().strip().split('\n')
            for line in lines[1:]:
                fields = line.split('\t')
                if len(fields) >= 4 and fields[3] == 'keep':
                    try:
                        bpb = float(fields[1])
                        if bpb > 0 and (best_bpb == "unknown" or bpb < float(best_bpb)):
                            best_bpb = str(bpb)
                    except ValueError:
                        pass

        task = (
            f"Edit train.py to improve val_bpb (validation bits per byte). "
            f"Current best: {best_bpb}. Make a single, targeted change to "
            f"a hyperparameter or architectural detail. Do not run training. "
            f"Only modify train.py."
        )

        # OpenHands needs the Ollama URL accessible from inside Docker.
        # On Linux: use host.docker.internal or the host's IP.
        # The user may need to configure this based on their Docker setup.
        ollama_host_url = self.ollama_url.replace(
            "localhost", "host.docker.internal"
        ).replace(
            "127.0.0.1", "host.docker.internal"
        )

        t0 = time.time()

        cmd = [
            "docker", "run", "--rm",
            "--add-host", "host.docker.internal:host-gateway",
            "-v", f"{workdir}:/workspace",
            "-w", "/workspace",
            self.image,
            "--headless",
            "--task", task,
            "--llm-model", f"ollama/{model}",
            "--llm-base-url", ollama_host_url,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return EditResult(
                success=False,
                time_s=time.time() - t0,
                error="OpenHands timed out after 300s",
            )

        elapsed = time.time() - t0
        modified = train_py.read_text()
        is_meaningful = original != modified

        if not is_meaningful:
            return EditResult(
                success=False,
                description="(no changes made)",
                time_s=elapsed,
                error="OpenHands made no changes to train.py",
                is_meaningful=False,
            )

        # Try to extract description from OpenHands JSONL output
        description = "openhands edit"
        for line in proc.stdout.strip().split('\n'):
            try:
                event = json.loads(line)
                if event.get("action") == "edit" or "edit" in str(event.get("message", "")):
                    description = str(event.get("message", "openhands edit"))[:100]
                    break
            except json.JSONDecodeError:
                continue

        # Syntax check
        try:
            py_compile.compile(str(train_py), doraise=True)
            syntax_ok = True
        except py_compile.PyCompileError:
            syntax_ok = False

        # Diff
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="train.py.orig",
            tofile="train.py",
        ))

        return EditResult(
            success=syntax_ok and is_meaningful,
            description=description,
            time_s=elapsed,
            syntax_ok=syntax_ok,
            is_meaningful=is_meaningful,
            diff=diff,
        )

    def is_available(self) -> bool:
        """Check if OpenHands Docker image is available."""
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True, timeout=10,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
