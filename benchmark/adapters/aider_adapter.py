"""Aider CLI adapter for benchmark harness.

Aider (https://aider.chat) is a terminal-first AI pair programmer with
native git integration and Ollama support. This adapter runs Aider in
single-shot mode to propose and apply one edit to train.py.

Requires: pip install aider-chat
"""

import difflib
import py_compile
import shutil
import subprocess
import time
from pathlib import Path

from .base import EditResult, HarnessAdapter


class AiderAdapter(HarnessAdapter):
    name = "aider"

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url

    def propose_and_apply(self, workdir: Path, model: str,
                          results_tsv: Path | None = None) -> EditResult:
        train_py = workdir / "train.py"
        original = train_py.read_text()

        # Build context message
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

        message = (
            f"Edit train.py to improve val_bpb (validation bits per byte). "
            f"The current best val_bpb is {best_bpb}. "
            f"Make a single, targeted change to a hyperparameter or architectural "
            f"detail that could lower val_bpb. The training budget is fixed at "
            f"5 minutes. Only modify train.py."
        )

        # Initialize a temporary git repo in workdir for Aider
        subprocess.run(["git", "init"], cwd=workdir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=workdir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=workdir, capture_output=True,
            env={**dict(__import__('os').environ),
                 "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@test",
                 "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@test"}
        )

        t0 = time.time()

        cmd = [
            "aider",
            "--model", f"ollama/{model}",
            "--yes",             # auto-approve edits
            "--no-auto-commits", # we track changes ourselves
            "--no-git",          # don't let Aider manage git
            "--message", message,
            str(train_py),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=180,
                env={**dict(__import__('os').environ),
                     "OLLAMA_API_BASE": self.ollama_url},
            )
        except subprocess.TimeoutExpired:
            return EditResult(
                success=False,
                time_s=time.time() - t0,
                error="Aider timed out after 180s",
            )

        elapsed = time.time() - t0
        modified = train_py.read_text()
        is_meaningful = original != modified

        if not is_meaningful:
            return EditResult(
                success=False,
                description="(no changes made)",
                time_s=elapsed,
                error="Aider made no changes to train.py",
                is_meaningful=False,
            )

        # Extract description from Aider output (first non-empty line of changes)
        description = "aider edit"
        for line in proc.stdout.split('\n'):
            if line.strip() and not line.startswith(('─', '>', 'Tokens:', 'Model:')):
                description = line.strip()[:100]
                break

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
        return shutil.which("aider") is not None
