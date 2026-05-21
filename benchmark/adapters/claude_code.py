"""Claude Code adapter for benchmark harness.

Uses Claude Code (Anthropic's agent framework) to propose and apply edits
to train.py. When used with Ollama as the backend, this tests the full
agent tool-use loop (Edit tool, Bash tool) with a local model.

Requires: npm install -g @anthropic-ai/claude-code
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


class ClaudeCodeAdapter(HarnessAdapter):
    name = "claude_code"

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url

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

        prompt = (
            f"Read train.py and make a single edit to improve val_bpb "
            f"(validation bits per byte). Current best: {best_bpb}. "
            f"Use the Edit tool to modify train.py. Change one hyperparameter "
            f"or architectural detail. Do not run training — just make the edit. "
            f"Only modify train.py."
        )

        # Write permissions for Claude Code
        claude_dir = workdir / ".claude"
        claude_dir.mkdir(exist_ok=True)
        settings = {
            "permissions": {
                "allow": ["Edit", "Read"],
                "deny": ["Bash", "Task", "Monitor", "Agent", "WebSearch", "WebFetch"]
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        env = {
            **os.environ,
            "ANTHROPIC_BASE_URL": self.ollama_url,
            "ANTHROPIC_AUTH_TOKEN": "ollama",
            "ANTHROPIC_API_KEY": "ollama",
            "ANTHROPIC_MODEL": model,
        }

        t0 = time.time()

        cmd = [
            "claude", "-p",
            "--permission-mode", "dontAsk",
            "--output-format", "json",
            prompt,
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return EditResult(
                success=False,
                time_s=time.time() - t0,
                error="Claude Code timed out after 180s",
            )

        elapsed = time.time() - t0
        modified = train_py.read_text()
        is_meaningful = original != modified

        if not is_meaningful:
            return EditResult(
                success=False,
                description="(no changes made)",
                time_s=elapsed,
                error="Claude Code made no changes to train.py",
                is_meaningful=False,
            )

        # Try to extract description from Claude Code output
        description = "claude code edit"
        try:
            output = json.loads(proc.stdout)
            if isinstance(output, dict) and "result" in output:
                description = str(output["result"])[:100]
        except (json.JSONDecodeError, KeyError):
            pass

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
        return shutil.which("claude") is not None
