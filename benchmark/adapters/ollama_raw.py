"""Raw API adapter — hypothesis generator pattern.

Uses hypothesis_generator.py to propose and apply edits via direct API
calls with JSON schema constraints. No agent framework needed.

Backend-agnostic: defaults to Ollama, but INFERENCE_BACKEND=openai (with
INFERENCE_URL) targets any OpenAI-compatible server — llama-server, vLLM,
ds4 — including engines with speculative decoding that Ollama lacks.
"""

import difflib
import json
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path

from .base import EditResult, HarnessAdapter

# hypothesis_generator.py lives in the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class OllamaRawAdapter(HarnessAdapter):
    name = "ollama_raw"

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 backend: str | None = None, url: str | None = None):
        self.backend = backend or os.environ.get("INFERENCE_BACKEND", "ollama")
        self.url = url or os.environ.get("INFERENCE_URL") or (
            ollama_url if self.backend == "ollama" else "http://localhost:8080/v1"
        )
        # Back-compat attribute (bench_edit_quality.py error message uses it)
        self.ollama_url = self.url

    def propose_and_apply(self, workdir: Path, model: str,
                          results_tsv: Path | None = None) -> EditResult:
        train_py = workdir / "train.py"
        original = train_py.read_text()
        results_path = str(results_tsv) if results_tsv else str(workdir / "results.tsv")

        t0 = time.time()

        # Step 1: Propose
        cmd = [
            sys.executable, str(REPO_ROOT / "hypothesis_generator.py"),
            "propose",
            "--model", model,
            "--backend", self.backend,
            "--url", self.url,
            "--train-py", str(train_py),
            "--results", results_path,
            "--retries", "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if proc.returncode != 0:
            return EditResult(
                success=False,
                time_s=time.time() - t0,
                error=f"propose failed: {proc.stderr[:200]}",
            )

        # Parse proposal
        try:
            proposal = json.loads(proc.stdout.strip())
        except json.JSONDecodeError as e:
            return EditResult(
                success=False,
                time_s=time.time() - t0,
                error=f"invalid JSON from propose: {e}",
            )

        if "error" in proposal:
            return EditResult(
                success=False,
                time_s=time.time() - t0,
                error=proposal["error"],
                json_valid=True,
            )

        description = proposal.get("description", "")
        edits = proposal.get("edits", [])
        tokens = proposal.get("_meta", {}).get("tokens_generated", 0)

        json_valid = True
        schema_valid = (
            isinstance(description, str) and
            isinstance(edits, list) and
            all(isinstance(e, dict) and "old_string" in e and "new_string" in e
                for e in edits)
        )

        if not edits:
            return EditResult(
                success=True,
                description=description,
                time_s=time.time() - t0,
                json_valid=json_valid,
                schema_valid=schema_valid,
                edits_apply=True,
                syntax_ok=True,
                is_meaningful=False,
                tokens_generated=tokens,
            )

        # Step 2: Apply
        cmd_apply = [
            sys.executable, str(REPO_ROOT / "hypothesis_generator.py"),
            "apply",
            "--edits-json", json.dumps(proposal),
            "--train-py", str(train_py),
        ]
        proc_apply = subprocess.run(cmd_apply, capture_output=True, text=True, timeout=10)
        edits_apply = proc_apply.returncode == 0

        if not edits_apply:
            return EditResult(
                success=False,
                description=description,
                time_s=time.time() - t0,
                error=f"apply failed: {proc_apply.stderr[:200]}",
                json_valid=json_valid,
                schema_valid=schema_valid,
                edits_apply=False,
                tokens_generated=tokens,
            )

        # Step 3: Syntax check
        try:
            py_compile.compile(str(train_py), doraise=True)
            syntax_ok = True
        except py_compile.PyCompileError:
            syntax_ok = False

        # Step 4: Diff
        modified = train_py.read_text()
        is_meaningful = original != modified
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="train.py.orig",
            tofile="train.py",
        ))

        return EditResult(
            success=edits_apply and syntax_ok and is_meaningful,
            description=description,
            time_s=time.time() - t0,
            json_valid=json_valid,
            schema_valid=schema_valid,
            edits_apply=edits_apply,
            syntax_ok=syntax_ok,
            is_meaningful=is_meaningful,
            tokens_generated=tokens,
            diff=diff,
        )

    def is_available(self) -> bool:
        try:
            import requests
            if self.backend == "openai":
                probe = f"{self.url.rstrip('/')}/models"
            else:
                probe = f"{self.url}/api/tags"
            resp = requests.get(probe, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
