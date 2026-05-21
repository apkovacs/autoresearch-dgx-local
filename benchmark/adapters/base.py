"""Abstract base class for benchmark harness adapters.

Each adapter wraps a different agent/tool framework (raw Ollama API, Aider,
Claude Code, OpenHands) and exposes a uniform interface for proposing and
applying a single edit to train.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EditResult:
    """Result of a single propose-and-apply cycle."""
    success: bool               # Did the edit apply and produce valid Python?
    description: str = ""       # What the model proposed
    time_s: float = 0.0         # Wall-clock time for the full cycle
    error: str | None = None    # Error message if success=False
    syntax_ok: bool = False     # Does the edited file pass py_compile?
    json_valid: bool = False    # Did the model produce valid JSON? (raw API only)
    schema_valid: bool = False  # Does the JSON match expected schema? (raw API only)
    edits_apply: bool = False   # Did all old_strings match? (raw API only)
    is_meaningful: bool = False # Is the edit a non-trivial change?
    tokens_generated: int = 0   # Output tokens (if available from API)
    diff: str = ""              # Unified diff of the change


class HarnessAdapter(ABC):
    """Base class for benchmark harness adapters."""

    name: str = "base"

    @abstractmethod
    def propose_and_apply(self, workdir: Path, model: str,
                          results_tsv: Path | None = None) -> EditResult:
        """Propose and apply a single edit to train.py in workdir.

        Args:
            workdir: Directory containing train.py (a copy — safe to modify)
            model: Ollama model name (e.g., "qwen3.6:27b")
            results_tsv: Optional path to results history for context

        Returns:
            EditResult with success/failure details
        """
        ...

    def setup(self) -> None:
        """One-time setup (install dependencies, pull images, etc.)."""
        pass

    def cleanup(self) -> None:
        """Cleanup after all trials."""
        pass

    def is_available(self) -> bool:
        """Check if this adapter's dependencies are installed."""
        return True
