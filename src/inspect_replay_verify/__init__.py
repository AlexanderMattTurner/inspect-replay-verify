"""Independent replay verification for Inspect AI evals.

Grade a sample on what an independent, model-free replay of its recorded tool calls
actually does in a fresh sandbox — not on the original run's own transcript or claims.
"""

from .adapters import ReplayAdapter, adapter_for
from .adapters.glovebox import GloveboxAdapter
from .checks import (
    Check,
    CheckOutcome,
    EvidenceCheck,
    EvidenceReader,
    LiveCheck,
    command_succeeds,
    custom_check,
    exit_code,
    file_contains,
    file_exists,
    file_hash,
    tests_pass,
)
from .metrics import inconclusive_rate, refuted_rate, verified_rate
from .offline import replay_log, replay_log_async
from .replay import PristineState, ReplayContext, ReplayReport, Verdict, replay
from .scorer import replay_scorer
from .transcript import DEFAULT_SKIP_TOOLS, RecordedCall, recorded_calls

__all__ = [
    "DEFAULT_SKIP_TOOLS",
    "Check",
    "CheckOutcome",
    "EvidenceCheck",
    "EvidenceReader",
    "GloveboxAdapter",
    "LiveCheck",
    "PristineState",
    "RecordedCall",
    "ReplayAdapter",
    "ReplayContext",
    "ReplayReport",
    "Verdict",
    "adapter_for",
    "command_succeeds",
    "custom_check",
    "exit_code",
    "file_contains",
    "file_exists",
    "file_hash",
    "inconclusive_rate",
    "recorded_calls",
    "refuted_rate",
    "replay",
    "replay_log",
    "replay_log_async",
    "replay_scorer",
    "tests_pass",
    "verified_rate",
]
