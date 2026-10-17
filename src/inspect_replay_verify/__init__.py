"""Independent replay verification for Inspect AI evals.

Grade a sample on what an independent, model-free replay of its recorded tool calls
actually does in a fresh sandbox — not on the original run's own transcript or claims.
"""

from .adapters import ReplayAdapter, SandboxReader, adapter_for
from .adapters.glovebox import GloveboxAdapter
from .checks import (
    Check,
    CheckOutcome,
    EvidenceReader,
    Sandboxes,
    command_succeeds,
    custom_check,
    exit_code,
    file_contains,
    file_exists,
    file_hash,
    tests_pass,
)
from .metrics import inconclusive_rate, refuted_rate, verified_rate
from .offline import logged_recording, replay_log, replay_log_async
from .replay import Recording, ReplayConfig, ReplayContext, replay, verify
from .report import CallReplay, Inconclusive, ReplayReport, Verdict
from .sandbox import PristineState, booted, pristine_state
from .scorer import live_recording, replay_scorer
from .transcript import DEFAULT_SKIP_TOOLS, RecordedCall, recorded_calls

__all__ = [
    "DEFAULT_SKIP_TOOLS",
    "CallReplay",
    "Check",
    "CheckOutcome",
    "EvidenceReader",
    "GloveboxAdapter",
    "Inconclusive",
    "PristineState",
    "RecordedCall",
    "Recording",
    "ReplayAdapter",
    "ReplayConfig",
    "ReplayContext",
    "ReplayReport",
    "SandboxReader",
    "Sandboxes",
    "Verdict",
    "adapter_for",
    "booted",
    "command_succeeds",
    "custom_check",
    "exit_code",
    "file_contains",
    "file_exists",
    "file_hash",
    "inconclusive_rate",
    "live_recording",
    "logged_recording",
    "pristine_state",
    "recorded_calls",
    "refuted_rate",
    "replay",
    "replay_log",
    "replay_log_async",
    "replay_scorer",
    "tests_pass",
    "verified_rate",
    "verify",
]
