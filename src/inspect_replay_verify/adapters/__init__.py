"""Backend adapters, chosen by the replay sandbox's type name."""

from __future__ import annotations

from importlib.metadata import entry_points

from .base import (
    DEFAULT_CREDENTIAL_PATTERNS,
    Collected,
    DefaultEvidenceReader,
    ReplayAdapter,
)
from .glovebox import GloveboxAdapter

ENTRY_POINT_GROUP = "inspect_replay_verify"

__all__ = [
    "DEFAULT_CREDENTIAL_PATTERNS",
    "ENTRY_POINT_GROUP",
    "Collected",
    "DefaultEvidenceReader",
    "GloveboxAdapter",
    "ReplayAdapter",
    "adapter_for",
]


def _registered() -> list[type[ReplayAdapter]]:
    found: list[type[ReplayAdapter]] = [GloveboxAdapter]
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
        except Exception:
            continue
        if isinstance(cls, type) and issubclass(cls, ReplayAdapter) and cls not in found:
            found.append(cls)
    return found


def adapter_for(sandbox_type: str) -> ReplayAdapter:
    """The adapter registered for `sandbox_type`, or the default adapter."""
    for cls in _registered():
        if sandbox_type in cls.sandbox_types:
            return cls()
    return ReplayAdapter()
