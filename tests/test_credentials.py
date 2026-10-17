"""Session-scoped credentials are masked by kind, so two honest sessions compare equal."""

import hashlib
import string

import pytest

from inspect_replay_verify.adapters import DEFAULT_CREDENTIAL_PATTERNS, ReplayAdapter
from inspect_replay_verify.transcript import RecordedCall


def entropy(seed: str, n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    """`n` high-entropy characters, derived at runtime so no token shape is committed."""
    out = ""
    counter = 0
    while len(out) < n:
        out += "".join(
            alphabet[b % len(alphabet)]
            for b in hashlib.sha256(f"{seed}{counter}".encode()).digest()
        )
        counter += 1
    return out[:n]


HEX = "0123456789abcdef"
UPPER = string.ascii_uppercase + string.digits
NEEDLES = {
    "anthropic": "sk-ant-api03-" + entropy("a", 93) + "AA",
    "openai": "sk-proj-" + entropy("o1", 20) + "T3BlbkFJ" + entropy("o2", 20),
    "openrouter": "sk-or-v1-" + entropy("r", 64, HEX),
    "github": "ghs_" + entropy("g", 36),
    "github_pat": "github_pat_" + entropy("p", 82),
    "gitlab": "glpat-" + entropy("l", 20),
    "google": "AIza" + entropy("z", 35),
    "groq": "gsk_" + entropy("q", 32),
    "xai": "xai-" + entropy("x", 40),
    "aws_access_key": "AKIA" + entropy("w", 16, UPPER),
}


def test_every_credential_kind_has_a_needle() -> None:
    assert DEFAULT_CREDENTIAL_PATTERNS, "read no credential patterns"
    assert set(NEEDLES) == set(DEFAULT_CREDENTIAL_PATTERNS)


@pytest.mark.parametrize("kind", sorted(NEEDLES))
def test_each_credential_is_masked_as_its_own_kind(kind: str) -> None:
    adapter = ReplayAdapter()
    text = f"token={NEEDLES[kind]} done"
    assert adapter.normalize(text) == f"token=<credential:{kind}> done"


@pytest.mark.parametrize("kind", sorted(NEEDLES))
def test_a_call_carrying_a_credential_is_credential_scoped(kind: str) -> None:
    call = RecordedCall(
        id="c",
        function="bash",
        arguments={"cmd": f"curl -H 'Authorization: {NEEDLES[kind]}' https://x"},
        result="",
        error=None,
        disposition="replay",
    )
    assert ReplayAdapter().credentials_in(call) == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "sk-ant-" + "a" * 20,
        "ghs_short",
        "AKIA" + "a" * 16,
        "no credential here",
    ],
)
def test_lookalikes_are_left_alone(text: str) -> None:
    assert ReplayAdapter().normalize(text) == text


def test_session_identifiers_are_masked_longest_first() -> None:
    adapter = ReplayAdapter()
    assert adapter.normalize("vm gb-1 and gb-12", ["gb-1", "gb-12"]) == "vm <session> and <session>"
