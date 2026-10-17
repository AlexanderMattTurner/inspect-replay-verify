# inspect-replay-verify

Independent replay verification for [Inspect AI](https://inspect.aisi.org.uk/) evals.

An Inspect sample is normally graded on the agent's own session: its transcript, its
final answer, the sandbox it ran in. Nothing in that loop catches a model that games its
own self-report, or that exploits a bug in the grading harness. For example, it might
satisfy a scorer through a side channel the transcript never shows, rewrite the tests it
is graded by, or claim success it never achieved. (agent-glovebox tracks this gap as
[#553](https://github.com/AlexanderMattTurner/agent-glovebox/blob/main/SECURITY.md).)

`replay_scorer()` grades a sample independently instead. It works in four steps:

1. Read the sample's recorded tool calls from its transcript (standard `.eval` format,
   from any sandbox provider).
2. Boot a **fresh instance of the task's own sandbox**, from the same pristine starting
   state: the sample's files and its setup script.
3. **Re-issue every recorded call mechanically**, with no model in the loop.
4. Score on what the replay itself observably did: an exit code, a file hash, a test
   result. It never relies on what the original run claimed.

A replay that cannot be completed or reproduced is **inconclusive, never a pass**.

```python
from inspect_ai import Task
from inspect_replay_verify import file_hash, replay_scorer, tests_pass

PRISTINE_TESTS = open("tests/test_solution.py").read()

Task(
    dataset=...,
    solver=...,
    sandbox="docker",
    scorer=[
        original_scorer(),  # what the run says about itself
        replay_scorer([     # what an independent replay shows
            tests_pass("pytest -q", restore_files={"tests/test_solution.py": PRISTINE_TESTS}),
            file_hash("dist/output.bin", sha256="9f86d0…"),
        ]),
    ],
)
```

```
pip install inspect-replay-verify
```

## Scope

This package is for tasks with a **checkable ground truth**: file state, exit codes, or
tests that pass or fail. It does not help with open-ended tasks that have no objective
effect to verify, so `replay_scorer([])` is always inconclusive.

## Verdicts

| verdict        | score value                    | meaning                                                         |
| -------------- | ------------------------------ | --------------------------------------------------------------- |
| `verified`     | `C`                            | complete replay, and every check passed                         |
| `refuted`      | `I`                            | complete replay, and at least one check definitely failed       |
| `inconclusive` | `N` (set `inconclusive_value`) | the replay could not be completed, reproduced, or judged        |

Any of these makes the verdict inconclusive:

- The sandbox type is not installed, the sandbox will not boot, or the setup script fails.
- A recorded call had an unknown effect: it was cancelled mid-flight, its sandbox became
  unavailable, it never completed, or it failed with a hard error. The replay stops there.
- A recorded tool is not available to the replay.
- A replayed call ends with a different outcome than the original did. Examples are a
  tool error the original did not raise, or an `exec` with another exit status. Whatever
  the replay then leaves behind cannot stand for the original.
- The sandbox was changed outside any tool call, and `untranscribed` is left at its
  default. See [Sandbox actions outside tool calls](#sandbox-actions-outside-tool-calls).
- A check raised or timed out, and no other check definitely failed.
- The adapter cannot trust the evidence. For glovebox, this means the guest was not
  confirmed quiescent.
- With `on_divergence="inconclusive"`, a replayed call's output differs from the recording.
- Only part of the transcript is readable (a bounded transcript with evicted events).

Each score's `metadata["replay"]` records every call's disposition, whether it diverged,
every check's observation, and adapter notes such as the quiescence census. The scorer
also reports the metrics `verified_rate`, `refuted_rate` and `inconclusive_rate`. Any
scores the run already had are copied to `metadata["original_scores"]` for comparison.

## Which calls are replayed

Calls come from the sample's `ToolEvent`s, so calls made inside sub-agents are included.
Calls from scorers, or from an earlier replay, are excluded. When a log has no events, the
calls come from the assistant messages instead.

| recorded call                                                         | replay                          |
| --------------------------------------------------------------------- | ------------------------------- |
| executed normally, or with a tool error (`timeout`, `permission`, …)  | re-issued                       |
| never executed: a `parsing` error, or rejected by an approver         | skipped                         |
| an agent handoff                                                      | skipped (its inner calls run)   |
| a tool in `skip_tools` (`think`, `submit`, `web_search`, …)           | skipped                         |
| `cancelled`, `sandbox_unavailable`, pending, or a hard failure        | stops the replay: inconclusive  |

Calls are re-issued one at a time, in the order they were recorded, through Inspect's own
`execute_tools`. Each call uses a **fresh instance** of the task's tool, rebuilt from its
registry parameters (same timeout, same user), so stateful tools such as `bash_session` do
not reach back into the original sandbox. Approvers are not consulted again: a call the
original approver rejected is skipped, and every other call already ran once.

Tools are resolved in this order:

1. `tools=` passed to the scorer
2. `state.tools`
3. Inspect's standard `bash`, `python`, `text_editor` and `bash_session`, with their default parameters

If the agent's tools were configured and are not on `state.tools`, as with
`react(tools=[bash(timeout=180)])`, pass them with `tools=`.

### Sandbox actions outside tool calls

A solver can change the sandbox without a tool call, through `sandbox().exec(...)` or
`write_file`. Harness setup does this, as in SWE-style tasks, and so does an agent that
runs inside the sandbox through the agent bridge. Inspect records each such action as a
`SandboxEvent`. A replay of the tool calls alone would start from a different state, so
`untranscribed=` decides what happens:

| `untranscribed`            | effect                                                                    |
| -------------------------- | ------------------------------------------------------------------------- |
| `"inconclusive"` (default) | refuse to judge, and name the action                                      |
| `"replay"`                 | re-issue each `exec` and `write_file` in transcript order, beside the tool calls |
| `"ignore"`                 | leave them out, so their effect counts against the agent                  |

Even with `"replay"`, some actions are unreproducible. One is an action whose recorded
input Inspect cut short: text of 10,000 characters or more, more than 20 lines, or any
binary content. The other is a command during which a model call ran. That last case is an agent
working inside the sandbox, so its effects are not a fixed command. Sandbox actions made
by an ordinary tool's own body, by Inspect's sample setup, or by scorers are never
counted. An agent's own sandbox calls are counted, whether the agent runs through a
handoff or as a tool.

Inspect records an action only once it returns. An `exec` or `write_file` that raises,
on a timeout for example, leaves no `SandboxEvent`. If a solver catches that error, the
action's effect is invisible to the replay.

### Divergence

A replayed tool call's output is compared with the recorded output. Before comparing, the
adapter masks session-scoped values: credentials matching known token formats, and
session identifiers such as VM names. A mismatch is a *divergence*. By default
(`on_divergence="record"`), divergences are counted in the metadata and the verdict comes
from the checks alone. Timestamps, PIDs and temporary names diverge in honest runs too.
Inspect's `bash` and `python` tools report a non-zero exit as plain output, not as an
error, so a command that fails only on replay shows up as a divergence. For a
deterministic task, set `on_divergence="inconclusive"` to refuse to judge any replay
whose observed outputs don't match the transcript.

## Checks

```python
from inspect_replay_verify import (
    exit_code, command_succeeds, tests_pass,      # live: run in the replay sandbox
    file_hash, file_exists, file_contains,        # evidence: read its state
    custom_check,
)
```

Live checks run first, while the sandbox still runs. Evidence checks run after them. The
adapter may freeze the sandbox between the two phases.

To stop an agent from passing by rewriting its own tests, pass `restore_files=` to a live
check. Pristine copies are then written over the agent's files before the check runs.

Every check is a `Check(name=..., phase=..., observe=...)`, and the built-ins above are
just constructors for one. `custom_check(fn, phase=...)` wraps your own async function.
A live check's function receives `Sandboxes` (`.environment(name)`), and an evidence
check's receives an `EvidenceReader` (`.read_file(path)`). Either returns a bool or a
`CheckOutcome(name=..., passed=...)`. A check that raises is inconclusive, never a pass.

Checks can also be built per sample from a `ReplayContext` (sample id, metadata, target):

```python
replay_scorer(lambda ctx: [file_contains("answer.txt", ctx.target[0])])
```

## Verifying a finished log

```python
from inspect_replay_verify import replay_log, file_hash

reports = replay_log("logs/my-eval.eval", [file_hash("out.txt", sha256="…")], dataset=task.dataset)
for (sample_id, epoch), report in reports.items():
    print(sample_id, report.verdict, report.explanation)
```

A log records the *names* of a sample's files, not their contents. A sample that shipped
files therefore needs its `dataset=` entry, and without one it is reported inconclusive.
The task's sandbox comes from the log header. Every other keyword is a `ReplayConfig`
field, as for `replay_scorer`, such as `sandbox=` or `untranscribed=`.

## Building on the pieces

The scorer and `replay_log` are thin. Each turns what a run left behind into a
`Recording` (`live_recording(state, target)` or `logged_recording(log, sample, dataset)`)
and calls `verify(recording, checks, ReplayConfig(...))`. Pass your own `Recording` to
verify a run from elsewhere. Below that, `replay(calls, pristine, checks, config)`
re-issues calls you already have, and the `booted(pristine, task_name)` context manager
gives you a fresh pristine sandbox. Any step that cannot stand for the original raises
`Inconclusive`, and the report turns it into the verdict.

## Backends

Adapters are chosen by sandbox type name. Pass `adapter=` to choose one explicitly.

### Docker, local, Kubernetes, …

These need no special handling. The default adapter reads evidence through the
sandbox's own `read_file`.

### glovebox

`GloveboxAdapter` is used for `sandbox="glovebox"`
([agent-glovebox](https://github.com/AlexanderMattTurner/agent-glovebox)'s
hypervisor-isolated microVM provider). It differs from the default adapter in two ways.

**It reads evidence only from a quiescent guest.** A replayed call can leave a process
running, a delayed writer, that rewrites a file just as a check reads it. glovebox's
breakout CTF already refuses evidence read from a guest it has not stopped. This adapter
mirrors that ordering:

1. Live checks run while the guest still admits commands. Their outcomes are captured on
   the host.
2. The run is quiesced with `DisposableRun.quiesce()`. Admissions stop, attached streams
   close, admitted execs drain, and guest root kills every process of every user the run
   ran a command as. A census then confirms that none survived.
3. Evidence is read by guest root through the trusted collector, addressed by sandbox
   name. This starts no agent code.

If quiescence is not established, the verdict is inconclusive. That covers survivors, a
drain that timed out, and a run whose user is root. It is inconclusive even when every
check would pass. Evidence inside a compose service's container cannot be read after the
microVM is quiesced, so a check aimed at a service is inconclusive. Check the VM's own
filesystem instead.

**It tolerates session-scoped credentials.** glovebox never puts a real credential in the
guest. The guest holds fixed public placeholders, and the host's egress proxy swaps in a
token minted for that session; GitHub App installation tokens live about an hour. A
replay VM gets its own fresh credentials from `sample_init`, and placeholders are
byte-identical across sessions, so honest replays need nothing extra. Any real token that
surfaced in a tool output is masked before outputs are compared. A call that carries such
a token literally in its arguments is marked `credential_scoped`, and its divergence is
tolerated rather than counted. To substitute a fresh value into such calls, pass
`GloveboxAdapter(remint=lambda kind, old: fresh_token(kind))`. Every adapter takes
`remint=`.

Note that the replay VM's own `sample_cleanup` publishes its egress record into the
running sample's `.eval`, alongside the original VM's record and under the replay VM's
own name.

### Writing an adapter

Subclass `ReplayAdapter`, set `sandbox_types`, and override any of these:

- `freeze(environments, notes)`: returns the `EvidenceReader` the evidence checks use.
  It runs after the live checks. Freeze the sandbox here if a writer the replay left
  running could change the evidence. Raise `Inconclusive` when the evidence cannot be
  trusted, and record what you observed in `notes`. Apart from masking its VM
  names, the glovebox adapter is this one method and its reader.
- `prepare(environments)`: called once the pristine state is reached, before any call.
- `rewrite_call(call)`: changes a call before it is re-issued.
- `session_identifiers(environments)` and `normalize(text, identifiers)`: mask values
  that differ between sessions before outputs are compared.

The engine owns the order: replay, then live checks, then `freeze`, then evidence
checks. An adapter cannot skip a phase or run one out of order.

Register the adapter so it is chosen without an explicit `adapter=`:

```toml
[project.entry-points.inspect_replay_verify]
mybackend = "my_package.replay:MyBackendAdapter"
```

## Costs and caveats

- A replay boots one extra sandbox per sample while the original is still alive. Cap
  concurrent replays with `max_concurrent=`.
- Replays are model-free, so they spend no tokens. They cost the wall-clock time of the
  sample's tool calls.
- Calls are re-issued sequentially, even ones the model made in parallel.
- Effects outside the sandbox (network services, external APIs) are not reset between the
  original run and the replay.
- If an approver *modified* a call (rather than approving or rejecting it), the replay
  re-issues the call as the model wrote it.
- Inspect exposes no public API for booting an extra sandbox instance or for reading the
  running sample's pristine `Sample`. The private seams this package uses are confined to
  `inspect_replay_verify._inspect_compat`, which fails at import and names any symbol a
  release has moved.

## Development

```
uv venv && uv pip install -e ".[dev]"
pytest                    # real evals on Inspect's local sandbox, and unit tests
pytest -m docker          # real evals in Docker containers (needs a Docker daemon;
                          # IRV_E2E_IMAGE picks another registry for the image)
pytest -m glovebox        # real glovebox microVMs (needs KVM and a provisioned host)
python tools/mutate.py    # break each decision the verdict rests on; a test must go red
ruff check . && ruff format --check . && mypy
```

CI runs each of these as its own job. The `e2e-glovebox` job provisions an
`ubuntu-latest` runner as a glovebox host, using agent-glovebox's own setup at a pinned
commit, and then runs `pytest -m glovebox` against real microVMs.

## License

Apache-2.0
