"""Tests for removing SubprocessRuntime, lazy Runtime provisioning, and zero-sandbox rollouts."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

import verifiers.v1 as vf
from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.runtimes import DockerRuntime, PrimeRuntime, make_runtime


class _ScriptedClient(vf.Client):
    """Deterministic in-memory client returning pre-scripted OpenAI chat-completions wire turns."""

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: vf.SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> vf.Response:
        self.calls.append(body)
        choice_msg = self._turns.pop(0)
        raw = {
            "id": f"cmpl-{len(self.calls)}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": choice_msg,
                    "finish_reason": (
                        "tool_calls" if choice_msg.get("tool_calls") else "stop"
                    ),
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        response = dialect.parse_response(dialect.validate_response(raw))
        response.raw = raw
        return response


class _FakeRuntime(vf.Runtime):
    """In-memory Runtime tracking exact `_start` / `cleanup` / op counts."""

    def __init__(self) -> None:
        super().__init__(name="fake-rt")
        self.config = vf.DockerConfig()
        self.start_calls = 0
        self.cleanup_calls = 0
        self.commands: list[list[str]] = []
        self.files: dict[str, bytes] = {}

    async def _start(self) -> None:
        await asyncio.sleep(0.01)
        self.start_calls += 1

    async def _run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        self.commands.append(argv)
        return vf.ProgramResult(exit_code=0, stdout="ran-in-sandbox\n", stderr="")

    async def _read(self, path: str) -> bytes:
        if path not in self.files:
            raise vf.SandboxError(f"missing {path}")
        return self.files[path]

    async def _write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def test_subprocess_runtime_removed_and_defaults_safe() -> None:
    assert not hasattr(vf, "SubprocessConfig")
    assert not hasattr(vf, "ModalConfig")
    adapter = TypeAdapter(vf.RuntimeConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "subprocess"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "modal"})

    assert isinstance(vf.HarnessConfig().runtime, vf.DockerConfig)
    assert vf.ToolsetConfig().runtime is None
    assert vf.UserConfig().runtime is None


async def test_runtime_provisions_lazily_and_skips_unstarted_teardown() -> None:
    rt = _FakeRuntime()
    assert not rt.started
    await rt.stop()
    assert rt.start_calls == 0
    assert rt.cleanup_calls == 0

    trace = vf.Trace(
        task=vf.Task(idx=0, prompt="q"),
        nodes=[
            vf.MessageNode(
                parent=None, message=vf.UserMessage(content="q"), sampled=False
            ),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content="from-trace"),
                sampled=True,
            ),
        ],
    )
    # Optional file fallback must NOT boot an unstarted container just to miss the file.
    assert (
        await vf.read_answer_file_or_last_reply(rt, "answer.txt", trace) == "from-trace"
    )
    assert not rt.started
    assert rt.start_calls == 0

    # Concurrent first touches provision exactly once.
    await asyncio.gather(rt.write("a.txt", b"hello"), rt.run(["echo", "hi"], {}))
    assert rt.started
    assert rt.start_calls == 1
    assert await rt.read("a.txt") == b"hello"

    await rt.stop()
    assert rt.cleanup_calls == 1


@pytest.mark.parametrize("runtime_type", ["docker", "prime"])
@pytest.mark.parametrize("harness_id", ["null", "default"])
async def test_single_turn_rollout_spawns_no_sandbox_or_tunnel(
    monkeypatch: pytest.MonkeyPatch, runtime_type: str, harness_id: str
) -> None:
    async def _forbid_start(self) -> None:
        raise AssertionError(f"{type(self).__name__}._start should not be called")

    monkeypatch.setattr(DockerRuntime, "_start", _forbid_start)
    monkeypatch.setattr(PrimeRuntime, "_start", _forbid_start)

    config = vf.EnvConfig(
        taskset={"id": "echo-v1"},
        harness={"id": harness_id, "runtime": {"type": runtime_type}},
    )
    env = vf.Environment(config)
    tasks = env.taskset.load_tasks()[:1]
    client = _ScriptedClient([{"role": "assistant", "content": "hello world"}])
    ctx = vf.RolloutContext(
        model="test-model", client=client, sampling=vf.SamplingConfig()
    )
    async with env.serving(tasks):
        ep = env.episode(tasks[0], ctx, n=1)
        (trace,) = await ep.run()
        assert ep.rollouts[0].runtime is not None
        assert not ep.rollouts[0].runtime.started

    assert trace.errors == []
    assert trace.reward == 1.0
    if harness_id == "null":
        assert "tools" not in client.calls[0]


async def test_in_process_mcp_tool_and_user_sim_spawn_no_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _forbid_start(self) -> None:
        raise AssertionError(f"{type(self).__name__}._start should not be called")

    monkeypatch.setattr(DockerRuntime, "_start", _forbid_start)

    # 1. Host MCP tool + per-rollout shared state (`counter-tool-v1`)
    tool_env = vf.Environment(
        vf.EnvConfig(taskset={"id": "counter-tool-v1"}, harness={"id": "null"})
    )
    tool_tasks = tool_env.taskset.load_tasks()[:1]
    tool_client = _ScriptedClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "counter_bump", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "counter_bump", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "<answer>done</answer>"},
        ]
    )
    ctx = vf.RolloutContext(
        model="test-model", client=tool_client, sampling=vf.SamplingConfig()
    )
    async with tool_env.serving(tool_tasks):
        ep = tool_env.episode(tool_tasks[0], ctx, n=1)
        (trace,) = await ep.run()
        assert ep.rollouts[0].runtime is not None
        assert not ep.rollouts[0].runtime.started
    assert trace.errors == []
    assert trace.reward == 1.0

    # 2. Multi-turn host user simulator (`echo-user-sim-v1`)
    user_env = vf.Environment(
        vf.EnvConfig(taskset={"id": "echo-user-sim-v1"}, harness={"id": "null"})
    )
    user_tasks = user_env.taskset.load_tasks()[:1]
    user_client = _ScriptedClient(
        [
            {"role": "assistant", "content": "hello world"},
            {"role": "assistant", "content": "goodbye world"},
        ]
    )
    user_ctx = vf.RolloutContext(
        model="test-model", client=user_client, sampling=vf.SamplingConfig()
    )
    async with user_env.serving(user_tasks):
        user_ep = user_env.episode(user_tasks[0], user_ctx, n=1)
        (user_trace,) = await user_ep.run()
        assert user_ep.rollouts[0].runtime is not None
        assert not user_ep.rollouts[0].runtime.started
    assert user_trace.errors == []
    assert user_trace.num_turns == 2
    assert user_trace.reward == 1.0


async def test_default_harness_lazily_starts_runtime_on_bash_or_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRuntime()
    fake.files["note.txt"] = b"old value\n"
    monkeypatch.setattr(
        "verifiers.v1.rollout.make_runtime", lambda config, name=None: fake
    )

    env = vf.Environment(
        vf.EnvConfig(taskset={"id": "echo-v1"}, harness={"id": "default"})
    )

    # Remove EchoTaskset's single_turn @stop for this multi-step tool test by allowing 3 turns:
    async def _allow_four_turns(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 4

    monkeypatch.setattr(type(env.taskset), "single_turn", vf.stop(_allow_four_turns))
    tasks = env.taskset.load_tasks()[:1]
    client = _ScriptedClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "b1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "echo hi"}',
                        },
                    },
                    {
                        "id": "e1",
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": (
                                '{"path": "note.txt", "old_str": "old value", '
                                '"new_str": "new value"}'
                            ),
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "hello world"},
        ]
    )
    ctx = vf.RolloutContext(
        model="test-model", client=client, sampling=vf.SamplingConfig()
    )
    assert not fake.started
    async with env.serving(tasks):
        ep = env.episode(tasks[0], ctx, n=1)
        (trace,) = await ep.run()

    assert trace.errors == []
    assert trace.reward == 1.0
    assert fake.started
    assert fake.start_calls == 1
    assert fake.commands == [["bash", "-c", "echo hi"]]
    assert fake.files["note.txt"] == b"new value\n"
    assert fake.cleanup_calls == 1
    assert isinstance(make_runtime(vf.DockerConfig()), DockerRuntime)
