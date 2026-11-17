"""The compacting harness: an in-process context-rewrite loop (every turn branches).

Carries `notes` across compactions and sends a fresh `[system, user]` each turn — the task on the
first turn, then only the carried-over notes plus (for one turn) the latest tool output — so the
prompt is rewritten rather than appended, and every turn becomes its own branch in the trace graph.
Within a compaction it can use the taskset's MCP tools (`mcp_urls`). Runs in-process on the host,
spawning no container or sandbox.
"""

import json
import re
from contextlib import AsyncExitStack
from typing import Any

from openai import APIError, AsyncOpenAI

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.mcp import call_mcp, connect_mcp
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

SYSTEM = (
    "You solve a task across several turns; your NOTES are your only lasting memory. The "
    "first turn shows the task; after that you see only your notes — plus, the turn right "
    "after you call a tool, that tool's output (shown ONCE, so copy what you need into your "
    "notes before it's gone). Each turn, reply with brief reasoning, then your COMPLETE "
    "updated notes in <notes>...</notes>, and either call a tool to gather more or give the "
    "final answer in <answer>...</answer>."
)


def _extract(tag: str, text: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


class CompactingHarnessConfig(HarnessConfig):
    """A context-rewrite harness: rebuilds its prompt from carried-over notes each turn instead of
    appending, so the trajectory branches at every compaction."""


class CompactingHarness(Harness[CompactingHarnessConfig]):
    SUPPORTS_MCP = True
    RUNS_ON_HOST = True

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        _, task_prompt = self.resolve_prompt(trace.task)
        notes: str | None = None
        tool_output: str | None = None
        async with (
            AsyncOpenAI(
                base_url=endpoint, api_key=secret, timeout=1800.0, max_retries=0
            ) as client,
            AsyncExitStack() as stack,
        ):
            tools, dispatch = await connect_mcp(stack, mcp_urls, runtime=runtime)
            while True:
                parts = [
                    f"Task: {task_prompt}" if notes is None else f"Notes:\n{notes}"
                ]
                if tool_output is not None:
                    parts.append(f"Latest tool output:\n{tool_output}")
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "\n\n".join(parts)},
                ]
                try:
                    completion = await client.chat.completions.create(
                        model=ctx.model,
                        messages=messages,
                        **({"tools": tools} if tools else {}),
                    )
                except APIError as e:
                    return ProgramResult(exit_code=1, stdout="", stderr=str(e))
                message = completion.choices[0].message
                notes = _extract("notes", message.content or "") or notes
                answer = _extract("answer", message.content or "")
                if answer is not None:
                    return ProgramResult(exit_code=0, stdout=answer, stderr="")
                tool_output = None
                if message.tool_calls:
                    results: list[str] = []
                    for call in message.tool_calls:
                        try:
                            args = json.loads(call.function.arguments or "{}")
                        except json.JSONDecodeError as e:
                            results.append(
                                f"{call.function.name}:\nerror: invalid JSON ({e})"
                            )
                            continue
                        if not isinstance(args, dict):
                            results.append(
                                f"{call.function.name}:\nerror: arguments must be a JSON object"
                            )
                            continue
                        if call.function.name in dispatch:
                            raw = await call_mcp(dispatch, call.function.name, args)
                            text = raw if isinstance(raw, str) else json.dumps(raw)
                        else:
                            text = f"error: unknown tool {call.function.name!r}"
                        results.append(f"{call.function.name}:\n{text}")
                    tool_output = "\n\n".join(results)
