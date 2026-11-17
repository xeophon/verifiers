"""The built-in null harness: an in-process chat loop with the taskset's MCP tools (none of its own).

Drives a growing-message-list chat loop directly in the host event loop against the rollout's
interception endpoint, exposing any taskset MCP tool servers (`mcp_urls`) and adding no local
tools of its own. Because it never touches `runtime`, pure-chat, judge-scored, user-simulated, and
host-MCP rollouts spawn no container or sandbox at all.
"""

import json
from contextlib import AsyncExitStack
from typing import Any

from openai import APIError, AsyncOpenAI

from verifiers.v1.clients import RolloutContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.mcp import call_mcp, connect_mcp
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace


class NullHarnessConfig(HarnessConfig):
    """The built-in tool-less chat harness. Runs in-process on the host — no container or sandbox
    is provisioned unless taskset hooks or colocated servers touch the runtime."""


class NullHarness(Harness[NullHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True
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
        system_prompt, prompt = self.resolve_prompt(trace.task)
        disabled = set(self.config.disabled_tools or ())
        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        if isinstance(prompt, str):
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(m) for m in prompt)

        # No client-side retries: re-sending a request whose turn the interception server already
        # committed would fork a duplicate branch on the trace.
        async with (
            AsyncOpenAI(
                base_url=endpoint, api_key=secret, timeout=1800.0, max_retries=0
            ) as client,
            AsyncExitStack() as stack,
        ):
            mcp_tools, dispatch = await connect_mcp(stack, mcp_urls, runtime=runtime)
            tools = [t for t in mcp_tools if t["function"]["name"] not in disabled]
            while True:
                try:
                    completion = await client.chat.completions.create(
                        model=ctx.model,
                        messages=messages,
                        **({"tools": tools} if tools else {}),
                    )
                except APIError as e:
                    return ProgramResult(exit_code=1, stdout="", stderr=str(e))
                message = completion.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break
                for call in message.tool_calls:
                    name = call.function.name
                    try:
                        tool_args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": (
                                    f"error: invalid JSON in tool arguments ({e}); "
                                    "resend the call with valid JSON"
                                ),
                            }
                        )
                        continue
                    if not isinstance(tool_args, dict):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": (
                                    "error: tool arguments must be a JSON object, got "
                                    f"{type(tool_args).__name__}; resend as an object"
                                ),
                            }
                        )
                        continue
                    if name in dispatch and name not in disabled:
                        content = await call_mcp(dispatch, name, tool_args)
                    else:
                        content = f"error: unknown tool {name!r}"
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": content}
                    )
        return ProgramResult(exit_code=0, stdout="", stderr="")
