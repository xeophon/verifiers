"""The built-in default harness: a chat loop with sandboxed `bash`/`edit` tools (+ optional `search`, MCP).

Drives a growing-message-list chat loop in-process on the host while executing `bash` and `edit`
tool calls inside the rollout's isolated `runtime` (`docker` / `prime`), plus any taskset
MCP tools and an optional Serper `search` tool. Because `runtime` provisions lazily on first use,
rollouts where the model answers without calling `bash` or `edit` spawn no container or sandbox at
all — and rollouts that do call `bash`/`edit` skip bootstrapping `uv`/`openai`/`mcp` in the sandbox,
while keeping credentials (`secret`, `SERPER_API_KEY`) strictly on the host.
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any

import httpx
from openai import APIError, AsyncOpenAI

from verifiers.v1.clients import RolloutContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import RolloutError, SandboxError
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.mcp import call_mcp, connect_mcp
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

SERPER_URL = "https://google.serper.dev/search"

# Frames the model as a coding agent and names its local tools (a pure-text chat loop gets no
# harness-injected prompt). Each clause is included only when that tool is enabled.
BASH_SYSTEM_PROMPT = (
    "You are a coding agent. You have access to a bash tool for running shell commands."
)
EDIT_SYSTEM_PROMPT = (
    "You also have an edit tool for single-occurrence string replacement in a file."
)
SEARCH_PROMPT = (
    "You also have a search tool that returns Google results (title, URL, snippet) for a query; "
    "use it to research, and use bash (e.g. curl) to read result pages in full when needed."
)

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                }
            },
            "required": ["command"],
        },
    },
}

EDIT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a unique string in a file. old_str must appear exactly once in the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path (relative to cwd or absolute).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find (must appear exactly once).",
                },
                "new_str": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Run a web search via Serper (Google) and return the top organic results as title, "
            "URL, and snippet. Issue focused queries and call it several times to cover different "
            "angles; use the bash tool (e.g. curl) to read a result page in full."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def _format_search_results(results: list[dict[str, Any]], query: str) -> str:
    sections = []
    for i, result in enumerate(results, 1):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  - {snippet}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


async def _run_search(query: str, api_key: str, num_results: Any = 5) -> str:
    if not api_key:
        return "Error: no Serper API key (SERPER_API_KEY was not set in the eval environment)"
    try:
        count = max(1, int(num_results))
    except (TypeError, ValueError):
        count = 5
    try:
        async with httpx.AsyncClient(timeout=45.0) as http:
            response = await http.post(
                SERPER_URL,
                json={"q": query},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            organic = response.json().get("organic") or []
            return _format_search_results(organic[:count], query)
    except Exception as e:
        return f"search failed ({e}). Try again or rephrase the query."


async def _run_bash(runtime: Runtime, command: Any, env: dict[str, str]) -> str:
    if not isinstance(command, str):
        return "error: 'command' must be a string"
    await runtime.start()
    try:
        result = await asyncio.wait_for(
            runtime.run(["bash", "-c", command], env), timeout=3600
        )
        return result.stdout + result.stderr
    except TimeoutError:
        return "error: command timed out after 3600s"
    except RolloutError:
        raise
    except Exception as e:
        return f"error: {e}"


async def _run_edit(runtime: Runtime, path: Any, old_str: Any, new_str: Any) -> str:
    if not isinstance(path, str) or not path:
        return "error: 'path' is required"
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return "error: 'old_str' and 'new_str' must be strings"
    if not old_str:
        return "error: 'old_str' must be a non-empty string"
    await runtime.start()
    try:
        raw = await runtime.read(path)
    except SandboxError as e:
        return f"error: could not read {path}: {e}"
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return f"error: could not read {path}: {e}"
    count = content.count(old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {path} (found {count})"
    try:
        await runtime.write(path, content.replace(old_str, new_str, 1).encode("utf-8"))
    except SandboxError as e:
        return f"error: could not write {path}: {e}"
    return f"Edited {path}"


class DefaultHarnessConfig(HarnessConfig):
    """The built-in default harness: drives the chat loop on the host and dispatches `bash` /
    `edit` tool calls into the configured `runtime` (`docker` / `prime`) on demand."""

    edit: bool = True
    """Offer the `edit` tool (single-occurrence string replacement in a file) alongside `bash`.
    On by default; set `--harness.edit false` for a bash-only agent."""

    search: bool = False
    """Offer a `search` tool (Google web results via serper.dev). Requires `SERPER_API_KEY` in the
    eval environment; executed on the host so the agent's sandbox never receives the key."""


class DefaultHarness(Harness[DefaultHarnessConfig]):
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
        enable_bash = "bash" not in disabled
        enable_edit = self.config.edit and "edit" not in disabled
        enable_search = self.config.search and "search" not in disabled

        fragments: list[str] = []
        if enable_bash:
            fragments.append(BASH_SYSTEM_PROMPT)
        if enable_edit:
            fragments.append(EDIT_SYSTEM_PROMPT)
        if enable_search:
            fragments.append(SEARCH_PROMPT)
        system_prompt = "\n\n".join(
            p for p in (" ".join(fragments), system_prompt) if p
        )

        env = {**self.config.resolved_env}
        serper_key = ""
        if self.config.search:
            # Pop SERPER_API_KEY from the sandbox env so `bash` commands never inherit it.
            popped = env.pop("SERPER_API_KEY", None)
            if popped is None:
                popped = os.environ.get("SERPER_API_KEY")
            if not popped:
                raise ValueError(
                    "default search=true requires SERPER_API_KEY in the eval environment "
                    "(the host env or --harness.env)"
                )
            serper_key = popped

        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        if isinstance(prompt, str):
            if prompt:
                messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(m) for m in prompt)

        async with (
            AsyncOpenAI(
                base_url=endpoint, api_key=secret, timeout=1800.0, max_retries=0
            ) as client,
            AsyncExitStack() as stack,
        ):
            mcp_tools, dispatch = await connect_mcp(stack, mcp_urls, runtime=runtime)
            tools: list[dict[str, Any]] = []
            if enable_bash:
                tools.append(BASH_TOOL)
            if enable_edit:
                tools.append(EDIT_TOOL)
            if enable_search:
                tools.append(SEARCH_TOOL)
            tools.extend(t for t in mcp_tools if t["function"]["name"] not in disabled)

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
                        content: str | list[dict[str, Any]] = await call_mcp(
                            dispatch, name, tool_args
                        )
                    elif name == "bash" and enable_bash:
                        content = await _run_bash(
                            runtime, tool_args.get("command", ""), env
                        )
                    elif name == "edit" and enable_edit:
                        content = await _run_edit(
                            runtime,
                            tool_args.get("path"),
                            tool_args.get("old_str"),
                            tool_args.get("new_str"),
                        )
                    elif name == "search" and enable_search:
                        content = await _run_search(
                            str(tool_args.get("query", "")),
                            serper_key,
                            tool_args.get("num_results", 5),
                        )
                    else:
                        content = f"error: unknown tool {name!r}"
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": content}
                    )
        return ProgramResult(exit_code=0, stdout="", stderr="")
