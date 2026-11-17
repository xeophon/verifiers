"""Host-side MCP client helpers shared by in-process harnesses (`null`, `default`, `compact`)."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from mcp import ClientSession

    from verifiers.v1.runtimes import Runtime

# Executes one HTTP request against loopback *inside* a remote sandbox (for colocated MCP servers
# on runtimes like Prime whose sandbox ports aren't directly exposed to the host).
_LOOPBACK_PROXY = """
import base64, json, os, sys, urllib.error, urllib.request
arg = sys.argv[1]
if arg.startswith("@"):
    with open(arg[1:], "rb") as f:
        req_data = json.loads(f.read())
    try:
        os.unlink(arg[1:])
    except OSError:
        pass
else:
    req_data = json.loads(arg)
body = base64.b64decode(req_data["body"]) if req_data["body"] else None
headers = {
    k: v for k, v in req_data["headers"]
    if k.lower() not in ("host", "content-length", "connection")
}
req = urllib.request.Request(req_data["url"], data=body, headers=headers, method=req_data["method"])
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        status, resp_headers, raw = resp.status, list(resp.headers.items()), resp.read()
except urllib.error.HTTPError as err:
    status, resp_headers, raw = err.code, list(err.headers.items()), err.read()
sys.stdout.write(json.dumps({
    "status": status,
    "headers": resp_headers,
    "body": base64.b64encode(raw).decode(),
}))
"""


class _RuntimeLoopbackTransport(httpx.AsyncBaseTransport):
    """Forwards HTTP requests to `127.0.0.1` inside a remote `Runtime` via `runtime.run`."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        payload = json.dumps(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": list(request.headers.multi_items()),
                "body": base64.b64encode(body).decode() if body else "",
            }
        )
        if len(payload) > 32_768:
            tmp = f"/tmp/vf-mcp-req-{uuid.uuid4().hex}.json"
            await self._runtime.write(tmp, payload.encode())
            arg = f"@{tmp}"
        else:
            arg = payload
        result = await self._runtime.run(["python3", "-c", _LOOPBACK_PROXY, arg], {})
        if result.exit_code != 0:
            raise httpx.ConnectError(
                f"in-runtime MCP request failed: {(result.stderr or result.stdout).strip()[-500:]}"
            )
        data = json.loads(result.stdout)
        return httpx.Response(
            status_code=data["status"],
            headers=data["headers"],
            content=base64.b64decode(data["body"]),
        )


def _is_remote_loopback(url: str, runtime: Runtime | None) -> bool:
    return (
        runtime is not None
        and not runtime.is_local
        and url.startswith(("http://127.0.0.1:", "http://localhost:"))
    )


async def connect_mcp(
    stack: AsyncExitStack,
    mcp_urls: Mapping[str, str],
    *,
    runtime: Runtime | None = None,
    headers_by_server: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[ClientSession, str]]]:
    """Connect to each MCP server in `mcp_urls` (`name -> url`); return `(tool_schemas, dispatch)`
    mapping `<name>_<tool>` to `(session, raw_tool_name)`."""
    if not mcp_urls:
        return [], {}
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    tool_schemas: list[dict[str, Any]] = []
    dispatch: dict[str, tuple[ClientSession, str]] = {}
    for name, url in mcp_urls.items():
        extra_headers = dict((headers_by_server or {}).get(name) or {}) or None
        if _is_remote_loopback(url, runtime):
            assert runtime is not None
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    transport=_RuntimeLoopbackTransport(runtime),
                    headers=extra_headers,
                    timeout=300.0,
                )
            )
        else:
            http_client = await stack.enter_async_context(
                create_mcp_http_client(headers=extra_headers)
            )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            full = f"{name}_{tool.name}"
            tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": full,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
            dispatch[full] = (session, tool.name)
    return tool_schemas, dispatch


def mcp_content_to_chat_content(blocks: list[Any]) -> str | list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            url = f"data:{block.mimeType};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": str(block)})
    if not parts:
        return str(blocks)
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


async def call_mcp(
    dispatch: Mapping[str, tuple[ClientSession, str]],
    name: str,
    arguments: dict[str, Any],
) -> str | list[dict[str, Any]]:
    session, raw = dispatch[name]
    result = await session.call_tool(raw, arguments)
    return mcp_content_to_chat_content(result.content)
