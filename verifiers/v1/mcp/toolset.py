"""`Toolset` + `ToolsetConfig`: a tool server authored as a vf-native class with `@vf.tool` methods.

A task gives the harness tools by declaring `Toolset`s from `Taskset.tools`. The config carries
placement (where the server runs); the class carries the `@vf.tool` methods the model calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import RuntimeConfig
from verifiers.v1.state import StateT

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class ToolsetConfig(BaseConfig):
    """Where one tool server runs (placement). The default — in-process on the host (`runtime=None`)
    — is the cheap one: the server runs inside the eval/worker event loop where `verifiers` and the
    env module are already loaded (no container or subprocess to spawn), and the harness reaches it
    over localhost or a tunnel. `colocated`, `shared`, and `runtime` trade that off:
      - default (`runtime=None`, `colocated=False`, `shared=False`): in-process on the host, per rollout.
      - `runtime={type="docker"|"prime"}`: its own isolated container/sandbox, per rollout.
      - `colocated=True`: inside the harness's OWN runtime, per rollout (uploads + installs `verifiers`
        and the env module in the sandbox).
      - `shared=True`: one instance for the whole eval (in-process on the host by default, or in its
        own `runtime`; per-rollout writable state stays isolated via `self.state`).
    See the placement/isolation section of `verifiers/v1/GUIDE.md` for the trade-offs of each.
    Subclass to add the server's own knobs (the data its `@tool` methods read). The server name is
    the class's `TOOL_PREFIX` ClassVar, not a field here — it's an identity (the model sees
    `<prefix>_<tool>`, baked into the taskset's prompt), not a tunable knob."""

    colocated: bool = False
    """Run the server inside the harness's runtime (reached in-sandbox, no tunnel). Off by
    default — in a sandbox the harness runtime must install the env package + `verifiers` (a
    per-rollout cost), so prefer the default host placement unless co-locating genuinely helps."""
    shared: bool = False
    """Run one server instance for the whole eval, shared across rollouts (on the host by default,
    or in its own `runtime`) — pays an expensive `setup` (a corpus/index) once. It may still be
    writable: each rollout reads and writes its OWN `self.state` (the framework tags the per-rollout
    state channel onto the shared server's URL), so concurrent rollouts don't corrupt each other.
    Mutually exclusive with `colocated`."""
    runtime: RuntimeConfig | None = None
    """An isolated container/sandbox runtime (`docker` / `prime`) for this server, used
    unless `colocated`. `None` (default) runs the server in-process on the host."""
    url: str | None = None
    """An already-running streamable-HTTP MCP endpoint to connect to instead of launching a
    server (e.g. a public remote like DeepWiki). When set, placement is ignored — the toolset
    needs no `@tool` methods, the model just sees the remote's tools as `<name>_<tool>`."""

    @model_validator(mode="after")
    def _validate_placement(self) -> "ToolsetConfig":
        if self.colocated and self.shared:
            raise ValueError("colocated and shared are mutually exclusive")
        return self


class Toolset(ServerBase[ConfigT, StateT]):
    """A tool server authored as a class: write `@vf.tool` methods (the model calls them as
    `<prefix>_<method>`; the docstring is the tool description), reading config off `self.config` and
    optionally the rollout's shared `self.state`. Example:

        class GlossaryToolsetConfig(vf.ToolsetConfig):
            facts: dict[str, str] = {}

        class GlossaryToolset(vf.Toolset[GlossaryToolsetConfig]):
            @vf.tool
            def lookup(self, name: str) -> str:
                return self.config.facts.get(name.lower(), "unknown")

    Parameterize a stateful toolset with its `State` subclass too — `Toolset[Config, MyState]` — so
    `self.state` is typed; it defaults to the base `State`.
    """

    def _register(self, mcp: FastMCP) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
