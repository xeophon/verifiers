"""Execution runtimes for harnesses and isolated servers.

Each runtime decides WHERE a container/sandbox runs and HOW it reaches the host
interception server: docker (local container) or prime (remote sandbox).
They share the `Runtime` contract and provision lazily on first use, so rollouts
that never touch the runtime spawn no container or sandbox. `RuntimeConfig` is the
discriminated config union and `make_runtime` builds the runtime matching a config.
"""

from typing import Annotated

from pydantic import Field

from verifiers.v1.runtimes.base import (
    HOST,
    ProgramResult,
    Runtime,
    host_endpoint,
    reachable_url,
)
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime

RuntimeConfig = Annotated[
    DockerConfig | PrimeConfig,
    Field(discriminator="type"),
]


def _runtime_cls(config: RuntimeConfig) -> type[Runtime]:
    if isinstance(config, PrimeConfig):
        return PrimeRuntime
    return DockerRuntime


def make_runtime(config: RuntimeConfig, name: str | None = None) -> Runtime:
    return _runtime_cls(config)(config, name)


def runtime_is_local(config: RuntimeConfig) -> bool:
    """Whether a runtime of this config shares the host network (so a program inside it reaches a
    host service at localhost, no tunnel) — read off the runtime class, without provisioning one.
    The interception pool / rollout use it to decide whether to tunnel their host port via
    `host_endpoint`."""
    return _runtime_cls(config).is_local


__all__ = [
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "make_runtime",
    "runtime_is_local",
    "host_endpoint",
    "reachable_url",
    "HOST",
    "DockerConfig",
    "DockerRuntime",
    "PrimeConfig",
    "PrimeRuntime",
]
