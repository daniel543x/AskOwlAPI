"""ResourceGate: per-resource async semaphores (PR 12 / decision 19).

Every LLM/search call passes a gate keyed by resource
(`model:<provider_id>`, `search:<config_id>`, ...) with a per-provider
`max_concurrency` (Ollama ships 1 — GPU-bound, serial; cloud providers
default higher). Foreground turns acquire their gates with a bounded
wait; a busy gate is what BUSY_POLICY reacts to (wait / background /
reject).
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ...tools.settings import settings

logger = logging.getLogger(__name__)


class GateBusy(Exception):
    """The resource gate is saturated and the bounded wait ran out."""


@dataclass(frozen=True)
class GateKey:
    """One gated resource: (kind, id) + its concurrency limit."""

    key: str
    limit: int


def model_gate(provider_id, max_concurrency: int) -> GateKey:
    return GateKey(f"model:{provider_id}", max(1, max_concurrency))


def search_gate(config_id, max_concurrency: int) -> GateKey:
    return GateKey(f"search:{config_id or 'default'}", max(1, max_concurrency))


class ResourceGateRegistry:
    """Process-local semaphores, created lazily per key.

    Single-node by design (decision 19); a Redis-backed gate arrives with
    the multi-node topology (same lazy pattern as ICache).
    """

    _semaphores: dict[str, asyncio.Semaphore] = {}

    @classmethod
    def semaphore(cls, key: GateKey) -> asyncio.Semaphore:
        semaphore = cls._semaphores.get(key.key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(key.limit)
            cls._semaphores[key.key] = semaphore
        return semaphore

    @classmethod
    def reset(cls) -> None:
        """Test hook: drop every gate."""
        cls._semaphores.clear()


async def acquire_gates(keys: list[GateKey], timeout: float | None) -> None:
    """Acquire every gate all-or-nothing within the bounded wait.

    Raises GateBusy when any wait exceeds `timeout` (already-acquired
    gates are released, so a rejected turn never wedges the resource).
    """
    if not keys:
        return
    acquired: list[asyncio.Semaphore] = []
    probed_busy: GateKey | None = None
    try:
        for key in keys:
            semaphore = ResourceGateRegistry.semaphore(key)
            if timeout is not None and timeout <= 0:
                # Non-blocking probe: locked() and acquire() run without
                # an intervening await, so the check is atomic on the loop.
                if semaphore.locked():
                    probed_busy = key
                    break
                await semaphore.acquire()
            elif timeout is None:
                await semaphore.acquire()
            else:
                await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
            acquired.append(semaphore)
    except TimeoutError:
        for semaphore in reversed(acquired):
            semaphore.release()
        keys_str = ", ".join(key.key for key in keys)
        logger.debug(
            "Gates busy (%s); bounded wait of %.1fs ran out.", keys_str, timeout or -1
        )
        raise GateBusy(
            f"Resource gate busy after {timeout}s wait: "
            f"{[key.key for key in keys]}"
        ) from None
    if probed_busy is not None:
        # All-or-nothing: give back what this probe already took.
        for semaphore in reversed(acquired):
            semaphore.release()
        raise GateBusy(
            f"Resource gate busy: {probed_busy.key} (of "
            f"{[key.key for key in keys]})"
        )


def release_gates(keys: list[GateKey]) -> None:
    for key in keys:
        ResourceGateRegistry.semaphore(key).release()


@asynccontextmanager
async def gated_turn(keys: list[GateKey], wait_timeout: float | None = None):
    """Hold the turn's gates for the whole body; busy raises GateBusy.

    `wait_timeout=None` waits forever (queued background jobs), the
    settings value bounds foreground waits.
    """
    timeout = (
        wait_timeout if wait_timeout is not None else settings.BUSY_WAIT_TIMEOUT_S
    )
    await acquire_gates(keys, timeout)
    try:
        yield
    finally:
        release_gates(keys)


@dataclass(frozen=True)
class TurnResources:
    """The gates one ask turn must hold (model + search), ready to wrap.

    `model_capabilities` rides along for ask-time gates that need model
    facts (e.g. the vision gate, PR 16) without re-resolving the model.
    """

    keys: list[GateKey]
    model_capabilities: list[str] = ()

    def gated(self, wait_timeout: float | None = None):
        return gated_turn(self.keys, wait_timeout=wait_timeout)


def provider_busy_detail() -> dict:
    """Coded 503 body for BUSY_POLICY=reject (mirrors the PR 9 codes)."""
    return {
        "code": "provider_busy",
        "message": "All resources for this turn are busy; retry or use "
        "background mode.",
    }
