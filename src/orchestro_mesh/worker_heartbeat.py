from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

from orchestro_mesh.models import Heartbeat, NodeInventory

logger = logging.getLogger(__name__)


async def send_heartbeat(
    gateway_url: str,
    token: str | None,
    beat: Heartbeat,
    timeout_s: float = 10.0,
) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = gateway_url.rstrip("/") + "/mesh/heartbeat"
    async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as client:
        response = await client.post(url, json=beat.model_dump(mode="json", exclude_none=True))
        response.raise_for_status()


class HeartbeatLoop:
    """Background loop that periodically POSTs the worker's heartbeat to a gateway.

    On heartbeat failure, applies capped exponential backoff before the next attempt.
    """

    def __init__(
        self,
        *,
        gateway_url: str,
        token: str | None,
        inventory_provider: Callable[[], NodeInventory | None],
        interval_s: float = 30.0,
        backoff_max_s: float = 300.0,
    ) -> None:
        self.gateway_url = gateway_url
        self.token = token
        self.inventory_provider = inventory_provider
        self.interval_s = interval_s
        self.backoff_max_s = backoff_max_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._consecutive_failures = 0

    async def _tick(self) -> None:
        inventory = self.inventory_provider()
        if inventory is None:
            return
        beat = Heartbeat(
            node_id=inventory.node_id,
            status=inventory.status,
            current_jobs=inventory.current_jobs,
            queue_depth=inventory.queue_depth,
        )
        await send_heartbeat(self.gateway_url, self.token, beat)

    def _next_delay(self) -> float:
        if self._consecutive_failures == 0:
            return self.interval_s
        backoff = self.interval_s * (2 ** min(self._consecutive_failures - 1, 8))
        return min(backoff, self.backoff_max_s)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick()
                self._consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                self._consecutive_failures += 1
                logger.warning(
                    "heartbeat.tick_failed consecutive=%d error=%s",
                    self._consecutive_failures,
                    exc,
                )
            delay = self._next_delay()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
