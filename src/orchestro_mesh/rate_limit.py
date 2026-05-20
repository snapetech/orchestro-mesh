from __future__ import annotations

import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    """In-process sliding-window per-requester rate limiter.

    Not safe across multiple gateway processes. For a multi-replica deployment,
    swap this for a shared store (Redis, Postgres-backed counter, etc.).
    """

    def __init__(
        self,
        per_minute: dict[str, int] | None = None,
        default_per_minute: int | None = None,
        clock: callable = time.monotonic,  # type: ignore[type-arg]
    ) -> None:
        self.per_minute = per_minute or {}
        self.default_per_minute = default_per_minute
        self._clock = clock
        self._history: dict[str, deque[float]] = defaultdict(deque)

    def limit_for(self, requester: str) -> int | None:
        if requester in self.per_minute:
            return self.per_minute[requester]
        return self.default_per_minute

    def check(self, requester: str) -> tuple[bool, int | None, float | None]:
        """Return (allowed, limit, retry_after_seconds).

        retry_after is only populated when the request is denied.
        """
        limit = self.limit_for(requester)
        if limit is None or limit <= 0:
            return True, None, None
        now = self._clock()
        window = self._history[requester]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit:
            retry_after = max(0.0, 60.0 - (now - window[0]))
            return False, limit, retry_after
        window.append(now)
        return True, limit, None
