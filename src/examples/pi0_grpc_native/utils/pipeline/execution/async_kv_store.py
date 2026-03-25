from __future__ import annotations

import asyncio
import time
from typing import Any


class AsyncKVStore:
    """Epoch-indexed KV snapshot store for async producer/consumer flow."""

    def __init__(self, *, max_snapshots: int, cache_ttl_ms: int) -> None:
        self._max_snapshots = max(1, max_snapshots)
        self._cache_ttl_ms = max(0, cache_ttl_ms)
        self._snapshots: dict[int, tuple[Any, tuple]] = {}
        self._created_at: dict[int, float] = {}
        self._latest_epoch = 0
        self._final_epoch = 0
        self._cond = asyncio.Condition()

    def _prune(self) -> None:
        if not self._snapshots:
            return
        now = time.time()
        if self._cache_ttl_ms > 0:
            cutoff = now - (self._cache_ttl_ms / 1000.0)
            for epoch in sorted(self._snapshots):
                if self._created_at.get(epoch, now) < cutoff and epoch != self._latest_epoch and epoch != self._final_epoch:
                    self._snapshots.pop(epoch, None)
                    self._created_at.pop(epoch, None)
        if len(self._snapshots) > self._max_snapshots:
            for epoch in sorted(self._snapshots):
                if len(self._snapshots) <= self._max_snapshots:
                    break
                if epoch in (self._latest_epoch, self._final_epoch):
                    continue
                self._snapshots.pop(epoch, None)
                self._created_at.pop(epoch, None)

    async def publish(self, *, epoch: int, snapshot: tuple[Any, tuple], final: bool = False) -> None:
        async with self._cond:
            if epoch < self._latest_epoch:
                return
            self._snapshots[epoch] = snapshot
            self._created_at[epoch] = time.time()
            self._latest_epoch = max(self._latest_epoch, epoch)
            if final:
                self._final_epoch = max(self._final_epoch, epoch)
            self._prune()
            self._cond.notify_all()

    async def wait_for(
        self,
        *,
        min_epoch: int,
        require_final: bool,
        timeout_s: float = 0.2,
    ) -> tuple[int, tuple[Any, tuple], bool] | None:
        async with self._cond:
            while True:
                if require_final and self._final_epoch >= min_epoch:
                    snap = self._snapshots.get(self._final_epoch)
                    if snap is not None:
                        return self._final_epoch, snap, True
                if not require_final and self._latest_epoch >= min_epoch:
                    snap = self._snapshots.get(self._latest_epoch)
                    if snap is not None:
                        return self._latest_epoch, snap, self._latest_epoch == self._final_epoch
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    return None
