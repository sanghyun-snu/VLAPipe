from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch


class LayerStatus(str, Enum):
    PENDING = "pending"
    RECEIVED = "received"
    CONSUMED = "consumed"


@dataclass(frozen=True)
class LayerSnapshot:
    layer_idx: int
    status: LayerStatus
    has_cache: bool


class LayerPayload(Protocol):
    request_id: str
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    prefix_pad_mask: torch.Tensor | None


class LayerState:
    """Tracks KV cache readiness and consumption per layer."""

    def __init__(self, layer_count: int) -> None:
        self._layer_count = layer_count
        self._status_by_request: dict[str, dict[int, LayerStatus]] = {}
        self._cache_by_request: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._prefix_pad_mask_by_request: dict[str, torch.Tensor] = {}
        self._lock = asyncio.Lock()

    def _ensure_session(self, request_id: str) -> None:
        if request_id not in self._status_by_request:
            self._status_by_request[request_id] = {idx: LayerStatus.PENDING for idx in range(self._layer_count)}
            self._cache_by_request[request_id] = {}

    async def ingest(self, payload: LayerPayload) -> None:
        async with self._lock:
            self._ensure_session(payload.request_id)
            self._cache_by_request[payload.request_id][payload.layer_idx] = (payload.key, payload.value)
            self._status_by_request[payload.request_id][payload.layer_idx] = LayerStatus.RECEIVED
            if payload.prefix_pad_mask is not None:
                self._prefix_pad_mask_by_request[payload.request_id] = payload.prefix_pad_mask

    async def consume(self, layer_idx: int, request_id: str = "default") -> tuple[torch.Tensor, torch.Tensor]:
        async with self._lock:
            self._ensure_session(request_id)
            if self._status_by_request[request_id][layer_idx] != LayerStatus.RECEIVED:
                raise RuntimeError(f"Layer {layer_idx} is not ready for request {request_id}")
            kv = self._cache_by_request[request_id][layer_idx]
            self._status_by_request[request_id][layer_idx] = LayerStatus.CONSUMED
            return kv

    async def is_ready(self, layer_idx: int, request_id: str = "default") -> bool:
        async with self._lock:
            self._ensure_session(request_id)
            return self._status_by_request[request_id][layer_idx] == LayerStatus.RECEIVED

    async def status(self, layer_idx: int, request_id: str = "default") -> LayerStatus:
        async with self._lock:
            self._ensure_session(request_id)
            return self._status_by_request[request_id][layer_idx]

    async def all_consumed(self, request_id: str = "default") -> bool:
        async with self._lock:
            self._ensure_session(request_id)
            return all(state == LayerStatus.CONSUMED for state in self._status_by_request[request_id].values())

    async def snapshots(self, indices: Iterable[int] | None = None, request_id: str = "default") -> list[LayerSnapshot]:
        layer_indices = list(indices) if indices is not None else list(range(self._layer_count))
        async with self._lock:
            self._ensure_session(request_id)
            return [
                LayerSnapshot(
                    layer_idx=layer_idx,
                    status=self._status_by_request[request_id][layer_idx],
                    has_cache=layer_idx in self._cache_by_request[request_id],
                )
                for layer_idx in layer_indices
            ]

    async def clear_session(self, request_id: str) -> None:
        async with self._lock:
            self._status_by_request.pop(request_id, None)
            self._cache_by_request.pop(request_id, None)
            self._prefix_pad_mask_by_request.pop(request_id, None)

    async def get_prefix_pad_mask(self, request_id: str) -> torch.Tensor:
        async with self._lock:
            if request_id not in self._prefix_pad_mask_by_request:
                raise KeyError(f"prefix_pad_mask missing for request {request_id}")
            return self._prefix_pad_mask_by_request[request_id]
