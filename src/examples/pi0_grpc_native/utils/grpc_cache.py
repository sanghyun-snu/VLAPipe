from __future__ import annotations

from dataclasses import dataclass

import grpc
import torch

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc

from .layer_state import LayerState
from .stream_protocol import proto_to_tensor

DEFAULT_HOST = "127.0.0.1"
DEFAULT_SUFFIX_PORT = 50061
DEFAULT_PREFIX_PORT = 50062


@dataclass(frozen=True)
class ServerAddress:
    host: str
    port: int

    @property
    def value(self) -> str:
        return f"{self.host}:{self.port}"


class PrefixClient:
    def __init__(self, address: str) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = pb2_grpc.PrefixServiceStub(self._channel)

    async def stream_prefix(self, request: pb2.PrefixRequest, timeout_s: float | None = None):
        async for chunk in self._stub.StreamPrefixKV(request, timeout=timeout_s):
            yield chunk

    async def close(self) -> None:
        await self._channel.close()


class SuffixClient:
    def __init__(self, address: str) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = pb2_grpc.SuffixServiceStub(self._channel)

    async def evaluate(self, request: pb2.EvalRequest, timeout_s: float | None = None) -> pb2.EvalResponse:
        return await self._stub.Evaluate(request, timeout=timeout_s)

    async def close(self) -> None:
        await self._channel.close()


class KVCacheReceiver:
    """Tracks request-scoped layer readiness for suffix execution."""

    def __init__(self, layer_count: int) -> None:
        self._state = LayerState(layer_count)

    async def ingest(self, request_id: str, layer_idx: int, key, value, prefix_pad_mask=None, has_prefix_pad_mask: bool = False) -> None:
        payload = _KVPayload(
            request_id=request_id,
            layer_idx=layer_idx,
            key=proto_to_tensor(key),
            value=proto_to_tensor(value),
            prefix_pad_mask=proto_to_tensor(prefix_pad_mask) if has_prefix_pad_mask and prefix_pad_mask is not None else None,
        )
        await self._state.ingest(payload)

    async def is_ready(self, request_id: str, layer_idx: int) -> bool:
        return await self._state.is_ready(layer_idx, request_id=request_id)

    async def consume(self, request_id: str, layer_idx: int):
        return await self._state.consume(layer_idx, request_id=request_id)

    async def status(self, request_id: str, layer_idx: int):
        return await self._state.status(layer_idx, request_id=request_id)

    async def clear_session(self, request_id: str) -> None:
        await self._state.clear_session(request_id)

    async def get_prefix_pad_mask(self, request_id: str) -> torch.Tensor:
        return await self._state.get_prefix_pad_mask(request_id)


@dataclass(frozen=True)
class _KVPayload:
    request_id: str
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    prefix_pad_mask: torch.Tensor | None = None
