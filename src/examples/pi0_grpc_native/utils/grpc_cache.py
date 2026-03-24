from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

import grpc

from .layer_state import LayerState
from .stream_protocol import KVCachePayload
from .stream_protocol import deserialize_kv_payload
from .stream_protocol import serialize_kv_payload

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50061
DEFAULT_POLL_INTERVAL_S = 0.05
SERVICE_NAME = "openpi.pi0.KVCacheService"
METHOD_NAME = "StreamKVCache"
METHOD_PATH = f"/{SERVICE_NAME}/{METHOD_NAME}"
PREFIX_SERVICE_NAME = "openpi.pi0.PrefixService"
PREFIX_METHOD_NAME = "StreamPrefixKV"
PREFIX_METHOD_PATH = f"/{PREFIX_SERVICE_NAME}/{PREFIX_METHOD_NAME}"
SUFFIX_SERVICE_NAME = "openpi.pi0.SuffixService"
SUFFIX_METHOD_NAME = "Evaluate"
SUFFIX_METHOD_PATH = f"/{SUFFIX_SERVICE_NAME}/{SUFFIX_METHOD_NAME}"
ACK_PREFIX = b"ACK:"


@dataclass(frozen=True)
class StreamConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class KVCacheReceiver:
    """Receiver-side state manager for per-layer KV cache readiness."""

    def __init__(self, layer_count: int) -> None:
        self._state = LayerState(layer_count)

    async def on_payload(self, payload: KVCachePayload) -> None:
        await self._state.ingest(payload)

    async def is_ready(self, layer_idx: int, request_id: str = "default") -> bool:
        return await self._state.is_ready(layer_idx, request_id=request_id)

    async def wait_until_ready(self, layer_idx: int, poll_interval_s: float, request_id: str = "default") -> None:
        while not await self.is_ready(layer_idx, request_id=request_id):
            await asyncio.sleep(poll_interval_s)

    async def consume(self, layer_idx: int, request_id: str = "default"):
        return await self._state.consume(layer_idx, request_id=request_id)

    async def status(self, layer_idx: int, request_id: str = "default"):
        return await self._state.status(layer_idx, request_id=request_id)

    async def all_consumed(self, request_id: str = "default") -> bool:
        return await self._state.all_consumed(request_id=request_id)

    async def snapshots(self, request_id: str = "default"):
        return await self._state.snapshots(request_id=request_id)

    async def clear_session(self, request_id: str) -> None:
        await self._state.clear_session(request_id)


class KVCacheGrpcServer:
    """gRPC server that receives layer KV payload stream."""

    def __init__(self, receiver: KVCacheReceiver, address: str) -> None:
        self._receiver = receiver
        self._address = address
        self._server = grpc.aio.server()
        handler = grpc.stream_stream_rpc_method_handler(self._stream_handler)
        generic_handler = grpc.method_handlers_generic_handler(SERVICE_NAME, {METHOD_NAME: handler})
        self._server.add_generic_rpc_handlers((generic_handler,))
        self._server.add_insecure_port(address)

    async def _stream_handler(self, request_iterator, _context):
        async for raw in request_iterator:
            payload = deserialize_kv_payload(raw)
            await self._receiver.on_payload(payload)
            yield ACK_PREFIX + str(payload.layer_idx).encode("utf-8")

    async def start(self) -> None:
        await self._server.start()

    async def stop(self, grace_s: float = 0.5) -> None:
        await self._server.stop(grace_s)


class KVCacheSender:
    """Prefix-side async sender that publishes per-layer KV payload."""

    _SENTINEL = object()

    def __init__(self, address: str) -> None:
        self._address = address
        self._channel = grpc.aio.insecure_channel(address)
        self._queue: asyncio.Queue[bytes | object] = asyncio.Queue()
        self._call = None
        self._ack_task: asyncio.Task | None = None

    async def connect(self, timeout_s: float = 5.0) -> None:
        await asyncio.wait_for(self._channel.channel_ready(), timeout=timeout_s)
        stream = self._channel.stream_stream(METHOD_PATH)
        self._call = stream(self._request_stream())
        self._ack_task = asyncio.create_task(self._consume_acks())

    async def send(self, payload: KVCachePayload) -> None:
        await self._queue.put(serialize_kv_payload(payload))

    async def close(self) -> None:
        await self._queue.put(self._SENTINEL)
        if self._ack_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._ack_task
        await self._channel.close()

    async def _request_stream(self):
        while True:
            item = await self._queue.get()
            if item is self._SENTINEL:
                break
            yield item

    async def _consume_acks(self) -> None:
        if self._call is None:
            return
        async for _ in self._call:
            pass


class PrefixStreamClient:
    """Suffix-side client that requests and consumes Prefix KV stream."""

    def __init__(self, address: str) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stream = self._channel.unary_stream(PREFIX_METHOD_PATH)

    async def stream_prefix(self, request_raw: bytes, timeout_s: float | None = None):
        async for raw in self._stream(request_raw, timeout=timeout_s):
            yield raw

    async def close(self) -> None:
        await self._channel.close()


class SuffixEvalClient:
    """Eval-side client that sends requests to the suffix server."""

    def __init__(self, address: str) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._unary = self._channel.unary_unary(SUFFIX_METHOD_PATH)

    async def evaluate(self, request_raw: bytes, timeout_s: float | None = None) -> bytes:
        return await self._unary(request_raw, timeout=timeout_s)

    async def close(self) -> None:
        await self._channel.close()
