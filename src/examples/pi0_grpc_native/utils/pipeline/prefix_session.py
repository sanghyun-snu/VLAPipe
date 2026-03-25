from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2

from .logging import log_prefix_backpressure
from .logging import log_prefix_emit
from .logging import log_prefix_end
from .logging import log_prefix_start
from .models import PrefixPipelineConfig
from .models import PrefixStreamState
from .profile import PipelineProfiler
from ..runtime.inference.runtime_inference import iter_prefix_cache_payloads_from_policy
from ..transport.gpu_ipc_bridge import PrefixGpuIpcPublisher
from ..transport.kv_transport import validate_kv_transfer_mode
from ..transport.stream_protocol import tensor_to_proto


class PrefixStreamSession:
    _QUEUE_POLL_INTERVAL_S = 0.2

    def __init__(
        self,
        loaded_component: Any,
        config: PrefixPipelineConfig,
        request_id: str,
        raw_policy_input: dict[str, Any],
        context,
        profiler: PipelineProfiler,
    ) -> None:
        self._loaded_component = loaded_component
        self._config = config
        self._request_id = request_id
        self._raw_policy_input = raw_policy_input
        self._context = context
        self._profiler = profiler
        self._start_t = self._profiler.now()
        self._state = PrefixStreamState()
        self._payload_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=self._config.stream_queue_size)
        self._sentinel = object()
        self._producer_task: asyncio.Task[None] | None = None
        self._kv_transfer_mode = validate_kv_transfer_mode(self._config.kv_transfer_mode)
        self._gpu_ipc_publisher: PrefixGpuIpcPublisher | None = None
        if self._kv_transfer_mode == "gpu_ipc":
            self._gpu_ipc_publisher = PrefixGpuIpcPublisher(self._config.gpu_ipc_prefix_sidecar_address)

    async def run(self):
        if self._loaded_component is None:
            raise RuntimeError("Prefix split component is not loaded. Start server with component checkpoint args.")
        log_prefix_start(self._request_id, self._config)
        self._profiler.event(request_id=self._request_id, pipeline="prefix", event="start")
        self._producer_task = asyncio.create_task(
            self._produce_payloads(),
            name=f"prefix-producer-{self._request_id}",
        )
        try:
            async for chunk in self._consume_chunks():
                yield chunk
            if self._state.producer_error is not None:
                err = self._state.producer_error
                self._profiler.event(
                    request_id=self._request_id,
                    pipeline="prefix",
                    event="producer_error",
                    details=str(err),
                )
                raise RuntimeError(f"Prefix producer failed request_id={self._request_id}: {err}") from err
            total_s = self._profiler.now() - self._start_t
            log_prefix_end(self._request_id, self._state, total_s)
            self._profiler.event(request_id=self._request_id, pipeline="prefix", event="end", value_s=total_s)
        finally:
            if self._producer_task is not None and not self._producer_task.done():
                self._producer_task.cancel()
            if self._producer_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._producer_task

    def _check_request_timeout(self) -> None:
        if self._config.request_timeout_s <= 0:
            return
        elapsed_s = self._profiler.now() - self._start_t
        if elapsed_s > self._config.request_timeout_s:
            self._profiler.event(
                request_id=self._request_id,
                pipeline="prefix",
                event="timeout",
                value_s=elapsed_s,
            )
            raise TimeoutError(
                f"Prefix stream timeout request_id={self._request_id} "
                f"elapsed={elapsed_s:.3f}s timeout={self._config.request_timeout_s:.3f}s"
            )

    async def _produce_payloads(self) -> None:
        try:
            for payload in iter_prefix_cache_payloads_from_policy(
                self._loaded_component,
                self._raw_policy_input,
                request_id=self._request_id,
                prefer_layerwise=self._config.prefer_layerwise,
                allow_fallback=self._config.allow_fallback,
            ):
                self._check_request_timeout()
                put_start_t = self._profiler.now()
                await self._payload_queue.put(payload)
                wait_s = self._profiler.now() - put_start_t
                self._state.queue_wait_s += wait_s
                self._state.produced_layers += 1
                self._profiler.event(
                    request_id=self._request_id,
                    pipeline="prefix",
                    event="produced_layer",
                    layer_idx=payload.layer_idx,
                )
                if wait_s * 1000.0 >= self._config.queue_wait_warn_ms:
                    log_prefix_backpressure(self._request_id, payload.layer_idx, wait_s * 1000.0)
                    self._profiler.event(
                        request_id=self._request_id,
                        pipeline="prefix",
                        event="queue_backpressure",
                        value_s=wait_s,
                        layer_idx=payload.layer_idx,
                    )
        except Exception as exc:  # noqa: BLE001
            self._state.producer_error = exc
        finally:
            await self._payload_queue.put(self._sentinel)

    async def _consume_chunks(self):
        while True:
            self._check_request_timeout()
            context_cancelled = getattr(self._context, "cancelled", None)
            if callable(context_cancelled) and context_cancelled():
                self._profiler.event(
                    request_id=self._request_id,
                    pipeline="prefix",
                    event="cancelled",
                )
                raise asyncio.CancelledError(f"Prefix stream cancelled by client request_id={self._request_id}")
            try:
                queued = await asyncio.wait_for(
                    self._payload_queue.get(),
                    timeout=self._QUEUE_POLL_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                # Periodically wake up so cancellation/timeout checks stay responsive
                # even when queue.get() is blocked.
                continue
            if queued is self._sentinel:
                break
            payload = queued
            emit_s = self._profiler.now() - self._start_t
            if self._state.first_emit_s is None:
                self._state.first_emit_s = emit_s
            self._state.last_emit_s = emit_s
            self._state.emitted_layers += 1
            log_prefix_emit(payload.request_id, payload.layer_idx)
            self._profiler.event(
                request_id=self._request_id,
                pipeline="prefix",
                event="emitted_layer",
                value_s=emit_s,
                layer_idx=payload.layer_idx,
            )
            if self._kv_transfer_mode == "gpu_ipc":
                if self._gpu_ipc_publisher is None:
                    raise RuntimeError("gpu_ipc publisher is not initialized")
                key_handle = self._gpu_ipc_publisher.publish_tensor(
                    payload.request_id, payload.layer_idx, "key", payload.key
                )
                value_handle = self._gpu_ipc_publisher.publish_tensor(
                    payload.request_id, payload.layer_idx, "value", payload.value
                )
                prefix_pad_mask_handle = (
                    self._gpu_ipc_publisher.publish_tensor(
                        payload.request_id,
                        payload.layer_idx,
                        "prefix_pad_mask",
                        payload.prefix_pad_mask,
                    )
                    if payload.prefix_pad_mask is not None
                    else pb2.GpuIpcHandle()
                )
                yield pb2.KVCacheChunk(
                    request_id=payload.request_id,
                    layer_idx=payload.layer_idx,
                    has_prefix_pad_mask=(payload.prefix_pad_mask is not None),
                    transfer_mode=pb2.KV_TRANSFER_MODE_GPU_IPC,
                    key_handle=key_handle.to_proto(),
                    value_handle=value_handle.to_proto(),
                    prefix_pad_mask_handle=(
                        prefix_pad_mask_handle.to_proto()
                        if payload.prefix_pad_mask is not None
                        else pb2.GpuIpcHandle()
                    ),
                )
            else:
                yield pb2.KVCacheChunk(
                    request_id=payload.request_id,
                    layer_idx=payload.layer_idx,
                    key=tensor_to_proto(payload.key),
                    value=tensor_to_proto(payload.value),
                    prefix_pad_mask=tensor_to_proto(payload.prefix_pad_mask)
                    if payload.prefix_pad_mask is not None
                    else pb2.NdArray(),
                    has_prefix_pad_mask=(payload.prefix_pad_mask is not None),
                    transfer_mode=pb2.KV_TRANSFER_MODE_PROTO_BYTES,
                )
            await asyncio.sleep(0)
