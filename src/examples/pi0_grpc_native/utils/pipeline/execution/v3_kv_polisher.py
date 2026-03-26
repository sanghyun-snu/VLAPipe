from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import grpc
import numpy as np
import torch

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2

from ..models import SuffixPipelineConfig
from ..profile import PipelineProfiler
from ...runtime.inference.runtime_inference import run_suffix_denoise_with_cache_provider
from ...transport.gpu_ipc_bridge import SuffixGpuIpcResolver
from ...transport.kv_transport import validate_kv_transfer_mode
from ...transport.stream_protocol import ndarray_to_proto
from ...transport.stream_protocol import proto_to_tensor


@dataclass(frozen=True)
class _LayerUpdate:
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    prefix_pad_mask: torch.Tensor | None


class V3KVPolisherStrategy:
    """Speculative suffix execution with in-place KV polishing."""

    def __init__(
        self,
        *,
        prefix_client,
        loaded_component: Any,
        config: SuffixPipelineConfig,
        profiler: PipelineProfiler,
    ) -> None:
        self._prefix_client = prefix_client
        self._loaded_component = loaded_component
        self._config = config
        self._profiler = profiler
        self._kv_transfer_mode = validate_kv_transfer_mode(config.kv_transfer_mode)
        self._gpu_ipc_resolver: SuffixGpuIpcResolver | None = None
        if self._kv_transfer_mode == "gpu_ipc":
            self._gpu_ipc_resolver = SuffixGpuIpcResolver(
                self._config.gpu_ipc_suffix_sidecar_address,
                resolve_mode=self._config.gpu_ipc_resolve_mode,
            )
        # Persistent static buffers across requests for fixed-memory execution.
        self._static_layer_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._static_prefix_pad_mask: torch.Tensor | None = None

    def _ensure_static_buffers(
        self,
        *,
        expected_num_layers: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if (
            self._static_layer_caches is None
            or len(self._static_layer_caches) != expected_num_layers
            or self._static_layer_caches[0][0].shape != key.shape
            or self._static_layer_caches[0][1].shape != value.shape
            or self._static_layer_caches[0][0].dtype != key.dtype
            or self._static_layer_caches[0][1].dtype != value.dtype
            or self._static_layer_caches[0][0].device != key.device
            or self._static_layer_caches[0][1].device != value.device
        ):
            self._static_layer_caches = [(key.detach().clone(), value.detach().clone()) for _ in range(expected_num_layers)]

    def _apply_update(self, *, update: _LayerUpdate, expected_num_layers: int) -> None:
        if update.layer_idx < 0 or update.layer_idx >= expected_num_layers:
            raise RuntimeError(
                f"layer_idx out of range: got={update.layer_idx} expected=[0,{expected_num_layers - 1}]"
            )
        self._ensure_static_buffers(
            expected_num_layers=expected_num_layers,
            key=update.key,
            value=update.value,
        )
        if self._static_layer_caches is None:
            raise RuntimeError("static layer cache initialization failed")
        key_buf, value_buf = self._static_layer_caches[update.layer_idx]
        key_buf.copy_(update.key)
        value_buf.copy_(update.value)
        if update.prefix_pad_mask is not None:
            if (
                self._static_prefix_pad_mask is None
                or self._static_prefix_pad_mask.shape != update.prefix_pad_mask.shape
                or self._static_prefix_pad_mask.dtype != update.prefix_pad_mask.dtype
                or self._static_prefix_pad_mask.device != update.prefix_pad_mask.device
            ):
                self._static_prefix_pad_mask = update.prefix_pad_mask.detach().clone()
            else:
                self._static_prefix_pad_mask.copy_(update.prefix_pad_mask)

    def _buffers_ready(self, expected_num_layers: int) -> bool:
        return (
            self._static_layer_caches is not None
            and len(self._static_layer_caches) == expected_num_layers
            and self._static_prefix_pad_mask is not None
        )

    async def run(
        self,
        *,
        request: pb2.EvalRequest,
        raw_policy_input: dict[str, Any],
        policy_name: str,
        context,
    ) -> pb2.EvalResponse:
        if self._loaded_component is None:
            raise RuntimeError("Suffix split component is not loaded. Start server with component checkpoint args.")
        if not request.request_id:
            raise ValueError("EvalRequest.request_id is required")

        request_id = request.request_id
        request_start_t = self._profiler.now()
        model_device = self._loaded_component.device
        expected_num_layers = len(self._loaded_component.model.paligemma_with_expert.gemma_expert.model.layers)
        prefix_request = pb2.PrefixRequest(request_id=request_id, eval_request=request)
        update_queue: asyncio.Queue[_LayerUpdate | object] = asyncio.Queue(maxsize=max(8, expected_num_layers))
        sentinel = object()
        receive_done = asyncio.Event()
        receive_error: Exception | None = None
        received_layers: set[int] = set()
        last_seen_layer_idx = -1
        highest_contiguous_layer = -1
        gpu_ipc_lookup_s = 0.0
        gpu_ipc_open_s = 0.0
        gpu_ipc_copy_s = 0.0

        async def _emit(event: str, *, layer_idx: int = -1, cache_epoch: int = 0, details: str = "") -> None:
            emit_event = getattr(context, "emit_event", None)
            if callable(emit_event):
                await emit_event(event=event, layer_idx=layer_idx, cache_epoch=cache_epoch, details=details)

        async def _producer() -> None:
            nonlocal receive_error, last_seen_layer_idx, highest_contiguous_layer
            nonlocal gpu_ipc_lookup_s, gpu_ipc_open_s, gpu_ipc_copy_s
            try:
                await _emit("prefix_stream_start")
                async for chunk in self._prefix_client.stream_prefix(
                    prefix_request,
                    timeout_s=self._config.prefix_stream_timeout_s,
                ):
                    layer_idx = int(chunk.layer_idx)
                    if chunk.request_id != request_id:
                        raise RuntimeError(f"Unexpected request_id in stream: expected={request_id} got={chunk.request_id}")
                    if self._config.strict_layer_ordering and layer_idx <= last_seen_layer_idx:
                        raise RuntimeError(
                            f"Out-of-order layer in stream: previous={last_seen_layer_idx} got={layer_idx}"
                        )
                    last_seen_layer_idx = layer_idx
                    if chunk.transfer_mode == pb2.KV_TRANSFER_MODE_GPU_IPC:
                        if self._gpu_ipc_resolver is None:
                            raise RuntimeError("gpu_ipc resolver is not initialized")
                        key_tensor, key_timing = self._gpu_ipc_resolver.resolve_tensor_timed(
                            chunk.request_id,
                            layer_idx,
                            "key",
                            chunk.key_handle,
                        )
                        value_tensor, value_timing = self._gpu_ipc_resolver.resolve_tensor_timed(
                            chunk.request_id,
                            layer_idx,
                            "value",
                            chunk.value_handle,
                        )
                        for timing in (key_timing, value_timing):
                            gpu_ipc_lookup_s += timing.sidecar_lookup_s
                            gpu_ipc_open_s += timing.ipc_open_s
                            gpu_ipc_copy_s += timing.d2d_copy_s
                        prefix_pad_mask = None
                        if chunk.has_prefix_pad_mask:
                            prefix_pad_mask, mask_timing = self._gpu_ipc_resolver.resolve_tensor_timed(
                                chunk.request_id,
                                layer_idx,
                                "prefix_pad_mask",
                                chunk.prefix_pad_mask_handle,
                            )
                            gpu_ipc_lookup_s += mask_timing.sidecar_lookup_s
                            gpu_ipc_open_s += mask_timing.ipc_open_s
                            gpu_ipc_copy_s += mask_timing.d2d_copy_s
                    else:
                        key_tensor = proto_to_tensor(chunk.key, device=model_device)
                        value_tensor = proto_to_tensor(chunk.value, device=model_device)
                        prefix_pad_mask = (
                            proto_to_tensor(chunk.prefix_pad_mask, device=model_device) if chunk.has_prefix_pad_mask else None
                        )
                    await update_queue.put(
                        _LayerUpdate(
                            layer_idx=layer_idx,
                            key=key_tensor,
                            value=value_tensor,
                            prefix_pad_mask=prefix_pad_mask,
                        )
                    )
                    received_layers.add(layer_idx)
                    while (highest_contiguous_layer + 1) in received_layers:
                        highest_contiguous_layer += 1
                    await _emit("layer_received", layer_idx=layer_idx, cache_epoch=highest_contiguous_layer + 1)
            except Exception as exc:  # noqa: BLE001
                receive_error = exc
            finally:
                receive_done.set()
                await update_queue.put(sentinel)

        async def _drain_updates(*, wait_for_update: bool) -> None:
            got_update = False
            while True:
                if receive_error is not None:
                    raise receive_error
                if wait_for_update and not got_update:
                    item = await update_queue.get()
                else:
                    try:
                        item = update_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if item is sentinel:
                    if wait_for_update and not got_update and receive_error is not None:
                        raise receive_error
                    break
                got_update = True
                self._apply_update(update=item, expected_num_layers=expected_num_layers)
            if wait_for_update and not got_update and receive_done.is_set() and receive_error is not None:
                raise receive_error

        async def _cache_provider(step_idx: int, num_steps: int) -> tuple[torch.Tensor, tuple]:
            del num_steps  # unused; kept for signature stability.
            if step_idx == 0:
                while not self._buffers_ready(expected_num_layers):
                    await _drain_updates(wait_for_update=True)
                    if receive_done.is_set() and not self._buffers_ready(expected_num_layers):
                        break
            else:
                await _drain_updates(wait_for_update=False)
            if not self._buffers_ready(expected_num_layers):
                if receive_done.is_set() and receive_error is not None:
                    raise receive_error
                raise RuntimeError("KV Polisher could not initialize static cache buffers")
            if self._static_prefix_pad_mask is None or self._static_layer_caches is None:
                raise RuntimeError("static cache buffer not ready")
            return self._static_prefix_pad_mask, tuple(self._static_layer_caches)

        producer_task = asyncio.create_task(_producer(), name=f"v3-producer-{request_id}")
        try:
            await _emit("suffix_denoise_start")
            cache_layers = expected_num_layers

            def _on_denoise_step(step_idx: int, total_steps: int, step_s: float, is_warmup: bool) -> None:
                self._profiler.event(
                    request_id=request_id,
                    pipeline="suffix",
                    event="denoise_step_v3",
                    value_s=step_s,
                    layer_idx=cache_layers,
                    details=(
                        f"iteration={step_idx + 1}/{total_steps},"
                        f"warmup={int(is_warmup)}"
                    ),
                )

            def _on_denoise_layer(step_idx: int, layer_idx: int, layer_s: float, is_warmup: bool) -> None:
                layer_elapsed_s = self._profiler.now() - request_start_t
                self._profiler.event(
                    request_id=request_id,
                    pipeline="suffix",
                    event="denoise_layer_v3",
                    value_s=layer_elapsed_s,
                    layer_idx=layer_idx,
                    details=(
                        f"iteration={step_idx + 1},"
                        f"warmup={int(is_warmup)},"
                        f"layer_s={layer_s:.9f}"
                    ),
                )

            actions = await run_suffix_denoise_with_cache_provider(
                self._loaded_component,
                raw_policy_input,
                cache_provider=_cache_provider,
                warmup_diffusion_steps=self._config.warmup_diffusion_steps,
                request_id=request_id,
                deterministic_noise=self._config.deterministic_noise,
                denoise_step_callback=_on_denoise_step,
                denoise_layer_callback=_on_denoise_layer,
                now_fn=self._profiler.now,
            )
            await _emit("suffix_denoise_done", cache_epoch=highest_contiguous_layer + 1)
            if self._kv_transfer_mode == "gpu_ipc" and received_layers:
                self._profiler.event(
                    request_id=request_id,
                    pipeline="suffix",
                    event="gpu_ipc_resolve_totals_v3",
                    value_s=gpu_ipc_lookup_s + gpu_ipc_open_s + gpu_ipc_copy_s,
                    details=(
                        f"lookup_s={gpu_ipc_lookup_s:.6f},open_s={gpu_ipc_open_s:.6f},"
                        f"copy_s={gpu_ipc_copy_s:.6f},layers={len(received_layers)}"
                    ),
                )
            actions = np.asarray(actions, dtype=np.float32)
            return pb2.EvalResponse(
                request_id=request_id,
                actions=ndarray_to_proto(actions),
                message=f"completed policy={policy_name} model=real_prefix_suffix mode=v3_kv_polisher",
            )
        except grpc.aio.AioRpcError as exc:
            details = (
                f"prefix stream RPC failed request_id={request_id} "
                f"code={exc.code().name} details={exc.details()}"
            )
            await _emit("error", details=details)
            await context.abort(grpc.StatusCode.UNAVAILABLE, details)
            raise RuntimeError(details)
        except Exception as exc:  # noqa: BLE001
            details = f"v3 kv_polisher execution failed request_id={request_id}: {exc}"
            await _emit("error", details=details)
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, details)
            raise RuntimeError(details)
        finally:
            if not producer_task.done():
                producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
            if self._gpu_ipc_resolver is not None:
                self._gpu_ipc_resolver.close()
