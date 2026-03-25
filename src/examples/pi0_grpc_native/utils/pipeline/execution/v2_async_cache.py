from __future__ import annotations

import asyncio
from typing import Any

import grpc
import numpy as np

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2

from .async_kv_store import AsyncKVStore
from ..models import SuffixPipelineConfig
from ..profile import PipelineProfiler
from ...runtime.inference.runtime_inference import run_suffix_denoise_with_cache
from ...transport.stream_protocol import ndarray_to_proto
from ...transport.stream_protocol import proto_to_tensor


class V2AsyncCacheStrategy:
    """Async cache execution with epoched snapshots and staleness policy."""

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
        model_device = self._loaded_component.device
        expected_num_layers = len(self._loaded_component.model.paligemma_with_expert.gemma_expert.model.layers)
        prefix_request = pb2.PrefixRequest(request_id=request_id, eval_request=request)
        kv_store = AsyncKVStore(
            max_snapshots=self._config.max_inflight_updates,
            cache_ttl_ms=self._config.cache_ttl_ms,
        )
        stop_updates = asyncio.Event()

        async def _emit(event: str, *, layer_idx: int = -1, cache_epoch: int = 0, details: str = "") -> None:
            emit_event = getattr(context, "emit_event", None)
            if callable(emit_event):
                await emit_event(event=event, layer_idx=layer_idx, cache_epoch=cache_epoch, details=details)

        receive_done = asyncio.Event()
        receive_error: Exception | None = None
        received_layers = 0
        layer_map: dict[int, tuple[Any, Any]] = {}
        prefix_pad_mask = None
        highest_contiguous_layer = -1
        last_seen_layer_idx = -1

        def _build_snapshot(epoch_layers: int) -> tuple[Any, tuple]:
            if prefix_pad_mask is None:
                raise RuntimeError("prefix_pad_mask not available yet")
            if epoch_layers <= 0:
                raise RuntimeError("epoch_layers must be > 0")
            if not self._config.allow_stale_cache and epoch_layers < expected_num_layers:
                raise RuntimeError(
                    f"stale cache snapshot not allowed: epoch_layers={epoch_layers} expected_layers={expected_num_layers}"
                )
            if 0 not in layer_map:
                raise RuntimeError("layer 0 cache missing")
            fallback_cache = layer_map[max(0, epoch_layers - 1)]
            caches: list[tuple[Any, Any]] = []
            for layer_idx in range(expected_num_layers):
                if layer_idx in layer_map:
                    caches.append(layer_map[layer_idx])
                else:
                    caches.append(fallback_cache)
            return prefix_pad_mask, tuple(caches)

        async def _producer() -> None:
            nonlocal received_layers, receive_error, highest_contiguous_layer, prefix_pad_mask, last_seen_layer_idx
            try:
                await _emit("prefix_stream_start")
                async for chunk in self._prefix_client.stream_prefix(
                    prefix_request,
                    timeout_s=self._config.prefix_stream_timeout_s,
                ):
                    if stop_updates.is_set() and self._config.drop_late_updates:
                        await _emit("late_update_dropped", details="consumer requested stop_updates")
                        break
                    if chunk.request_id != request_id:
                        raise RuntimeError(f"Unexpected request_id in stream: expected={request_id} got={chunk.request_id}")
                    layer_idx = int(chunk.layer_idx)
                    if self._config.strict_layer_ordering and layer_idx <= last_seen_layer_idx:
                        raise RuntimeError(
                            f"Out-of-order layer in stream: previous={last_seen_layer_idx} got={layer_idx}"
                        )
                    last_seen_layer_idx = layer_idx
                    layer_map[layer_idx] = (
                        proto_to_tensor(chunk.key, device=model_device),
                        proto_to_tensor(chunk.value, device=model_device),
                    )
                    if chunk.has_prefix_pad_mask:
                        prefix_pad_mask = proto_to_tensor(chunk.prefix_pad_mask, device=model_device)
                    received_layers += 1
                    while (highest_contiguous_layer + 1) in layer_map:
                        highest_contiguous_layer += 1
                    epoch_layers = highest_contiguous_layer + 1
                    await _emit("layer_received", layer_idx=layer_idx, cache_epoch=epoch_layers)
                    if prefix_pad_mask is None or epoch_layers <= 0:
                        continue
                    snapshot = _build_snapshot(epoch_layers)
                    is_final = epoch_layers >= expected_num_layers
                    await kv_store.publish(
                        epoch=epoch_layers,
                        snapshot=snapshot,
                        final=is_final,
                    )
                    await _emit(
                        "cache_published",
                        cache_epoch=epoch_layers,
                        details="final" if is_final else "partial",
                    )
            except Exception as exc:  # noqa: BLE001
                receive_error = exc
            finally:
                receive_done.set()

        async def _consumer() -> pb2.EvalResponse:
            await _emit("suffix_waiting_cache")
            min_epoch = expected_num_layers
            require_final = True
            if self._config.allow_stale_cache:
                min_epoch = max(1, expected_num_layers - self._config.max_staleness_layers)
                require_final = False
            while True:
                maybe_snapshot = await kv_store.wait_for(
                    min_epoch=min_epoch,
                    require_final=require_final,
                    timeout_s=0.2,
                )
                if maybe_snapshot is not None:
                    epoch, snapshot, is_final = maybe_snapshot
                    break
                if receive_done.is_set() and receive_error is not None:
                    raise receive_error
                if receive_done.is_set() and highest_contiguous_layer + 1 < min_epoch:
                    raise RuntimeError(
                        f"insufficient cache epochs: received={highest_contiguous_layer + 1}, required={min_epoch}"
                    )
            prefix_pad_masks, layer_caches = snapshot
            await _emit("cache_selected", cache_epoch=epoch, details="final" if is_final else "stale")
            if self._config.drop_late_updates and not is_final:
                stop_updates.set()
                await _emit("late_updates_stop_requested", cache_epoch=epoch)
            await _emit("suffix_denoise_start")
            actions = run_suffix_denoise_with_cache(
                self._loaded_component,
                raw_policy_input,
                prefix_pad_masks,
                layer_caches,
                warmup_diffusion_steps=self._config.warmup_diffusion_steps,
            )
            await _emit("suffix_denoise_done")
            actions = np.asarray(actions, dtype=np.float32)
            return pb2.EvalResponse(
                request_id=request_id,
                actions=ndarray_to_proto(actions),
                message=f"completed policy={policy_name} model=real_prefix_suffix mode=v2_async_cache",
            )

        producer_task = asyncio.create_task(_producer(), name=f"v2-producer-{request_id}")
        try:
            result = await _consumer()
            await _emit("done", cache_epoch=highest_contiguous_layer + 1)
            return result
        except grpc.aio.AioRpcError as exc:
            details = (
                f"prefix stream RPC failed request_id={request_id} "
                f"code={exc.code().name} details={exc.details()}"
            )
            await _emit("error", details=details)
            await context.abort(grpc.StatusCode.UNAVAILABLE, details)
            raise RuntimeError(details)
        except Exception as exc:  # noqa: BLE001
            if receive_error is not None:
                details = f"v2 async cache receive failed request_id={request_id}: {receive_error}"
                await _emit("error", details=details)
                await context.abort(grpc.StatusCode.UNAVAILABLE, details)
                raise RuntimeError(details) from receive_error
            details = f"v2 async cache execution failed request_id={request_id}: {exc}"
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
