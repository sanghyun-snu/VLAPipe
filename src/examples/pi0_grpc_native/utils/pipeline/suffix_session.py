from __future__ import annotations

from typing import Any

import grpc
import numpy as np

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from openpi.models_pytorch.layer_scheduler import LayerCacheCollector
from openpi.models_pytorch.layer_scheduler import LayerKVPayload

from .logging import log_suffix_end
from .logging import log_suffix_error
from .logging import log_suffix_receive
from .logging import log_suffix_received_done
from .logging import log_suffix_start
from .models import SuffixEvalState
from .models import SuffixPipelineConfig
from .profile import PipelineProfiler
from ..transport.gpu_ipc_bridge import SuffixGpuIpcResolver
from ..transport.kv_transport import validate_kv_transfer_mode
from ..runtime.inference.runtime_inference import run_suffix_denoise_with_cache
from ..transport.stream_protocol import ndarray_to_proto
from ..transport.stream_protocol import proto_to_tensor


class SuffixEvalSession:
    def __init__(
        self,
        prefix_client,
        loaded_component: Any,
        config: SuffixPipelineConfig,
        request: pb2.EvalRequest,
        raw_policy_input: dict[str, Any],
        policy_name: str,
        context,
        profiler: PipelineProfiler,
    ) -> None:
        self._prefix_client = prefix_client
        self._loaded_component = loaded_component
        self._config = config
        self._request = request
        self._raw_policy_input = raw_policy_input
        self._policy_name = policy_name
        self._context = context
        self._profiler = profiler
        self._state = SuffixEvalState()
        self._start_t = self._profiler.now()
        self._kv_transfer_mode = validate_kv_transfer_mode(self._config.kv_transfer_mode)
        self._gpu_ipc_resolver: SuffixGpuIpcResolver | None = None
        if self._kv_transfer_mode == "gpu_ipc":
            self._gpu_ipc_resolver = SuffixGpuIpcResolver(self._config.gpu_ipc_suffix_sidecar_address)

    async def run(self) -> pb2.EvalResponse:
        if self._loaded_component is None:
            raise RuntimeError("Suffix split component is not loaded. Start server with component checkpoint args.")
        if not self._request.request_id:
            raise ValueError("EvalRequest.request_id is required")

        request_id = self._request.request_id
        model_device = self._loaded_component.device
        expected_num_layers = len(self._loaded_component.model.paligemma_with_expert.gemma_expert.model.layers)
        log_suffix_start(request_id, expected_num_layers, self._config)
        self._profiler.event(request_id=request_id, pipeline="suffix", event="start")

        scheduler = LayerCacheCollector(
            expected_request_id=request_id,
            enforce_increasing_layer=self._config.strict_layer_ordering,
        )
        prefix_request = pb2.PrefixRequest(request_id=request_id, eval_request=self._request)

        try:
            receive_start_t = self._profiler.now()
            async for chunk in self._prefix_client.stream_prefix(
                prefix_request, timeout_s=self._config.prefix_stream_timeout_s
            ):
                if chunk.transfer_mode == pb2.KV_TRANSFER_MODE_GPU_IPC:
                    if self._gpu_ipc_resolver is None:
                        raise RuntimeError("gpu_ipc resolver is not initialized")
                    key_tensor = self._gpu_ipc_resolver.resolve_tensor(
                        chunk.request_id,
                        int(chunk.layer_idx),
                        "key",
                        chunk.key_handle,
                    )
                    value_tensor = self._gpu_ipc_resolver.resolve_tensor(
                        chunk.request_id,
                        int(chunk.layer_idx),
                        "value",
                        chunk.value_handle,
                    )
                    prefix_pad_mask_tensor = (
                        self._gpu_ipc_resolver.resolve_tensor(
                            chunk.request_id,
                            int(chunk.layer_idx),
                            "prefix_pad_mask",
                            chunk.prefix_pad_mask_handle,
                        )
                        if chunk.has_prefix_pad_mask
                        else None
                    )
                else:
                    key_tensor = proto_to_tensor(chunk.key, device=model_device)
                    value_tensor = proto_to_tensor(chunk.value, device=model_device)
                    prefix_pad_mask_tensor = (
                        proto_to_tensor(chunk.prefix_pad_mask, device=model_device) if chunk.has_prefix_pad_mask else None
                    )
                scheduler.ingest(
                    LayerKVPayload(
                        request_id=chunk.request_id,
                        layer_idx=chunk.layer_idx,
                        key=key_tensor,
                        value=value_tensor,
                        prefix_pad_mask=prefix_pad_mask_tensor,
                    )
                )
                self._state.received_layers += 1
                log_suffix_receive(chunk.request_id, chunk.layer_idx)
                self._profiler.event(
                    request_id=request_id,
                    pipeline="suffix",
                    event="received_layer",
                    layer_idx=chunk.layer_idx,
                )
            receive_done_t = self._profiler.now()
            finalize_start_t = self._profiler.now()
            prefix_pad_masks, layer_caches = scheduler.finalize(expected_num_layers=expected_num_layers)
            finalize_done_t = self._profiler.now()
            self._state.receive_s = receive_done_t - receive_start_t
            self._state.finalize_s = finalize_done_t - finalize_start_t
            log_suffix_received_done(request_id, self._state)
            self._profiler.event(
                request_id=request_id,
                pipeline="suffix",
                event="received_done",
                value_s=self._state.receive_s,
                details=f"layers={self._state.received_layers}",
            )
            self._profiler.event(
                request_id=request_id,
                pipeline="suffix",
                event="finalize_done",
                value_s=self._state.finalize_s,
            )

            denoise_start_t = self._profiler.now()
            actions = run_suffix_denoise_with_cache(
                self._loaded_component,
                self._raw_policy_input,
                prefix_pad_masks,
                layer_caches,
                warmup_diffusion_steps=self._config.warmup_diffusion_steps,
            )
            denoise_done_t = self._profiler.now()
            self._state.denoise_s = denoise_done_t - denoise_start_t
            self._state.total_s = self._profiler.now() - self._start_t
            log_suffix_end(request_id, self._state)
            self._profiler.event(
                request_id=request_id,
                pipeline="suffix",
                event="denoise_done",
                value_s=self._state.denoise_s,
            )
            self._profiler.event(
                request_id=request_id,
                pipeline="suffix",
                event="end",
                value_s=self._state.total_s,
            )

            actions = np.asarray(actions, dtype=np.float32)
            message = f"completed policy={self._policy_name} model=real_prefix_suffix"
            return pb2.EvalResponse(
                request_id=request_id,
                actions=ndarray_to_proto(actions),
                message=message,
            )
        except grpc.aio.AioRpcError as exc:
            rpc_code = exc.code()
            details = (
                f"prefix stream RPC failed request_id={request_id} "
                f"code={rpc_code.name} details={exc.details()}"
            )
            log_suffix_error(details)
            self._profiler.event(request_id=request_id, pipeline="suffix", event="rpc_error", details=details)
            if rpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                await self._context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, details)
            elif rpc_code == grpc.StatusCode.CANCELLED:
                await self._context.abort(grpc.StatusCode.CANCELLED, details)
            elif rpc_code == grpc.StatusCode.FAILED_PRECONDITION:
                await self._context.abort(grpc.StatusCode.FAILED_PRECONDITION, details)
            elif rpc_code == grpc.StatusCode.INVALID_ARGUMENT:
                await self._context.abort(grpc.StatusCode.INVALID_ARGUMENT, details)
            await self._context.abort(grpc.StatusCode.UNAVAILABLE, details)
        except TimeoutError as exc:
            details = f"prefix stream timeout request_id={request_id}: {exc}"
            log_suffix_error(details)
            self._profiler.event(request_id=request_id, pipeline="suffix", event="timeout", details=details)
            await self._context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, details)
        except (RuntimeError, ValueError, TypeError) as exc:
            details = f"invalid prefix cache stream request_id={request_id}: {exc}"
            log_suffix_error(details)
            self._profiler.event(request_id=request_id, pipeline="suffix", event="invalid_stream", details=details)
            await self._context.abort(grpc.StatusCode.FAILED_PRECONDITION, details)
        raise RuntimeError(f"Suffix pipeline aborted request_id={request_id}")
