from __future__ import annotations

import argparse
import asyncio

import grpc

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc
from examples.pi0_grpc_native.utils import PrefixClient
from examples.pi0_grpc_native.utils import RuntimePolicyArgs
from examples.pi0_grpc_native.utils import SuffixPipeline
from examples.pi0_grpc_native.utils import SuffixServiceOptions
from examples.pi0_grpc_native.utils import adapt_eval_request_to_policy_input
from examples.pi0_grpc_native.utils import load_suffix_component
from examples.pi0_grpc_native.utils.runtime import run_suffix_endpoint_startup_warmup
from .utils.pipeline.execution import V2AsyncOperationManager  # pyright: ignore[reportMissingImports]
from .utils.pipeline.execution import resolve_execution_overrides  # pyright: ignore[reportMissingImports]

DEFAULT_SUFFIX_HOST = "127.0.0.1"
DEFAULT_SUFFIX_PORT = 50061
DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062
DEFAULT_SUFFIX_SERVICE_OPTIONS = SuffixServiceOptions()


class SuffixService(pb2_grpc.SuffixServiceServicer):
    def __init__(
        self,
        *,
        prefix_address: str,
        loaded_component=None,
        options: SuffixServiceOptions = DEFAULT_SUFFIX_SERVICE_OPTIONS,
    ) -> None:
        if options.prefix_stream_timeout_s <= 0:
            raise ValueError(f"prefix_stream_timeout_s must be > 0, got {options.prefix_stream_timeout_s}")
        if options.warmup_diffusion_steps < 0:
            raise ValueError(f"warmup_diffusion_steps must be >= 0, got {options.warmup_diffusion_steps}")
        self._service_options = options
        self._pipeline = SuffixPipeline(
            prefix_client=PrefixClient(address=prefix_address),
            loaded_component=loaded_component,
            config=self._service_options.to_pipeline_config(),
        )
        self._v2_ops = V2AsyncOperationManager()

    async def Evaluate(self, request: pb2.EvalRequest, context) -> pb2.EvalResponse:
        adapted = adapt_eval_request_to_policy_input(request)
        return await self._pipeline.evaluate(
            request=request,
            raw_policy_input=adapted.raw_policy_input,
            policy_name=adapted.policy_name,
            context=context,
        )

    async def EvaluateLayerPipeline(self, request: pb2.EvaluatePipelineRequest, context) -> pb2.EvalResponse:
        eval_request = request.eval_request
        adapted = adapt_eval_request_to_policy_input(eval_request)
        overrides = resolve_execution_overrides(
            request.execution,
            default_execution_mode=self._service_options.execution_mode,
            default_warmup_diffusion_steps=self._service_options.warmup_diffusion_steps,
            default_strict_layer_ordering=self._service_options.strict_layer_ordering,
            default_max_inflight_updates=self._service_options.max_inflight_updates,
            default_cache_ttl_ms=self._service_options.cache_ttl_ms,
            default_allow_stale_cache=self._service_options.allow_stale_cache,
            default_max_staleness_layers=self._service_options.max_staleness_layers,
            default_drop_late_updates=self._service_options.drop_late_updates,
        )
        return await self._pipeline.evaluate(
            request=eval_request,
            raw_policy_input=adapted.raw_policy_input,
            policy_name=adapted.policy_name,
            context=context,
            execution_mode=overrides.execution_mode,
            warmup_diffusion_steps=overrides.warmup_diffusion_steps,
            strict_layer_ordering=overrides.strict_layer_ordering,
            max_inflight_updates=overrides.max_inflight_updates,
            cache_ttl_ms=overrides.cache_ttl_ms,
            allow_stale_cache=overrides.allow_stale_cache,
            max_staleness_layers=overrides.max_staleness_layers,
            drop_late_updates=overrides.drop_late_updates,
        )

    async def SubmitEvaluate(self, request: pb2.SubmitEvaluateRequest, context) -> pb2.SubmitEvaluateResponse:
        return await self._v2_ops.submit(
            request=request,
            evaluate_pipeline=self.EvaluateLayerPipeline,
            context=context,
        )

    async def GetEvaluateResult(self, request: pb2.GetEvaluateResultRequest, context) -> pb2.GetEvaluateResultResponse:
        return await self._v2_ops.get_result(request=request, context=context)

    async def WatchEvaluate(self, request: pb2.WatchEvaluateRequest, context):
        async for event in self._v2_ops.watch(request=request, context=context):
            yield event

    async def CancelEvaluate(self, request: pb2.CancelEvaluateRequest, context) -> pb2.CancelEvaluateResponse:
        return await self._v2_ops.cancel(request=request)

    async def close(self) -> None:
        await self._v2_ops.close()
        await self._pipeline.close()


class SuffixServer:
    def __init__(
        self,
        host: str,
        port: int,
        prefix_host: str,
        prefix_port: int,
        loaded_component=None,
        *,
        options: SuffixServiceOptions = DEFAULT_SUFFIX_SERVICE_OPTIONS,
    ) -> None:
        self._address = f"{host}:{port}"
        self._service = SuffixService(
            prefix_address=f"{prefix_host}:{prefix_port}",
            loaded_component=loaded_component,
            options=options,
        )
        self._server = grpc.aio.server()
        pb2_grpc.add_SuffixServiceServicer_to_server(self._service, self._server)
        self._server.add_insecure_port(self._address)

    async def serve(self) -> None:
        await self._server.start()
        print(f"[suffix] listening on {self._address}")
        await self._server.wait_for_termination()

    async def start(self) -> None:
        await self._server.start()
        print(f"[suffix] listening on {self._address}")

    async def stop(self, grace_s: float = 0.5) -> None:
        await self._server.stop(grace_s)
        await self._service.close()

    async def wait_for_termination(self) -> None:
        await self._server.wait_for_termination()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suffix gRPC server (protobuf stub)")
    parser.add_argument("--host", default=DEFAULT_SUFFIX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SUFFIX_PORT)
    parser.add_argument("--prefix-host", default=DEFAULT_PREFIX_HOST)
    parser.add_argument("--prefix-port", type=int, default=DEFAULT_PREFIX_PORT)
    parser.add_argument("--policy-train-config", default="")
    parser.add_argument("--policy-checkpoint-dir", default="")
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--policy-name", choices=["aloha", "libero"], default="libero")
    parser.add_argument("--checkpoint-map-json", default="")
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--force-download-checkpoint", action="store_true")
    parser.add_argument("--auto-convert-checkpoint", action="store_true")
    parser.add_argument("--converted-checkpoint-dir", default="")
    parser.add_argument("--convert-precision", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--prefix-stream-timeout-s", type=float, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.prefix_stream_timeout_s)
    parser.add_argument(
        "--strict-layer-ordering",
        dest="strict_layer_ordering",
        action="store_true",
    )
    parser.add_argument(
        "--disable-strict-layer-ordering",
        dest="strict_layer_ordering",
        action="store_false",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["v1_layer_pipeline", "v2_async_cache", "v3_kv_polisher"],
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.execution_mode,
    )
    parser.add_argument("--warmup-diffusion-steps", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.warmup_diffusion_steps)
    parser.add_argument("--max-inflight-updates", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.max_inflight_updates)
    parser.add_argument("--cache-ttl-ms", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.cache_ttl_ms)
    parser.add_argument("--allow-stale-cache", action="store_true")
    parser.add_argument("--disable-allow-stale-cache", dest="allow_stale_cache", action="store_false")
    parser.add_argument("--max-staleness-layers", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.max_staleness_layers)
    parser.add_argument("--drop-late-updates", action="store_true")
    parser.add_argument("--disable-drop-late-updates", dest="drop_late_updates", action="store_false")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--profile-log-path", default=DEFAULT_SUFFIX_SERVICE_OPTIONS.profile_log_path)
    parser.add_argument(
        "--kv-transfer-mode",
        choices=["proto_bytes", "gpu_ipc"],
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.kv_transfer_mode,
    )
    parser.add_argument(
        "--gpu-ipc-suffix-sidecar-address",
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.gpu_ipc_suffix_sidecar_address,
    )
    parser.add_argument(
        "--gpu-ipc-resolve-mode",
        choices=["direct", "sidecar_fallback", "sidecar_only"],
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.gpu_ipc_resolve_mode,
    )
    parser.add_argument("--deterministic-noise", action="store_true")
    parser.add_argument("--startup-warmup", dest="startup_warmup", action="store_true")
    parser.add_argument("--disable-startup-warmup", dest="startup_warmup", action="store_false")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-timeout-s", type=float, default=90.0)
    parser.set_defaults(
        strict_layer_ordering=DEFAULT_SUFFIX_SERVICE_OPTIONS.strict_layer_ordering,
        allow_stale_cache=DEFAULT_SUFFIX_SERVICE_OPTIONS.allow_stale_cache,
        drop_late_updates=DEFAULT_SUFFIX_SERVICE_OPTIONS.drop_late_updates,
        startup_warmup=True,
    )
    return parser


def _runtime_policy_args(args: argparse.Namespace) -> RuntimePolicyArgs:
    return RuntimePolicyArgs(
        policy_train_config=args.policy_train_config,
        policy_checkpoint_dir=args.policy_checkpoint_dir,
        policy_name=args.policy_name,
        checkpoint_map_json=args.checkpoint_map_json,
        auto_download_checkpoint=args.auto_download_checkpoint,
        force_download_checkpoint=args.force_download_checkpoint,
        policy_device=args.policy_device,
        auto_convert_checkpoint=args.auto_convert_checkpoint,
        converted_checkpoint_dir=args.converted_checkpoint_dir,
        convert_precision=args.convert_precision,
    )


def _service_options_from_args(args: argparse.Namespace) -> SuffixServiceOptions:
    return SuffixServiceOptions(
        prefix_stream_timeout_s=args.prefix_stream_timeout_s,
        strict_layer_ordering=args.strict_layer_ordering,
        execution_mode=args.execution_mode,
        warmup_diffusion_steps=args.warmup_diffusion_steps,
        max_inflight_updates=args.max_inflight_updates,
        cache_ttl_ms=args.cache_ttl_ms,
        allow_stale_cache=args.allow_stale_cache,
        max_staleness_layers=args.max_staleness_layers,
        drop_late_updates=args.drop_late_updates,
        enable_profiling=args.enable_profiling,
        profile_log_path=args.profile_log_path,
        kv_transfer_mode=args.kv_transfer_mode,
        gpu_ipc_suffix_sidecar_address=args.gpu_ipc_suffix_sidecar_address,
        gpu_ipc_resolve_mode=args.gpu_ipc_resolve_mode,
        deterministic_noise=args.deterministic_noise,
    )


async def main_async(args: argparse.Namespace) -> None:
    loaded_component = load_suffix_component(_runtime_policy_args(args))
    server = SuffixServer(
        host=args.host,
        port=args.port,
        prefix_host=args.prefix_host,
        prefix_port=args.prefix_port,
        loaded_component=loaded_component,
        options=_service_options_from_args(args),
    )
    await server.start()
    if loaded_component is not None and args.startup_warmup:
        warmup_runs = max(1, int(args.warmup_runs))
        warmup_address = f"{args.host}:{args.port}"
        try:
            warmup_s = await run_suffix_endpoint_startup_warmup(
                address=warmup_address,
                policy_name=args.policy_name,
                runs=warmup_runs,
                timeout_s=float(args.warmup_timeout_s),
            )
            print(
                f"[suffix] startup warmup done policy={args.policy_name} "
                f"runs={warmup_runs} warmup_latency_s={warmup_s:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[suffix] startup warmup skipped due to error: {exc}")
    elif loaded_component is not None:
        print("[suffix] startup warmup disabled")
    else:
        print("[suffix] startup warmup skipped: policy is not loaded")
    await server.wait_for_termination()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
