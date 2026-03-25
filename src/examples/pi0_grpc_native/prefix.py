from __future__ import annotations

import argparse
import asyncio

import grpc

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc

from .utils import PrefixPipeline
from .utils import PrefixPipelineConfig
from .utils.policy_adapter import adapt_eval_request_to_policy_input
from .utils.policy_runtime_loader import RuntimePolicyArgs
from .utils.split_policy_components import load_prefix_component

DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062
DEFAULT_STREAM_QUEUE_SIZE = 2
DEFAULT_QUEUE_WAIT_WARN_MS = 10.0
DEFAULT_PREFIX_REQUEST_TIMEOUT_S = 0.0
DEFAULT_ENABLE_PROFILING = False
DEFAULT_PROFILE_LOG_PATH = ""


class PrefixService(pb2_grpc.PrefixServiceServicer):
    def __init__(
        self,
        loaded_component=None,
        *,
        stream_queue_size: int = DEFAULT_STREAM_QUEUE_SIZE,
        prefer_layerwise: bool = True,
        allow_fallback: bool = True,
        queue_wait_warn_ms: float = DEFAULT_QUEUE_WAIT_WARN_MS,
        request_timeout_s: float = DEFAULT_PREFIX_REQUEST_TIMEOUT_S,
        enable_profiling: bool = DEFAULT_ENABLE_PROFILING,
        profile_log_path: str = DEFAULT_PROFILE_LOG_PATH,
    ) -> None:
        if stream_queue_size <= 0:
            raise ValueError(f"stream_queue_size must be > 0, got {stream_queue_size}")
        if queue_wait_warn_ms < 0:
            raise ValueError(f"queue_wait_warn_ms must be >= 0, got {queue_wait_warn_ms}")
        if request_timeout_s < 0:
            raise ValueError(f"request_timeout_s must be >= 0, got {request_timeout_s}")
        self._pipeline = PrefixPipeline(
            loaded_component=loaded_component,
            config=PrefixPipelineConfig(
                stream_queue_size=stream_queue_size,
                prefer_layerwise=prefer_layerwise,
                allow_fallback=allow_fallback,
                queue_wait_warn_ms=queue_wait_warn_ms,
                request_timeout_s=request_timeout_s,
                enable_profiling=enable_profiling,
                profile_log_path=profile_log_path,
            ),
        )

    async def StreamPrefixKV(self, request: pb2.PrefixRequest, context):
        eval_request = request.eval_request
        if not eval_request.request_id:
            raise ValueError("PrefixRequest.eval_request is required")
        adapted = adapt_eval_request_to_policy_input(eval_request)
        async for chunk in self._pipeline.stream_kv(
            request_id=eval_request.request_id,
            raw_policy_input=adapted.raw_policy_input,
            context=context,
        ):
            yield chunk


class PrefixServer:
    def __init__(
        self,
        host: str,
        port: int,
        loaded_component=None,
        *,
        stream_queue_size: int = DEFAULT_STREAM_QUEUE_SIZE,
        prefer_layerwise: bool = True,
        allow_fallback: bool = True,
        queue_wait_warn_ms: float = DEFAULT_QUEUE_WAIT_WARN_MS,
        request_timeout_s: float = DEFAULT_PREFIX_REQUEST_TIMEOUT_S,
        enable_profiling: bool = DEFAULT_ENABLE_PROFILING,
        profile_log_path: str = DEFAULT_PROFILE_LOG_PATH,
    ) -> None:
        self._address = f"{host}:{port}"
        self._server = grpc.aio.server()
        pb2_grpc.add_PrefixServiceServicer_to_server(
            PrefixService(
                loaded_component=loaded_component,
                stream_queue_size=stream_queue_size,
                prefer_layerwise=prefer_layerwise,
                allow_fallback=allow_fallback,
                queue_wait_warn_ms=queue_wait_warn_ms,
                request_timeout_s=request_timeout_s,
                enable_profiling=enable_profiling,
                profile_log_path=profile_log_path,
            ),
            self._server,
        )
        self._server.add_insecure_port(self._address)

    async def serve(self) -> None:
        await self._server.start()
        print(f"[prefix] listening on {self._address}")
        await self._server.wait_for_termination()

    async def start(self) -> None:
        await self._server.start()
        print(f"[prefix] listening on {self._address}")

    async def stop(self, grace_s: float = 0.5) -> None:
        await self._server.stop(grace_s)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prefix gRPC server (protobuf stub)")
    parser.add_argument("--host", default=DEFAULT_PREFIX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PREFIX_PORT)
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
    parser.add_argument("--stream-queue-size", type=int, default=DEFAULT_STREAM_QUEUE_SIZE)
    parser.add_argument("--queue-wait-warn-ms", type=float, default=DEFAULT_QUEUE_WAIT_WARN_MS)
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_PREFIX_REQUEST_TIMEOUT_S)
    parser.add_argument("--prefer-layerwise", dest="prefer_layerwise", action="store_true")
    parser.add_argument("--disable-layerwise", dest="prefer_layerwise", action="store_false")
    parser.add_argument("--allow-fallback", dest="allow_fallback", action="store_true")
    parser.add_argument("--disable-fallback", dest="allow_fallback", action="store_false")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--profile-log-path", default=DEFAULT_PROFILE_LOG_PATH)
    parser.set_defaults(prefer_layerwise=True, allow_fallback=True)
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


async def main_async(args: argparse.Namespace) -> None:
    loaded_component = load_prefix_component(_runtime_policy_args(args))
    await PrefixServer(
        host=args.host,
        port=args.port,
        loaded_component=loaded_component,
        stream_queue_size=args.stream_queue_size,
        prefer_layerwise=args.prefer_layerwise,
        allow_fallback=args.allow_fallback,
        queue_wait_warn_ms=args.queue_wait_warn_ms,
        request_timeout_s=args.request_timeout_s,
        enable_profiling=args.enable_profiling,
        profile_log_path=args.profile_log_path,
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
