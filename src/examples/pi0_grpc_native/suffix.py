from __future__ import annotations

import argparse
import asyncio

import grpc

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc

from .utils import SuffixPipeline
from .utils import SuffixPipelineConfig
from .utils.grpc_cache import PrefixClient
from .utils.policy_adapter import adapt_eval_request_to_policy_input
from .utils.policy_runtime_loader import RuntimePolicyArgs
from .utils.split_policy_components import load_suffix_component

DEFAULT_SUFFIX_HOST = "127.0.0.1"
DEFAULT_SUFFIX_PORT = 50061
DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062
DEFAULT_PREFIX_STREAM_TIMEOUT_S = 30.0
DEFAULT_STRICT_LAYER_ORDERING = True


class SuffixService(pb2_grpc.SuffixServiceServicer):
    def __init__(
        self,
        *,
        prefix_address: str,
        loaded_component=None,
        prefix_stream_timeout_s: float = DEFAULT_PREFIX_STREAM_TIMEOUT_S,
        strict_layer_ordering: bool = DEFAULT_STRICT_LAYER_ORDERING,
    ) -> None:
        if prefix_stream_timeout_s <= 0:
            raise ValueError(f"prefix_stream_timeout_s must be > 0, got {prefix_stream_timeout_s}")
        self._pipeline = SuffixPipeline(
            prefix_client=PrefixClient(address=prefix_address),
            loaded_component=loaded_component,
            config=SuffixPipelineConfig(
                prefix_stream_timeout_s=prefix_stream_timeout_s,
                strict_layer_ordering=strict_layer_ordering,
            ),
        )

    async def Evaluate(self, request: pb2.EvalRequest, context) -> pb2.EvalResponse:
        adapted = adapt_eval_request_to_policy_input(request)
        return await self._pipeline.evaluate(
            request=request,
            raw_policy_input=adapted.raw_policy_input,
            policy_name=adapted.policy_name,
            context=context,
        )

    async def close(self) -> None:
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
        prefix_stream_timeout_s: float = DEFAULT_PREFIX_STREAM_TIMEOUT_S,
        strict_layer_ordering: bool = DEFAULT_STRICT_LAYER_ORDERING,
    ) -> None:
        self._address = f"{host}:{port}"
        self._service = SuffixService(
            prefix_address=f"{prefix_host}:{prefix_port}",
            loaded_component=loaded_component,
            prefix_stream_timeout_s=prefix_stream_timeout_s,
            strict_layer_ordering=strict_layer_ordering,
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
    parser.add_argument("--prefix-stream-timeout-s", type=float, default=DEFAULT_PREFIX_STREAM_TIMEOUT_S)
    parser.add_argument("--strict-layer-ordering", dest="strict_layer_ordering", action="store_true")
    parser.add_argument("--disable-strict-layer-ordering", dest="strict_layer_ordering", action="store_false")
    parser.set_defaults(strict_layer_ordering=DEFAULT_STRICT_LAYER_ORDERING)
    return parser


async def main_async(args: argparse.Namespace) -> None:
    loaded_component = load_suffix_component(
        RuntimePolicyArgs(
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
    )
    await SuffixServer(
        host=args.host,
        port=args.port,
        prefix_host=args.prefix_host,
        prefix_port=args.prefix_port,
        loaded_component=loaded_component,
        prefix_stream_timeout_s=args.prefix_stream_timeout_s,
        strict_layer_ordering=args.strict_layer_ordering,
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
