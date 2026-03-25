from __future__ import annotations

import argparse
import asyncio

import grpc
import numpy as np

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc
from openpi.models_pytorch.layer_scheduler import LayerCacheCollector
from openpi.models_pytorch.layer_scheduler import LayerKVPayload

from .utils.grpc_cache import PrefixClient
from .utils.policy_adapter import adapt_eval_request_to_policy_input
from .utils.policy_runtime_loader import RuntimePolicyArgs
from .utils.runtime_inference import run_suffix_denoise_with_cache
from .utils.split_policy_components import load_suffix_component
from .utils.stream_protocol import proto_to_tensor
from .utils.stream_protocol import ndarray_to_proto

DEFAULT_SUFFIX_HOST = "127.0.0.1"
DEFAULT_SUFFIX_PORT = 50061
DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062
PREFIX_STREAM_TIMEOUT_S = 30.0


class SuffixService(pb2_grpc.SuffixServiceServicer):
    def __init__(
        self,
        *,
        prefix_address: str,
        loaded_component=None,
    ) -> None:
        self._prefix_client = PrefixClient(address=prefix_address)
        self._loaded_component = loaded_component

    async def Evaluate(self, request: pb2.EvalRequest, _context) -> pb2.EvalResponse:
        if self._loaded_component is None:
            raise RuntimeError("Suffix split component is not loaded. Start server with component checkpoint args.")
        model_device = self._loaded_component.device
        adapted = adapt_eval_request_to_policy_input(request)
        prefix_request = pb2.PrefixRequest(
            request_id=request.request_id,
            eval_request=request,
        )
        scheduler = LayerCacheCollector()
        async for chunk in self._prefix_client.stream_prefix(prefix_request, timeout_s=PREFIX_STREAM_TIMEOUT_S):
            scheduler.ingest(
                LayerKVPayload(
                    request_id=chunk.request_id,
                    layer_idx=chunk.layer_idx,
                    key=proto_to_tensor(chunk.key, device=model_device),
                    value=proto_to_tensor(chunk.value, device=model_device),
                    prefix_pad_mask=proto_to_tensor(chunk.prefix_pad_mask, device=model_device)
                    if chunk.has_prefix_pad_mask
                    else None,
                )
            )
            print(f"[suffix] received kv request={chunk.request_id} layer={chunk.layer_idx}")

        prefix_pad_masks, layer_caches = scheduler.finalize()

        actions = run_suffix_denoise_with_cache(
            self._loaded_component,
            adapted.raw_policy_input,
            prefix_pad_masks,
            layer_caches,
        )
        actions = np.asarray(actions, dtype=np.float32)
        message = f"completed policy={adapted.policy_name} model=real_prefix_suffix"

        return pb2.EvalResponse(
            request_id=request.request_id,
            actions=ndarray_to_proto(actions),
            message=message,
        )

    async def close(self) -> None:
        await self._prefix_client.close()


class SuffixServer:
    def __init__(
        self,
        host: str,
        port: int,
        prefix_host: str,
        prefix_port: int,
        loaded_component=None,
    ) -> None:
        self._address = f"{host}:{port}"
        self._service = SuffixService(prefix_address=f"{prefix_host}:{prefix_port}", loaded_component=loaded_component)
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
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
