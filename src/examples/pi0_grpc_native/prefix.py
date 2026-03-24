from __future__ import annotations

import argparse
import asyncio

import grpc

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc

from .utils.policy_adapter import adapt_eval_request_to_policy_input
from .utils.policy_runtime_loader import RuntimePolicyArgs
from .utils.runtime_inference import compute_prefix_cache_from_policy
from .utils.runtime_inference import extract_layer_kv
from .utils.split_policy_components import load_prefix_component
from .utils.stream_protocol import tensor_to_proto

DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062


class PrefixService(pb2_grpc.PrefixServiceServicer):
    def __init__(self, loaded_policy=None) -> None:
        self._loaded_policy = loaded_policy

    async def StreamPrefixKV(self, request: pb2.PrefixRequest, _context):
        eval_request = request.eval_request
        if not eval_request.request_id:
            raise ValueError("PrefixRequest.eval_request is required")
        if self._loaded_policy is None:
            raise RuntimeError("Prefix split component is not loaded. Start server with policy checkpoint args.")
        adapted = adapt_eval_request_to_policy_input(eval_request)
        print(f"[prefix] start request={request.request_id or eval_request.request_id}")
        prefix_pad_masks, past_key_values = compute_prefix_cache_from_policy(
            self._loaded_policy, adapted.raw_policy_input
        )
        for layer_idx in range(len(past_key_values)):
            key, value = extract_layer_kv(past_key_values, layer_idx)
            yield pb2.KVCacheChunk(
                request_id=eval_request.request_id,
                layer_idx=layer_idx,
                key=tensor_to_proto(key),
                value=tensor_to_proto(value),
                prefix_pad_mask=tensor_to_proto(prefix_pad_masks) if layer_idx == 0 else pb2.NdArray(),
                has_prefix_pad_mask=(layer_idx == 0),
            )
            await asyncio.sleep(0)
        print(f"[prefix] end request={request.request_id or eval_request.request_id}")


class PrefixServer:
    def __init__(self, host: str, port: int, loaded_policy=None) -> None:
        self._address = f"{host}:{port}"
        self._server = grpc.aio.server()
        pb2_grpc.add_PrefixServiceServicer_to_server(PrefixService(loaded_policy=loaded_policy), self._server)
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
    parser.add_argument("--policy-name", choices=["aloha", "libero"], default=None)
    parser.add_argument("--checkpoint-map-json", default="")
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--force-download-checkpoint", action="store_true")
    parser.add_argument("--auto-convert-checkpoint", action="store_true")
    parser.add_argument("--converted-checkpoint-dir", default="")
    parser.add_argument("--convert-precision", choices=["float32", "bfloat16", "float16"], default="bfloat16")
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
    await PrefixServer(host=args.host, port=args.port, loaded_policy=loaded_component).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
