from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator

import grpc

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc

from .utils.model_helpers import PipelineConfig
from .utils.model_helpers import make_prefix_hidden_from_state
from .utils.model_helpers import run_prefix_layer
from .utils.stream_protocol import proto_to_tensor
from .utils.stream_protocol import tensor_to_proto

DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062


def iter_prefix_layers(config: PipelineConfig, request_id: str, state_tensor) -> Iterator[tuple[int, object, object]]:
    hidden = make_prefix_hidden_from_state(config, state=state_tensor, request_id=request_id)
    for layer_idx in range(config.num_layers):
        hidden, (key, value) = run_prefix_layer(layer_idx, hidden)
        yield layer_idx, key, value


class PrefixService(pb2_grpc.PrefixServiceServicer):
    async def StreamPrefixKV(self, request: pb2.PrefixRequest, _context):
        config = PipelineConfig(
            num_layers=request.inference.num_layers,
            hidden_size=request.inference.hidden_size,
            prefix_tokens=request.inference.prefix_tokens,
            suffix_tokens=request.inference.suffix_tokens,
            compute_delay_s=request.inference.compute_delay_s,
            seed=request.inference.seed,
        )
        state_tensor = proto_to_tensor(request.normalized_state)
        layer_iter = iter_prefix_layers(config, request_id=request.request_id, state_tensor=state_tensor)
        print(f"[prefix] start request={request.request_id}")
        while True:
            try:
                layer_idx, key, value = next(layer_iter)
            except StopIteration:
                break
            yield pb2.KVCacheChunk(
                request_id=request.request_id,
                layer_idx=layer_idx,
                key=tensor_to_proto(key),
                value=tensor_to_proto(value),
            )
            await asyncio.sleep(config.compute_delay_s)
        print(f"[prefix] end request={request.request_id}")


class PrefixServer:
    def __init__(self, host: str, port: int) -> None:
        self._address = f"{host}:{port}"
        self._server = grpc.aio.server()
        pb2_grpc.add_PrefixServiceServicer_to_server(PrefixService(), self._server)
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
    return parser


async def main_async(args: argparse.Namespace) -> None:
    await PrefixServer(host=args.host, port=args.port).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
