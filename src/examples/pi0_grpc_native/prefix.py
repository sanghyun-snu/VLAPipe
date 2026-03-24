from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator

import grpc

from .utils.grpc_cache import PREFIX_METHOD_NAME
from .utils.grpc_cache import PREFIX_SERVICE_NAME
from .utils.model_helpers import PipelineConfig
from .utils.model_helpers import make_prefix_hidden_from_state
from .utils.model_helpers import run_prefix_layer
from .utils.stream_protocol import deserialize_prefix_request
from .utils.stream_protocol import serialize_kv_payload

DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062


def iter_prefix_layers(config: PipelineConfig, request_id: str, state) -> Iterator[bytes]:
    """Yield serialized KV payload per layer for one request."""
    hidden = make_prefix_hidden_from_state(config, state=state, request_id=request_id)
    for layer_idx in range(config.num_layers):
        hidden, kv_payload = run_prefix_layer(layer_idx, hidden, request_id=request_id)
        yield serialize_kv_payload(kv_payload)


class PrefixServer:
    """Prefix model server that streams layer KV cache to suffix."""

    def __init__(self, host: str, port: int) -> None:
        self._address = f"{host}:{port}"
        self._server = grpc.aio.server()
        handler = grpc.unary_stream_rpc_method_handler(self._stream_prefix)
        generic_handler = grpc.method_handlers_generic_handler(PREFIX_SERVICE_NAME, {PREFIX_METHOD_NAME: handler})
        self._server.add_generic_rpc_handlers((generic_handler,))
        self._server.add_insecure_port(self._address)

    async def _stream_prefix(self, raw_request: bytes, _context):
        request = deserialize_prefix_request(raw_request)
        config = PipelineConfig(
            num_layers=request.num_layers,
            hidden_size=request.hidden_size,
            prefix_tokens=request.prefix_tokens,
            suffix_tokens=request.suffix_tokens,
            compute_delay_s=request.compute_delay_s,
            seed=request.seed,
        )
        layer_iter = iter_prefix_layers(config, request_id=request.request_id, state=request.state)
        print(f"[prefix] start request={request.request_id} policy={request.policy_name}")
        while True:
            try:
                raw_kv = next(layer_iter)
            except StopIteration:
                break
            yield raw_kv
            await asyncio.sleep(config.compute_delay_s)
        print(f"[prefix] end request={request.request_id}")

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
    parser = argparse.ArgumentParser(description="Prefix gRPC server (layer-wise KV streaming)")
    parser.add_argument("--host", default=DEFAULT_PREFIX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PREFIX_PORT)
    return parser


async def main_async(args: argparse.Namespace) -> None:
    server = PrefixServer(host=args.host, port=args.port)
    await server.serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
