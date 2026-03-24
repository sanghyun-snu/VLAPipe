from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections.abc import Iterator

import grpc

from .utils.grpc_cache import SUFFIX_METHOD_NAME
from .utils.grpc_cache import SUFFIX_SERVICE_NAME
from .utils.grpc_cache import KVCacheReceiver
from .utils.grpc_cache import PrefixStreamClient
from .utils.layer_state import LayerStatus
from .utils.model_helpers import PipelineConfig
from .utils.model_helpers import finalize_actions
from .utils.model_helpers import make_suffix_query_from_state
from .utils.model_helpers import run_suffix_layer
from .utils.stream_protocol import EvalResponse
from .utils.stream_protocol import PrefixRequest
from .utils.stream_protocol import deserialize_eval_request
from .utils.stream_protocol import deserialize_kv_payload
from .utils.stream_protocol import serialize_eval_response
from .utils.stream_protocol import serialize_prefix_request

DEFAULT_SUFFIX_HOST = "127.0.0.1"
DEFAULT_SUFFIX_PORT = 50061
DEFAULT_PREFIX_HOST = "127.0.0.1"
DEFAULT_PREFIX_PORT = 50062


def iter_suffix_layers(num_layers: int) -> Iterator[int]:
    for layer_idx in range(num_layers):
        yield layer_idx


class SuffixServer:
    """Eval-facing suffix server that forwards prefix request and returns final actions."""

    def __init__(self, host: str, port: int, prefix_host: str, prefix_port: int) -> None:
        self._address = f"{host}:{port}"
        self._prefix_address = f"{prefix_host}:{prefix_port}"
        self._receiver = KVCacheReceiver(layer_count=1)
        self._prefix_client = PrefixStreamClient(address=self._prefix_address)
        self._server = grpc.aio.server()
        handler = grpc.unary_unary_rpc_method_handler(self._evaluate)
        generic_handler = grpc.method_handlers_generic_handler(SUFFIX_SERVICE_NAME, {SUFFIX_METHOD_NAME: handler})
        self._server.add_generic_rpc_handlers((generic_handler,))
        self._server.add_insecure_port(self._address)

    async def _drain_prefix_stream(self, prefix_request: PrefixRequest) -> None:
        raw_request = serialize_prefix_request(prefix_request)
        async for raw_kv in self._prefix_client.stream_prefix(raw_request, timeout_s=30.0):
            payload = deserialize_kv_payload(raw_kv)
            await self._receiver.on_payload(payload)
            print(f"[suffix] received kv request={payload.request_id} layer={payload.layer_idx}")

    async def _evaluate(self, raw_request: bytes, _context) -> bytes:
        request = deserialize_eval_request(raw_request)
        pipeline_config = PipelineConfig(
            num_layers=request.num_layers,
            hidden_size=request.hidden_size,
            prefix_tokens=request.prefix_tokens,
            suffix_tokens=request.suffix_tokens,
            compute_delay_s=request.compute_delay_s,
            seed=request.seed,
        )
        self._receiver = KVCacheReceiver(layer_count=pipeline_config.num_layers)
        prefix_request = PrefixRequest(
            request_id=request.request_id,
            policy_name=request.policy_name,
            state=request.state,
            num_layers=request.num_layers,
            hidden_size=request.hidden_size,
            prefix_tokens=request.prefix_tokens,
            suffix_tokens=request.suffix_tokens,
            compute_delay_s=request.compute_delay_s,
            seed=request.seed,
        )
        stream_task = asyncio.create_task(self._drain_prefix_stream(prefix_request))
        query = make_suffix_query_from_state(pipeline_config, request.state, request_id=request.request_id)
        layer_iter = iter_suffix_layers(pipeline_config.num_layers)

        try:
            while True:
                try:
                    layer_idx = next(layer_iter)
                except StopIteration:
                    break

                while not await self._receiver.is_ready(layer_idx, request_id=request.request_id):
                    status = await self._receiver.status(layer_idx, request_id=request.request_id)
                    print(f"[suffix] waiting request={request.request_id} layer={layer_idx} status={status.value}")
                    await asyncio.sleep(request.poll_interval_s)

                kv = await self._receiver.consume(layer_idx, request_id=request.request_id)
                query = run_suffix_layer(layer_idx, query, kv)
                status = await self._receiver.status(layer_idx, request_id=request.request_id)
                if status != LayerStatus.CONSUMED:
                    raise RuntimeError(
                        f"Invalid suffix state transition request={request.request_id} layer={layer_idx} status={status}"
                    )
                print(f"[suffix] consumed request={request.request_id} layer={layer_idx}")
            await stream_task
        finally:
            if not stream_task.done():
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            await self._receiver.clear_session(request.request_id)

        response = EvalResponse(
            request_id=request.request_id,
            actions=finalize_actions(query),
            message=f"completed policy={request.policy_name}",
        )
        return serialize_eval_response(response)

    async def serve(self) -> None:
        await self._server.start()
        print(f"[suffix] listening on {self._address} forwarding to {self._prefix_address}")
        await self._server.wait_for_termination()

    async def start(self) -> None:
        await self._server.start()
        print(f"[suffix] listening on {self._address} forwarding to {self._prefix_address}")

    async def stop(self, grace_s: float = 0.5) -> None:
        await self._server.stop(grace_s)
        await self._prefix_client.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suffix gRPC server (eval-facing)")
    parser.add_argument("--host", default=DEFAULT_SUFFIX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SUFFIX_PORT)
    parser.add_argument("--prefix-host", default=DEFAULT_PREFIX_HOST)
    parser.add_argument("--prefix-port", type=int, default=DEFAULT_PREFIX_PORT)
    return parser


async def main_async(args: argparse.Namespace) -> None:
    server = SuffixServer(
        host=args.host,
        port=args.port,
        prefix_host=args.prefix_host,
        prefix_port=args.prefix_port,
    )
    await server.serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
