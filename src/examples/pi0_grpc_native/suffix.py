from __future__ import annotations

import argparse
import asyncio
import contextlib

import grpc
import numpy as np
import torch

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc

from .utils.grpc_cache import KVCacheReceiver
from .utils.grpc_cache import PrefixClient
from .utils.layer_state import LayerStatus
from .utils.model_helpers import PipelineConfig
from .utils.model_helpers import finalize_actions
from .utils.model_helpers import make_suffix_query_from_state
from .utils.model_helpers import run_suffix_layer
from .utils.policy_adapter import adapt_eval_request_to_policy_input
from .utils.policy_runtime_loader import RuntimePolicyArgs
from .utils.policy_runtime_loader import load_runtime_policy
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
        loaded_policy=None,
    ) -> None:
        self._prefix_client = PrefixClient(address=prefix_address)
        self._loaded_policy = loaded_policy

    async def _drain_prefix_stream(self, prefix_request: pb2.PrefixRequest, receiver: KVCacheReceiver) -> None:
        async for chunk in self._prefix_client.stream_prefix(prefix_request, timeout_s=PREFIX_STREAM_TIMEOUT_S):
            await receiver.ingest(
                request_id=chunk.request_id,
                layer_idx=chunk.layer_idx,
                key=chunk.key,
                value=chunk.value,
            )
            print(f"[suffix] received kv request={chunk.request_id} layer={chunk.layer_idx}")

    async def Evaluate(self, request: pb2.EvalRequest, _context) -> pb2.EvalResponse:
        adapted = adapt_eval_request_to_policy_input(request)
        config = PipelineConfig(
            num_layers=request.inference.num_layers,
            hidden_size=request.inference.hidden_size,
            prefix_tokens=request.inference.prefix_tokens,
            suffix_tokens=request.inference.suffix_tokens,
            compute_delay_s=request.inference.compute_delay_s,
            seed=request.inference.seed,
        )
        receiver = KVCacheReceiver(layer_count=config.num_layers)
        prefix_request = pb2.PrefixRequest(
            request_id=request.request_id,
            policy_type=request.policy_type,
            inference=request.inference,
            normalized_state=ndarray_to_proto(adapted.normalized_state.astype(np.float32)),
        )
        stream_task = asyncio.create_task(self._drain_prefix_stream(prefix_request, receiver))
        query = make_suffix_query_from_state(
            config,
            state=torch.from_numpy(adapted.normalized_state).to(dtype=torch.float32),
            request_id=request.request_id,
        )

        try:
            for layer_idx in range(config.num_layers):
                while not await receiver.is_ready(request.request_id, layer_idx):
                    status = await receiver.status(request.request_id, layer_idx)
                    print(f"[suffix] waiting request={request.request_id} layer={layer_idx} status={status.value}")
                    await asyncio.sleep(request.inference.poll_interval_s)

                kv = await receiver.consume(request.request_id, layer_idx)
                query = run_suffix_layer(layer_idx, query, kv)
                status = await receiver.status(request.request_id, layer_idx)
                if status != LayerStatus.CONSUMED:
                    raise RuntimeError(f"Invalid transition request={request.request_id} layer={layer_idx} status={status}")
                print(f"[suffix] consumed request={request.request_id} layer={layer_idx}")
            await stream_task
        finally:
            if not stream_task.done():
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            await receiver.clear_session(request.request_id)

        if self._loaded_policy is not None:
            policy_out = self._loaded_policy.infer(adapted.raw_policy_input)
            actions = np.asarray(policy_out["actions"], dtype=np.float32)
            message = f"completed policy={adapted.policy_name} model=real_policy"
        else:
            actions = finalize_actions(query).detach().cpu().numpy()
            message = f"completed policy={adapted.policy_name} model=toy_suffix"

        return pb2.EvalResponse(
            request_id=request.request_id,
            actions=ndarray_to_proto(actions),
            message=message,
        )

    async def close(self) -> None:
        await self._prefix_client.close()


class SuffixServer:
    def __init__(self, host: str, port: int, prefix_host: str, prefix_port: int, loaded_policy=None) -> None:
        self._address = f"{host}:{port}"
        self._service = SuffixService(prefix_address=f"{prefix_host}:{prefix_port}", loaded_policy=loaded_policy)
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
    parser.add_argument("--policy-name", choices=["aloha", "libero"], default=None)
    parser.add_argument("--checkpoint-map-json", default="")
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--force-download-checkpoint", action="store_true")
    return parser


def _maybe_load_policy(args: argparse.Namespace):
    runtime_args = RuntimePolicyArgs(
        policy_train_config=args.policy_train_config,
        policy_checkpoint_dir=args.policy_checkpoint_dir,
        policy_name=args.policy_name,
        checkpoint_map_json=args.checkpoint_map_json,
        auto_download_checkpoint=args.auto_download_checkpoint,
        force_download_checkpoint=args.force_download_checkpoint,
        policy_device=args.policy_device,
    )
    return load_runtime_policy(runtime_args)


async def main_async(args: argparse.Namespace) -> None:
    loaded_policy = _maybe_load_policy(args)
    await SuffixServer(
        host=args.host,
        port=args.port,
        prefix_host=args.prefix_host,
        prefix_port=args.prefix_port,
        loaded_policy=loaded_policy,
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
