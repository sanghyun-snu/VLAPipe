from __future__ import annotations
# pyright: reportMissingImports=false

import argparse
import asyncio
import time

import grpc
import numpy as np

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc
from examples.pi0_grpc_native.utils import RuntimePolicyArgs
from examples.pi0_grpc_native.utils import adapt_eval_request_to_policy_input
from examples.pi0_grpc_native.utils.runtime import run_full_policy_startup_warmup
from examples.pi0_grpc_native.utils.runtime import load_runtime_policy
from examples.pi0_grpc_native.utils.runtime.inference.noise_facade import (
    build_numpy_deterministic_noise,  # pyright: ignore[reportMissingImports]
)
from examples.pi0_grpc_native.utils.transport import ndarray_to_proto

DEFAULT_FULL_HOST = "127.0.0.1"
DEFAULT_FULL_PORT = 50063


class FullService(pb2_grpc.SuffixServiceServicer):
    def __init__(self, loaded_policy=None, *, deterministic_noise: bool = False) -> None:
        self._loaded_policy = loaded_policy
        self._deterministic_noise = deterministic_noise

    async def Evaluate(self, request: pb2.EvalRequest, context) -> pb2.EvalResponse:
        if self._loaded_policy is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Full policy is not loaded. Start full server with policy checkpoint args.",
            )
        if not request.request_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "EvalRequest.request_id is required")

        adapted = adapt_eval_request_to_policy_input(request)
        infer_start_t = time.perf_counter()
        try:
            noise = None
            if self._deterministic_noise:
                model = getattr(self._loaded_policy, "_model", None)
                if model is None:
                    raise RuntimeError("loaded policy does not expose _model for deterministic noise")
                noise = build_numpy_deterministic_noise(
                    request.request_id,
                    action_horizon=int(model.config.action_horizon),
                    action_dim=int(model.config.action_dim),
                )
            outputs = self._loaded_policy.infer(adapted.raw_policy_input, noise=noise)
            actions = np.asarray(outputs["actions"], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            await context.abort(grpc.StatusCode.INTERNAL, f"full inference failed request_id={request.request_id}: {exc}")
        infer_latency_s = time.perf_counter() - infer_start_t
        print(f"[full] infer_latency_s={infer_latency_s:.4f} request_id={request.request_id}")

        return pb2.EvalResponse(
            request_id=request.request_id,
            actions=ndarray_to_proto(actions),
            message=f"completed policy={adapted.policy_name} model=full",
        )


class FullServer:
    def __init__(self, host: str, port: int, loaded_policy=None, *, deterministic_noise: bool = False) -> None:
        self._address = f"{host}:{port}"
        self._server = grpc.aio.server()
        pb2_grpc.add_SuffixServiceServicer_to_server(
            FullService(loaded_policy=loaded_policy, deterministic_noise=deterministic_noise),
            self._server,
        )
        self._server.add_insecure_port(self._address)

    async def serve(self) -> None:
        await self._server.start()
        print(f"[full] listening on {self._address}")
        await self._server.wait_for_termination()

    async def start(self) -> None:
        await self._server.start()
        print(f"[full] listening on {self._address}")

    async def stop(self, grace_s: float = 0.5) -> None:
        await self._server.stop(grace_s)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full PI0 gRPC server (no prefix/suffix split)")
    parser.add_argument("--host", default=DEFAULT_FULL_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_FULL_PORT)
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
    parser.add_argument("--deterministic-noise", action="store_true")
    parser.add_argument("--startup-warmup", dest="startup_warmup", action="store_true")
    parser.add_argument("--disable-startup-warmup", dest="startup_warmup", action="store_false")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.set_defaults(startup_warmup=True)
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
    loaded_policy = load_runtime_policy(_runtime_policy_args(args))
    startup_warmup = bool(getattr(args, "startup_warmup", True))
    warmup_runs = max(1, int(getattr(args, "warmup_runs", 1)))
    if loaded_policy is not None and startup_warmup:
        warmup_s = run_full_policy_startup_warmup(loaded_policy, args.policy_name, runs=warmup_runs)
        print(
            f"[full] startup warmup done policy={args.policy_name} "
            f"runs={warmup_runs} warmup_latency_s={warmup_s:.4f}"
        )
    elif loaded_policy is not None:
        print("[full] startup warmup disabled")
    else:
        print("[full] startup warmup skipped: policy is not loaded")
    await FullServer(
        host=args.host,
        port=args.port,
        loaded_policy=loaded_policy,
        deterministic_noise=args.deterministic_noise,
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
