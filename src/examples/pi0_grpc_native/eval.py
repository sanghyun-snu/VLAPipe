from __future__ import annotations

import argparse
import asyncio
import uuid

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example

from .utils.grpc_cache import SuffixClient
from .utils.stream_protocol import POLICY_TYPE_NAME_TO_ENUM
from .utils.stream_protocol import ndarray_to_proto
from .utils.stream_protocol import proto_to_ndarray

DEFAULT_SUFFIX_HOST = "127.0.0.1"
DEFAULT_SUFFIX_PORT = 50061


def _build_policy_input(policy_name: str):
    if policy_name == "droid":
        example = make_droid_example()
        return pb2.DroidInput(
            exterior_image_left=ndarray_to_proto(example["observation/exterior_image_1_left"]),
            wrist_image_left=ndarray_to_proto(example["observation/wrist_image_left"]),
            joint_position=ndarray_to_proto(example["observation/joint_position"]),
            gripper_position=ndarray_to_proto(example["observation/gripper_position"]),
            prompt=example.get("prompt", ""),
        )
    if policy_name == "aloha":
        example = make_aloha_example()
        return pb2.AlohaInput(
            state=ndarray_to_proto(example["state"]),
            images={name: ndarray_to_proto(img) for name, img in example["images"].items()},
            prompt=example.get("prompt", ""),
        )
    if policy_name == "libero":
        example = make_libero_example()
        return pb2.LiberoInput(
            state=ndarray_to_proto(example["observation/state"]),
            image=ndarray_to_proto(example["observation/image"]),
            wrist_image=ndarray_to_proto(example["observation/wrist_image"]),
            prompt=example.get("prompt", ""),
        )
    raise ValueError(f"Unsupported policy: {policy_name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval client for suffix server (protobuf stub)")
    parser.add_argument("--host", default=DEFAULT_SUFFIX_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SUFFIX_PORT)
    parser.add_argument("--policy", choices=["droid", "aloha", "libero"], default="droid")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--compute-delay-s", type=float, default=0.05)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser


async def main_async(args: argparse.Namespace) -> None:
    request_id = args.request_id or str(uuid.uuid4())
    policy_input = _build_policy_input(args.policy)
    inference = pb2.InferenceConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        prefix_tokens=args.prefix_tokens,
        suffix_tokens=args.suffix_tokens,
        compute_delay_s=args.compute_delay_s,
        poll_interval_s=args.poll_interval_s,
        seed=args.seed,
    )
    request = pb2.EvalRequest(
        request_id=request_id,
        policy_type=POLICY_TYPE_NAME_TO_ENUM[args.policy],
        inference=inference,
    )
    if args.policy == "droid":
        request.droid.CopyFrom(policy_input)
    elif args.policy == "aloha":
        request.aloha.CopyFrom(policy_input)
    else:
        request.libero.CopyFrom(policy_input)

    client = SuffixClient(address=f"{args.host}:{args.port}")
    try:
        response = await client.evaluate(request, timeout_s=args.timeout_s)
        actions = proto_to_ndarray(response.actions)
        print(f"[eval] request_id={response.request_id}")
        print(f"[eval] message={response.message}")
        print(f"[eval] actions={actions.tolist()}")
    finally:
        await client.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
