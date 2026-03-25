#!/usr/bin/env python3
from __future__ import annotations

import argparse
import uuid

import grpc
import numpy as np

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc
from examples.pi0_grpc_native.utils.transport import POLICY_TYPE_NAME_TO_ENUM
from examples.pi0_grpc_native.utils.transport import ndarray_to_proto
from examples.pi0_grpc_native.utils.transport import proto_to_ndarray
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example


def _build_request(policy_name: str) -> pb2.EvalRequest:
    request = pb2.EvalRequest(
        request_id=str(uuid.uuid4()),
        policy_type=POLICY_TYPE_NAME_TO_ENUM[policy_name],
    )
    if policy_name == "droid":
        example = make_droid_example()
        request.droid.CopyFrom(
            pb2.DroidInput(
                exterior_image_left=ndarray_to_proto(example["observation/exterior_image_1_left"]),
                wrist_image_left=ndarray_to_proto(example["observation/wrist_image_left"]),
                joint_position=ndarray_to_proto(example["observation/joint_position"]),
                gripper_position=ndarray_to_proto(example["observation/gripper_position"]),
                prompt=example.get("prompt", ""),
            )
        )
    elif policy_name == "aloha":
        example = make_aloha_example()
        request.aloha.CopyFrom(
            pb2.AlohaInput(
                state=ndarray_to_proto(example["state"]),
                images={name: ndarray_to_proto(img) for name, img in example["images"].items()},
                prompt=example.get("prompt", ""),
            )
        )
    elif policy_name == "libero":
        example = make_libero_example()
        request.libero.CopyFrom(
            pb2.LiberoInput(
                state=ndarray_to_proto(example["observation/state"]),
                image=ndarray_to_proto(example["observation/image"]),
                wrist_image=ndarray_to_proto(example["observation/wrist_image"]),
                prompt=example.get("prompt", ""),
            )
        )
    else:
        raise ValueError(f"unsupported policy={policy_name}")
    return request


def _infer_actions(stub: pb2_grpc.SuffixServiceStub, request: pb2.EvalRequest, timeout_s: float) -> np.ndarray:
    response = stub.Evaluate(request, timeout=timeout_s)
    return np.asarray(proto_to_ndarray(response.actions), dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare action parity between two suffix endpoints.")
    parser.add_argument("--baseline-host", default="127.0.0.1")
    parser.add_argument("--baseline-port", type=int, default=50061)
    parser.add_argument("--test-host", default="127.0.0.1")
    parser.add_argument("--test-port", type=int, default=50063)
    parser.add_argument("--policy", choices=["droid", "aloha", "libero"], default="libero")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--different-request-id", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    baseline_addr = f"{args.baseline_host}:{args.baseline_port}"
    test_addr = f"{args.test_host}:{args.test_port}"

    max_abs_diff = 0.0
    mean_abs_diff_acc = 0.0
    cosine_acc = 0.0

    with grpc.insecure_channel(baseline_addr) as baseline_ch, grpc.insecure_channel(test_addr) as test_ch:
        baseline_stub = pb2_grpc.SuffixServiceStub(baseline_ch)
        test_stub = pb2_grpc.SuffixServiceStub(test_ch)
        for idx in range(args.num_requests):
            request = _build_request(args.policy)
            baseline_actions = _infer_actions(baseline_stub, request, args.timeout_s)
            # Keep policy input identical; by default keep request_id identical as well
            # so request-id-derived deterministic noise can be compared reliably.
            test_request = pb2.EvalRequest()
            test_request.CopyFrom(request)
            if args.different_request_id:
                test_request.request_id = str(uuid.uuid4())
            test_actions = _infer_actions(test_stub, test_request, args.timeout_s)

            if baseline_actions.shape != test_actions.shape:
                raise RuntimeError(
                    f"shape mismatch at sample={idx}: baseline={baseline_actions.shape} test={test_actions.shape}"
                )

            diff = np.abs(baseline_actions - test_actions)
            sample_max = float(np.max(diff))
            sample_mean = float(np.mean(diff))
            baseline_flat = baseline_actions.reshape(-1)
            test_flat = test_actions.reshape(-1)
            denom = float(np.linalg.norm(baseline_flat) * np.linalg.norm(test_flat))
            cosine = float(np.dot(baseline_flat, test_flat) / denom) if denom > 0 else 1.0

            max_abs_diff = max(max_abs_diff, sample_max)
            mean_abs_diff_acc += sample_mean
            cosine_acc += cosine
            print(
                f"[parity] sample={idx} max_abs_diff={sample_max:.6e} "
                f"mean_abs_diff={sample_mean:.6e} cosine={cosine:.8f}"
            )

    mean_abs_diff = mean_abs_diff_acc / float(max(1, args.num_requests))
    mean_cosine = cosine_acc / float(max(1, args.num_requests))
    passed = max_abs_diff <= args.atol
    print(
        f"[parity] summary baseline={baseline_addr} test={test_addr} "
        f"samples={args.num_requests} max_abs_diff={max_abs_diff:.6e} "
        f"mean_abs_diff={mean_abs_diff:.6e} mean_cosine={mean_cosine:.8f} atol={args.atol:.6e}"
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

