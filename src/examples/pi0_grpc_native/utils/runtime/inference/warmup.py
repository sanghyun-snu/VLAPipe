from __future__ import annotations

import time
import uuid
from typing import Any

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example

from ...transport.grpc_cache import SuffixClient
from ...transport.stream_protocol import ndarray_to_proto
from .runtime_inference import compute_prefix_cache_from_policy


def build_warmup_policy_input(policy_name: str) -> dict[str, Any]:
    if policy_name == "droid":
        return make_droid_example()
    if policy_name == "aloha":
        return make_aloha_example()
    if policy_name == "libero":
        return make_libero_example()
    raise ValueError(f"unsupported policy_name for warmup: {policy_name}")


def run_full_policy_startup_warmup(loaded_policy: Any, policy_name: str, *, runs: int = 1) -> float:
    if runs <= 0:
        return 0.0
    warmup_input = build_warmup_policy_input(policy_name)
    start_t = time.perf_counter()
    for _ in range(runs):
        outputs = loaded_policy.infer(warmup_input)
        _ = outputs["actions"]
    return time.perf_counter() - start_t


def run_prefix_component_startup_warmup(component: Any, policy_name: str, *, runs: int = 1) -> float:
    if runs <= 0:
        return 0.0
    warmup_input = build_warmup_policy_input(policy_name)
    start_t = time.perf_counter()
    for _ in range(runs):
        _prefix_pad_masks, _past_key_values = compute_prefix_cache_from_policy(component, warmup_input)
    return time.perf_counter() - start_t


def build_warmup_eval_request(policy_name: str, *, request_id: str | None = None) -> pb2.EvalRequest:
    request_id = request_id or f"startup-warmup-{uuid.uuid4()}"
    request = pb2.EvalRequest(request_id=request_id)
    if policy_name == "droid":
        example = make_droid_example()
        request.policy_type = pb2.POLICY_TYPE_DROID
        request.droid.CopyFrom(
            pb2.DroidInput(
                exterior_image_left=ndarray_to_proto(example["observation/exterior_image_1_left"]),
                wrist_image_left=ndarray_to_proto(example["observation/wrist_image_left"]),
                joint_position=ndarray_to_proto(example["observation/joint_position"]),
                gripper_position=ndarray_to_proto(example["observation/gripper_position"]),
                prompt=example.get("prompt", ""),
            )
        )
        return request
    if policy_name == "aloha":
        example = make_aloha_example()
        request.policy_type = pb2.POLICY_TYPE_ALOHA
        request.aloha.CopyFrom(
            pb2.AlohaInput(
                state=ndarray_to_proto(example["state"]),
                images={name: ndarray_to_proto(img) for name, img in example["images"].items()},
                prompt=example.get("prompt", ""),
            )
        )
        return request
    if policy_name == "libero":
        example = make_libero_example()
        request.policy_type = pb2.POLICY_TYPE_LIBERO
        request.libero.CopyFrom(
            pb2.LiberoInput(
                state=ndarray_to_proto(example["observation/state"]),
                image=ndarray_to_proto(example["observation/image"]),
                wrist_image=ndarray_to_proto(example["observation/wrist_image"]),
                prompt=example.get("prompt", ""),
            )
        )
        return request
    raise ValueError(f"unsupported policy_name for warmup: {policy_name}")


async def run_suffix_endpoint_startup_warmup(
    *,
    address: str,
    policy_name: str,
    runs: int = 1,
    timeout_s: float = 60.0,
) -> float:
    if runs <= 0:
        return 0.0
    client = SuffixClient(address=address)
    start_t = time.perf_counter()
    try:
        for run_idx in range(runs):
            request = build_warmup_eval_request(policy_name, request_id=f"startup-warmup-{run_idx}")
            _ = await client.evaluate(request, timeout_s=timeout_s)
    finally:
        await client.close()
    return time.perf_counter() - start_t
