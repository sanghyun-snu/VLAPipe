from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from openpi.models import model as model_base
from openpi.policies.aloha_policy import AlohaInputs
from openpi.policies.droid_policy import DroidInputs
from openpi.policies.libero_policy import LiberoInputs

from .stream_protocol import POLICY_TYPE_ENUM_TO_NAME
from .stream_protocol import proto_to_ndarray


@dataclass(frozen=True)
class AdaptedPolicyInput:
    policy_name: str
    raw_policy_input: dict[str, Any]
    model_input: dict[str, Any]
    normalized_state: np.ndarray


def _policy_name(policy_type: int) -> str:
    if policy_type not in POLICY_TYPE_ENUM_TO_NAME:
        raise ValueError(f"Unsupported policy_type enum: {policy_type}")
    return POLICY_TYPE_ENUM_TO_NAME[policy_type]


def adapt_eval_request_to_policy_input(
    request: pb2.EvalRequest,
    *,
    model_type: model_base.ModelType = model_base.ModelType.PI0,
) -> AdaptedPolicyInput:
    """Convert proto EvalRequest to policy-specific raw/model-ready dictionaries."""
    policy_name = _policy_name(request.policy_type)

    if request.policy_type == pb2.POLICY_TYPE_DROID:
        raw = {
            "observation/exterior_image_1_left": proto_to_ndarray(request.droid.exterior_image_left),
            "observation/wrist_image_left": proto_to_ndarray(request.droid.wrist_image_left),
            "observation/joint_position": proto_to_ndarray(request.droid.joint_position),
            "observation/gripper_position": proto_to_ndarray(request.droid.gripper_position),
            "prompt": request.droid.prompt,
        }
        model_input = DroidInputs(model_type=model_type)(raw.copy())
        normalized_state = np.asarray(model_input["state"], dtype=np.float32)
        return AdaptedPolicyInput(policy_name=policy_name, raw_policy_input=raw, model_input=model_input, normalized_state=normalized_state)

    if request.policy_type == pb2.POLICY_TYPE_ALOHA:
        raw = {
            "state": proto_to_ndarray(request.aloha.state),
            "images": {name: proto_to_ndarray(img) for name, img in request.aloha.images.items()},
            "prompt": request.aloha.prompt,
        }
        model_input = AlohaInputs(adapt_to_pi=True)(raw.copy())
        normalized_state = np.asarray(model_input["state"], dtype=np.float32)
        return AdaptedPolicyInput(policy_name=policy_name, raw_policy_input=raw, model_input=model_input, normalized_state=normalized_state)

    if request.policy_type == pb2.POLICY_TYPE_LIBERO:
        raw = {
            "observation/state": proto_to_ndarray(request.libero.state),
            "observation/image": proto_to_ndarray(request.libero.image),
            "observation/wrist_image": proto_to_ndarray(request.libero.wrist_image),
            "prompt": request.libero.prompt,
        }
        model_input = LiberoInputs(model_type=model_type)(raw.copy())
        normalized_state = np.asarray(model_input["state"], dtype=np.float32)
        return AdaptedPolicyInput(policy_name=policy_name, raw_policy_input=raw, model_input=model_input, normalized_state=normalized_state)

    raise ValueError(f"Unsupported policy_type enum: {request.policy_type}")
