from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from openpi.models import model as model_base
from openpi.models_pytorch.prefix_scheduler import iter_prefix_forward_payloads
from openpi.models_pytorch.prefix_scheduler import run_prefix_forward
from openpi.models_pytorch.suffix_scheduler import preprocess_suffix_observation
from openpi.models_pytorch.suffix_scheduler import run_suffix_denoise

from .noise_facade import build_torch_deterministic_noise  # pyright: ignore[reportMissingImports]


def _to_tensor_batch(value: Any, *, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value)).to(device)[None, ...]


def _tensorize_inputs(data: dict[str, Any], *, device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[key] = {sub_key: _to_tensor_batch(sub_value, device=device) for sub_key, sub_value in value.items()}
        else:
            out[key] = _to_tensor_batch(value, device=device)
    return out


def _prepare_component_inputs(component, raw_policy_input: dict[str, Any]) -> tuple[dict[str, Any], model_base.Observation]:
    inputs = component.input_transform(dict(raw_policy_input))
    inputs_torch = _tensorize_inputs(inputs, device=component.device)
    return inputs, model_base.Observation.from_dict(inputs_torch)


def build_observation_from_raw(component, raw_policy_input: dict[str, Any]):
    _inputs, observation = _prepare_component_inputs(component, raw_policy_input)
    return observation


def compute_prefix_cache_from_policy(component, raw_policy_input: dict[str, Any]) -> tuple[torch.Tensor, tuple]:
    observation = build_observation_from_raw(component, raw_policy_input)
    return run_prefix_forward(component.model, observation)


def iter_prefix_cache_payloads_from_policy(
    component,
    raw_policy_input: dict[str, Any],
    request_id: str,
    *,
    prefer_layerwise: bool = True,
    allow_fallback: bool = True,
):
    if not request_id:
        raise ValueError("request_id must be non-empty")
    observation = build_observation_from_raw(component, raw_policy_input)
    yield from iter_prefix_forward_payloads(
        component.model,
        observation,
        request_id=request_id,
        prefer_layerwise=prefer_layerwise,
        allow_fallback=allow_fallback,
    )


def run_suffix_denoise_with_cache(
    component,
    raw_policy_input: dict[str, Any],
    prefix_pad_masks: torch.Tensor,
    past_key_values: tuple,
    *,
    warmup_diffusion_steps: int = 0,
    request_id: str = "",
    deterministic_noise: bool = False,
    denoise_step_callback: Callable[[int, int, float, bool], None] | None = None,
) -> np.ndarray:
    model = component.model
    transformed_inputs, observation = _prepare_component_inputs(component, raw_policy_input)
    state, _device = preprocess_suffix_observation(model, observation)
    num_steps = int(component.sample_kwargs.get("num_steps", 10))
    noise = None
    if deterministic_noise:
        if not request_id:
            raise ValueError("request_id is required when deterministic_noise is enabled")
        actions_shape = (state.shape[0], model.config.action_horizon, model.config.action_dim)
        noise = build_torch_deterministic_noise(
            request_id,
            shape=actions_shape,
            device=state.device,
        )
    x_t = run_suffix_denoise(
        model,
        state,
        prefix_pad_masks,
        past_key_values,
        num_steps=num_steps,
        warmup_diffusion_steps=warmup_diffusion_steps,
        noise=noise,
        on_step=denoise_step_callback,
    )
    actions = x_t.detach().cpu().numpy()
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    transformed_outputs = component.output_transform(
        {
            "state": np.asarray(transformed_inputs["state"]),
            "actions": np.asarray(actions),
        }
    )
    return np.asarray(transformed_outputs["actions"], dtype=np.float32)
