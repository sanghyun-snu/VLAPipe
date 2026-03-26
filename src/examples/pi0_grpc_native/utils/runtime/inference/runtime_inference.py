from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

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
    denoise_layer_callback: Callable[[int, int, float, bool], None] | None = None,
    now_fn: Callable[[], float] | None = None,
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
        on_step_layer=denoise_layer_callback,
        now_fn=now_fn,
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


async def run_suffix_denoise_with_cache_provider(
    component,
    raw_policy_input: dict[str, Any],
    *,
    cache_provider: Callable[[int, int], Awaitable[tuple[torch.Tensor, tuple]]],
    request_id: str = "",
    deterministic_noise: bool = False,
    warmup_diffusion_steps: int = 0,
    denoise_step_callback: Callable[[int, int, float, bool], None] | None = None,
    denoise_layer_callback: Callable[[int, int, float, bool], None] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> np.ndarray:
    """Run denoise loop with per-step cache selection.

    `cache_provider(step_idx, num_steps)` must return `(prefix_pad_masks, past_key_values)`.
    """
    if now_fn is None:
        import time

        now_fn = time.perf_counter
    model = component.model
    transformed_inputs, observation = _prepare_component_inputs(component, raw_policy_input)
    state, _device = preprocess_suffix_observation(model, observation)
    num_steps = int(component.sample_kwargs.get("num_steps", 10))
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    warmup_diffusion_steps = min(max(0, int(warmup_diffusion_steps)), num_steps)
    bsize = state.shape[0]
    device = state.device
    actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
    noise = None
    if deterministic_noise:
        if not request_id:
            raise ValueError("request_id is required when deterministic_noise is enabled")
        noise = build_torch_deterministic_noise(
            request_id,
            shape=actions_shape,
            device=state.device,
        )
    if noise is None:
        x_t = model.sample_noise(actions_shape, device)
    else:
        if tuple(noise.shape) != actions_shape:
            raise ValueError(f"noise shape mismatch: expected={actions_shape} got={tuple(noise.shape)}")
        x_t = noise.to(device=device, dtype=torch.float32)

    for step_idx in range(num_steps):
        step_start_t = now_fn()
        last_layer_t = step_start_t
        prefix_pad_masks, past_key_values = await cache_provider(step_idx, num_steps)
        model_cache = DynamicCache.from_legacy_cache(past_key_values)

        def _on_layer(layer_idx: int) -> None:
            nonlocal last_layer_t
            if denoise_layer_callback is None:
                return
            current_t = now_fn()
            layer_s = current_t - last_layer_t
            last_layer_t = current_t
            denoise_layer_callback(step_idx, layer_idx, layer_s, step_idx < warmup_diffusion_steps)

        time_value = 1.0 - (step_idx / float(num_steps))
        expanded_time = torch.full((bsize,), time_value, dtype=torch.float32, device=device)
        v_t = model.denoise_step(
            state,
            prefix_pad_masks,
            model_cache,
            x_t,
            expanded_time,
            on_layer=_on_layer if denoise_layer_callback is not None else None,
        )
        x_t = x_t + (-1.0 / num_steps) * v_t
        if denoise_step_callback is not None:
            denoise_step_callback(step_idx, num_steps, now_fn() - step_start_t, step_idx < warmup_diffusion_steps)

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
