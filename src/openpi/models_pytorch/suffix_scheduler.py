from __future__ import annotations

import time
from collections.abc import Callable

import torch
from transformers.cache_utils import DynamicCache

from openpi.models import model as model_base
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def preprocess_suffix_observation(
    model: PI0Pytorch,
    observation: model_base.Observation,
) -> tuple[torch.Tensor, torch.device]:
    """Prepare state tensor for suffix denoising."""
    _images, _img_masks, _lang_tokens, _lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    return state, state.device


def run_suffix_denoise(
    model: PI0Pytorch,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values: tuple,
    *,
    num_steps: int,
    warmup_diffusion_steps: int = 0,
    noise: torch.Tensor | None = None,
    on_step: Callable[[int, int, float, bool], None] | None = None,
    on_step_layer: Callable[[int, int, float, bool], None] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> torch.Tensor:
    """Run denoising loop using external prefix KV cache."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    if warmup_diffusion_steps < 0:
        raise ValueError(f"warmup_diffusion_steps must be >= 0, got {warmup_diffusion_steps}")
    warmup_diffusion_steps = min(warmup_diffusion_steps, num_steps)
    if now_fn is None:
        now_fn = time.perf_counter

    bsize = state.shape[0]
    device = state.device
    actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
    if noise is None:
        noise = model.sample_noise(actions_shape, device)
    else:
        if tuple(noise.shape) != actions_shape:
            raise ValueError(f"noise shape mismatch: expected={actions_shape} got={tuple(noise.shape)}")
        noise = noise.to(device=device, dtype=torch.float32)
    model_cache = DynamicCache.from_legacy_cache(past_key_values)
    x_t = noise

    def _run_step_range(start_step: int, end_step: int, x: torch.Tensor) -> torch.Tensor:
        for step_idx in range(start_step, end_step):
            step_start_t = now_fn()
            last_layer_t = step_start_t

            def _on_layer(layer_idx: int) -> None:
                nonlocal last_layer_t
                if on_step_layer is None:
                    return
                current_t = now_fn()
                layer_s = current_t - last_layer_t
                last_layer_t = current_t
                on_step_layer(step_idx, layer_idx, layer_s, step_idx < warmup_diffusion_steps)

            time_value = 1.0 - (step_idx / float(num_steps))
            expanded_time = torch.full((bsize,), time_value, dtype=torch.float32, device=device)
            v_t = model.denoise_step(
                state,
                prefix_pad_masks,
                model_cache,
                x,
                expanded_time,
                on_layer=_on_layer if on_step_layer is not None else None,
            )
            x = x + (-1.0 / num_steps) * v_t
            if on_step is not None:
                step_elapsed_s = now_fn() - step_start_t
                on_step(step_idx, num_steps, step_elapsed_s, step_idx < warmup_diffusion_steps)
        return x

    x_t = _run_step_range(0, warmup_diffusion_steps, x_t)
    x_t = _run_step_range(warmup_diffusion_steps, num_steps, x_t)
    return x_t
