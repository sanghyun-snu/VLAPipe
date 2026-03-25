from __future__ import annotations

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
) -> torch.Tensor:
    """Run denoising loop using external prefix KV cache."""
    bsize = state.shape[0]
    device = state.device
    actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
    noise = model.sample_noise(actions_shape, device)
    model_cache = DynamicCache.from_legacy_cache(past_key_values)

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = model.denoise_step(state, prefix_pad_masks, model_cache, x_t, expanded_time)
        x_t = x_t + dt * v_t
        time += dt
    return x_t
