from __future__ import annotations

from typing import Any

import numpy as np
import torch

from openpi.models import model as model_base
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


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


def build_observation_from_raw(component, raw_policy_input: dict[str, Any]):
    """Apply policy transforms and build Observation tensor object."""
    inputs = component.input_transform(dict(raw_policy_input))
    inputs_torch = _tensorize_inputs(inputs, device=component.device)
    return model_base.Observation.from_dict(inputs_torch)


def compute_prefix_cache_from_policy(component, raw_policy_input: dict[str, Any]) -> tuple[torch.Tensor, tuple]:
    """Run real prefix path and return `(prefix_pad_masks, past_key_values)`."""
    model = component.model
    observation = build_observation_from_raw(component, raw_policy_input)
    images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, tuple(past_key_values)


def extract_layer_kv(past_key_values: tuple, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer_cache = past_key_values[layer_idx]
    if not isinstance(layer_cache, tuple) or len(layer_cache) < 2:
        raise ValueError(f"Unexpected layer cache type at index {layer_idx}: {type(layer_cache)}")
    return layer_cache[0], layer_cache[1]


def run_suffix_denoise_with_cache(component, raw_policy_input: dict[str, Any], prefix_pad_masks: torch.Tensor, past_key_values: tuple) -> np.ndarray:
    """Run real suffix denoising using externally provided prefix KV cache."""
    model = component.model
    observation = build_observation_from_raw(component, raw_policy_input)
    _images, _img_masks, _lang_tokens, _lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    bsize = state.shape[0]
    device = state.device
    actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
    noise = model.sample_noise(actions_shape, device)
    num_steps = int(component.sample_kwargs.get("num_steps", 10))

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        time += dt
    return x_t.detach().cpu().numpy()
