from __future__ import annotations

from collections.abc import Callable

import torch

def run_gemma_suffix_layerwise(
    *,
    gemma_model,
    suffix_embs: torch.Tensor,
    full_att_2d_masks_4d: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    adarms_cond: torch.Tensor | None = None,
    on_layer: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Run gemma expert layers explicitly, reporting per-layer timing via callback."""
    hidden_states = suffix_embs
    if len(gemma_model.layers) > 0 and gemma_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        hidden_states = hidden_states.to(torch.bfloat16)

    cache_position = torch.arange(suffix_embs.shape[1], device=suffix_embs.device)
    position_embeddings = gemma_model.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(gemma_model.layers[: gemma_model.config.num_hidden_layers]):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            adarms_cond=adarms_cond,
        )
        hidden_states = layer_outputs[0]
        if on_layer is not None:
            on_layer(layer_idx)

    hidden_states, _ = gemma_model.norm(hidden_states, adarms_cond)
    return hidden_states
