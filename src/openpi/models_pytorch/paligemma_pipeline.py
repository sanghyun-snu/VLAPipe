from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.layer_scheduler import LayerKVPayload
from openpi.models_pytorch.layer_scheduler import validate_layer_payload


def iter_paligemma_prefix_kv_layerwise(
    *,
    language_model,
    prefix_embs: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    prefix_att_2d_masks_4d: torch.Tensor,
    request_id: str,
):
    """Yield per-layer KV payloads from PaliGemma language model prefix pass."""
    hidden_states = prefix_embs
    if len(language_model.layers) > 0 and language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        hidden_states = hidden_states.to(torch.bfloat16)

    cache_position = torch.arange(prefix_embs.shape[1], device=prefix_embs.device)
    position_embeddings = language_model.rotary_emb(hidden_states, prefix_position_ids)
    past_key_values = DynamicCache()

    for layer_idx, decoder_layer in enumerate(language_model.layers[: language_model.config.num_hidden_layers]):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_value=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
        layer_key, layer_value = past_key_values[layer_idx]
        payload = LayerKVPayload(
            request_id=request_id,
            layer_idx=layer_idx,
            key=layer_key,
            value=layer_value,
            prefix_pad_mask=prefix_pad_masks if layer_idx == 0 else None,
        )
        validate_layer_payload(payload)
        yield payload
