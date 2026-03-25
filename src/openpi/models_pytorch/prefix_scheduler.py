from __future__ import annotations

import torch

from openpi.models import model as model_base
from openpi.models_pytorch.layer_scheduler import LayerKVPayload
from openpi.models_pytorch.layer_scheduler import iter_layer_kv_payloads
from openpi.models_pytorch.paligemma_pipeline import iter_paligemma_prefix_kv_layerwise
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


def _prepare_prefix_forward_inputs(
    model: PI0Pytorch,
    observation: model_base.Observation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    return prefix_embs, prefix_pad_masks, prefix_position_ids, prefix_att_2d_masks_4d


def run_prefix_forward(model: PI0Pytorch, observation: model_base.Observation) -> tuple[torch.Tensor, tuple]:
    """Compute prefix KV cache for a PI0Pytorch model."""
    prefix_embs, prefix_pad_masks, prefix_position_ids, prefix_att_2d_masks_4d = _prepare_prefix_forward_inputs(
        model,
        observation,
    )
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, tuple(past_key_values)


def iter_prefix_forward_layerwise_payloads(
    model: PI0Pytorch,
    observation: model_base.Observation,
    *,
    request_id: str,
):
    """Yield KV payloads as each prefix decoder layer completes."""
    prefix_embs, prefix_pad_masks, prefix_position_ids, prefix_att_2d_masks_4d = _prepare_prefix_forward_inputs(
        model,
        observation,
    )
    language_model = model.paligemma_with_expert.paligemma.language_model
    language_model.config._attn_implementation = "eager"  # noqa: SLF001
    yield from iter_paligemma_prefix_kv_layerwise(
        language_model=language_model,
        prefix_embs=prefix_embs,
        prefix_pad_masks=prefix_pad_masks,
        prefix_position_ids=prefix_position_ids,
        prefix_att_2d_masks_4d=prefix_att_2d_masks_4d,
        request_id=request_id,
    )


def iter_prefix_forward_payloads(
    model: PI0Pytorch,
    observation: model_base.Observation,
    request_id: str,
    *,
    prefer_layerwise: bool = True,
    allow_fallback: bool = True,
):
    """Yield per-layer KV payloads for prefix forward pass, prioritizing layerwise execution."""
    if prefer_layerwise:
        try:
            yield from iter_prefix_forward_layerwise_payloads(model, observation, request_id=request_id)
            return
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            print(f"[prefix-scheduler] layerwise path failed, fallback to batch path: {exc}")

    prefix_pad_masks, past_key_values = run_prefix_forward(model, observation)
    yield from iter_layer_kv_payloads(
        request_id=request_id,
        past_key_values=past_key_values,
        prefix_pad_mask=prefix_pad_masks,
    )
