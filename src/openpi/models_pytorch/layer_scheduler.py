from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerKVPayload:
    request_id: str
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    prefix_pad_mask: torch.Tensor | None = None


def _validate_layer_kv_tensor_shapes(layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
    if key.ndim != 4:
        raise ValueError(f"Layer {layer_idx}: key must be rank-4 [B, H, T, D], got shape={tuple(key.shape)}")
    if value.ndim != 4:
        raise ValueError(f"Layer {layer_idx}: value must be rank-4 [B, H, T, D], got shape={tuple(value.shape)}")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError(
            f"Layer {layer_idx}: key/value leading dims mismatch, key={tuple(key.shape)} value={tuple(value.shape)}"
        )
    if key.device != value.device:
        raise ValueError(
            f"Layer {layer_idx}: key/value device mismatch, key={key.device} value={value.device}"
        )


def validate_layer_payload(payload: LayerKVPayload) -> None:
    if payload.layer_idx < 0:
        raise ValueError(f"layer_idx must be >= 0, got {payload.layer_idx}")
    if not payload.request_id:
        raise ValueError("request_id must be non-empty")
    if not isinstance(payload.key, torch.Tensor):
        raise TypeError(f"key must be torch.Tensor, got {type(payload.key)}")
    if not isinstance(payload.value, torch.Tensor):
        raise TypeError(f"value must be torch.Tensor, got {type(payload.value)}")
    _validate_layer_kv_tensor_shapes(payload.layer_idx, payload.key, payload.value)
    if payload.prefix_pad_mask is not None:
        if not isinstance(payload.prefix_pad_mask, torch.Tensor):
            raise TypeError(f"prefix_pad_mask must be torch.Tensor, got {type(payload.prefix_pad_mask)}")
        if payload.prefix_pad_mask.ndim != 2:
            raise ValueError(
                f"prefix_pad_mask must be rank-2 [B, T], got shape={tuple(payload.prefix_pad_mask.shape)}"
            )
        if payload.prefix_pad_mask.shape[0] != payload.key.shape[0]:
            raise ValueError(
                "Batch mismatch between prefix_pad_mask and KV tensors: "
                f"mask_batch={payload.prefix_pad_mask.shape[0]} kv_batch={payload.key.shape[0]}"
            )


def iter_layer_kv_payloads(
    request_id: str,
    past_key_values: tuple,
    prefix_pad_mask: torch.Tensor | None = None,
):
    """Yield layer KV payloads in ascending layer order."""
    for layer_idx, layer_cache in enumerate(past_key_values):
        if not isinstance(layer_cache, tuple) or len(layer_cache) < 2:
            raise ValueError(f"Unexpected layer cache type at index {layer_idx}: {type(layer_cache)}")
        key, value = layer_cache[0], layer_cache[1]
        payload = LayerKVPayload(
            request_id=request_id,
            layer_idx=layer_idx,
            key=key,
            value=value,
            prefix_pad_mask=prefix_pad_mask if layer_idx == 0 else None,
        )
        validate_layer_payload(payload)
        yield payload


class LayerCacheCollector:
    """Collect streamed per-layer KV payloads and build final cache tuple."""

    def __init__(
        self,
        *,
        expected_request_id: str | None = None,
        enforce_increasing_layer: bool = True,
    ) -> None:
        self._expected_request_id = expected_request_id
        self._enforce_increasing_layer = enforce_increasing_layer
        self._layer_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._prefix_pad_mask: torch.Tensor | None = None
        self._last_seen_layer_idx = -1

    def _request_context(self) -> str:
        request_id = self._expected_request_id if self._expected_request_id is not None else "<unknown>"
        return f"request_id={request_id}"

    def ingest(self, payload: LayerKVPayload) -> None:
        validate_layer_payload(payload)
        if self._expected_request_id is None:
            self._expected_request_id = payload.request_id
        if payload.request_id != self._expected_request_id:
            raise RuntimeError(
                "Unexpected request_id in KV payload "
                f"({self._request_context()}): expected={self._expected_request_id}, got={payload.request_id}"
            )
        if payload.layer_idx in self._layer_map:
            raise RuntimeError(f"Duplicate KV payload ({self._request_context()}) for layer {payload.layer_idx}")
        if self._enforce_increasing_layer and payload.layer_idx <= self._last_seen_layer_idx:
            raise RuntimeError(
                "Out-of-order KV payload "
                f"({self._request_context()}): previous_layer={self._last_seen_layer_idx}, got_layer={payload.layer_idx}"
            )
        self._layer_map[payload.layer_idx] = (payload.key, payload.value)
        self._last_seen_layer_idx = payload.layer_idx
        if payload.prefix_pad_mask is not None:
            if self._prefix_pad_mask is not None:
                raise RuntimeError(f"prefix_pad_mask received more than once ({self._request_context()})")
            self._prefix_pad_mask = payload.prefix_pad_mask

    def finalize(
        self, *, expected_num_layers: int | None = None
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if not self._layer_map:
            raise RuntimeError(f"No KV payloads received ({self._request_context()})")
        if expected_num_layers is not None and expected_num_layers <= 0:
            raise ValueError(f"expected_num_layers must be > 0, got {expected_num_layers}")
        if expected_num_layers is None:
            max_idx = max(self._layer_map.keys())
        else:
            max_idx = expected_num_layers - 1
            unexpected_layers = sorted(idx for idx in self._layer_map if idx > max_idx)
            if unexpected_layers:
                raise RuntimeError(
                    "Received unexpected KV payload layers beyond expected range "
                    f"({self._request_context()}): {unexpected_layers}"
                )
        layer_caches = []
        for idx in range(max_idx + 1):
            if idx not in self._layer_map:
                raise RuntimeError(f"Missing KV payload ({self._request_context()}) at layer {idx}")
            layer_caches.append(self._layer_map[idx])
        if self._prefix_pad_mask is None:
            raise RuntimeError(f"Prefix pad mask was not received ({self._request_context()})")
        return self._prefix_pad_mask, tuple(layer_caches)
