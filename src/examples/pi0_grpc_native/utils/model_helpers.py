from __future__ import annotations

from dataclasses import dataclass

import torch

from .stream_protocol import KVCachePayload

DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = torch.float32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 8
DEFAULT_PREFIX_TOKENS = 32
DEFAULT_SUFFIX_TOKENS = 8
DEFAULT_COMPUTE_DELAY_S = 0.05
DEFAULT_ACTION_DIM = 8


@dataclass(frozen=True)
class PipelineConfig:
    num_layers: int = DEFAULT_NUM_LAYERS
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    prefix_tokens: int = DEFAULT_PREFIX_TOKENS
    suffix_tokens: int = DEFAULT_SUFFIX_TOKENS
    device: str = DEFAULT_DEVICE
    dtype: torch.dtype = DEFAULT_DTYPE
    compute_delay_s: float = DEFAULT_COMPUTE_DELAY_S
    seed: int = 7


def _stable_seed(base_seed: int, request_id: str) -> int:
    return base_seed + sum(ord(ch) for ch in request_id)


def make_prefix_hidden(config: PipelineConfig) -> torch.Tensor:
    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed)
    return torch.randn(
        1,
        config.prefix_tokens,
        config.hidden_size,
        generator=generator,
        device=config.device,
        dtype=config.dtype,
    )


def make_prefix_hidden_from_state(config: PipelineConfig, state: torch.Tensor, request_id: str) -> torch.Tensor:
    generator = torch.Generator(device=config.device)
    generator.manual_seed(_stable_seed(config.seed, request_id))
    noise = torch.randn(
        1,
        config.prefix_tokens,
        config.hidden_size,
        generator=generator,
        device=config.device,
        dtype=config.dtype,
    )
    state_bias = state.flatten().to(device=config.device, dtype=config.dtype).mean()
    return noise + (0.01 * state_bias)


def make_suffix_query(config: PipelineConfig) -> torch.Tensor:
    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed + 1)
    return torch.randn(
        1,
        config.suffix_tokens,
        config.hidden_size,
        generator=generator,
        device=config.device,
        dtype=config.dtype,
    )


def make_suffix_query_from_state(config: PipelineConfig, state: torch.Tensor, request_id: str) -> torch.Tensor:
    generator = torch.Generator(device=config.device)
    generator.manual_seed(_stable_seed(config.seed + 1, request_id))
    noise = torch.randn(
        1,
        config.suffix_tokens,
        config.hidden_size,
        generator=generator,
        device=config.device,
        dtype=config.dtype,
    )
    state_bias = state.flatten().to(device=config.device, dtype=config.dtype).mean()
    return noise + (0.02 * state_bias)


def run_prefix_layer(layer_idx: int, hidden: torch.Tensor, request_id: str = "default") -> tuple[torch.Tensor, KVCachePayload]:
    """Toy PyTorch-native prefix pass that emits per-layer KV cache."""
    scale = 1.0 + layer_idx * 0.01
    key = hidden * scale
    value = torch.tanh(hidden) * scale
    next_hidden = hidden + 0.1 * value
    return next_hidden, KVCachePayload(request_id=request_id, layer_idx=layer_idx, key=key, value=value)


def run_suffix_layer(layer_idx: int, query: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Toy PyTorch-native suffix layer consuming KV cache for one layer."""
    key, value = kv
    context = (key.mean(dim=1, keepdim=True) + value.mean(dim=1, keepdim=True)) / 2.0
    projected = torch.tanh(query + context)
    return projected + (0.001 * layer_idx)


def finalize_actions(query: torch.Tensor, action_dim: int = DEFAULT_ACTION_DIM) -> torch.Tensor:
    pooled = query.mean(dim=1)
    return pooled[:, :action_dim].to(dtype=torch.float32)
