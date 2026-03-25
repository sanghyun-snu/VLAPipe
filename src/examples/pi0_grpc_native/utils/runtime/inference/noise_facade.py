from __future__ import annotations

import hashlib

import numpy as np
import torch


def deterministic_seed_from_request_id(request_id: str) -> int:
    if not request_id:
        raise ValueError("request_id must be non-empty for deterministic noise")
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_numpy_deterministic_noise_with_shape(request_id: str, *, shape: tuple[int, ...]) -> np.ndarray:
    seed = deterministic_seed_from_request_id(request_id)
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal(shape), dtype=np.float32)


def build_numpy_deterministic_noise(
    request_id: str,
    *,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    return build_numpy_deterministic_noise_with_shape(
        request_id,
        shape=(action_horizon, action_dim),
    )


def build_torch_deterministic_noise(
    request_id: str,
    *,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> torch.Tensor:
    noise_np = build_numpy_deterministic_noise_with_shape(request_id, shape=shape)
    return torch.from_numpy(noise_np).to(device=device, dtype=torch.float32)
