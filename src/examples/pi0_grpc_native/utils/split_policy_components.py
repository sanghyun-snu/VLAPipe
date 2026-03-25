from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import safetensors
import torch

from openpi import transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import checkpoints as checkpoints_lib
from openpi.training import config as training_config

from .checkpoint_conversion import ensure_transformers_replace_installed
from .policy_runtime_loader import RuntimePolicyArgs
from .policy_runtime_loader import resolve_runtime_policy


PREFIX_WEIGHT_PREFIXES = ("paligemma_with_expert.paligemma.",)
SUFFIX_WEIGHT_PREFIXES = (
    "paligemma_with_expert.gemma_expert.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
)


@dataclass(frozen=True)
class PrefixComponent:
    model: PI0Pytorch
    input_transform: Any
    device: str


@dataclass(frozen=True)
class SuffixComponent:
    model: PI0Pytorch
    input_transform: Any
    output_transform: Any
    device: str
    sample_kwargs: dict[str, Any]


def _make_input_transform(train_cfg, checkpoint_dir: Path):
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats")
    norm_stats = checkpoints_lib.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _make_output_transform(train_cfg, checkpoint_dir: Path):
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats")
    norm_stats = checkpoints_lib.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )


def _load_filtered_weights(model: PI0Pytorch, checkpoint_dir: Path, allowed_prefixes: tuple[str, ...]) -> None:
    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"model.safetensors not found: {model_path}")
    target_keys = set(model.state_dict().keys())
    load_dict: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if any(key.startswith(prefix) for prefix in allowed_prefixes) and key in target_keys:
                load_dict[key] = f.get_tensor(key)
    result = model.load_state_dict(load_dict, strict=False)
    print(
        f"[split-load] loaded={len(load_dict)} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
    )


def _prune_for_prefix(model: PI0Pytorch) -> None:
    model.paligemma_with_expert.gemma_expert = None
    model.action_in_proj = None
    model.action_out_proj = None
    if hasattr(model, "time_mlp_in"):
        model.time_mlp_in = None
    if hasattr(model, "time_mlp_out"):
        model.time_mlp_out = None
    if hasattr(model, "state_proj"):
        model.state_proj = None
    if hasattr(model, "action_time_mlp_in"):
        model.action_time_mlp_in = None
    if hasattr(model, "action_time_mlp_out"):
        model.action_time_mlp_out = None


def _prune_for_suffix(model: PI0Pytorch) -> None:
    model.paligemma_with_expert.paligemma = None


def _build_split_model(
    train_cfg,
    checkpoint_dir: Path,
    device: str,
    allowed_prefixes: tuple[str, ...],
    prune_fn,
) -> PI0Pytorch:
    ensure_transformers_replace_installed()
    model = PI0Pytorch(train_cfg.model)
    _load_filtered_weights(model, checkpoint_dir, allowed_prefixes)
    prune_fn(model)
    model = model.to(device)
    model.eval()
    return model


def load_prefix_component(args: RuntimePolicyArgs) -> PrefixComponent | None:
    resolved = resolve_runtime_policy(args)
    if resolved is None:
        return None
    train_cfg = training_config.get_config(resolved.train_config_name)
    checkpoint_dir = Path(resolved.checkpoint_dir).expanduser().resolve()
    device = args.policy_device or "cuda"
    input_transform = _make_input_transform(train_cfg, checkpoint_dir)
    model = _build_split_model(train_cfg, checkpoint_dir, device, PREFIX_WEIGHT_PREFIXES, _prune_for_prefix)
    return PrefixComponent(model=model, input_transform=input_transform, device=device)


def load_suffix_component(args: RuntimePolicyArgs) -> SuffixComponent | None:
    resolved = resolve_runtime_policy(args)
    if resolved is None:
        return None
    train_cfg = training_config.get_config(resolved.train_config_name)
    checkpoint_dir = Path(resolved.checkpoint_dir).expanduser().resolve()
    device = args.policy_device or "cuda"
    input_transform = _make_input_transform(train_cfg, checkpoint_dir)
    output_transform = _make_output_transform(train_cfg, checkpoint_dir)
    model = _build_split_model(train_cfg, checkpoint_dir, device, SUFFIX_WEIGHT_PREFIXES, _prune_for_suffix)
    return SuffixComponent(
        model=model,
        input_transform=input_transform,
        output_transform=output_transform,
        device=device,
        sample_kwargs={},
    )
