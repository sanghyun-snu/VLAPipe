from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint_conversion import convert_jax_checkpoint_to_pytorch
from .checkpoint_conversion import default_converted_checkpoint_dir
from .checkpoint_runtime import resolve_runtime_checkpoint
from .model_loader import PolicyLoadConfig
from .model_loader import load_openpi_policy


@dataclass(frozen=True)
class RuntimePolicyArgs:
    policy_train_config: str = ""
    policy_checkpoint_dir: str = ""
    policy_name: str | None = None
    checkpoint_map_json: str = ""
    auto_download_checkpoint: bool = False
    force_download_checkpoint: bool = False
    policy_device: str | None = None
    require_pytorch_model: bool = True
    auto_convert_checkpoint: bool = False
    converted_checkpoint_dir: str = ""
    convert_precision: str = "bfloat16"


@dataclass(frozen=True)
class RuntimeResolvedPolicy:
    policy_name: str
    train_config_name: str
    checkpoint_dir: Path
    policy_device: str | None


def _ensure_pytorch_checkpoint(checkpoint_dir: str, *, require_pytorch_model: bool) -> None:
    if not require_pytorch_model:
        return
    model_path = Path(checkpoint_dir).expanduser().resolve() / "model.safetensors"
    if not model_path.exists():
        raise RuntimeError(
            f"PyTorch checkpoint required but model file not found: {model_path}. "
            "Use converted PyTorch checkpoint containing model.safetensors."
        )


def _resolve_pytorch_checkpoint(checkpoint_dir: str | Path, train_config_name: str, args: RuntimePolicyArgs) -> Path:
    raw_path = Path(checkpoint_dir).expanduser().resolve()
    model_path = raw_path / "model.safetensors"
    if model_path.exists():
        return raw_path

    if not args.require_pytorch_model:
        return raw_path

    converted_dir = (
        Path(args.converted_checkpoint_dir).expanduser().resolve()
        if args.converted_checkpoint_dir
        else default_converted_checkpoint_dir(raw_path)
    )
    if (converted_dir / "model.safetensors").exists():
        return converted_dir

    if args.auto_convert_checkpoint:
        print(f"[runtime] converting checkpoint to PyTorch: src={raw_path} dst={converted_dir}")
        converted_dir = convert_jax_checkpoint_to_pytorch(
            checkpoint_dir=raw_path,
            config_name=train_config_name,
            output_dir=converted_dir,
            precision=args.convert_precision,
        )
        return converted_dir

    _ensure_pytorch_checkpoint(str(raw_path), require_pytorch_model=args.require_pytorch_model)
    return raw_path


def resolve_runtime_policy(args: RuntimePolicyArgs) -> RuntimeResolvedPolicy | None:
    if args.policy_train_config and args.policy_checkpoint_dir:
        checkpoint_dir = _resolve_pytorch_checkpoint(args.policy_checkpoint_dir, args.policy_train_config, args)
        print(f"[runtime] loading explicit policy config={args.policy_train_config} checkpoint={checkpoint_dir}")
        return RuntimeResolvedPolicy(
            policy_name=args.policy_name or "custom",
            train_config_name=args.policy_train_config,
            checkpoint_dir=checkpoint_dir,
            policy_device=args.policy_device,
        )

    if args.policy_name is None:
        return None

    resolved = resolve_runtime_checkpoint(
        args.policy_name,
        mapping_json_path=args.checkpoint_map_json or None,
        auto_download=args.auto_download_checkpoint,
        force_download=args.force_download_checkpoint,
    )
    checkpoint_dir = _resolve_pytorch_checkpoint(resolved.checkpoint_dir, resolved.train_config_name, args)
    print(
        f"[runtime] loading policy policy_name={resolved.policy_name} "
        f"config={resolved.train_config_name} checkpoint={checkpoint_dir}"
    )
    return RuntimeResolvedPolicy(
        policy_name=resolved.policy_name,
        train_config_name=resolved.train_config_name,
        checkpoint_dir=checkpoint_dir,
        policy_device=args.policy_device,
    )


def load_runtime_policy(args: RuntimePolicyArgs) -> Any | None:
    resolved = resolve_runtime_policy(args)
    if resolved is None:
        return None
    load_config = PolicyLoadConfig(
        train_config_name=resolved.train_config_name,
        checkpoint_dir=resolved.checkpoint_dir,
        pytorch_device=resolved.policy_device,
    )
    return load_openpi_policy(load_config)
