from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def load_runtime_policy(args: RuntimePolicyArgs) -> Any | None:
    """Resolve checkpoint and load a real OpenPI policy when configured."""
    if args.policy_train_config and args.policy_checkpoint_dir:
        load_config = PolicyLoadConfig(
            train_config_name=args.policy_train_config,
            checkpoint_dir=args.policy_checkpoint_dir,
            pytorch_device=args.policy_device,
        )
        print(
            f"[runtime] loading explicit policy config={args.policy_train_config} "
            f"checkpoint={args.policy_checkpoint_dir}"
        )
        return load_openpi_policy(load_config)

    if args.policy_name is None:
        return None

    resolved = resolve_runtime_checkpoint(
        args.policy_name,
        mapping_json_path=args.checkpoint_map_json or None,
        auto_download=args.auto_download_checkpoint,
        force_download=args.force_download_checkpoint,
    )
    load_config = PolicyLoadConfig(
        train_config_name=resolved.train_config_name,
        checkpoint_dir=resolved.checkpoint_dir,
        pytorch_device=args.policy_device,
    )
    print(
        f"[runtime] loading policy policy_name={resolved.policy_name} "
        f"config={resolved.train_config_name} checkpoint={resolved.checkpoint_dir}"
    )
    return load_openpi_policy(load_config)
