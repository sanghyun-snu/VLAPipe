from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.training import config as training_config


@dataclass(frozen=True)
class PolicyLoadConfig:
    train_config_name: str
    checkpoint_dir: str | Path
    pytorch_device: str | None = None
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def load_openpi_policy(load_config: PolicyLoadConfig) -> policy_module.Policy:
    """Load a real OpenPI policy for runtime experiments."""
    train_cfg = training_config.get_config(load_config.train_config_name)
    return policy_config.create_trained_policy(
        train_cfg,
        load_config.checkpoint_dir,
        sample_kwargs=load_config.sample_kwargs,
        default_prompt=load_config.default_prompt,
        pytorch_device=load_config.pytorch_device,
    )
