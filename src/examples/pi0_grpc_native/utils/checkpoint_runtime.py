from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import urllib.parse

from openpi.shared import download


@dataclass(frozen=True)
class RuntimeCheckpointSpec:
    policy_name: str
    train_config_name: str
    checkpoint_url: str


@dataclass(frozen=True)
class ResolvedRuntimeCheckpoint:
    policy_name: str
    train_config_name: str
    checkpoint_url: str
    checkpoint_dir: Path


RUNTIME_CHECKPOINTS: dict[str, RuntimeCheckpointSpec] = {
    "aloha": RuntimeCheckpointSpec(
        policy_name="aloha",
        train_config_name="pi0_aloha_sim",
        checkpoint_url="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    "libero": RuntimeCheckpointSpec(
        policy_name="libero",
        train_config_name="pi05_libero",
        checkpoint_url="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _cache_path_for_url(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    return (download.get_cache_dir() / parsed.netloc / parsed.path.strip("/")).resolve()


def _load_mapping_file(mapping_path: Path) -> dict:
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint_dir_from_mapping(entry: dict) -> Path:
    # Support both old and new mapping schemas.
    candidate = (
        entry.get("pytorch_checkpoint_dir")
        or entry.get("checkpoint_dir")
        or entry.get("jax_checkpoint_dir")
    )
    if not candidate:
        raise KeyError(
            "Expected one of: pytorch_checkpoint_dir, checkpoint_dir, jax_checkpoint_dir in mapping entry."
        )
    return Path(candidate).expanduser().resolve()


def download_runtime_checkpoint(policy_name: str, *, force_download: bool = False) -> ResolvedRuntimeCheckpoint:
    if policy_name not in RUNTIME_CHECKPOINTS:
        raise ValueError(f"Unsupported runtime policy '{policy_name}'. Expected one of: {tuple(RUNTIME_CHECKPOINTS)}")
    spec = RUNTIME_CHECKPOINTS[policy_name]
    local_path = download.maybe_download(spec.checkpoint_url, force_download=force_download)
    return ResolvedRuntimeCheckpoint(
        policy_name=spec.policy_name,
        train_config_name=spec.train_config_name,
        checkpoint_url=spec.checkpoint_url,
        checkpoint_dir=local_path,
    )


def resolve_runtime_checkpoint(
    policy_name: str,
    *,
    mapping_json_path: str | None = None,
    auto_download: bool = False,
    force_download: bool = False,
) -> ResolvedRuntimeCheckpoint:
    if policy_name not in RUNTIME_CHECKPOINTS:
        raise ValueError(f"Unsupported runtime policy '{policy_name}'. Expected one of: {tuple(RUNTIME_CHECKPOINTS)}")
    spec = RUNTIME_CHECKPOINTS[policy_name]

    if mapping_json_path:
        mapping_path = Path(mapping_json_path).expanduser().resolve()
        if not mapping_path.exists():
            raise FileNotFoundError(f"Checkpoint mapping json not found: {mapping_path}")
        mapping = _load_mapping_file(mapping_path)
        entry = mapping.get(policy_name)
        if entry:
            checkpoint_dir = _resolve_checkpoint_dir_from_mapping(entry)
            train_config_name = entry.get("train_config", spec.train_config_name)
            checkpoint_url = entry.get("checkpoint_url", spec.checkpoint_url)
            if checkpoint_dir.exists():
                return ResolvedRuntimeCheckpoint(
                    policy_name=policy_name,
                    train_config_name=train_config_name,
                    checkpoint_url=checkpoint_url,
                    checkpoint_dir=checkpoint_dir,
                )

    if auto_download:
        return download_runtime_checkpoint(policy_name, force_download=force_download)

    cached_path = _cache_path_for_url(spec.checkpoint_url)
    if not cached_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for policy '{policy_name}' not found in cache: {cached_path}. "
            "Run download script first or use --auto-download-checkpoint."
        )

    return ResolvedRuntimeCheckpoint(
        policy_name=policy_name,
        train_config_name=spec.train_config_name,
        checkpoint_url=spec.checkpoint_url,
        checkpoint_dir=cached_path,
    )
