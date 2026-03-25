from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import subprocess
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def ensure_transformers_replace_installed(repo_root: Path | None = None) -> None:
    """Install local transformers patches required by PI0Pytorch."""
    repo_root = _repo_root() if repo_root is None else repo_root
    transformers = importlib.import_module("transformers")
    source_root = repo_root / "src" / "openpi" / "models_pytorch" / "transformers_replace"
    target_root = Path(transformers.__file__).resolve().parent
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        target = target_root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def default_converted_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    ckpt = Path(checkpoint_dir).expanduser().resolve()
    return ckpt.parent / f"{ckpt.name}_pytorch"


def convert_jax_checkpoint_to_pytorch(
    *,
    checkpoint_dir: str | Path,
    config_name: str,
    output_dir: str | Path | None = None,
    precision: str = "bfloat16",
) -> Path:
    """Invoke examples converter script and return converted checkpoint dir."""
    repo_root = _repo_root()
    source_dir = Path(checkpoint_dir).expanduser().resolve()
    target_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else default_converted_checkpoint_dir(source_dir)
    )
    converter = repo_root / "examples" / "convert_jax_model_to_pytorch.py"
    ensure_transformers_replace_installed(repo_root)
    command = [
        sys.executable,
        str(converter),
        "--checkpoint_dir",
        str(source_dir),
        "--config_name",
        config_name,
        "--output_path",
        str(target_dir),
        "--precision",
        precision,
    ]
    subprocess.run(command, check=True, cwd=repo_root)
    source_assets = source_dir / "assets"
    if source_assets.exists():
        target_assets = target_dir / "assets"
        if target_assets.exists():
            shutil.rmtree(target_assets)
        shutil.copytree(source_assets, target_assets)
    return target_dir
