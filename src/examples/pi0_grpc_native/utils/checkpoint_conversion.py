from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


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
    return target_dir
