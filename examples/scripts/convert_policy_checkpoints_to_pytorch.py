#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.pi0_grpc_native.utils.checkpoint_conversion import convert_jax_checkpoint_to_pytorch
from examples.pi0_grpc_native.utils.checkpoint_conversion import default_converted_checkpoint_dir
from examples.pi0_grpc_native.utils.checkpoint_runtime import resolve_runtime_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert downloaded OpenPI policy checkpoints to PyTorch safetensors.")
    parser.add_argument("--policy", choices=["aloha", "libero", "all"], default="all")
    parser.add_argument("--checkpoint-map-json", default="")
    parser.add_argument("--precision", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--write-json", default="")
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--force-download-checkpoint", action="store_true")
    return parser


def _targets(policy: str) -> list[str]:
    if policy == "all":
        return ["aloha", "libero"]
    return [policy]


def main() -> None:
    args = build_arg_parser().parse_args()
    out: dict[str, dict[str, str]] = {}
    for policy_name in _targets(args.policy):
        resolved = resolve_runtime_checkpoint(
            policy_name,
            mapping_json_path=args.checkpoint_map_json or None,
            auto_download=args.auto_download_checkpoint,
            force_download=args.force_download_checkpoint,
        )
        target_dir = default_converted_checkpoint_dir(resolved.checkpoint_dir)
        converted = convert_jax_checkpoint_to_pytorch(
            checkpoint_dir=resolved.checkpoint_dir,
            config_name=resolved.train_config_name,
            output_dir=target_dir,
            precision=args.precision,
        )
        out[policy_name] = {
            "checkpoint_url": resolved.checkpoint_url,
            "train_config": resolved.train_config_name,
            "jax_checkpoint_dir": str(resolved.checkpoint_dir),
            "pytorch_checkpoint_dir": str(converted),
        }
        print(f"[convert] {policy_name}: {resolved.checkpoint_dir} -> {converted}")

    if args.write_json:
        output_path = Path(args.write_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[convert] wrote mapping json: {output_path}")

    print("[convert] done")


if __name__ == "__main__":
    main()
