#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.pi0_grpc_native.utils.checkpoint_runtime import download_runtime_checkpoint
from examples.pi0_grpc_native.utils.checkpoint_runtime import RUNTIME_CHECKPOINTS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download OpenPI policy checkpoints for suffix server experiments.")
    parser.add_argument(
        "--policy",
        choices=["aloha", "libero", "all"],
        default="all",
        help="Which policy checkpoint to download.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download checkpoint even when cached locally.",
    )
    parser.add_argument(
        "--write-json",
        default="",
        help="Optional output json path with local checkpoint_dir and train_config mapping.",
    )
    return parser


def _targets(policy: str) -> list[str]:
    if policy == "all":
        return sorted(RUNTIME_CHECKPOINTS.keys())
    return [policy]


def main() -> None:
    args = build_arg_parser().parse_args()
    targets = _targets(args.policy)
    result: dict[str, dict[str, str]] = {}

    for name in targets:
        resolved = download_runtime_checkpoint(name, force_download=args.force_download)
        result[name] = {
            "checkpoint_url": resolved.checkpoint_url,
            "checkpoint_dir": str(resolved.checkpoint_dir),
            "train_config": resolved.train_config_name,
        }
        print(f"[download] {name}: {resolved.checkpoint_dir}")

    if args.write_json:
        out_path = Path(args.write_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[download] wrote mapping json: {out_path}")

    print("[download] done")


if __name__ == "__main__":
    main()
