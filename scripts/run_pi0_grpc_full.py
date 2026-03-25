#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio

from examples.pi0_grpc_native.full import main_async


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full PI0 gRPC server (no prefix/suffix split).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50063)
    parser.add_argument("--policy-train-config", default="")
    parser.add_argument("--policy-checkpoint-dir", default="")
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--policy-name", choices=["aloha", "libero"], default="libero")
    parser.add_argument("--checkpoint-map-json", default="")
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--force-download-checkpoint", action="store_true")
    parser.add_argument("--auto-convert-checkpoint", action="store_true")
    parser.add_argument("--converted-checkpoint-dir", default="")
    parser.add_argument("--convert-precision", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--deterministic-noise", action="store_true")
    parser.add_argument("--startup-warmup", dest="startup_warmup", action="store_true")
    parser.add_argument("--disable-startup-warmup", dest="startup_warmup", action="store_false")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.set_defaults(startup_warmup=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
