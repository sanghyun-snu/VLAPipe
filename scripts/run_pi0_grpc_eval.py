#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio

from examples.pi0_grpc_native.eval import main_async


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pi0 gRPC eval client for a policy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50061)
    parser.add_argument("--policy", choices=["droid", "aloha", "libero"], default="droid")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--compute-delay-s", type=float, default=0.05)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
