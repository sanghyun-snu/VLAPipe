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
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
