#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio

from examples.pi0_grpc_native.prefix import PrefixServer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pi0 gRPC prefix server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50062)
    return parser


async def main_async(args: argparse.Namespace) -> None:
    await PrefixServer(host=args.host, port=args.port).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
