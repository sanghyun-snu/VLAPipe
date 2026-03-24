#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio

from examples.pi0_grpc_native.suffix import SuffixServer
from examples.pi0_grpc_native.utils.policy_runtime_loader import RuntimePolicyArgs
from examples.pi0_grpc_native.utils.policy_runtime_loader import load_runtime_policy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pi0 gRPC suffix server with runtime policy loading.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50061)
    parser.add_argument("--prefix-host", default="127.0.0.1")
    parser.add_argument("--prefix-port", type=int, default=50062)

    parser.add_argument("--policy-train-config", default="")
    parser.add_argument("--policy-checkpoint-dir", default="")
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--policy-name", choices=["aloha", "libero"], default=None)
    parser.add_argument("--checkpoint-map-json", default="")
    parser.add_argument("--auto-download-checkpoint", action="store_true")
    parser.add_argument("--force-download-checkpoint", action="store_true")
    return parser


def _runtime_policy_args(args: argparse.Namespace) -> RuntimePolicyArgs:
    return RuntimePolicyArgs(
        policy_train_config=args.policy_train_config,
        policy_checkpoint_dir=args.policy_checkpoint_dir,
        policy_name=args.policy_name,
        checkpoint_map_json=args.checkpoint_map_json,
        auto_download_checkpoint=args.auto_download_checkpoint,
        force_download_checkpoint=args.force_download_checkpoint,
        policy_device=args.policy_device,
    )


async def main_async(args: argparse.Namespace) -> None:
    loaded_policy = load_runtime_policy(_runtime_policy_args(args))
    await SuffixServer(
        host=args.host,
        port=args.port,
        prefix_host=args.prefix_host,
        prefix_port=args.prefix_port,
        loaded_policy=loaded_policy,
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
