#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio

from examples.pi0_grpc_native.prefix import PrefixServer
from examples.pi0_grpc_native.utils import RuntimePolicyArgs
from examples.pi0_grpc_native.utils import load_prefix_component


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pi0 gRPC prefix server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50062)
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
    parser.add_argument("--stream-queue-size", type=int, default=2)
    parser.add_argument("--queue-wait-warn-ms", type=float, default=10.0)
    parser.add_argument("--request-timeout-s", type=float, default=0.0)
    parser.add_argument("--prefer-layerwise", dest="prefer_layerwise", action="store_true")
    parser.add_argument("--disable-layerwise", dest="prefer_layerwise", action="store_false")
    parser.add_argument("--allow-fallback", dest="allow_fallback", action="store_true")
    parser.add_argument("--disable-fallback", dest="allow_fallback", action="store_false")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--profile-log-path", default="")
    parser.set_defaults(prefer_layerwise=True, allow_fallback=True)
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
        auto_convert_checkpoint=args.auto_convert_checkpoint,
        converted_checkpoint_dir=args.converted_checkpoint_dir,
        convert_precision=args.convert_precision,
    )


async def main_async(args: argparse.Namespace) -> None:
    loaded_component = load_prefix_component(_runtime_policy_args(args))
    await PrefixServer(
        host=args.host,
        port=args.port,
        loaded_component=loaded_component,
        stream_queue_size=args.stream_queue_size,
        prefer_layerwise=args.prefer_layerwise,
        allow_fallback=args.allow_fallback,
        queue_wait_warn_ms=args.queue_wait_warn_ms,
        request_timeout_s=args.request_timeout_s,
        enable_profiling=args.enable_profiling,
        profile_log_path=args.profile_log_path,
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
