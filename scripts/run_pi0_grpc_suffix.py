#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio

from examples.pi0_grpc_native.suffix import SuffixServer
from examples.pi0_grpc_native.utils import RuntimePolicyArgs
from examples.pi0_grpc_native.utils import SuffixServiceOptions
from examples.pi0_grpc_native.utils import load_suffix_component

DEFAULT_SUFFIX_SERVICE_OPTIONS = SuffixServiceOptions()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pi0 gRPC suffix server with runtime policy loading.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50061)
    parser.add_argument("--prefix-host", default="127.0.0.1")
    parser.add_argument("--prefix-port", type=int, default=50062)

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
    parser.add_argument("--prefix-stream-timeout-s", type=float, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.prefix_stream_timeout_s)
    parser.add_argument("--strict-layer-ordering", dest="strict_layer_ordering", action="store_true")
    parser.add_argument("--disable-strict-layer-ordering", dest="strict_layer_ordering", action="store_false")
    parser.add_argument(
        "--execution-mode",
        choices=["v1_layer_pipeline", "v2_async_cache"],
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.execution_mode,
    )
    parser.add_argument("--warmup-diffusion-steps", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.warmup_diffusion_steps)
    parser.add_argument("--max-inflight-updates", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.max_inflight_updates)
    parser.add_argument("--cache-ttl-ms", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.cache_ttl_ms)
    parser.add_argument("--allow-stale-cache", action="store_true")
    parser.add_argument("--disable-allow-stale-cache", dest="allow_stale_cache", action="store_false")
    parser.add_argument("--max-staleness-layers", type=int, default=DEFAULT_SUFFIX_SERVICE_OPTIONS.max_staleness_layers)
    parser.add_argument("--drop-late-updates", action="store_true")
    parser.add_argument("--disable-drop-late-updates", dest="drop_late_updates", action="store_false")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--profile-log-path", default=DEFAULT_SUFFIX_SERVICE_OPTIONS.profile_log_path)
    parser.add_argument(
        "--kv-transfer-mode",
        choices=["proto_bytes", "gpu_ipc"],
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.kv_transfer_mode,
    )
    parser.add_argument(
        "--gpu-ipc-suffix-sidecar-address",
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.gpu_ipc_suffix_sidecar_address,
    )
    parser.set_defaults(
        strict_layer_ordering=DEFAULT_SUFFIX_SERVICE_OPTIONS.strict_layer_ordering,
        allow_stale_cache=DEFAULT_SUFFIX_SERVICE_OPTIONS.allow_stale_cache,
        drop_late_updates=DEFAULT_SUFFIX_SERVICE_OPTIONS.drop_late_updates,
    )
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


def _service_options_from_args(args: argparse.Namespace) -> SuffixServiceOptions:
    return SuffixServiceOptions(
        prefix_stream_timeout_s=args.prefix_stream_timeout_s,
        strict_layer_ordering=args.strict_layer_ordering,
        execution_mode=args.execution_mode,
        warmup_diffusion_steps=args.warmup_diffusion_steps,
        max_inflight_updates=args.max_inflight_updates,
        cache_ttl_ms=args.cache_ttl_ms,
        allow_stale_cache=args.allow_stale_cache,
        max_staleness_layers=args.max_staleness_layers,
        drop_late_updates=args.drop_late_updates,
        enable_profiling=args.enable_profiling,
        profile_log_path=args.profile_log_path,
        kv_transfer_mode=args.kv_transfer_mode,
        gpu_ipc_suffix_sidecar_address=args.gpu_ipc_suffix_sidecar_address,
    )


async def main_async(args: argparse.Namespace) -> None:
    loaded_component = load_suffix_component(_runtime_policy_args(args))
    await SuffixServer(
        host=args.host,
        port=args.port,
        prefix_host=args.prefix_host,
        prefix_port=args.prefix_port,
        loaded_component=loaded_component,
        options=_service_options_from_args(args),
    ).serve()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
