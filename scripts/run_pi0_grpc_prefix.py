#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
from pathlib import Path
import socket

from examples.pi0_grpc_native.prefix import PrefixServer
from examples.pi0_grpc_native.utils import PrefixServiceOptions
from examples.pi0_grpc_native.utils import RuntimePolicyArgs
from examples.pi0_grpc_native.utils import load_prefix_component
from examples.pi0_grpc_native.utils.runtime import run_prefix_component_startup_warmup

DEFAULT_PREFIX_SERVICE_OPTIONS = PrefixServiceOptions()
DEFAULT_PREFIX_SIDECAR_BIN = Path(__file__).resolve().parents[1] / "build" / "pi0_sidecar" / "prefix_sidecar"


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
    parser.add_argument("--stream-queue-size", type=int, default=DEFAULT_PREFIX_SERVICE_OPTIONS.stream_queue_size)
    parser.add_argument("--queue-wait-warn-ms", type=float, default=DEFAULT_PREFIX_SERVICE_OPTIONS.queue_wait_warn_ms)
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_PREFIX_SERVICE_OPTIONS.request_timeout_s)
    parser.add_argument("--prefer-layerwise", dest="prefer_layerwise", action="store_true")
    parser.add_argument("--disable-layerwise", dest="prefer_layerwise", action="store_false")
    parser.add_argument("--allow-fallback", dest="allow_fallback", action="store_true")
    parser.add_argument("--disable-fallback", dest="allow_fallback", action="store_false")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--profile-log-path", default=DEFAULT_PREFIX_SERVICE_OPTIONS.profile_log_path)
    parser.add_argument(
        "--kv-transfer-mode",
        choices=["proto_bytes", "gpu_ipc"],
        default=DEFAULT_PREFIX_SERVICE_OPTIONS.kv_transfer_mode,
    )
    parser.add_argument(
        "--gpu-ipc-prefix-sidecar-address",
        default=DEFAULT_PREFIX_SERVICE_OPTIONS.gpu_ipc_prefix_sidecar_address,
    )
    parser.add_argument("--gpu-ipc-publish-sidecar", dest="gpu_ipc_publish_sidecar", action="store_true")
    parser.add_argument("--gpu-ipc-publish-direct-only", dest="gpu_ipc_publish_sidecar", action="store_false")
    parser.add_argument("--with-sidecar", dest="with_sidecar", action="store_true")
    parser.add_argument("--without-sidecar", dest="with_sidecar", action="store_false")
    parser.add_argument("--sidecar-bin", default=str(DEFAULT_PREFIX_SIDECAR_BIN))
    parser.add_argument("--sidecar-address", default=DEFAULT_PREFIX_SERVICE_OPTIONS.gpu_ipc_prefix_sidecar_address)
    parser.add_argument(
        "--sidecar-upstream-address",
        default="",
        help="Reserved for option symmetry with suffix runner; unused in prefix runner.",
    )
    parser.add_argument("--sidecar-ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--startup-warmup", dest="startup_warmup", action="store_true")
    parser.add_argument("--disable-startup-warmup", dest="startup_warmup", action="store_false")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.set_defaults(
        prefer_layerwise=DEFAULT_PREFIX_SERVICE_OPTIONS.prefer_layerwise,
        allow_fallback=DEFAULT_PREFIX_SERVICE_OPTIONS.allow_fallback,
        with_sidecar=None,
        gpu_ipc_publish_sidecar=DEFAULT_PREFIX_SERVICE_OPTIONS.gpu_ipc_publish_sidecar,
        startup_warmup=True,
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


def _service_options_from_args(args: argparse.Namespace) -> PrefixServiceOptions:
    return PrefixServiceOptions(
        stream_queue_size=args.stream_queue_size,
        prefer_layerwise=args.prefer_layerwise,
        allow_fallback=args.allow_fallback,
        queue_wait_warn_ms=args.queue_wait_warn_ms,
        request_timeout_s=args.request_timeout_s,
        enable_profiling=args.enable_profiling,
        profile_log_path=args.profile_log_path,
        kv_transfer_mode=args.kv_transfer_mode,
        gpu_ipc_prefix_sidecar_address=args.sidecar_address or args.gpu_ipc_prefix_sidecar_address,
        gpu_ipc_publish_sidecar=args.gpu_ipc_publish_sidecar,
    )


def _parse_host_port(address: str) -> tuple[str, int]:
    try:
        host, port_text = address.rsplit(":", 1)
        return host, int(port_text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid sidecar address={address!r}; expected host:port") from exc


async def _wait_for_port_ready(address: str, timeout_s: float) -> None:
    host, port = _parse_host_port(address)
    deadline = asyncio.get_event_loop().time() + timeout_s
    last_error: Exception | None = None
    while asyncio.get_event_loop().time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            sock.connect((host, port))
            sock.close()
            return
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.1)
        finally:
            sock.close()
    raise TimeoutError(f"sidecar not ready at {address} within {timeout_s:.1f}s: {last_error}")


async def _start_sidecar_process(sidecar_bin: str, sidecar_address: str, ready_timeout_s: float) -> asyncio.subprocess.Process:
    bin_path = Path(sidecar_bin).expanduser().resolve()
    if not bin_path.exists():
        raise FileNotFoundError(
            f"prefix sidecar binary not found: {bin_path}. "
            "Build with: cmake -S native/pi0_sidecar -B build/pi0_sidecar && cmake --build build/pi0_sidecar -j"
        )
    process = await asyncio.create_subprocess_exec(str(bin_path), sidecar_address)
    try:
        await _wait_for_port_ready(sidecar_address, ready_timeout_s)
    except Exception:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=2.0)
        raise
    print(f"[prefix-runner] sidecar started pid={process.pid} address={sidecar_address}")
    return process


async def _stop_sidecar_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _should_launch_sidecar(args: argparse.Namespace) -> bool:
    if args.with_sidecar is not None:
        return bool(args.with_sidecar)
    return args.kv_transfer_mode == "gpu_ipc" and bool(args.gpu_ipc_publish_sidecar)


async def main_async(args: argparse.Namespace) -> None:
    sidecar_process: asyncio.subprocess.Process | None = None
    if _should_launch_sidecar(args):
        sidecar_process = await _start_sidecar_process(
            args.sidecar_bin,
            args.sidecar_address,
            args.sidecar_ready_timeout_s,
        )
    elif args.kv_transfer_mode == "gpu_ipc" and args.gpu_ipc_publish_sidecar:
        await _wait_for_port_ready(args.sidecar_address, args.sidecar_ready_timeout_s)
        print(f"[prefix-runner] using existing sidecar address={args.sidecar_address}")

    loaded_component = load_prefix_component(_runtime_policy_args(args))
    if loaded_component is not None and args.startup_warmup:
        warmup_runs = max(1, int(args.warmup_runs))
        warmup_s = run_prefix_component_startup_warmup(loaded_component, args.policy_name, runs=warmup_runs)
        print(
            f"[prefix-runner] startup warmup done policy={args.policy_name} "
            f"runs={warmup_runs} warmup_latency_s={warmup_s:.4f}"
        )
    elif loaded_component is not None:
        print("[prefix-runner] startup warmup disabled")
    else:
        print("[prefix-runner] startup warmup skipped: policy is not loaded")
    try:
        await PrefixServer(
            host=args.host,
            port=args.port,
            loaded_component=loaded_component,
            options=_service_options_from_args(args),
        ).serve()
    finally:
        await _stop_sidecar_process(sidecar_process)


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
