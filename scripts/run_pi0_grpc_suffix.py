#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from pathlib import Path
import socket

from examples.pi0_grpc_native.suffix import SuffixServer
from examples.pi0_grpc_native.utils import RuntimePolicyArgs
from examples.pi0_grpc_native.utils import SuffixServiceOptions
from examples.pi0_grpc_native.utils import load_suffix_component
from examples.pi0_grpc_native.utils.runtime import run_suffix_endpoint_startup_warmup

DEFAULT_SUFFIX_SERVICE_OPTIONS = SuffixServiceOptions()
DEFAULT_SUFFIX_SIDECAR_BIN = Path(__file__).resolve().parents[1] / "build" / "pi0_sidecar" / "suffix_sidecar"


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
    parser.add_argument(
        "--gpu-ipc-resolve-mode",
        choices=["direct", "sidecar_fallback", "sidecar_only"],
        default=DEFAULT_SUFFIX_SERVICE_OPTIONS.gpu_ipc_resolve_mode,
    )
    parser.add_argument("--deterministic-noise", action="store_true")
    parser.add_argument("--with-sidecar", dest="with_sidecar", action="store_true")
    parser.add_argument("--without-sidecar", dest="with_sidecar", action="store_false")
    parser.add_argument("--sidecar-bin", default=str(DEFAULT_SUFFIX_SIDECAR_BIN))
    parser.add_argument("--sidecar-address", default=DEFAULT_SUFFIX_SERVICE_OPTIONS.gpu_ipc_suffix_sidecar_address)
    parser.add_argument(
        "--sidecar-upstream-address",
        default="127.0.0.1:55062",
        help="Upstream sidecar address for handle resolve miss forwarding.",
    )
    parser.add_argument("--sidecar-ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--startup-warmup", dest="startup_warmup", action="store_true")
    parser.add_argument("--disable-startup-warmup", dest="startup_warmup", action="store_false")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-timeout-s", type=float, default=90.0)
    parser.set_defaults(
        strict_layer_ordering=DEFAULT_SUFFIX_SERVICE_OPTIONS.strict_layer_ordering,
        allow_stale_cache=DEFAULT_SUFFIX_SERVICE_OPTIONS.allow_stale_cache,
        drop_late_updates=DEFAULT_SUFFIX_SERVICE_OPTIONS.drop_late_updates,
        with_sidecar=None,
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
        gpu_ipc_suffix_sidecar_address=args.sidecar_address or args.gpu_ipc_suffix_sidecar_address,
        gpu_ipc_resolve_mode=args.gpu_ipc_resolve_mode,
        deterministic_noise=args.deterministic_noise,
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


async def _start_sidecar_process(
    sidecar_bin: str,
    sidecar_address: str,
    ready_timeout_s: float,
    *,
    sidecar_upstream_address: str = "",
) -> asyncio.subprocess.Process:
    bin_path = Path(sidecar_bin).expanduser().resolve()
    if not bin_path.exists():
        raise FileNotFoundError(
            f"suffix sidecar binary not found: {bin_path}. "
            "Build with: cmake -S native/pi0_sidecar -B build/pi0_sidecar && cmake --build build/pi0_sidecar -j"
        )
    env = None
    if sidecar_upstream_address:
        env = dict(**os.environ, PI0_GPU_IPC_UPSTREAM=sidecar_upstream_address)
    process = await asyncio.create_subprocess_exec(str(bin_path), sidecar_address, env=env)
    try:
        await _wait_for_port_ready(sidecar_address, ready_timeout_s)
    except Exception:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=2.0)
        raise
    print(f"[suffix-runner] sidecar started pid={process.pid} address={sidecar_address}")
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
    return args.kv_transfer_mode == "gpu_ipc" and args.gpu_ipc_resolve_mode != "direct"


async def main_async(args: argparse.Namespace) -> None:
    sidecar_process: asyncio.subprocess.Process | None = None
    if _should_launch_sidecar(args):
        sidecar_process = await _start_sidecar_process(
            args.sidecar_bin,
            args.sidecar_address,
            args.sidecar_ready_timeout_s,
            sidecar_upstream_address=args.sidecar_upstream_address,
        )
    elif args.kv_transfer_mode == "gpu_ipc" and args.gpu_ipc_resolve_mode != "direct":
        await _wait_for_port_ready(args.sidecar_address, args.sidecar_ready_timeout_s)
        print(f"[suffix-runner] using existing sidecar address={args.sidecar_address}")

    loaded_component = load_suffix_component(_runtime_policy_args(args))
    server = SuffixServer(
        host=args.host,
        port=args.port,
        prefix_host=args.prefix_host,
        prefix_port=args.prefix_port,
        loaded_component=loaded_component,
        options=_service_options_from_args(args),
    )
    try:
        await server.start()
        if loaded_component is not None and args.startup_warmup:
            warmup_runs = max(1, int(args.warmup_runs))
            warmup_address = f"{args.host}:{args.port}"
            try:
                warmup_s = await run_suffix_endpoint_startup_warmup(
                    address=warmup_address,
                    policy_name=args.policy_name,
                    runs=warmup_runs,
                    timeout_s=float(args.warmup_timeout_s),
                )
                print(
                    f"[suffix-runner] startup warmup done policy={args.policy_name} "
                    f"runs={warmup_runs} warmup_latency_s={warmup_s:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[suffix-runner] startup warmup skipped due to error: {exc}")
        elif loaded_component is not None:
            print("[suffix-runner] startup warmup disabled")
        else:
            print("[suffix-runner] startup warmup skipped: policy is not loaded")
        await server.wait_for_termination()
    finally:
        await _stop_sidecar_process(sidecar_process)


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
