#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2_grpc as pb2_grpc
from examples.pi0_grpc_native.utils.transport import POLICY_TYPE_NAME_TO_ENUM
from examples.pi0_grpc_native.utils.transport import ndarray_to_proto
from examples.pi0_grpc_native.utils.transport import proto_to_ndarray
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example


ROOT = Path(__file__).resolve().parents[1]
PREFIX_SCRIPT = ROOT / "scripts" / "run_pi0_grpc_prefix.py"
SUFFIX_SCRIPT = ROOT / "scripts" / "run_pi0_grpc_suffix.py"


@dataclass(frozen=True)
class Stats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


def _build_request(policy_name: str) -> pb2.EvalRequest:
    request = pb2.EvalRequest(
        request_id=str(uuid.uuid4()),
        policy_type=POLICY_TYPE_NAME_TO_ENUM[policy_name],
    )
    if policy_name == "droid":
        example = make_droid_example()
        request.droid.CopyFrom(
            pb2.DroidInput(
                exterior_image_left=ndarray_to_proto(example["observation/exterior_image_1_left"]),
                wrist_image_left=ndarray_to_proto(example["observation/wrist_image_left"]),
                joint_position=ndarray_to_proto(example["observation/joint_position"]),
                gripper_position=ndarray_to_proto(example["observation/gripper_position"]),
                prompt=example.get("prompt", ""),
            )
        )
    elif policy_name == "aloha":
        example = make_aloha_example()
        request.aloha.CopyFrom(
            pb2.AlohaInput(
                state=ndarray_to_proto(example["state"]),
                images={name: ndarray_to_proto(img) for name, img in example["images"].items()},
                prompt=example.get("prompt", ""),
            )
        )
    elif policy_name == "libero":
        example = make_libero_example()
        request.libero.CopyFrom(
            pb2.LiberoInput(
                state=ndarray_to_proto(example["observation/state"]),
                image=ndarray_to_proto(example["observation/image"]),
                wrist_image=ndarray_to_proto(example["observation/wrist_image"]),
                prompt=example.get("prompt", ""),
            )
        )
    else:
        raise ValueError(f"unsupported policy={policy_name}")
    return request


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            sock.connect((host, port))
            sock.close()
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
        finally:
            sock.close()
    raise TimeoutError(f"port not ready {host}:{port} within {timeout_s:.1f}s ({last_error})")


def _start_process(command: list[str], log_path: Path) -> subprocess.Popen:
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


def _run_eval_requests(host: str, port: int, policy: str, num_requests: int, timeout_s: float) -> float:
    address = f"{host}:{port}"
    latencies_s: list[float] = []
    with grpc.insecure_channel(address) as channel:
        stub = pb2_grpc.SuffixServiceStub(channel)
        for _ in range(num_requests):
            request = _build_request(policy)
            start_t = time.perf_counter()
            response = stub.Evaluate(request, timeout=timeout_s)
            latency_s = time.perf_counter() - start_t
            # Ensure response tensor is materialized and type-consistent.
            _ = np.asarray(proto_to_ndarray(response.actions), dtype=np.float32)
            latencies_s.append(latency_s)
    return (sum(latencies_s) / max(1, len(latencies_s))) * 1000.0


def _load_emit_overhead_stats(csv_path: Path) -> Stats:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event") != "emit_overhead_layer":
                continue
            value = row.get("value_s", "")
            if value:
                values.append(float(value) * 1000.0)
    if not values:
        return Stats(count=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, max_ms=0.0)
    p95_index = max(0, min(len(values) - 1, int(round(0.95 * (len(values) - 1)))))
    sorted_vals = sorted(values)
    return Stats(
        count=len(values),
        mean_ms=sum(values) / len(values),
        p50_ms=statistics.median(values),
        p95_ms=sorted_vals[p95_index],
        max_ms=max(values),
    )


def _trial(
    *,
    policy: str,
    checkpoint_map_json: str,
    policy_device: str,
    prefix_port: int,
    suffix_port: int,
    num_requests: int,
    timeout_s: float,
    publish_sidecar: bool,
    warmup_runs: int,
) -> tuple[Stats, float, Path, Path, Path]:
    suffix_process: subprocess.Popen | None = None
    prefix_process: subprocess.Popen | None = None
    with tempfile.TemporaryDirectory(prefix="pi0_changed_only_") as tempdir:
        temp_root = Path(tempdir)
        profile_path = temp_root / ("prefix_sidecar_on.csv" if publish_sidecar else "prefix_direct_only.csv")
        prefix_log = temp_root / ("prefix_sidecar_on.log" if publish_sidecar else "prefix_direct_only.log")
        suffix_log = temp_root / ("suffix_sidecar_on.log" if publish_sidecar else "suffix_direct_only.log")

        publish_flag = "--gpu-ipc-publish-sidecar" if publish_sidecar else "--gpu-ipc-publish-direct-only"

        prefix_cmd = [
            sys.executable,
            str(PREFIX_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(prefix_port),
            "--policy-name",
            policy,
            "--checkpoint-map-json",
            checkpoint_map_json,
            "--policy-device",
            policy_device,
            "--kv-transfer-mode",
            "gpu_ipc",
            publish_flag,
            "--enable-profiling",
            "--profile-log-path",
            str(profile_path),
            "--startup-warmup",
            "--warmup-runs",
            str(max(1, warmup_runs)),
        ]
        suffix_cmd = [
            sys.executable,
            str(SUFFIX_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(suffix_port),
            "--prefix-host",
            "127.0.0.1",
            "--prefix-port",
            str(prefix_port),
            "--policy-name",
            policy,
            "--checkpoint-map-json",
            checkpoint_map_json,
            "--policy-device",
            policy_device,
            "--kv-transfer-mode",
            "gpu_ipc",
            "--gpu-ipc-resolve-mode",
            "direct",
            "--deterministic-noise",
            "--startup-warmup",
            "--warmup-runs",
            str(max(1, warmup_runs)),
        ]

        prefix_process = _start_process(prefix_cmd, prefix_log)
        _wait_for_port("127.0.0.1", prefix_port, timeout_s=120.0)
        suffix_process = _start_process(suffix_cmd, suffix_log)
        _wait_for_port("127.0.0.1", suffix_port, timeout_s=120.0)

        mean_eval_ms = _run_eval_requests("127.0.0.1", suffix_port, policy, num_requests, timeout_s)
        # Give profiler flush a brief chance before shutdown.
        time.sleep(0.5)

        _stop_process(suffix_process)
        _stop_process(prefix_process)
        suffix_process = None
        prefix_process = None

        stats = _load_emit_overhead_stats(profile_path)
        # Persist artifacts into /tmp to allow inspection after tempdir cleanup.
        keep_root = Path("/tmp") / f"pi0_changed_only_{'sidecar_on' if publish_sidecar else 'direct_only'}_{int(time.time())}"
        keep_root.mkdir(parents=True, exist_ok=True)
        kept_profile = keep_root / profile_path.name
        kept_prefix_log = keep_root / prefix_log.name
        kept_suffix_log = keep_root / suffix_log.name
        kept_profile.write_bytes(profile_path.read_bytes())
        kept_prefix_log.write_bytes(prefix_log.read_bytes())
        kept_suffix_log.write_bytes(suffix_log.read_bytes())
        return stats, mean_eval_ms, kept_profile, kept_prefix_log, kept_suffix_log


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark only the changed prefix emit structure: sidecar publish ON vs direct-only."
    )
    parser.add_argument("--policy", choices=["droid", "aloha", "libero"], default="libero")
    parser.add_argument("--checkpoint-map-json", default="examples/checkpoint_map_pytorch.json")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--prefix-port", type=int, default=50062)
    parser.add_argument("--suffix-port", type=int, default=50061)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print("[bench] trial A: sidecar publish ON")
    a_stats, a_eval_ms, a_profile, a_prefix_log, a_suffix_log = _trial(
        policy=args.policy,
        checkpoint_map_json=args.checkpoint_map_json,
        policy_device=args.policy_device,
        prefix_port=args.prefix_port,
        suffix_port=args.suffix_port,
        num_requests=args.num_requests,
        timeout_s=args.timeout_s,
        publish_sidecar=True,
        warmup_runs=args.warmup_runs,
    )
    print("[bench] trial B: sidecar publish OFF (direct-only)")
    b_stats, b_eval_ms, b_profile, b_prefix_log, b_suffix_log = _trial(
        policy=args.policy,
        checkpoint_map_json=args.checkpoint_map_json,
        policy_device=args.policy_device,
        prefix_port=args.prefix_port,
        suffix_port=args.suffix_port,
        num_requests=args.num_requests,
        timeout_s=args.timeout_s,
        publish_sidecar=False,
        warmup_runs=args.warmup_runs,
    )

    print("")
    print("=== Changed-Only Benchmark Summary ===")
    print(
        f"A sidecar_on: emit_overhead count={a_stats.count} mean={a_stats.mean_ms:.3f}ms "
        f"p50={a_stats.p50_ms:.3f}ms p95={a_stats.p95_ms:.3f}ms max={a_stats.max_ms:.3f}ms "
        f"mean_eval={a_eval_ms:.3f}ms"
    )
    print(
        f"B direct_only: emit_overhead count={b_stats.count} mean={b_stats.mean_ms:.3f}ms "
        f"p50={b_stats.p50_ms:.3f}ms p95={b_stats.p95_ms:.3f}ms max={b_stats.max_ms:.3f}ms "
        f"mean_eval={b_eval_ms:.3f}ms"
    )
    if a_stats.count > 0 and b_stats.count > 0:
        print(
            f"delta_emit_mean_ms (A-B) = {a_stats.mean_ms - b_stats.mean_ms:.3f} "
            f"(positive means direct_only is faster)"
        )
    print(f"A artifacts: profile={a_profile} prefix_log={a_prefix_log} suffix_log={a_suffix_log}")
    print(f"B artifacts: profile={b_profile} prefix_log={b_prefix_log} suffix_log={b_suffix_log}")


if __name__ == "__main__":
    main()
