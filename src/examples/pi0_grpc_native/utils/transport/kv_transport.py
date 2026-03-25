from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


KVTransferMode = Literal["proto_bytes", "gpu_ipc"]
GpuIpcResolveMode = Literal["direct", "sidecar_fallback", "sidecar_only"]
DEFAULT_KV_TRANSFER_MODE: KVTransferMode = "gpu_ipc"
DEFAULT_GPU_IPC_RESOLVE_MODE: GpuIpcResolveMode = "direct"


def validate_kv_transfer_mode(mode: str) -> KVTransferMode:
    if mode == "proto_bytes":
        return "proto_bytes"
    if mode == "gpu_ipc":
        return "gpu_ipc"
    raise ValueError(f"unsupported kv_transfer_mode={mode!r}; expected one of: proto_bytes, gpu_ipc")


def validate_gpu_ipc_resolve_mode(mode: str) -> GpuIpcResolveMode:
    if mode == "direct":
        return "direct"
    if mode == "sidecar_fallback":
        return "sidecar_fallback"
    if mode == "sidecar_only":
        return "sidecar_only"
    raise ValueError(
        f"unsupported gpu_ipc_resolve_mode={mode!r}; expected one of: direct, sidecar_fallback, sidecar_only"
    )


@dataclass(frozen=True)
class GpuIpcOptions:
    enabled: bool = False
    prefix_sidecar_address: str = "127.0.0.1:55062"
    suffix_sidecar_address: str = "127.0.0.1:55061"

