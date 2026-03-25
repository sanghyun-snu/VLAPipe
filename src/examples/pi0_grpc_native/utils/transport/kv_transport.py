from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


KVTransferMode = Literal["proto_bytes", "gpu_ipc"]
DEFAULT_KV_TRANSFER_MODE: KVTransferMode = "gpu_ipc"


def validate_kv_transfer_mode(mode: str) -> KVTransferMode:
    if mode == "proto_bytes":
        return "proto_bytes"
    if mode == "gpu_ipc":
        return "gpu_ipc"
    raise ValueError(f"unsupported kv_transfer_mode={mode!r}; expected one of: proto_bytes, gpu_ipc")


@dataclass(frozen=True)
class GpuIpcOptions:
    enabled: bool = False
    prefix_sidecar_address: str = "127.0.0.1:55062"
    suffix_sidecar_address: str = "127.0.0.1:55061"

