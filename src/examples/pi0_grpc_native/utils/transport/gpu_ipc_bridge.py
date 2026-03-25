from __future__ import annotations
# pyright: reportMissingImports=false

import base64
import contextlib
import ctypes
import ctypes.util
from dataclasses import dataclass
import socket
import time
from collections import OrderedDict
from typing import Literal

import grpc
import torch

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.utils.transport.sidecar_proto_gen import gpu_ipc_sidecar_pb2 as sidecar_pb2  # pyright: ignore[reportMissingImports]
from examples.pi0_grpc_native.utils.transport.sidecar_proto_gen import (
    gpu_ipc_sidecar_pb2_grpc as sidecar_pb2_grpc,  # pyright: ignore[reportMissingImports]
)
from .kv_transport import validate_gpu_ipc_resolve_mode


@dataclass(frozen=True)
class GpuIpcTensorHandle:
    handle_id: str
    shape: tuple[int, ...]
    dtype: str
    device_index: int
    nbytes: int

    def to_proto(self) -> pb2.GpuIpcHandle:
        return pb2.GpuIpcHandle(
            handle_id=self.handle_id,
            shape=list(self.shape),
            dtype=self.dtype,
            device_index=self.device_index,
            nbytes=self.nbytes,
        )


@dataclass(frozen=True)
class _OpenedDevicePointer:
    dev_ptr: int
    opened_at_s: float
    last_used_s: float


@dataclass(frozen=True)
class ResolveTiming:
    source: Literal["direct", "sidecar", "sidecar_fallback"]
    sidecar_lookup_s: float
    ipc_open_s: float
    d2d_copy_s: float


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _CudaRuntime:
    _CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
    _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1

    def __init__(self) -> None:
        self._lib = self._load_libcudart()
        self._bind()

    def _load_libcudart(self):
        candidates = []
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.extend(
            [
                "libcudart.so",
                "libcudart.so.12",
                "libcudart.so.11.0",
                "/usr/local/cuda/lib64/libcudart.so",
            ]
        )
        last_error: Exception | None = None
        for name in candidates:
            try:
                return ctypes.CDLL(name)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError("failed to load CUDA runtime library libcudart") from last_error

    def _bind(self) -> None:
        self._lib.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(_CudaIpcMemHandle), ctypes.c_void_p]
        self._lib.cudaIpcGetMemHandle.restype = ctypes.c_int
        self._lib.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _CudaIpcMemHandle, ctypes.c_uint]
        self._lib.cudaIpcOpenMemHandle.restype = ctypes.c_int
        self._lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self._lib.cudaIpcCloseMemHandle.restype = ctypes.c_int
        self._lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._lib.cudaMemcpy.restype = ctypes.c_int
        self._lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code: int, op: str) -> None:
        if code == 0:
            return
        err = self._lib.cudaGetErrorString(code)
        msg = err.decode("utf-8") if err else f"cuda error code {code}"
        raise RuntimeError(f"{op} failed: {msg}")

    def get_mem_handle_bytes(self, dev_ptr: int) -> bytes:
        handle = _CudaIpcMemHandle()
        code = self._lib.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(dev_ptr))
        self._check(code, "cudaIpcGetMemHandle")
        return ctypes.string_at(ctypes.addressof(handle), 64)

    def open_mem_handle(self, handle_bytes: bytes) -> int:
        if len(handle_bytes) != 64:
            raise ValueError(f"invalid CUDA IPC handle size={len(handle_bytes)} expected=64")
        handle = _CudaIpcMemHandle()
        ctypes.memmove(ctypes.addressof(handle), handle_bytes, 64)
        dev_ptr = ctypes.c_void_p()
        code = self._lib.cudaIpcOpenMemHandle(
            ctypes.byref(dev_ptr),
            handle,
            ctypes.c_uint(self._CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
        )
        self._check(code, "cudaIpcOpenMemHandle")
        return int(dev_ptr.value)

    def close_mem_handle(self, dev_ptr: int) -> None:
        code = self._lib.cudaIpcCloseMemHandle(ctypes.c_void_p(dev_ptr))
        self._check(code, "cudaIpcCloseMemHandle")

    def memcpy_d2d(self, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        code = self._lib.cudaMemcpy(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(src_ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(self._CUDA_MEMCPY_DEVICE_TO_DEVICE),
        )
        self._check(code, "cudaMemcpyDeviceToDevice")


_CUDA_RUNTIME: _CudaRuntime | None = None


def _cuda_runtime() -> _CudaRuntime:
    global _CUDA_RUNTIME  # noqa: PLW0603
    if _CUDA_RUNTIME is None:
        _CUDA_RUNTIME = _CudaRuntime()
    return _CUDA_RUNTIME


def _encode_handle_id(handle_bytes: bytes) -> str:
    return base64.urlsafe_b64encode(handle_bytes).decode("ascii")


def _decode_handle_id(handle_id: str) -> bytes:
    return base64.urlsafe_b64decode(handle_id.encode("ascii"))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "bool": torch.bool,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype in gpu_ipc handle: {name}")
    return mapping[name]


def _require_sidecar_reachable(sidecar_address: str) -> None:
    try:
        host, port_text = sidecar_address.rsplit(":", 1)
        port = int(port_text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid sidecar address: {sidecar_address!r}, expected host:port") from exc
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return
    except OSError as exc:
        raise RuntimeError(
            f"gpu_ipc requires running sidecar at {sidecar_address}; connection failed: {exc}"
        ) from exc


class PrefixGpuIpcPublisher:
    """Publish CUDA IPC handles from prefix tensors."""

    def __init__(self, sidecar_address: str) -> None:
        self._sidecar_address = sidecar_address
        _require_sidecar_reachable(self._sidecar_address)
        self._sidecar_channel = grpc.insecure_channel(self._sidecar_address)
        self._sidecar_stub = sidecar_pb2_grpc.GpuIpcSidecarStub(self._sidecar_channel)
        self._live_tensors: OrderedDict[str, tuple[torch.Tensor, float]] = OrderedDict()
        self._max_live_handles = 4096
        self._ttl_s = 30.0

    def publish_tensor(self, request_id: str, layer_idx: int, name: str, tensor: torch.Tensor) -> GpuIpcTensorHandle:
        if not tensor.is_cuda:
            raise RuntimeError(
                f"gpu_ipc requires CUDA tensor request_id={request_id} layer_idx={layer_idx} name={name} "
                f"got device={tensor.device}"
            )
        # CUDA IPC handles are most robust when exported from a base contiguous allocation.
        # For safety, materialize a fresh contiguous tensor when storage offset is non-zero.
        if tensor.is_contiguous() and tensor.storage_offset() == 0:
            contiguous = tensor
        else:
            contiguous = tensor.contiguous().clone()

        device_index = contiguous.device.index if contiguous.device.index is not None else 0
        # Correctness-first barrier: ensure producer-side writes are visible before exporting handle.
        torch.cuda.synchronize(device_index)
        handle_bytes = _cuda_runtime().get_mem_handle_bytes(contiguous.data_ptr())
        handle_id = _encode_handle_id(handle_bytes)
        self._live_tensors[handle_id] = (contiguous, time.monotonic())
        self._prune_live_tensors()
        result = GpuIpcTensorHandle(
            handle_id=handle_id,
            shape=tuple(int(v) for v in contiguous.shape),
            dtype=_dtype_name(contiguous.dtype),
            device_index=int(device_index),
            nbytes=int(contiguous.numel() * contiguous.element_size()),
        )
        self._publish_to_sidecar(request_id, layer_idx, name, result)
        return result

    def _prune_live_tensors(self) -> None:
        now = time.monotonic()
        while self._live_tensors:
            oldest_key = next(iter(self._live_tensors))
            _tensor, ts = self._live_tensors[oldest_key]
            if len(self._live_tensors) > self._max_live_handles or (now - ts) > self._ttl_s:
                self._live_tensors.pop(oldest_key)
                continue
            break

    def _publish_to_sidecar(self, request_id: str, layer_idx: int, name: str, handle: GpuIpcTensorHandle) -> None:
        request = sidecar_pb2.PublishHandleRequest(
            request_id=request_id,
            layer_idx=int(layer_idx),
            tensor_name=name,
            handle=sidecar_pb2.GpuIpcHandle(
                handle_id=handle.handle_id,
                shape=list(handle.shape),
                dtype=handle.dtype,
                device_index=handle.device_index,
                nbytes=handle.nbytes,
            ),
        )
        response = self._sidecar_stub.PublishHandle(request, timeout=0.5)
        if not response.ok:
            raise RuntimeError(
                f"sidecar PublishHandle failed request_id={request_id} layer_idx={layer_idx} "
                f"name={name}: {response.message}"
            )


class SuffixGpuIpcResolver:
    """Resolve CUDA IPC handles to local CUDA tensors."""

    def __init__(self, sidecar_address: str, *, resolve_mode: str = "direct") -> None:
        self._sidecar_address = sidecar_address
        _require_sidecar_reachable(self._sidecar_address)
        self._resolve_mode = validate_gpu_ipc_resolve_mode(resolve_mode)
        self._sidecar_channel = grpc.insecure_channel(self._sidecar_address)
        self._sidecar_stub = sidecar_pb2_grpc.GpuIpcSidecarStub(self._sidecar_channel)
        self._opened_cache: OrderedDict[str, _OpenedDevicePointer] = OrderedDict()
        self._max_opened_handles = 4096
        self._opened_ttl_s = 60.0

    def resolve_tensor(self, request_id: str, layer_idx: int, name: str, handle: pb2.GpuIpcHandle) -> torch.Tensor:
        tensor, _timing = self.resolve_tensor_timed(request_id, layer_idx, name, handle)
        return tensor

    def resolve_tensor_timed(
        self,
        request_id: str,
        layer_idx: int,
        name: str,
        handle: pb2.GpuIpcHandle,
    ) -> tuple[torch.Tensor, ResolveTiming]:
        sidecar_lookup_s = 0.0
        source: Literal["direct", "sidecar", "sidecar_fallback"] = "direct"
        effective_handle = handle
        if self._resolve_mode != "direct":
            lookup_start_t = time.perf_counter()
            resolved = self._resolve_handle_from_sidecar(request_id, layer_idx, name)
            sidecar_lookup_s = time.perf_counter() - lookup_start_t
            if self._resolve_mode == "sidecar_only":
                if resolved is None:
                    raise RuntimeError(
                        f"sidecar-only resolve failed request_id={request_id} layer_idx={layer_idx} name={name}"
                    )
                effective_handle = resolved
                source = "sidecar"
            elif resolved is not None:
                effective_handle = resolved
                source = "sidecar"
            else:
                source = "sidecar_fallback"
        if not effective_handle.handle_id:
            raise ValueError(
                f"gpu_ipc handle_id is required request_id={request_id} layer_idx={layer_idx} name={name}"
            )
        device_index = int(effective_handle.device_index)
        dtype = _dtype_from_name(effective_handle.dtype)
        shape = tuple(int(v) for v in effective_handle.shape)
        if not shape:
            raise ValueError(f"invalid empty shape in gpu_ipc handle request_id={request_id} layer_idx={layer_idx}")
        torch.cuda.set_device(device_index)
        out = torch.empty(shape, dtype=dtype, device=torch.device("cuda", device_index))
        src_ptr, ipc_open_s = self._get_or_open_device_ptr(effective_handle.handle_id)
        copy_start_t = time.perf_counter()
        _cuda_runtime().memcpy_d2d(out.data_ptr(), src_ptr, int(effective_handle.nbytes))
        d2d_copy_s = time.perf_counter() - copy_start_t
        self._prune_opened_cache()
        return out, ResolveTiming(
            source=source,
            sidecar_lookup_s=sidecar_lookup_s,
            ipc_open_s=ipc_open_s,
            d2d_copy_s=d2d_copy_s,
        )

    def _resolve_handle_from_sidecar(
        self, request_id: str, layer_idx: int, name: str
    ) -> sidecar_pb2.GpuIpcHandle | None:
        try:
            response = self._sidecar_stub.ResolveHandle(
                sidecar_pb2.ResolveHandleRequest(
                    request_id=request_id,
                    layer_idx=int(layer_idx),
                    tensor_name=name,
                ),
                timeout=0.5,
            )
        except grpc.RpcError:
            return None
        if not response.found:
            return None
        return response.handle

    def _get_or_open_device_ptr(self, handle_id: str) -> tuple[int, float]:
        now = time.monotonic()
        cached = self._opened_cache.get(handle_id)
        if cached is not None:
            self._opened_cache[handle_id] = _OpenedDevicePointer(
                dev_ptr=cached.dev_ptr,
                opened_at_s=cached.opened_at_s,
                last_used_s=now,
            )
            self._opened_cache.move_to_end(handle_id, last=True)
            return cached.dev_ptr, 0.0
        open_start_t = time.perf_counter()
        dev_ptr = _cuda_runtime().open_mem_handle(_decode_handle_id(handle_id))
        ipc_open_s = time.perf_counter() - open_start_t
        self._opened_cache[handle_id] = _OpenedDevicePointer(
            dev_ptr=dev_ptr,
            opened_at_s=now,
            last_used_s=now,
        )
        return dev_ptr, ipc_open_s

    def _prune_opened_cache(self) -> None:
        now = time.monotonic()
        while self._opened_cache and len(self._opened_cache) > self._max_opened_handles:
            _oldest_key, oldest = self._opened_cache.popitem(last=False)
            _cuda_runtime().close_mem_handle(oldest.dev_ptr)
        while self._opened_cache:
            oldest_key = next(iter(self._opened_cache))
            oldest = self._opened_cache[oldest_key]
            if (now - oldest.last_used_s) <= self._opened_ttl_s:
                break
            _cuda_runtime().close_mem_handle(oldest.dev_ptr)
            self._opened_cache.pop(oldest_key, None)

    def close(self) -> None:
        while self._opened_cache:
            key = next(iter(self._opened_cache))
            opened = self._opened_cache.pop(key)
            _cuda_runtime().close_mem_handle(opened.dev_ptr)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

