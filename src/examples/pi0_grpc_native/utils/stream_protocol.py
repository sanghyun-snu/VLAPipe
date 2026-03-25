from __future__ import annotations

import numpy as np
import torch

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2

POLICY_TYPE_NAME_TO_ENUM = {
    "droid": pb2.POLICY_TYPE_DROID,
    "aloha": pb2.POLICY_TYPE_ALOHA,
    "libero": pb2.POLICY_TYPE_LIBERO,
}
POLICY_TYPE_ENUM_TO_NAME = {value: key for key, value in POLICY_TYPE_NAME_TO_ENUM.items()}
BFloat16_DTYPE_NAME = "bfloat16"


def ndarray_to_proto(array: np.ndarray) -> pb2.NdArray:
    arr = np.ascontiguousarray(array)
    return pb2.NdArray(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def proto_to_ndarray(array_msg: pb2.NdArray) -> np.ndarray:
    if array_msg.dtype == BFloat16_DTYPE_NAME:
        shape = tuple(array_msg.shape)
        raw = np.frombuffer(array_msg.data, dtype=np.uint16)
        if not shape:
            return raw.copy()
        return raw.reshape(shape).copy()
    dtype = np.dtype(array_msg.dtype)
    shape = tuple(array_msg.shape)
    out = np.frombuffer(array_msg.data, dtype=dtype)
    if not shape:
        return np.array(out[0], dtype=dtype)
    return out.reshape(shape).copy()


def tensor_to_proto(array: torch.Tensor) -> pb2.NdArray:
    tensor = array.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        raw = tensor.view(torch.uint16).numpy()
        return pb2.NdArray(data=np.ascontiguousarray(raw).tobytes(), shape=list(tensor.shape), dtype=BFloat16_DTYPE_NAME)
    return ndarray_to_proto(tensor.numpy())


def proto_to_tensor(array_msg: pb2.NdArray, device: str = "cpu") -> torch.Tensor:
    if array_msg.dtype == BFloat16_DTYPE_NAME:
        raw = proto_to_ndarray(array_msg)
        return torch.from_numpy(raw).view(torch.bfloat16).to(device=device)
    return torch.from_numpy(proto_to_ndarray(array_msg)).to(device=device)
