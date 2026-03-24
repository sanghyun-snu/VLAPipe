from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass

import torch

PROTO_MAGIC = b"KV01"
HEADER_LEN_STRUCT = struct.Struct("!I")
PREFIX_REQ_MAGIC = b"PRQ1"
EVAL_REQ_MAGIC = b"ERQ1"
EVAL_RES_MAGIC = b"ERS1"


@dataclass(frozen=True)
class KVCachePayload:
    """Layer KV cache payload exchanged over gRPC."""

    request_id: str
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class PrefixRequest:
    request_id: str
    policy_name: str
    state: torch.Tensor
    num_layers: int
    hidden_size: int
    prefix_tokens: int
    suffix_tokens: int
    compute_delay_s: float
    seed: int


@dataclass(frozen=True)
class EvalRequest:
    request_id: str
    policy_name: str
    state: torch.Tensor
    num_layers: int
    hidden_size: int
    prefix_tokens: int
    suffix_tokens: int
    compute_delay_s: float
    poll_interval_s: float
    seed: int


@dataclass(frozen=True)
class EvalResponse:
    request_id: str
    actions: torch.Tensor
    message: str


def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", copy=True)


def serialize_kv_payload(payload: KVCachePayload) -> bytes:
    """Serialize KV payload as [magic][header_len][json_header][torch_blob]."""
    key = _to_cpu_tensor(payload.key)
    value = _to_cpu_tensor(payload.value)
    header = {
        "request_id": payload.request_id,
        "layer_idx": payload.layer_idx,
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "key_dtype": str(key.dtype),
        "value_dtype": str(value.dtype),
    }
    header_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob_buffer = io.BytesIO()
    torch.save({"key": key, "value": value}, blob_buffer)
    return PROTO_MAGIC + HEADER_LEN_STRUCT.pack(len(header_raw)) + header_raw + blob_buffer.getvalue()


def deserialize_kv_payload(raw: bytes, device: str | torch.device = "cpu") -> KVCachePayload:
    """Deserialize wire payload into KV cache tensors."""
    if len(raw) < len(PROTO_MAGIC) + HEADER_LEN_STRUCT.size:
        raise ValueError("Invalid payload: message too short")
    if raw[: len(PROTO_MAGIC)] != PROTO_MAGIC:
        raise ValueError("Invalid payload: magic mismatch")

    cursor = len(PROTO_MAGIC)
    (header_len,) = HEADER_LEN_STRUCT.unpack(raw[cursor : cursor + HEADER_LEN_STRUCT.size])
    cursor += HEADER_LEN_STRUCT.size
    header_raw = raw[cursor : cursor + header_len]
    cursor += header_len
    header = json.loads(header_raw.decode("utf-8"))

    blob_buffer = io.BytesIO(raw[cursor:])
    kv = torch.load(blob_buffer, map_location=device)
    return KVCachePayload(
        request_id=str(header["request_id"]),
        layer_idx=int(header["layer_idx"]),
        key=kv["key"],
        value=kv["value"],
    )


def _serialize_object(magic: bytes, payload: dict) -> bytes:
    blob_buffer = io.BytesIO()
    torch.save(payload, blob_buffer)
    return magic + blob_buffer.getvalue()


def _deserialize_object(magic: bytes, raw: bytes, device: str | torch.device = "cpu") -> dict:
    if not raw.startswith(magic):
        raise ValueError("Invalid payload magic")
    blob_buffer = io.BytesIO(raw[len(magic) :])
    return torch.load(blob_buffer, map_location=device)


def serialize_prefix_request(request: PrefixRequest) -> bytes:
    return _serialize_object(
        PREFIX_REQ_MAGIC,
        {
            "request_id": request.request_id,
            "policy_name": request.policy_name,
            "state": _to_cpu_tensor(request.state),
            "num_layers": request.num_layers,
            "hidden_size": request.hidden_size,
            "prefix_tokens": request.prefix_tokens,
            "suffix_tokens": request.suffix_tokens,
            "compute_delay_s": request.compute_delay_s,
            "seed": request.seed,
        },
    )


def deserialize_prefix_request(raw: bytes, device: str | torch.device = "cpu") -> PrefixRequest:
    payload = _deserialize_object(PREFIX_REQ_MAGIC, raw, device=device)
    return PrefixRequest(
        request_id=payload["request_id"],
        policy_name=payload["policy_name"],
        state=payload["state"],
        num_layers=int(payload["num_layers"]),
        hidden_size=int(payload["hidden_size"]),
        prefix_tokens=int(payload["prefix_tokens"]),
        suffix_tokens=int(payload["suffix_tokens"]),
        compute_delay_s=float(payload["compute_delay_s"]),
        seed=int(payload["seed"]),
    )


def serialize_eval_request(request: EvalRequest) -> bytes:
    return _serialize_object(
        EVAL_REQ_MAGIC,
        {
            "request_id": request.request_id,
            "policy_name": request.policy_name,
            "state": _to_cpu_tensor(request.state),
            "num_layers": request.num_layers,
            "hidden_size": request.hidden_size,
            "prefix_tokens": request.prefix_tokens,
            "suffix_tokens": request.suffix_tokens,
            "compute_delay_s": request.compute_delay_s,
            "poll_interval_s": request.poll_interval_s,
            "seed": request.seed,
        },
    )


def deserialize_eval_request(raw: bytes, device: str | torch.device = "cpu") -> EvalRequest:
    payload = _deserialize_object(EVAL_REQ_MAGIC, raw, device=device)
    return EvalRequest(
        request_id=payload["request_id"],
        policy_name=payload["policy_name"],
        state=payload["state"],
        num_layers=int(payload["num_layers"]),
        hidden_size=int(payload["hidden_size"]),
        prefix_tokens=int(payload["prefix_tokens"]),
        suffix_tokens=int(payload["suffix_tokens"]),
        compute_delay_s=float(payload["compute_delay_s"]),
        poll_interval_s=float(payload["poll_interval_s"]),
        seed=int(payload["seed"]),
    )


def serialize_eval_response(response: EvalResponse) -> bytes:
    return _serialize_object(
        EVAL_RES_MAGIC,
        {
            "request_id": response.request_id,
            "actions": _to_cpu_tensor(response.actions),
            "message": response.message,
        },
    )


def deserialize_eval_response(raw: bytes, device: str | torch.device = "cpu") -> EvalResponse:
    payload = _deserialize_object(EVAL_RES_MAGIC, raw, device=device)
    return EvalResponse(
        request_id=payload["request_id"],
        actions=payload["actions"],
        message=payload["message"],
    )
