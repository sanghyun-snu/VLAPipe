from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixPipelineConfig:
    stream_queue_size: int = 2
    prefer_layerwise: bool = True
    allow_fallback: bool = True
    queue_wait_warn_ms: float = 10.0
    request_timeout_s: float = 0.0


@dataclass
class PrefixStreamState:
    produced_layers: int = 0
    emitted_layers: int = 0
    queue_wait_s: float = 0.0
    first_emit_s: float | None = None
    last_emit_s: float | None = None
    producer_error: Exception | None = None


@dataclass(frozen=True)
class SuffixPipelineConfig:
    prefix_stream_timeout_s: float = 30.0
    strict_layer_ordering: bool = True


@dataclass
class SuffixEvalState:
    received_layers: int = 0
    receive_s: float = 0.0
    finalize_s: float = 0.0
    denoise_s: float = 0.0
    total_s: float = 0.0
