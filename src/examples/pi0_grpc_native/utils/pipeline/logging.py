from __future__ import annotations

from .models import PrefixPipelineConfig
from .models import PrefixStreamState
from .models import SuffixEvalState
from .models import SuffixPipelineConfig


def log_prefix_start(request_id: str, config: PrefixPipelineConfig) -> None:
    print(
        f"[prefix] start request={request_id} queue={config.stream_queue_size} "
        f"prefer_layerwise={config.prefer_layerwise} allow_fallback={config.allow_fallback}"
    )


def log_prefix_backpressure(request_id: str, layer_idx: int, queue_wait_ms: float) -> None:
    print(f"[prefix] backpressure request={request_id} layer={layer_idx} queue_wait_ms={queue_wait_ms:.3f}")


def log_prefix_emit(request_id: str, layer_idx: int) -> None:
    print(f"[prefix] emit kv request={request_id} layer={layer_idx}")


def log_prefix_end(request_id: str, state: PrefixStreamState, total_s: float) -> None:
    first_emit_s = state.first_emit_s if state.first_emit_s is not None else -1.0
    last_emit_s = state.last_emit_s if state.last_emit_s is not None else -1.0
    print(
        f"[prefix] end request={request_id} produced_layers={state.produced_layers} emitted_layers={state.emitted_layers} "
        f"t_first_emit_s={first_emit_s:.6f} t_last_emit_s={last_emit_s:.6f} "
        f"queue_wait_total_ms={state.queue_wait_s * 1000.0:.3f} total_s={total_s:.6f}"
    )


def log_suffix_start(request_id: str, expected_layers: int, config: SuffixPipelineConfig) -> None:
    print(
        f"[suffix] start request={request_id} expected_layers={expected_layers} "
        f"strict_layer_ordering={config.strict_layer_ordering} timeout_s={config.prefix_stream_timeout_s}"
    )


def log_suffix_receive(request_id: str, layer_idx: int) -> None:
    print(f"[suffix] received kv request={request_id} layer={layer_idx}")


def log_suffix_received_done(request_id: str, state: SuffixEvalState) -> None:
    print(
        f"[suffix] received_done request={request_id} layers={state.received_layers} "
        f"receive_s={state.receive_s:.6f} finalize_s={state.finalize_s:.6f}"
    )


def log_suffix_end(request_id: str, state: SuffixEvalState) -> None:
    print(f"[suffix] end request={request_id} denoise_s={state.denoise_s:.6f} total_s={state.total_s:.6f}")


def log_suffix_error(details: str) -> None:
    print(f"[suffix] error {details}")
