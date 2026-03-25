from __future__ import annotations

from dataclasses import dataclass

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2


@dataclass(frozen=True)
class ExecutionOverrides:
    execution_mode: str
    warmup_diffusion_steps: int
    strict_layer_ordering: bool
    max_inflight_updates: int
    cache_ttl_ms: int
    allow_stale_cache: bool
    max_staleness_layers: int
    drop_late_updates: bool


def resolve_execution_overrides(
    execution: pb2.ExecutionConfig,
    *,
    default_execution_mode: str,
    default_warmup_diffusion_steps: int,
    default_strict_layer_ordering: bool,
    default_max_inflight_updates: int,
    default_cache_ttl_ms: int,
    default_allow_stale_cache: bool,
    default_max_staleness_layers: int,
    default_drop_late_updates: bool,
) -> ExecutionOverrides:
    execution_mode = default_execution_mode
    warmup_steps = default_warmup_diffusion_steps
    strict_layer_ordering = default_strict_layer_ordering
    max_inflight_updates = default_max_inflight_updates
    cache_ttl_ms = default_cache_ttl_ms
    allow_stale_cache = default_allow_stale_cache
    max_staleness_layers = default_max_staleness_layers
    drop_late_updates = default_drop_late_updates

    if execution.mode == pb2.SUFFIX_EXECUTION_MODE_LAYER_PIPELINE_V1:
        execution_mode = "v1_layer_pipeline"
    elif execution.mode == pb2.SUFFIX_EXECUTION_MODE_ASYNC_CACHE_V2:
        execution_mode = "v2_async_cache"

    if execution.HasField("v1"):
        warmup_steps = int(execution.v1.warmup_diffusion_steps)
        strict_layer_ordering = bool(execution.v1.strict_layer_ordering)
    if execution.HasField("v2"):
        max_inflight_updates = int(execution.v2.max_inflight_updates or max_inflight_updates)
        cache_ttl_ms = int(execution.v2.cache_ttl_ms)
        allow_stale_cache = bool(execution.v2.allow_stale_cache)
        max_staleness_layers = int(execution.v2.max_staleness_layers)
        drop_late_updates = bool(execution.v2.drop_late_updates)

    return ExecutionOverrides(
        execution_mode=execution_mode,
        warmup_diffusion_steps=warmup_steps,
        strict_layer_ordering=strict_layer_ordering,
        max_inflight_updates=max_inflight_updates,
        cache_ttl_ms=cache_ttl_ms,
        allow_stale_cache=allow_stale_cache,
        max_staleness_layers=max_staleness_layers,
        drop_late_updates=drop_late_updates,
    )
