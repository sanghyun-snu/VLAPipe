from __future__ import annotations

from dataclasses import dataclass

from .models import PrefixPipelineConfig
from .models import SuffixPipelineConfig


@dataclass(frozen=True)
class PrefixServiceOptions:
    stream_queue_size: int = 2
    prefer_layerwise: bool = True
    allow_fallback: bool = True
    queue_wait_warn_ms: float = 10.0
    request_timeout_s: float = 0.0
    enable_profiling: bool = False
    profile_log_path: str = ""
    kv_transfer_mode: str = "gpu_ipc"
    gpu_ipc_prefix_sidecar_address: str = "127.0.0.1:55062"

    def to_pipeline_config(self) -> PrefixPipelineConfig:
        return PrefixPipelineConfig(
            stream_queue_size=self.stream_queue_size,
            prefer_layerwise=self.prefer_layerwise,
            allow_fallback=self.allow_fallback,
            queue_wait_warn_ms=self.queue_wait_warn_ms,
            request_timeout_s=self.request_timeout_s,
            enable_profiling=self.enable_profiling,
            profile_log_path=self.profile_log_path,
            kv_transfer_mode=self.kv_transfer_mode,
            gpu_ipc_prefix_sidecar_address=self.gpu_ipc_prefix_sidecar_address,
        )


@dataclass(frozen=True)
class SuffixServiceOptions:
    prefix_stream_timeout_s: float = 30.0
    strict_layer_ordering: bool = True
    execution_mode: str = "v1_layer_pipeline"
    warmup_diffusion_steps: int = 1
    max_inflight_updates: int = 2
    cache_ttl_ms: int = 0
    allow_stale_cache: bool = False
    max_staleness_layers: int = 0
    drop_late_updates: bool = False
    enable_profiling: bool = False
    profile_log_path: str = ""
    kv_transfer_mode: str = "gpu_ipc"
    gpu_ipc_suffix_sidecar_address: str = "127.0.0.1:55061"

    def to_pipeline_config(self) -> SuffixPipelineConfig:
        return SuffixPipelineConfig(
            prefix_stream_timeout_s=self.prefix_stream_timeout_s,
            strict_layer_ordering=self.strict_layer_ordering,
            execution_mode=self.execution_mode,
            warmup_diffusion_steps=self.warmup_diffusion_steps,
            max_inflight_updates=self.max_inflight_updates,
            cache_ttl_ms=self.cache_ttl_ms,
            allow_stale_cache=self.allow_stale_cache,
            max_staleness_layers=self.max_staleness_layers,
            drop_late_updates=self.drop_late_updates,
            enable_profiling=self.enable_profiling,
            profile_log_path=self.profile_log_path,
            kv_transfer_mode=self.kv_transfer_mode,
            gpu_ipc_suffix_sidecar_address=self.gpu_ipc_suffix_sidecar_address,
        )
