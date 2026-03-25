from __future__ import annotations

import dataclasses
from typing import Any

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2

from ..transport.grpc_cache import PrefixClient
from .execution import make_suffix_execution_strategy
from .models import PrefixPipelineConfig
from .models import SuffixPipelineConfig
from .prefix_session import PrefixStreamSession
from .profile import PipelineProfiler
from .profile import PipelineProfilerConfig


class PrefixPipeline:
    def __init__(self, loaded_component: Any, config: PrefixPipelineConfig) -> None:
        if config.stream_queue_size <= 0:
            raise ValueError(f"stream_queue_size must be > 0, got {config.stream_queue_size}")
        if config.queue_wait_warn_ms < 0:
            raise ValueError(f"queue_wait_warn_ms must be >= 0, got {config.queue_wait_warn_ms}")
        if config.request_timeout_s < 0:
            raise ValueError(f"request_timeout_s must be >= 0, got {config.request_timeout_s}")
        self._loaded_component = loaded_component
        self._config = config
        self._profiler = PipelineProfiler(
            PipelineProfilerConfig(enabled=config.enable_profiling, log_path=config.profile_log_path)
        )

    async def stream_kv(self, request_id: str, raw_policy_input: dict[str, Any], context):
        session = PrefixStreamSession(
            loaded_component=self._loaded_component,
            config=self._config,
            request_id=request_id,
            raw_policy_input=raw_policy_input,
            context=context,
            profiler=self._profiler,
        )
        async for chunk in session.run():
            yield chunk


class SuffixPipeline:
    def __init__(self, prefix_client: PrefixClient, loaded_component: Any, config: SuffixPipelineConfig) -> None:
        if config.prefix_stream_timeout_s <= 0:
            raise ValueError(f"prefix_stream_timeout_s must be > 0, got {config.prefix_stream_timeout_s}")
        if config.warmup_diffusion_steps < 0:
            raise ValueError(f"warmup_diffusion_steps must be >= 0, got {config.warmup_diffusion_steps}")
        if config.max_inflight_updates <= 0:
            raise ValueError(f"max_inflight_updates must be > 0, got {config.max_inflight_updates}")
        if config.cache_ttl_ms < 0:
            raise ValueError(f"cache_ttl_ms must be >= 0, got {config.cache_ttl_ms}")
        if config.max_staleness_layers < 0:
            raise ValueError(f"max_staleness_layers must be >= 0, got {config.max_staleness_layers}")
        self._prefix_client = prefix_client
        self._loaded_component = loaded_component
        self._config = config
        self._profiler = PipelineProfiler(
            PipelineProfilerConfig(enabled=config.enable_profiling, log_path=config.profile_log_path)
        )

    async def evaluate(
        self,
        request: pb2.EvalRequest,
        raw_policy_input: dict[str, Any],
        policy_name: str,
        context,
        *,
        execution_mode: str | None = None,
        warmup_diffusion_steps: int | None = None,
        strict_layer_ordering: bool | None = None,
        max_inflight_updates: int | None = None,
        cache_ttl_ms: int | None = None,
        allow_stale_cache: bool | None = None,
        max_staleness_layers: int | None = None,
        drop_late_updates: bool | None = None,
    ):
        config = dataclasses.replace(
            self._config,
            execution_mode=(self._config.execution_mode if execution_mode is None else execution_mode),
            warmup_diffusion_steps=(
                self._config.warmup_diffusion_steps if warmup_diffusion_steps is None else warmup_diffusion_steps
            ),
            strict_layer_ordering=(
                self._config.strict_layer_ordering if strict_layer_ordering is None else strict_layer_ordering
            ),
            max_inflight_updates=(
                self._config.max_inflight_updates if max_inflight_updates is None else max_inflight_updates
            ),
            cache_ttl_ms=(self._config.cache_ttl_ms if cache_ttl_ms is None else cache_ttl_ms),
            allow_stale_cache=(
                self._config.allow_stale_cache if allow_stale_cache is None else allow_stale_cache
            ),
            max_staleness_layers=(
                self._config.max_staleness_layers if max_staleness_layers is None else max_staleness_layers
            ),
            drop_late_updates=(
                self._config.drop_late_updates if drop_late_updates is None else drop_late_updates
            ),
        )
        strategy = make_suffix_execution_strategy(
            mode=config.execution_mode,
            prefix_client=self._prefix_client,
            loaded_component=self._loaded_component,
            config=config,
            profiler=self._profiler,
        )
        return await strategy.run(
            request=request,
            raw_policy_input=raw_policy_input,
            policy_name=policy_name,
            context=context,
        )

    async def close(self) -> None:
        await self._prefix_client.close()
