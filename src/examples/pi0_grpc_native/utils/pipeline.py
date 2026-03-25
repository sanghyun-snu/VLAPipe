from __future__ import annotations

from typing import Any

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2
from examples.pi0_grpc_native.utils.pipeline_models import PrefixPipelineConfig
from examples.pi0_grpc_native.utils.pipeline_models import SuffixPipelineConfig
from examples.pi0_grpc_native.utils.pipeline_prefix import PrefixStreamSession
from examples.pi0_grpc_native.utils.pipeline_suffix import SuffixEvalSession

from .grpc_cache import PrefixClient


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

    async def stream_kv(self, request_id: str, raw_policy_input: dict[str, Any], context):
        session = PrefixStreamSession(
            loaded_component=self._loaded_component,
            config=self._config,
            request_id=request_id,
            raw_policy_input=raw_policy_input,
            context=context,
        )
        async for chunk in session.run():
            yield chunk


class SuffixPipeline:
    def __init__(self, prefix_client: PrefixClient, loaded_component: Any, config: SuffixPipelineConfig) -> None:
        if config.prefix_stream_timeout_s <= 0:
            raise ValueError(f"prefix_stream_timeout_s must be > 0, got {config.prefix_stream_timeout_s}")
        self._prefix_client = prefix_client
        self._loaded_component = loaded_component
        self._config = config

    async def evaluate(self, request: pb2.EvalRequest, raw_policy_input: dict[str, Any], policy_name: str, context):
        session = SuffixEvalSession(
            prefix_client=self._prefix_client,
            loaded_component=self._loaded_component,
            config=self._config,
            request=request,
            raw_policy_input=raw_policy_input,
            policy_name=policy_name,
            context=context,
        )
        return await session.run()

    async def close(self) -> None:
        await self._prefix_client.close()
