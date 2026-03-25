from __future__ import annotations

from typing import Any

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2

from ..models import SuffixPipelineConfig
from ..profile import PipelineProfiler
from ..suffix_session import SuffixEvalSession


class V1LayerPipelineStrategy:
    """Baseline-compatible suffix execution using layer-ordered prefix stream."""

    def __init__(
        self,
        *,
        prefix_client,
        loaded_component: Any,
        config: SuffixPipelineConfig,
        profiler: PipelineProfiler,
    ) -> None:
        self._prefix_client = prefix_client
        self._loaded_component = loaded_component
        self._config = config
        self._profiler = profiler

    async def run(
        self,
        *,
        request: pb2.EvalRequest,
        raw_policy_input: dict[str, Any],
        policy_name: str,
        context,
    ) -> pb2.EvalResponse:
        session = SuffixEvalSession(
            prefix_client=self._prefix_client,
            loaded_component=self._loaded_component,
            config=self._config,
            request=request,
            raw_policy_input=raw_policy_input,
            policy_name=policy_name,
            context=context,
            profiler=self._profiler,
        )
        return await session.run()
