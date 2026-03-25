from __future__ import annotations

from typing import Any
from typing import Protocol

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2


class SuffixExecutionStrategy(Protocol):
    async def run(
        self,
        *,
        request: pb2.EvalRequest,
        raw_policy_input: dict[str, Any],
        policy_name: str,
        context,
    ) -> pb2.EvalResponse:
        ...
