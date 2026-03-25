from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2


class BackgroundContext:
    def __init__(
        self,
        *,
        event_emitter: Callable[[str, int, int, str], Awaitable[None]] | None = None,
    ) -> None:
        self._event_emitter = event_emitter

    def cancelled(self) -> bool:
        return False

    async def abort(self, code, details: str):
        raise RuntimeError(f"background operation aborted code={code.name}: {details}")

    async def emit_event(
        self,
        *,
        event: str,
        layer_idx: int = -1,
        cache_epoch: int = 0,
        details: str = "",
    ) -> None:
        if self._event_emitter is None:
            return
        await self._event_emitter(event, layer_idx, cache_epoch, details)
