from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
import time
import uuid

import grpc

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2

from .background_context import BackgroundContext
from .operation_store import OperationStore


class V2AsyncOperationManager:
    """Manages async submit/poll/watch/cancel operation lifecycle for v2 RPCs."""

    def __init__(self, *, poll_interval_s: float = 0.2) -> None:
        self._poll_interval_s = poll_interval_s
        self._store = OperationStore()

    async def _emit_event(
        self,
        *,
        operation_id: str,
        state: int,
        event: str,
        layer_idx: int = -1,
        cache_epoch: int = 0,
        details: str = "",
    ) -> None:
        queue_depth = await self._store.event_queue_depth(operation_id)
        ts_ms = int(time.time() * 1000)
        meta = f"ts_ms={ts_ms} queued_events={queue_depth}"
        merged_details = meta if not details else f"{meta} | {details}"
        await self._store.enqueue_event(
            operation_id,
            pb2.EvaluateEvent(
                operation_id=operation_id,
                state=state,
                layer_idx=layer_idx,
                cache_epoch=cache_epoch,
                event=event,
                details=merged_details,
            ),
        )

    async def _run_operation(
        self,
        operation_id: str,
        request: pb2.EvaluatePipelineRequest,
        evaluate_pipeline: Callable[[pb2.EvaluatePipelineRequest, Any], Awaitable[pb2.EvalResponse]],
    ) -> None:
        await self._store.mark_running(operation_id)
        await self._emit_event(
            operation_id=operation_id,
            state=pb2.EVAL_OP_STATE_RUNNING,
            event="state_changed",
        )
        try:
            result = await evaluate_pipeline(
                request,
                BackgroundContext(
                    event_emitter=lambda event, layer_idx, cache_epoch, details: self._emit_event(
                        operation_id=operation_id,
                        state=pb2.EVAL_OP_STATE_RUNNING,
                        event=event,
                        layer_idx=layer_idx,
                        cache_epoch=cache_epoch,
                        details=details,
                    )
                ),
            )
            await self._store.mark_succeeded(operation_id, result)
            await self._emit_event(
                operation_id=operation_id,
                state=pb2.EVAL_OP_STATE_SUCCEEDED,
                event="state_changed",
            )
        except asyncio.CancelledError:
            await self._store.mark_cancelled(operation_id)
            await self._emit_event(
                operation_id=operation_id,
                state=pb2.EVAL_OP_STATE_CANCELLED,
                event="state_changed",
                details="operation cancelled",
            )
            raise
        except Exception as exc:  # noqa: BLE001
            await self._store.mark_failed(operation_id, str(exc))
            await self._emit_event(
                operation_id=operation_id,
                state=pb2.EVAL_OP_STATE_FAILED,
                event="state_changed",
                details=str(exc),
            )

    async def submit(
        self,
        *,
        request: pb2.SubmitEvaluateRequest,
        evaluate_pipeline: Callable[[pb2.EvaluatePipelineRequest, Any], Awaitable[pb2.EvalResponse]],
        context,
    ) -> pb2.SubmitEvaluateResponse:
        if request.wait_for_result:
            result = await evaluate_pipeline(request.request, context)
            return pb2.SubmitEvaluateResponse(
                operation_id="",
                state=pb2.EVAL_OP_STATE_SUCCEEDED,
                result=result,
                message="completed synchronously",
            )

        operation_id = uuid.uuid4().hex
        task = asyncio.create_task(
            self._run_operation(operation_id, request.request, evaluate_pipeline),
            name=f"suffix-op-{operation_id}",
        )
        await self._store.create(operation_id, task)
        await self._emit_event(
            operation_id=operation_id,
            state=pb2.EVAL_OP_STATE_QUEUED,
            event="state_changed",
        )
        return pb2.SubmitEvaluateResponse(
            operation_id=operation_id,
            state=pb2.EVAL_OP_STATE_QUEUED,
            message="operation queued",
        )

    async def get_result(self, *, request: pb2.GetEvaluateResultRequest, context) -> pb2.GetEvaluateResultResponse:
        record = await self._store.get(request.operation_id)
        if record is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"unknown operation_id={request.operation_id}")
        return pb2.GetEvaluateResultResponse(
            operation_id=request.operation_id,
            state=record.state,
            result=record.result if record.result is not None else pb2.EvalResponse(),
            error=record.error,
        )

    async def watch(self, *, request: pb2.WatchEvaluateRequest, context):
        queue = await self._store.get_events_queue(request.operation_id)
        if queue is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"unknown operation_id={request.operation_id}")
        while True:
            record = await self._store.get(request.operation_id)
            if record is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"unknown operation_id={request.operation_id}")
            try:
                event = await asyncio.wait_for(queue.get(), timeout=self._poll_interval_s)
                yield event
            except asyncio.TimeoutError:
                pass
            if record.state in (
                pb2.EVAL_OP_STATE_SUCCEEDED,
                pb2.EVAL_OP_STATE_FAILED,
                pb2.EVAL_OP_STATE_CANCELLED,
            ) and queue.empty():
                break

    async def cancel(self, *, request: pb2.CancelEvaluateRequest) -> pb2.CancelEvaluateResponse:
        cancelled = await self._store.cancel(request.operation_id)
        if cancelled:
            await self._emit_event(
                operation_id=request.operation_id,
                state=pb2.EVAL_OP_STATE_CANCELLED,
                event="cancel_requested",
                details="operation cancelled by client",
            )
        return pb2.CancelEvaluateResponse(cancelled=cancelled)

    async def close(self) -> None:
        tasks = await self._store.active_tasks()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
