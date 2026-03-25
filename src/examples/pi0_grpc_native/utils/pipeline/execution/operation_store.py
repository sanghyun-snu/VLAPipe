from __future__ import annotations

import asyncio
from dataclasses import dataclass

from examples.pi0_grpc_native.proto_gen import pi0_pipeline_pb2 as pb2


@dataclass
class OperationRecord:
    state: int
    result: pb2.EvalResponse | None = None
    error: str = ""
    task: asyncio.Task[None] | None = None
    events: asyncio.Queue[pb2.EvaluateEvent] | None = None


class OperationStore:
    def __init__(self) -> None:
        self._records: dict[str, OperationRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, operation_id: str, task: asyncio.Task[None]) -> None:
        async with self._lock:
            self._records[operation_id] = OperationRecord(
                state=pb2.EVAL_OP_STATE_QUEUED,
                task=task,
                events=asyncio.Queue(),
            )

    async def mark_running(self, operation_id: str) -> None:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is not None:
                record.state = pb2.EVAL_OP_STATE_RUNNING

    async def mark_succeeded(self, operation_id: str, result: pb2.EvalResponse) -> None:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is not None:
                record.state = pb2.EVAL_OP_STATE_SUCCEEDED
                record.result = result
                record.error = ""

    async def mark_failed(self, operation_id: str, error: str) -> None:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is not None:
                record.state = pb2.EVAL_OP_STATE_FAILED
                record.error = error

    async def mark_cancelled(self, operation_id: str, error: str = "operation cancelled") -> None:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is not None:
                record.state = pb2.EVAL_OP_STATE_CANCELLED
                record.error = error

    async def get(self, operation_id: str) -> OperationRecord | None:
        async with self._lock:
            return self._records.get(operation_id)

    async def cancel(self, operation_id: str) -> bool:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is None:
                return False
            if record.task is not None and not record.task.done():
                record.task.cancel()
                return True
            return False

    async def enqueue_event(self, operation_id: str, event: pb2.EvaluateEvent) -> None:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is None or record.events is None:
                return
            queue = record.events
        await queue.put(event)

    async def event_queue_depth(self, operation_id: str) -> int:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is None or record.events is None:
                return -1
            return record.events.qsize()

    async def get_events_queue(self, operation_id: str) -> asyncio.Queue[pb2.EvaluateEvent] | None:
        async with self._lock:
            record = self._records.get(operation_id)
            if record is None:
                return None
            return record.events

    async def active_tasks(self) -> list[asyncio.Task[None]]:
        async with self._lock:
            return [
                record.task
                for record in self._records.values()
                if record.task is not None and not record.task.done()
            ]
