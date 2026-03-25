from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import threading
import time


@dataclass(frozen=True)
class PipelineProfilerConfig:
    enabled: bool = False
    log_path: str = ""


class PipelineProfiler:
    def __init__(self, config: PipelineProfilerConfig) -> None:
        self._enabled = config.enabled
        self._log_path = config.log_path
        self._lock = threading.Lock()
        self._header_written = False
        if self._enabled:
            if not self._log_path:
                raise ValueError("profile_log_path is required when profiling is enabled")
            log_file = Path(self._log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._header_written = log_file.exists() and log_file.stat().st_size > 0

    def now(self) -> float:
        return time.perf_counter()

    def enabled(self) -> bool:
        return self._enabled

    def event(
        self,
        *,
        request_id: str,
        pipeline: str,
        event: str,
        value_s: float | None = None,
        layer_idx: int | None = None,
        details: str = "",
    ) -> None:
        if not self._enabled:
            return
        row = {
            "unix_ts": f"{time.time():.6f}",
            "request_id": request_id,
            "pipeline": pipeline,
            "event": event,
            "value_s": "" if value_s is None else f"{value_s:.9f}",
            "layer_idx": "" if layer_idx is None else str(layer_idx),
            "details": details,
        }
        self._append_row(row)

    def _append_row(self, row: dict[str, str]) -> None:
        with self._lock:
            with Path(self._log_path).open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["unix_ts", "request_id", "pipeline", "event", "value_s", "layer_idx", "details"],
                )
                if not self._header_written:
                    writer.writeheader()
                    self._header_written = True
                writer.writerow(row)
