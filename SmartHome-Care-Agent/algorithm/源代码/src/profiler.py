"""推理性能监控模块 - 记录延迟与内存占用"""
from __future__ import annotations
import time
import psutil
import os
from dataclasses import dataclass, field


@dataclass
class InferenceRecord:
    query: str
    latency_ms: float
    memory_mb: float
    tool_calls: int
    success: bool


@dataclass
class Profiler:
    records: list[InferenceRecord] = field(default_factory=list)
    _process: psutil.Process = field(default_factory=lambda: psutil.Process(os.getpid()))

    def start(self) -> tuple[float, float]:
        mem = self._process.memory_info().rss / 1024 / 1024
        return time.perf_counter(), mem

    def end(self, t0: float, m0: float, query: str, tool_calls: int, success: bool):
        latency_ms = (time.perf_counter() - t0) * 1000
        mem_now = self._process.memory_info().rss / 1024 / 1024
        mem_delta = mem_now - m0
        rec = InferenceRecord(
            query=query[:40],
            latency_ms=latency_ms,
            memory_mb=mem_now,
            tool_calls=tool_calls,
            success=success,
        )
        self.records.append(rec)
        return latency_ms, mem_delta

    def report(self) -> dict:
        if not self.records:
            return {}
        latencies = [r.latency_ms for r in self.records]
        return {
            "total_queries": len(self.records),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "min_latency_ms": round(min(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "peak_memory_mb": round(max(r.memory_mb for r in self.records), 1),
            "success_rate": f"{sum(r.success for r in self.records)/len(self.records)*100:.0f}%",
            "total_tool_calls": sum(r.tool_calls for r in self.records),
        }
