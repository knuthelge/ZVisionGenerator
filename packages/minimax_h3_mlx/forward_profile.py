"""Opt-in forward-pass profiling helpers for MiniMax-H3 MLX generation.

The default generation path does not import or activate a profiler.  Callers that
need component timing install a :class:`ForwardPassProfiler` with
:func:`active_profiler`; instrumented call sites then add explicit MLX
synchronization boundaries so recorded times are real wall-clock observations
rather than lazy graph construction costs.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import statistics
import time
from typing import Any, Callable, Iterator

import mlx.core as mx

_ACTIVE_PROFILER: ContextVar["ForwardPassProfiler | None"] = ContextVar(
    "minimax_h3_forward_profiler",
    default=None,
)

# These totals intentionally overlap detailed leaf timings and are excluded from
# share denominators while still being reported for denoise/forward timing.
OVERLAPPING_CATEGORIES = {"denoise_step_total", "dit_forward_total"}


def _now_monotonic() -> float:
    return time.perf_counter()


def _eval_mlx_tree(value: Any) -> None:
    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, (tuple, list)):
        arrays = [item for item in value if isinstance(item, mx.array)]
        if arrays:
            mx.eval(*arrays)
    elif isinstance(value, dict):
        arrays = [item for item in value.values() if isinstance(item, mx.array)]
        if arrays:
            mx.eval(*arrays)


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (tuple, list)):
            out[key] = [str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for v in value]
        else:
            out[key] = str(value)
    return out


@dataclass
class ForwardPassProfiler:
    """Collect wall-clock timings for an explicitly profiled generation run."""

    synchronize: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at_monotonic: float = field(default_factory=_now_monotonic)

    def call(
        self,
        label: str,
        category: str,
        fn: Callable[[], Any],
        *,
        eval_output: bool = True,
        synchronize: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Run ``fn`` and record elapsed seconds under ``label``/``category``."""

        do_sync = self.synchronize if synchronize is None else bool(synchronize)
        if do_sync:
            mx.synchronize()
        started = _now_monotonic()
        ok = False
        try:
            out = fn()
            if eval_output:
                _eval_mlx_tree(out)
            if do_sync:
                mx.synchronize()
            ok = True
            return out
        finally:
            elapsed = _now_monotonic() - started
            self.events.append(
                {
                    "label": str(label),
                    "category": str(category),
                    "seconds": float(elapsed),
                    "ok": bool(ok),
                    "metadata": _safe_metadata(metadata),
                }
            )

    @contextmanager
    def block(
        self,
        label: str,
        category: str,
        *,
        synchronize: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record a non-returning block, useful for media muxing and file I/O."""

        do_sync = self.synchronize if synchronize is None else bool(synchronize)
        if do_sync:
            mx.synchronize()
        started = _now_monotonic()
        ok = False
        try:
            yield
            if do_sync:
                mx.synchronize()
            ok = True
        finally:
            elapsed = _now_monotonic() - started
            self.events.append(
                {
                    "label": str(label),
                    "category": str(category),
                    "seconds": float(elapsed),
                    "ok": bool(ok),
                    "metadata": _safe_metadata(metadata),
                }
            )

    def record_phase(self, label: str, category: str, seconds: float, *, metadata: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "label": str(label),
                "category": str(category),
                "seconds": float(seconds),
                "ok": True,
                "metadata": _safe_metadata(metadata),
            }
        )

    def summary(self) -> dict[str, Any]:
        by_category: dict[str, dict[str, Any]] = {}
        by_label: dict[str, dict[str, Any]] = {}
        for event in self.events:
            seconds = float(event.get("seconds") or 0.0)
            for table, key in ((by_category, str(event["category"])), (by_label, str(event["label"]))):
                row = table.setdefault(key, {"count": 0, "total_seconds": 0.0, "samples": []})
                row["count"] += 1
                row["total_seconds"] += seconds
                row["samples"].append(seconds)

        def finalize(table: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for key, row in sorted(table.items()):
                samples = [float(v) for v in row.pop("samples")]
                total = float(row["total_seconds"])
                out[key] = {
                    "count": int(row["count"]),
                    "total_seconds": total,
                    "mean_seconds": total / max(int(row["count"]), 1),
                    "median_seconds": float(statistics.median(samples)) if samples else None,
                    "min_seconds": float(min(samples)) if samples else None,
                    "max_seconds": float(max(samples)) if samples else None,
                }
            return out

        category_summary = finalize(by_category)
        label_summary = finalize(by_label)
        share_denominator = sum(
            row["total_seconds"]
            for category, row in category_summary.items()
            if category not in OVERLAPPING_CATEGORIES and math.isfinite(row["total_seconds"])
        )
        shares = {
            category: (row["total_seconds"] / share_denominator if share_denominator else None)
            for category, row in category_summary.items()
            if category not in OVERLAPPING_CATEGORIES
        }
        return {
            "event_count": len(self.events),
            "elapsed_since_profiler_start_seconds": _now_monotonic() - self.started_at_monotonic,
            "category_totals": category_summary,
            "label_totals": label_summary,
            "non_overlapping_share_denominator_seconds": share_denominator,
            "category_shares_non_overlapping": shares,
            "overlapping_categories_excluded_from_shares": sorted(OVERLAPPING_CATEGORIES),
            "synchronization_policy": "mx.synchronize before/after profiled calls; MLX array outputs are mx.eval'd before stop",
        }

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "profiler": "minimax_h3_forward_pass_opt_in",
            "summary": self.summary(),
        }
        if include_events:
            payload["events"] = self.events
        return payload


@contextmanager
def active_profiler(profiler: ForwardPassProfiler | None) -> Iterator[ForwardPassProfiler | None]:
    """Install ``profiler`` for the current context if profiling was requested."""

    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield profiler
    finally:
        _ACTIVE_PROFILER.reset(token)


def current_profiler() -> ForwardPassProfiler | None:
    return _ACTIVE_PROFILER.get()


def profiled_call(
    label: str,
    category: str,
    fn: Callable[[], Any],
    *,
    eval_output: bool = True,
    synchronize: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    profiler = current_profiler()
    if profiler is None:
        return fn()
    return profiler.call(
        label,
        category,
        fn,
        eval_output=eval_output,
        synchronize=synchronize,
        metadata=metadata,
    )


@contextmanager
def profiled_block(
    label: str,
    category: str,
    *,
    synchronize: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    profiler = current_profiler()
    if profiler is None:
        yield
        return
    with profiler.block(label, category, synchronize=synchronize, metadata=metadata):
        yield
