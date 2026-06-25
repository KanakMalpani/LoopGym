"""Loop Trace 1.0 — episode trace documents for LoopGym runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_loop_trace(
    spec: dict[str, Any],
    *,
    loop_id: str,
    success: bool,
    termination_reason: str,
    history: list[dict[str, Any]],
    total_cost_usd: float = 0.0,
    started_at: str,
    ended_at: str | None = None,
    spec_path: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Loop Trace 1.0 document from episode history."""
    workers = spec.get("workers") or []
    default_worker = workers[0].get("id", workers[0].get("role", "worker")) if workers else "worker"
    ended = ended_at or utc_now()
    per_iter_cost = total_cost_usd / max(len(history), 1)

    iterations: list[dict[str, Any]] = []
    for rec in history:
        idx = int(rec.get("iteration", len(iterations) + 1))
        q = float(rec.get("quality_score", 0.0))
        entry: dict[str, Any] = {
            "iteration": max(0, idx - 1),
            "timestamp": started_at,
            "worker_id": rec.get("worker_id", default_worker),
            "evaluator_scores": {"quality": q},
            "cost_usd": round(per_iter_cost, 6),
        }
        if rec.get("output"):
            entry["worker_output"] = str(rec["output"])[:2000]
        if rec.get("feedback"):
            entry["feedback"] = [str(rec["feedback"])[:500]]
        iterations.append(entry)

    if not iterations:
        iterations.append(
            {
                "iteration": 0,
                "timestamp": started_at,
                "worker_id": default_worker,
                "evaluator_scores": {},
                "cost_usd": 0.0,
            }
        )

    meta = {"runtime": "loopgym/sim"}
    if metadata:
        meta.update(metadata)

    return {
        "trace_version": "1.0",
        "loop_id": loop_id,
        "loop_name": spec.get("loop_name", "unknown"),
        "spec_path": spec_path,
        "started_at": started_at,
        "ended_at": ended,
        "success": success,
        "termination_reason": termination_reason,
        "total_cost_usd": round(total_cost_usd, 6),
        "iterations": iterations,
        "metadata": meta,
    }


def write_loop_trace(trace: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    return out
