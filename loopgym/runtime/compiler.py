"""LSS compiler — parse specs into a runtime graph (v0.1 subset)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loopgym.runtime.loop_runtime import load_lss_spec


@dataclass
class WorkerNode:
    """Compiled worker node."""

    id: str
    role: str
    depends_on: list[str] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorNode:
    """Compiled evaluator node."""

    id: str
    type: str
    runs_after: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeGraph:
    """Compiled runtime graph from an LSS spec."""

    loop_name: str
    version: str
    objective: str
    workers: list[WorkerNode]
    evaluators: list[EvaluatorNode]
    max_iterations: int
    quality_threshold: float
    spec: dict[str, Any]

    @property
    def worker_ids(self) -> list[str]:
        return [w.id for w in self.workers]


def compile_lss(spec: dict[str, Any]) -> RuntimeGraph:
    """Compile an LSS dict into a RuntimeGraph."""
    workers = [
        WorkerNode(
            id=str(w["id"]),
            role=str(w.get("role", w["id"])),
            depends_on=list(w.get("depends_on") or []),
            model=dict(w.get("model") or {}),
        )
        for w in spec.get("workers") or []
    ]
    evaluators = [
        EvaluatorNode(
            id=str(e["id"]),
            type=str(e.get("type", "llm_rubric")),
            runs_after=[str(x) for x in (e.get("runs_after") or [])],
            config={k: v for k, v in e.items() if k not in ("id", "type", "runs_after")},
        )
        for e in spec.get("evaluators") or []
    ]

    term = spec.get("termination_conditions") or {}
    max_iter = 10
    for failure in term.get("failure") or []:
        if failure.get("type") == "max_iterations":
            max_iter = int(failure.get("value", 10))
            break
    opt = spec.get("optimization_strategy") or {}
    max_iter = int(opt.get("max_steps") or max_iter)

    quality = 0.8
    for success in term.get("success") or []:
        if success.get("operator") in ("gte", "gt") and success.get("value") is not None:
            quality = float(success["value"])
            break

    return RuntimeGraph(
        loop_name=str(spec.get("loop_name", "unnamed")),
        version=str(spec.get("version", "0.0.0")),
        objective=str(spec.get("objective", "")),
        workers=workers,
        evaluators=evaluators,
        max_iterations=max_iter,
        quality_threshold=quality,
        spec=spec,
    )


def compile_lss_file(path: str | Path) -> RuntimeGraph:
    """Load and compile an LSS YAML file."""
    return compile_lss(load_lss_spec(path))
