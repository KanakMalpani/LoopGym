"""Build LoopNet ln/record-v1 records from LoopGym SimEnv episodes."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loopgym.export.les import les_from_trajectory
from loopgym.runtime.loop_runtime import _get_max_iterations, _get_quality_threshold

PATTERN_SLUGS = {
    "reflection-loop",
    "critique-loop",
    "planning-loop",
    "verification-loop",
    "research-loop",
    "simulation-loop",
    "debate-loop",
    "exploration-loop",
    "optimization-loop",
    "memory-augmented-loop",
    "human-in-the-loop",
    "safety-constrained-loop",
    "multi-agent-coordination",
    "recursive-improvement-loop",
}

ENV_PATTERN_HINTS: dict[str, list[str]] = {
    "loopbench/code-repair-v1": ["verification-loop", "reflection-loop"],
    "loopbench/research-synthesis-v1": ["research-loop", "critique-loop"],
    "loopbench/multi-agent-debate-v1": ["debate-loop", "multi-agent-coordination"],
}


def _record_id() -> str:
    return f"ln-{uuid.uuid4()}"


def _normalize_loop_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if len(slug) < 3:
        slug = f"loop-{slug or 'run'}"
    return slug[:64]


def _map_termination(raw: str, *, success: bool) -> str:
    text = raw.lower()
    if success or "quality_threshold" in text or "goal_met" in text:
        return "goal_met"
    if "max_iterations" in text or "max iterations" in text:
        return "max_iterations"
    if "cost" in text:
        return "cost_exceeded"
    if "safety" in text:
        return "safety_violation"
    if "timeout" in text:
        return "timeout"
    if "error" in text:
        return "error"
    if "stall" in text:
        return "stall"
    return "budget_exhausted"


def _infer_outcome(*, success: bool, goal_final: float, goal_target: float) -> str:
    if success:
        return "success"
    if goal_final >= goal_target * 0.85:
        return "partial"
    return "failure"


def _infer_failure_mode(
    *,
    outcome: str,
    termination_reason: str,
    trajectory: list[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    if outcome == "success":
        return None, []

    regressions = sum(
        1
        for i in range(1, len(trajectory))
        if trajectory[i]["goal_score"] < trajectory[i - 1]["goal_score"]
    )
    goal_final = trajectory[-1]["goal_score"]
    goal_0 = trajectory[0]["goal_score"]

    if "max_iterations" in termination_reason:
        primary = "fail.tau_omission"
    elif regressions >= 2:
        primary = "fail.oscillation"
    elif goal_final > goal_0 + 0.15 and outcome == "failure":
        primary = "fail.false_pass"
    elif goal_final < goal_0:
        primary = "fail.evaluator_drift"
    elif outcome == "partial":
        primary = "fail.open_loop"
    else:
        primary = "fail.self_grade"

    modes = [primary]
    if outcome == "partial" and primary not in modes:
        modes.append("fail.open_loop")
    return primary, modes


def _patterns_from_spec(spec: dict[str, Any], env_id: str) -> list[str]:
    extensions = spec.get("extensions") or {}
    raw = extensions.get("patterns") or []
    patterns = [p for p in raw if p in PATTERN_SLUGS]
    if patterns:
        return patterns

    hinted = ENV_PATTERN_HINTS.get(env_id, [])
    if hinted:
        return hinted

    loop_name = str(spec.get("loop_name", "reflection-loop"))
    for slug in PATTERN_SLUGS:
        if slug.replace("-loop", "") in loop_name:
            return [slug]
    return ["reflection-loop"]


def _episode_trajectory(
    episode: dict[str, Any],
    *,
    elapsed_seconds: float,
    tokens_used: int,
) -> list[dict[str, Any]]:
    steps = episode.get("trajectory") or []
    if not steps:
        return []

    per_step_latency = round(max(elapsed_seconds / len(steps), 0.05), 3)
    per_step_cost = round(max(0.01, tokens_used * 0.00002 / len(steps)), 4)
    trajectory: list[dict[str, Any]] = []

    for step in steps:
        goal_score = round(float(step.get("quality_score", 0.0)), 4)
        trajectory.append(
            {
                "iteration": int(step["iteration"]),
                "goal_score": goal_score,
                "primary_quality": goal_score,
                "cost_usd": per_step_cost,
                "latency_seconds": per_step_latency,
                "tokens": max(50, tokens_used // len(steps)),
                "failure_codes": [],
                "safety_events": 0,
                "human_intervention": False,
            }
        )
    return trajectory


def episode_to_record(
    episode: dict[str, Any],
    *,
    spec: dict[str, Any],
    env_id: str,
    source: str = "case_study",
    split: str = "train",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a SimEnv run_episode() result into a LoopNet record."""
    goal_target = _get_quality_threshold(spec)
    max_iterations = _get_max_iterations(spec)
    elapsed = float(episode.get("elapsed_seconds") or 1.0)
    tokens_used = int(episode.get("tokens_used") or 500)

    trajectory = _episode_trajectory(episode, elapsed_seconds=elapsed, tokens_used=tokens_used)
    if not trajectory:
        raise ValueError("episode has empty trajectory")

    success = bool(episode.get("success"))
    goal_final = trajectory[-1]["goal_score"]
    termination_raw = str(episode.get("termination_reason") or "")
    termination_reason = _map_termination(termination_raw, success=success)
    outcome = _infer_outcome(success=success, goal_final=goal_final, goal_target=goal_target)
    failure_mode, failure_modes = _infer_failure_mode(
        outcome=outcome,
        termination_reason=termination_raw,
        trajectory=trajectory,
    )

    patterns = _patterns_from_spec(spec, env_id)
    loop_name = _normalize_loop_name(str(spec.get("loop_name") or patterns[0].replace("-loop", "")))
    objective = str(spec.get("objective") or "Loop objective not specified in spec.")

    les_observed = les_from_trajectory(
        trajectory,
        goal_target=goal_target,
        outcome=outcome,
        failure_mode=failure_mode,
        max_iterations_budget=max_iterations,
    )

    regressions = sum(
        1
        for i in range(1, len(trajectory))
        if trajectory[i]["goal_score"] < trajectory[i - 1]["goal_score"]
    )

    record: dict[str, Any] = {
        "record_id": _record_id(),
        "schema_version": "ln/record-v1",
        "spec_pins": {"lss": "lss@1.0.0", "les": "les@1.0.0"},
        "created_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "source": source,
        "split": split,
        "patterns": patterns,
        "loop_name": loop_name,
        "objective": objective,
        "loop_spec": {
            "loop_name": spec.get("loop_name"),
            "version": spec.get("version", "1.0.0"),
            "workers": spec.get("workers") or [],
            "evaluators": spec.get("evaluators") or [],
            "termination_conditions": spec.get("termination_conditions"),
            "extensions": {"env_id": env_id, "task_id": episode.get("task_id")},
        },
        "outcome": outcome,
        "termination_reason": termination_reason,
        "trajectory": trajectory,
        "les_observed": les_observed,
        "metadata": {
            "iteration_count": len(trajectory),
            "cost_total_usd": round(sum(s["cost_usd"] for s in trajectory), 4),
            "goal_target": goal_target,
            "goal_final": goal_final,
            "worker_count": max(1, len(spec.get("workers") or [])),
            "evaluator_count": max(1, len(spec.get("evaluators") or [])),
            "max_iterations_budget": max_iterations,
            "regression_count": regressions,
            "tags": tags or ["captured", "loopgym", env_id.split("/")[-1]],
        },
        "redaction": {"level": "none", "fields_removed": []},
    }

    if failure_modes:
        record["failure_modes"] = failure_modes
    if failure_mode:
        record["failure_mode"] = failure_mode

    return record


def capture_env_episodes(
    env_id: str,
    *,
    task_ids: list[str],
    seeds: list[int],
    spec_path: str | Path | None = None,
    source: str = "case_study",
    split: str = "train",
    failure_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run SimEnv episodes and return LoopNet records."""
    import copy

    import loopgym as lg

    failure_seeds = set(failure_seeds or [])
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        for seed in seeds:
            if seed in failure_seeds:
                base = lg.make(env_id, spec_path=spec_path, seed=seed)
                spec = copy.deepcopy(getattr(base, "spec", {}))
                spec["termination_conditions"] = {
                    "success": [{"type": "quality_threshold", "threshold": 0.99}],
                    "failure": [{"type": "max_iterations", "value": 1}],
                }
                env = lg.make(env_id, spec=spec, seed=seed)
            else:
                env = lg.make(env_id, spec_path=spec_path, seed=seed)
            if not hasattr(env, "run_episode"):
                raise TypeError(f"{env_id} does not support run_episode()")
            episode = env.run_episode(task_id=task_id, seed=seed)
            spec = getattr(env, "spec", {})
            records.append(
                episode_to_record(
                    episode,
                    spec=spec,
                    env_id=env_id,
                    source=source,
                    split=split,
                )
            )
    return records


def append_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
