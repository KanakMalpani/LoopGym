"""LES-1.0 heuristics for LoopNet export (mirrors loopnet.les — keep in sync)."""

from __future__ import annotations

from typing import Any

LES_CATEGORIES = (
    "effectiveness",
    "speed",
    "cost",
    "robustness",
    "scalability",
    "safety",
    "adaptability",
    "autonomy",
)

LES_WEIGHTS = {
    "effectiveness": 0.25,
    "speed": 0.15,
    "cost": 0.15,
    "robustness": 0.15,
    "scalability": 0.10,
    "safety": 0.10,
    "adaptability": 0.05,
    "autonomy": 0.05,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def composite_les(categories: dict[str, float]) -> float:
    return sum(LES_WEIGHTS[cat] * clamp(categories[cat]) for cat in LES_CATEGORIES)


def les_from_trajectory(
    trajectory: list[dict[str, Any]],
    *,
    goal_target: float,
    outcome: str,
    failure_mode: str | None = None,
    max_iterations_budget: int,
) -> dict[str, Any]:
    goal_final = trajectory[-1]["goal_score"]
    goal_0 = trajectory[0]["goal_score"]
    iteration_count = len(trajectory)
    cost_total = sum(step["cost_usd"] for step in trajectory)
    latencies = [step["latency_seconds"] for step in trajectory]
    median_latency = sorted(latencies)[len(latencies) // 2]
    regressions = sum(
        1
        for i in range(1, len(trajectory))
        if trajectory[i]["goal_score"] < trajectory[i - 1]["goal_score"]
    )
    safety_events = sum(step.get("safety_events", 0) for step in trajectory)
    human_steps = sum(1 for step in trajectory if step.get("human_intervention"))

    effectiveness = clamp(goal_final / max(goal_target, 0.01))
    if outcome == "failure" and goal_final < goal_target:
        effectiveness *= 0.6

    speed = clamp(1.0 / (1.0 + median_latency / 30.0))
    if any(lat > 3 * median_latency for lat in latencies):
        speed *= 0.85

    delta_g = goal_final - goal_0
    cost_eff = (delta_g / cost_total) if cost_total > 0 and delta_g > 0 else 0.0
    cost_score = clamp(min(cost_eff * 2.0, 1.0))

    robustness = clamp(0.75 - 0.05 * regressions)
    scalability = clamp(0.55 + 0.05 * min(iteration_count, 6))
    safety = (
        0.0
        if failure_mode == "fail.safety_bypass"
        else clamp(1.0 - min(safety_events * 0.2, 0.8))
    )
    adaptability = clamp(0.45 + 0.1 * (goal_final - goal_0))
    autonomy = clamp(1.0 - min(human_steps / max(iteration_count, 1), 0.6))

    categories = {
        "effectiveness": round(effectiveness, 4),
        "speed": round(speed, 4),
        "cost": round(cost_score, 4),
        "robustness": round(robustness, 4),
        "scalability": round(scalability, 4),
        "safety": round(safety, 4),
        "adaptability": round(adaptability, 4),
        "autonomy": round(autonomy, 4),
    }

    if iteration_count < 2 and outcome == "failure":
        categories = {
            k: round(v * 0.5, 4) if k != "safety" else v for k, v in categories.items()
        }

    les_normalized = round(composite_les(categories), 4)
    return {
        "les_normalized": les_normalized,
        "les_display": round(les_normalized * 100, 1),
        "categories": categories,
        "partial": iteration_count < max_iterations_budget and outcome != "success",
    }
