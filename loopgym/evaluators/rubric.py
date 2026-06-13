"""LLM rubric evaluator helpers."""

from __future__ import annotations

from typing import Any, Protocol


class RubricLLM(Protocol):
    def complete(self, prompt: str, role: str = "default") -> str: ...


def run_rubric(
    llm: RubricLLM,
    output: str,
    objective: str,
    rubric: dict[str, Any],
    role: str = "evaluator",
) -> dict[str, Any]:
    """Score output against an LSS rubric using an LLM backend."""
    threshold = float(rubric.get("pass_threshold", 0.8))
    dimensions = rubric.get("dimensions") or []
    dim_names = ", ".join(d.get("name", "quality") for d in dimensions) or "quality"

    feedback = llm.complete(
        f"Evaluate ({dim_names}) against objective '{objective}':\n{output}",
        role=role,
    )

    score = 0.0
    for token in feedback.split():
        try:
            val = float(token.rstrip("."))
            if 0.0 <= val <= 1.0:
                score = val
        except ValueError:
            pass

    passed = score >= threshold
    return {
        "passed": passed,
        "score": score,
        "feedback": feedback,
        "failure_codes": [] if passed else ["fail.self_grade"],
        "dimension_scores": {d.get("name", "quality"): score for d in dimensions},
    }
