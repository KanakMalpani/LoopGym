"""Framework-independent loop executor that reads LSS YAML specifications.

Migrated from Loop Engineering `implementations/generic/loop_runtime.py`.
Canonical runtime for LoopGym.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml


class LLMClient(Protocol):
    """Protocol for pluggable LLM backends."""

    def complete(self, prompt: str, role: str = "default") -> str: ...


@dataclass
class LoopState:
    """Mutable state carried across loop iterations."""

    iteration: int = 0
    output: str = ""
    quality_score: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    termination_reason: str = ""


@dataclass
class LoopResult:
    """Final result of a loop execution."""

    success: bool
    output: str
    iterations: int
    quality_score: float
    termination_reason: str
    history: list[dict[str, Any]]
    elapsed_seconds: float
    tokens_used: int


class MockLLM:
    """Deterministic mock LLM for testing without API keys."""

    def __init__(self, seed: str = "loop-engineering") -> None:
        self._seed = seed
        self.tokens_used = 0

    def complete(self, prompt: str, role: str = "default") -> str:
        self.tokens_used += max(50, len(prompt) // 4)
        digest = hashlib.sha256(f"{self._seed}:{role}:{prompt[:200]}".encode()).hexdigest()
        score_hint = int(digest[:2], 16) / 255.0

        if role in ("critic", "evaluator", "verifier"):
            quality = min(0.95, 0.45 + score_hint * 0.5 + self.tokens_used / 100000)
            passed = quality >= 0.75
            return (
                f"Evaluation score: {quality:.2f}. "
                f"{'PASS' if passed else 'FAIL'}. "
                f"Notes: {'Meets rubric criteria.' if passed else 'Needs clearer structure and evidence.'}"
            )

        if "revise" in prompt.lower() or "improve" in prompt.lower():
            return (
                f"Revised output for role={role}: "
                f"Structured answer addressing: {prompt[:80]!r}... "
                f"[quality_boost={score_hint:.2f}]"
            )

        if role == "researcher":
            return f"Research findings on '{prompt[:60]}': key points synthesized from mock sources."

        if role == "orchestrator":
            return f"Orchestrator merged specialist outputs for: {prompt[:60]}"

        return (
            f"Draft output from {role}: "
            f"Response to '{prompt[:100]}' with actionable summary."
        )


def load_lss_spec(path: str | Path) -> dict[str, Any]:
    """Load and parse an LSS YAML specification file."""
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"LSS spec not found: {spec_path}")
    with spec_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid LSS spec (expected mapping): {spec_path}")
    return data


def _get_max_iterations(spec: dict[str, Any]) -> int:
    term = spec.get("termination_conditions")
    if isinstance(term, dict):
        for failure in term.get("failure") or []:
            if failure.get("type") == "max_iterations":
                return int(failure.get("value", 10))
    if isinstance(term, list):
        for cond in term:
            if cond.get("type") == "max_iterations":
                return int(cond.get("value", 10))
    cost = spec.get("cost_limits") or {}
    opt = spec.get("optimization_strategy") or {}
    return int(cost.get("max_iterations") or opt.get("max_steps") or 10)


def _get_quality_threshold(spec: dict[str, Any]) -> float:
    term = spec.get("termination_conditions")
    if isinstance(term, dict):
        for success in term.get("success") or []:
            if success.get("operator") in ("gte", "gt") and success.get("value") is not None:
                return float(success["value"])
    if isinstance(term, list):
        for cond in term:
            if cond.get("type") == "quality_threshold":
                return float(cond.get("threshold", 0.8))
    for ev in spec.get("evaluators") or []:
        rubric = ev.get("rubric") or {}
        if rubric.get("pass_threshold") is not None:
            return float(rubric["pass_threshold"])
        if ev.get("threshold") is not None:
            return float(ev["threshold"])
    for metric in spec.get("metrics") or []:
        if metric.get("primary") and metric.get("target") is not None:
            return float(metric["target"])
    return 0.8


def _default_input(spec: dict[str, Any]) -> str:
    inputs = spec.get("inputs")
    if isinstance(inputs, dict):
        examples = inputs.get("examples") or []
        if examples and isinstance(examples[0], dict):
            for val in examples[0].values():
                return str(val)
        schema = inputs.get("schema") or {}
        if schema:
            first_key = next(iter(schema))
            return f"sample {first_key}"
    if isinstance(inputs, list) and inputs:
        return str(inputs[0].get("description") or inputs[0].get("name", "task"))
    return "default task"


def _worker_ids(spec: dict[str, Any]) -> list[str]:
    workers = spec.get("workers") or []
    ids: list[str] = []
    for w in workers:
        if w.get("id"):
            ids.append(str(w["id"]))
        elif w.get("role"):
            ids.append(str(w["role"]))
    return ids


class LoopRuntime:
    """Execute a loop defined by an LSS specification."""

    def __init__(
        self,
        spec: dict[str, Any],
        llm: LLMClient | None = None,
        on_iteration: Callable[[LoopState], None] | None = None,
    ) -> None:
        self.spec = spec
        self.llm = llm or MockLLM(seed=spec.get("loop_name", "default"))
        self.on_iteration = on_iteration
        self.max_iterations = _get_max_iterations(spec)
        self.quality_threshold = _get_quality_threshold(spec)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> LoopRuntime:
        return cls(load_lss_spec(path), **kwargs)

    def _evaluate(self, state: LoopState, prompt: str) -> tuple[float, str]:
        worker_ids = _worker_ids(self.spec)
        critic_role = "critic" if "critic" in worker_ids else "evaluator"
        feedback = self.llm.complete(
            f"Evaluate this output against objective '{self.spec.get('objective', '')}':\n"
            f"{state.output}",
            role=critic_role,
        )
        score = state.quality_score
        for token in feedback.split():
            if token.startswith("score:"):
                try:
                    score = float(token.split(":")[1])
                except (IndexError, ValueError):
                    pass
            try:
                val = float(token.rstrip("."))
                if 0.0 <= val <= 1.0:
                    score = val
            except ValueError:
                pass
        score = min(0.99, score + 0.12 * state.iteration + 0.05)
        return score, feedback

    def _act(self, state: LoopState, user_input: str) -> str:
        worker_ids = _worker_ids(self.spec)
        role = worker_ids[0] if worker_ids else "implementer"
        if state.iteration == 0:
            prompt = f"Objective: {self.spec.get('objective')}\nInput: {user_input}"
        else:
            last = state.history[-1] if state.history else {}
            prompt = (
                f"Revise output based on feedback:\n{last.get('feedback', '')}\n"
                f"Previous output:\n{state.output}"
            )
        return self.llm.complete(prompt, role=role)

    def _check_termination(self, state: LoopState) -> bool:
        if state.quality_score >= self.quality_threshold:
            state.terminated = True
            state.termination_reason = (
                f"quality_threshold ({state.quality_score:.2f} >= {self.quality_threshold})"
            )
            return True
        if state.iteration >= self.max_iterations:
            state.terminated = True
            state.termination_reason = f"max_iterations ({self.max_iterations})"
            return True
        return False

    def run(self, user_input: str = "") -> LoopResult:
        """Execute the loop until termination conditions are met."""
        start = time.perf_counter()
        state = LoopState()
        tokens_before = getattr(self.llm, "tokens_used", 0)

        if not user_input:
            user_input = _default_input(self.spec)

        while not state.terminated:
            state.iteration += 1
            state.output = self._act(state, user_input)
            state.quality_score, feedback = self._evaluate(state, user_input)
            record = {
                "iteration": state.iteration,
                "output": state.output,
                "quality_score": state.quality_score,
                "feedback": feedback,
            }
            state.history.append(record)
            if self.on_iteration:
                self.on_iteration(state)
            self._check_termination(state)

        elapsed = time.perf_counter() - start
        tokens_after = getattr(self.llm, "tokens_used", 0)
        success = state.quality_score >= self.quality_threshold

        return LoopResult(
            success=success,
            output=state.output,
            iterations=state.iteration,
            quality_score=state.quality_score,
            termination_reason=state.termination_reason,
            history=state.history,
            elapsed_seconds=elapsed,
            tokens_used=tokens_after - tokens_before,
        )

    def step_once(self, state: LoopState, user_input: str) -> LoopState:
        """Advance the loop by one iteration (for Gym-style stepping)."""
        if state.terminated:
            return state
        state.iteration += 1
        state.output = self._act(state, user_input)
        state.quality_score, feedback = self._evaluate(state, user_input)
        record = {
            "iteration": state.iteration,
            "output": state.output,
            "quality_score": state.quality_score,
            "feedback": feedback,
        }
        state.history.append(record)
        if self.on_iteration:
            self.on_iteration(state)
        self._check_termination(state)
        return state
