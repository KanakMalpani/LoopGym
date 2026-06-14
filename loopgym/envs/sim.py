"""SimEnv — mock LLM + mock oracles (no API keys)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loopgym.envs.base import LoopEnv, Observation
from loopgym.runtime.compiler import compile_lss
from loopgym.runtime.loop_runtime import LoopRuntime, LoopState, MockLLM, load_lss_spec


def _task_input(spec: dict[str, Any], task_id: str, tasks_path: Path | None) -> str:
    """Resolve task input from tasks.json or spec examples."""
    if tasks_path and tasks_path.exists():
        with tasks_path.open(encoding="utf-8") as fh:
            tasks = json.load(fh)
        for task in tasks.get("tasks", []):
            if task.get("id") == task_id:
                payload = task.get("input") or task
                return json.dumps(payload) if isinstance(payload, dict) else str(payload)
    inputs = spec.get("inputs") or {}
    examples = inputs.get("examples") or []
    if examples:
        first = examples[0]
        if isinstance(first, dict):
            return json.dumps(first)
        return str(first)
    return f"task:{task_id or 'default'}"


def _mock_llm_seed(env_seed: int, task_id: str, loop_name: str) -> str:
    """Deterministic seed for reproducible trajectories."""
    return f"{env_seed}:{task_id}:{loop_name}"


class SimEnv(LoopEnv):
    """Simulation environment with MockLLM — no API keys required."""

    def __init__(
        self,
        env_id: str,
        spec: dict[str, Any],
        spec_path: Path | None = None,
        tasks_path: Path | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(env_id)
        self.spec = spec
        self.spec_path = spec_path
        self.tasks_path = tasks_path
        self.seed = seed
        self._graph = compile_lss(spec)
        self._runtime: LoopRuntime | None = None
        self._state: LoopState | None = None
        self._user_input = ""

    @classmethod
    def from_spec_file(
        cls,
        env_id: str,
        spec_path: str | Path,
        tasks_path: str | Path | None = None,
        seed: int = 0,
    ) -> SimEnv:
        path = Path(spec_path)
        tasks = Path(tasks_path) if tasks_path else path.parent / "tasks.json"
        return cls(
            env_id=env_id,
            spec=load_lss_spec(path),
            spec_path=path,
            tasks_path=tasks if tasks.exists() else None,
            seed=seed,
        )

    def reset(self, task_id: str = "", seed: int | None = None, **kwargs: Any) -> Observation:
        if seed is not None:
            self.seed = seed
        self._task_id = task_id or "default"
        self._user_input = _task_input(self.spec, self._task_id, self.tasks_path)
        llm_seed = _mock_llm_seed(self.seed, self._task_id, self._graph.loop_name)
        llm = MockLLM(seed=llm_seed)
        self._runtime = LoopRuntime(self.spec, llm=llm)
        self._state = LoopState()
        self._done = False

        self._obs = Observation(
            task_id=self._task_id,
            iteration=0,
            output="",
            quality_score=0.0,
            objective=self._graph.objective,
            done=False,
            info={
                "env_id": self.env_id,
                "seed": self.seed,
                "loop_name": self._graph.loop_name,
                "user_input": self._user_input,
            },
        )
        return self._obs

    def step(self, action: Any = None) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._runtime is None or self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            obs = self._obs or Observation(
                task_id=self._task_id,
                iteration=0,
                output="",
                quality_score=0.0,
                objective=self._graph.objective,
                done=True,
            )
            return obs, 0.0, True, {"reason": "already_done"}

        if isinstance(action, dict) and action.get("output"):
            self._state.output = str(action["output"])
            self._state.quality_score, feedback = self._runtime._evaluate(self._state, self._user_input)
            self._state.iteration += 1
            self._state.history.append(
                {
                    "iteration": self._state.iteration,
                    "output": self._state.output,
                    "quality_score": self._state.quality_score,
                    "feedback": feedback,
                    "agent_override": True,
                }
            )
            self._runtime._check_termination(self._state)
        else:
            self._state = self._runtime.step_once(self._state, self._user_input)

        self._done = self._state.terminated
        success = self._state.quality_score >= self._runtime.quality_threshold
        reward = self._reward(self._state.quality_score, success)

        info = {
            "iteration": self._state.iteration,
            "termination_reason": self._state.termination_reason,
            "success": success,
            "history_len": len(self._state.history),
        }
        if self._state.history:
            info["last_feedback"] = self._state.history[-1].get("feedback", "")

        self._obs = Observation(
            task_id=self._task_id,
            iteration=self._state.iteration,
            output=self._state.output,
            quality_score=self._state.quality_score,
            objective=self._graph.objective,
            done=self._done,
            info=info,
        )
        return self._obs, reward, self._done, info

    def run_episode(self, task_id: str = "", seed: int | None = None) -> dict[str, Any]:
        """Run full episode until done (convenience for benchmarks)."""
        import time

        self.reset(task_id=task_id, seed=seed)
        total_reward = 0.0
        steps = 0
        start = time.perf_counter()
        while not self.done:
            _, reward, _, info = self.step()
            total_reward += reward
            steps += 1
        elapsed = time.perf_counter() - start
        tokens_used = getattr(self._runtime.llm, "tokens_used", 0) if self._runtime else 0
        return {
            "task_id": self._task_id,
            "seed": self.seed,
            "env_id": self.env_id,
            "steps": steps,
            "total_reward": total_reward,
            "success": info.get("success", False),
            "quality_score": self._obs.quality_score if self._obs else 0.0,
            "termination_reason": (
                self._state.termination_reason if self._state else info.get("termination_reason", "")
            ),
            "elapsed_seconds": round(elapsed, 3),
            "tokens_used": tokens_used,
            "trajectory": [
                {
                    "iteration": h["iteration"],
                    "output": h["output"],
                    "quality_score": h["quality_score"],
                }
                for h in (self._state.history if self._state else [])
            ],
        }
