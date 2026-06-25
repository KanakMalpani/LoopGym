"""ComposedSimEnv — parallel branch simulation for LB-COMP-1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loopgym.envs.base import LoopEnv, Observation
from loopgym.envs.sim import SimEnv, _task_input
from loopgym.runtime.loop_runtime import load_lss_spec


class ComposedSimEnv(LoopEnv):
    """Run parallel branch SimEnvs, merge outputs, then orchestrator SimEnv."""

    def __init__(
        self,
        env_id: str,
        orchestrator_spec: dict[str, Any],
        orchestrator_spec_path: Path,
        tasks_path: Path | None,
        branches: list[tuple[str, Path, Path | None]],
        seed: int = 0,
    ) -> None:
        super().__init__(env_id)
        self.orchestrator_spec = orchestrator_spec
        self.orchestrator_spec_path = orchestrator_spec_path
        self.tasks_path = tasks_path
        self.branches = branches
        self.seed = seed
        self._orch: SimEnv | None = None
        self._branch_episodes: list[dict[str, Any]] = []
        self._obs: Observation | None = None
        self._done = False
        self._task_id = ""

    def _run_branches(self, task_id: str) -> tuple[str, list[dict[str, Any]]]:
        episodes: list[dict[str, Any]] = []
        dissent: list[str] = []
        base_task = _task_input(self.orchestrator_spec, task_id, self.tasks_path)

        for branch_id, spec_path, branch_tasks in self.branches:
            branch_spec = load_lss_spec(spec_path)
            branch_seed = self.seed + (hash(branch_id) % 997)
            branch_env = SimEnv(
                f"{self.env_id}/{branch_id}",
                branch_spec,
                spec_path=spec_path,
                tasks_path=branch_tasks if branch_tasks and branch_tasks.exists() else None,
                seed=branch_seed,
            )
            episode = branch_env.run_episode(task_id=task_id, seed=branch_seed)
            episodes.append({"branch_id": branch_id, **episode})
            if episode.get("quality_score", 0.0) < 0.75:
                dissent.append(branch_id)

        branch_payload = []
        for ep in episodes:
            traj = ep.get("trajectory") or []
            last_out = traj[-1]["output"] if traj else ""
            branch_payload.append(
                {
                    "id": ep["branch_id"],
                    "quality_score": ep.get("quality_score", 0.0),
                    "output": last_out[:500],
                }
            )

        merged = {
            "task": base_task,
            "branches": branch_payload,
            "dissent": dissent,
            "preserve_dissent": True,
        }
        return json.dumps(merged), episodes

    def reset(self, task_id: str = "", seed: int | None = None, **kwargs: Any) -> Observation:
        if seed is not None:
            self.seed = seed
        self._task_id = task_id or "comp-001"
        merged_input, self._branch_episodes = self._run_branches(self._task_id)

        self._orch = SimEnv(
            self.env_id,
            self.orchestrator_spec,
            spec_path=self.orchestrator_spec_path,
            tasks_path=self.tasks_path,
            seed=self.seed,
        )
        obs = self._orch.reset(task_id=self._task_id, seed=self.seed)
        self._orch._user_input = merged_input
        self._done = False
        self._obs = Observation(
            task_id=obs.task_id,
            iteration=obs.iteration,
            output=obs.output,
            quality_score=obs.quality_score,
            objective=obs.objective,
            done=obs.done,
            info={
                **obs.info,
                "branches_run": len(self._branch_episodes),
                "dissent": json.loads(merged_input).get("dissent", []),
            },
        )
        return self._obs

    def step(self, action: Any = None) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._orch is None:
            raise RuntimeError("Call reset() before step()")
        obs, reward, done, info = self._orch.step(action)
        self._done = done
        info = {**info, "branches_run": len(self._branch_episodes)}
        self._obs = Observation(
            task_id=obs.task_id,
            iteration=obs.iteration,
            output=obs.output,
            quality_score=obs.quality_score,
            objective=obs.objective,
            done=obs.done,
            info=info,
        )
        return self._obs, reward, done, info

    @property
    def done(self) -> bool:
        return self._done or (self._orch.done if self._orch else False)

    def run_episode(
        self,
        task_id: str = "",
        seed: int | None = None,
        *,
        trace_path: str | Path | None = None,
    ) -> dict[str, Any]:
        import time

        from loopgym.trace import build_loop_trace, utc_now, write_loop_trace

        started_at = utc_now()
        self.reset(task_id=task_id, seed=seed)
        assert self._orch is not None
        total_reward = 0.0
        steps = 0
        start = time.perf_counter()
        last_info: dict[str, Any] = {}
        while not self.done:
            _, reward, _, info = self.step()
            total_reward += reward
            steps += 1
            last_info = info
        elapsed = time.perf_counter() - start

        branch_scores = [b.get("quality_score", 0.0) for b in self._branch_episodes]
        orch_score = self._obs.quality_score if self._obs else 0.0
        composite = min(0.99, (sum(branch_scores) / max(len(branch_scores), 1)) * 0.55 + orch_score * 0.45)
        branch_tokens = sum(b.get("tokens_used", 0) for b in self._branch_episodes)
        orch_tokens = getattr(self._orch._runtime.llm, "tokens_used", 0) if self._orch._runtime else 0
        success = last_info.get("success", False) or composite >= 0.80
        termination = (
            self._orch._state.termination_reason if self._orch._state else last_info.get("termination_reason", "")
        )
        total_cost = (branch_tokens + orch_tokens) * 0.000002
        loop_id = f"{self.env_id}:{self._task_id}:{self.seed}"

        result = {
            "task_id": self._task_id,
            "seed": self.seed,
            "env_id": self.env_id,
            "steps": steps + sum(b.get("steps", 0) for b in self._branch_episodes),
            "total_reward": total_reward,
            "success": success,
            "quality_score": round(composite, 4),
            "orchestrator_score": orch_score,
            "branch_scores": branch_scores,
            "termination_reason": termination,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_used": branch_tokens + orch_tokens,
            "branches": self._branch_episodes,
            "trajectory": [
                {
                    "iteration": h["iteration"],
                    "output": h["output"],
                    "quality_score": h["quality_score"],
                }
                for h in (self._orch._state.history if self._orch._state else [])
            ],
        }

        trace = build_loop_trace(
            self.orchestrator_spec,
            loop_id=loop_id,
            success=bool(success),
            termination_reason=termination or "unknown",
            history=self._orch._state.history if self._orch._state else [],
            total_cost_usd=total_cost,
            started_at=started_at,
            spec_path=str(self.orchestrator_spec_path),
            metadata={
                "env_id": self.env_id,
                "composition": "parallel",
                "branch_count": len(self._branch_episodes),
            },
        )
        result["loop_trace"] = trace
        if trace_path is not None:
            result["trace_path"] = str(write_loop_trace(trace, trace_path))
        return result
