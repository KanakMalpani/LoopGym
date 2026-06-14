"""ReplayEnv — replay LoopNet trajectories from ln/record-v1 JSONL."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from loopgym.envs.base import LoopEnv, Observation


def is_captured_record(record: dict[str, Any]) -> bool:
    """True when the record came from LoopGym capture (not synthetic seed)."""
    tags = (record.get("metadata") or {}).get("tags") or []
    if "captured" in tags:
        return True
    return record.get("source") == "case_study"


def _resolve_loopnet_corpus_path() -> Path | None:
    """Resolve LoopNet JSONL from env, sibling clone, or CI deps layout."""
    loopgym_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    for env_key in ("LOOPNET_RECORDS_PATH", "LOOPNET_SEED_PATH"):
        env_path = os.environ.get(env_key)
        if env_path:
            candidates.append(Path(env_path))
    candidates.extend(
        [
            loopgym_root.parent / "04-loopnet" / "data" / "v0.2" / "records.jsonl",
            loopgym_root.parent / "loopnet" / "data" / "v0.2" / "records.jsonl",
            loopgym_root / "deps" / "loopnet" / "data" / "v0.2" / "records.jsonl",
            loopgym_root.parent / "04-loopnet" / "data" / "seed" / "records.jsonl",
            loopgym_root.parent / "loopnet" / "data" / "seed" / "records.jsonl",
            loopgym_root / "deps" / "loopnet" / "data" / "seed" / "records.jsonl",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_loopnet_seed_path() -> Path | None:
    """Backward-compatible alias for corpus resolution."""
    return _resolve_loopnet_corpus_path()


def load_loopnet_records(path: Path) -> list[dict[str, Any]]:
    """Load LoopNet records from JSONL."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
    return records


def find_captured_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records produced by LoopGym capture."""
    return [record for record in records if is_captured_record(record)]


def record_to_trajectory(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Map ln/record-v1 trajectory steps to ReplayEnv step dicts."""
    steps: list[dict[str, Any]] = []
    objective = str(record.get("objective", ""))
    for step in record.get("trajectory") or []:
        goal = float(step.get("goal_score", step.get("primary_quality", 0.0)))
        steps.append(
            {
                "iteration": int(step.get("iteration", len(steps) + 1)),
                "output": (
                    f"[loopnet replay] {objective[:80]}… "
                    f"iter={step.get('iteration')} goal={goal:.3f}"
                ),
                "quality_score": goal,
                "cost_usd": step.get("cost_usd"),
                "latency_seconds": step.get("latency_seconds"),
                "failure_codes": list(step.get("failure_codes") or []),
            }
        )
    return steps


class ReplayEnv(LoopEnv):
    """Replay recorded trajectories from LoopNet ln/record-v1 records."""

    def __init__(
        self,
        env_id: str,
        trajectory_path: str | Path | None = None,
        records_path: str | Path | None = None,
    ) -> None:
        super().__init__(env_id)
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None
        resolved_records = Path(records_path) if records_path else None
        if resolved_records and resolved_records.exists():
            self.records_path = resolved_records
        else:
            self.records_path = _resolve_loopnet_corpus_path()
        self._records: list[dict[str, Any]] = []
        self._record: dict[str, Any] | None = None
        self._trajectory: list[dict[str, Any]] = []
        self._index = 0

    def _load_records_corpus(self) -> None:
        if self._records or not self.records_path or not self.records_path.exists():
            return
        self._records = load_loopnet_records(self.records_path)

    def _select_record(self, task_id: str, seed: int | None) -> dict[str, Any] | None:
        self._load_records_corpus()
        if not self._records:
            return None

        if task_id.startswith("ln-"):
            for record in self._records:
                if record.get("record_id") == task_id:
                    return record

        if task_id and task_id != "default":
            for record in self._records:
                if record.get("loop_name") == task_id:
                    return record

        rng = random.Random(seed if seed is not None else 0)
        return rng.choice(self._records)

    def reset(self, task_id: str = "", seed: int | None = None, **kwargs: Any) -> Observation:
        record_id = kwargs.get("record_id")
        self._task_id = task_id or "default"
        self._index = 0
        self._trajectory = []
        self._done = False
        self._record = None

        if self.trajectory_path and self.trajectory_path.exists():
            with self.trajectory_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            self._trajectory = list(data.get("steps") or data.get("history") or [])
        else:
            if record_id:
                self._load_records_corpus()
                for record in self._records:
                    if record.get("record_id") == record_id:
                        self._record = record
                        break
            else:
                self._record = self._select_record(self._task_id, seed)

            if self._record:
                self._trajectory = record_to_trajectory(self._record)
                self._task_id = str(self._record.get("record_id", self._task_id))

        if not self._trajectory:
            self._trajectory = [
                {"iteration": 1, "output": "replay fallback step 1", "quality_score": 0.5},
                {"iteration": 2, "output": "replay fallback step 2", "quality_score": 0.85},
            ]

        step = self._trajectory[0]
        objective = str((self._record or {}).get("objective", "Replay LoopNet trajectory"))
        record_meta = self._record or {}
        les = record_meta.get("les_observed") or {}
        self._obs = Observation(
            task_id=self._task_id,
            iteration=int(step.get("iteration", 1)),
            output=str(step.get("output", "")),
            quality_score=float(step.get("quality_score", 0.0)),
            objective=objective,
            done=False,
            info={
                "mode": "replay",
                "total_steps": len(self._trajectory),
                "record_id": record_meta.get("record_id"),
                "outcome": record_meta.get("outcome"),
                "source": record_meta.get("source"),
                "captured": is_captured_record(record_meta) if record_meta else False,
                "les_observed": les.get("les_normalized"),
                "env_id": (record_meta.get("loop_spec") or {}).get("extensions", {}).get("env_id"),
            },
        )
        return self._obs

    def step(self, action: Any = None) -> tuple[Observation, float, bool, dict[str, Any]]:
        self._index += 1
        if self._index >= len(self._trajectory):
            self._done = True
            obs = self._obs or Observation(
                task_id=self._task_id,
                iteration=0,
                output="",
                quality_score=0.0,
                objective="Replay LoopNet trajectory",
                done=True,
            )
            return obs, 0.0, True, {"reason": "trajectory_exhausted"}

        step = self._trajectory[self._index]
        quality = float(step.get("quality_score", 0.0))
        self._done = self._index >= len(self._trajectory) - 1
        objective = str((self._record or {}).get("objective", "Replay LoopNet trajectory"))
        record_meta = self._record or {}
        outcome = record_meta.get("outcome")
        success = outcome == "success" or quality >= float(
            (record_meta.get("metadata") or {}).get("goal_target", 0.0)
        )
        self._obs = Observation(
            task_id=self._task_id,
            iteration=int(step.get("iteration", self._index + 1)),
            output=str(step.get("output", "")),
            quality_score=quality,
            objective=objective,
            done=self._done,
            info={
                "mode": "replay",
                "step_index": self._index,
                "record_id": record_meta.get("record_id"),
                "failure_codes": step.get("failure_codes", []),
                "captured": is_captured_record(record_meta) if record_meta else False,
                "success": success if self._done else False,
            },
        )
        info: dict[str, Any] = {"step_index": self._index}
        if self._done:
            info["success"] = success
            info["termination_reason"] = record_meta.get("termination_reason", "trajectory_exhausted")
        return self._obs, quality, self._done, info

    def run_episode(
        self,
        task_id: str = "",
        seed: int | None = None,
        *,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        """Run full replay until the stored trajectory is exhausted."""
        reset_kwargs: dict[str, Any] = {}
        if record_id:
            reset_kwargs["record_id"] = record_id
        self.reset(task_id=task_id, seed=seed, **reset_kwargs)
        total_reward = 0.0
        steps = 0
        start = time.perf_counter()
        info: dict[str, Any] = {}
        while not self.done:
            _, reward, _, info = self.step()
            total_reward += reward
            steps += 1
        elapsed = time.perf_counter() - start
        record_meta = self._record or {}
        final_quality = self._obs.quality_score if self._obs else 0.0
        goal_target = float((record_meta.get("metadata") or {}).get("goal_target", 0.0))
        outcome = record_meta.get("outcome")
        success = info.get("success", False) or outcome == "success" or final_quality >= goal_target
        return {
            "task_id": self._task_id,
            "seed": seed,
            "env_id": self.env_id,
            "record_id": record_meta.get("record_id"),
            "captured": is_captured_record(record_meta) if record_meta else False,
            "steps": steps,
            "total_reward": total_reward,
            "success": success,
            "quality_score": final_quality,
            "termination_reason": info.get("termination_reason", record_meta.get("termination_reason", "")),
            "elapsed_seconds": round(elapsed, 3),
            "tokens_used": 0,
            "outcome": outcome,
            "les_observed": (record_meta.get("les_observed") or {}).get("les_normalized"),
            "trajectory": [
                {
                    "iteration": step.get("iteration"),
                    "output": step.get("output"),
                    "quality_score": step.get("quality_score"),
                }
                for step in self._trajectory
            ],
        }
