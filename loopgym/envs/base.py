"""Loop environment base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """Gym-style observation returned by reset/step."""

    task_id: str
    iteration: int
    output: str
    quality_score: float
    objective: str
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "iteration": self.iteration,
            "output": self.output,
            "quality_score": self.quality_score,
            "objective": self.objective,
            "done": self.done,
            "info": self.info,
        }


class LoopEnv(ABC):
    """Abstract loop environment (Gym-style API)."""

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self._done = False
        self._task_id = ""
        self._obs: Observation | None = None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def observation(self) -> Observation | None:
        return self._obs

    @abstractmethod
    def reset(self, task_id: str = "", seed: int | None = None, **kwargs: Any) -> Observation:
        """Reset environment for a new episode."""

    @abstractmethod
    def step(self, action: Any = None) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Advance one loop iteration. Returns (obs, reward, done, info)."""

    def close(self) -> None:
        """Release resources."""

    def _reward(self, quality_score: float, success: bool) -> float:
        return quality_score if success else quality_score - 0.5
