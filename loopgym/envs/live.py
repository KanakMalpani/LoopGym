"""LiveEnv — real LLM APIs (optional, user-provided keys)."""

from __future__ import annotations

import os
from typing import Any

from loopgym.envs.base import LoopEnv, Observation
from loopgym.envs.sim import SimEnv
from loopgym.runtime.loop_runtime import LLMClient


class _OpenAILiveLLM:
    """Thin wrapper for OpenAI-compatible APIs (optional dependency)."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.tokens_used = 0
        if not self.api_key:
            raise ValueError(
                "LiveEnv requires OPENAI_API_KEY or api_key= parameter. "
                "Use SimEnv for keyless testing."
            )
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        self._client = openai.OpenAI(api_key=self.api_key)

    def complete(self, prompt: str, role: str = "default") -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a {role} agent in a loop."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""
        if response.usage:
            self.tokens_used += response.usage.total_tokens
        return text


class LiveEnv(LoopEnv):
    """Live environment using real LLM APIs. Falls back to documenting key requirement."""

    def __init__(
        self,
        env_id: str,
        spec: dict[str, Any],
        llm: LLMClient | None = None,
        **sim_kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        self._sim = SimEnv(env_id, spec, **sim_kwargs)
        self._llm = llm
        if llm is not None:
            self._sim._runtime = None  # will be set on reset with custom llm

    def reset(self, task_id: str = "", seed: int | None = None, **kwargs: Any) -> Observation:
        obs = self._sim.reset(task_id=task_id, seed=seed, **kwargs)
        if self._llm and self._sim._runtime:
            self._sim._runtime.llm = self._llm
        return obs

    def step(self, action: Any = None) -> tuple[Observation, float, bool, dict[str, Any]]:
        return self._sim.step(action)

    @classmethod
    def with_openai(
        cls,
        env_id: str,
        spec: dict[str, Any],
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        **sim_kwargs: Any,
    ) -> LiveEnv:
        return cls(env_id, spec, llm=_OpenAILiveLLM(model=model, api_key=api_key), **sim_kwargs)
