"""Environment registry and make() factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loopgym.envs.composed import ComposedSimEnv
from loopgym.envs.live import LiveEnv
from loopgym.envs.replay import ReplayEnv
from loopgym.envs.sim import SimEnv
from loopgym.runtime.loop_runtime import load_lss_spec

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_ENVS_ROOT = _PACKAGE_ROOT / "envs" / "loopbench"

_REGISTRY: dict[str, dict[str, Any]] = {
    "loopbench/code-repair-v1": {
        "backend": "sim",
        "spec": _ENVS_ROOT / "code-repair-v1" / "spec.yaml",
        "tasks": _ENVS_ROOT / "code-repair-v1" / "tasks.json",
        "description": "Verify-driven code repair loop (LB-CR-1)",
    },
    "loopbench/research-synthesis-v1": {
        "backend": "sim",
        "spec": _ENVS_ROOT / "research-synthesis-v1" / "spec.yaml",
        "tasks": _ENVS_ROOT / "research-synthesis-v1" / "tasks.json",
        "description": "Research brief synthesis loop (LB-RS-1)",
    },
    "loopbench/multi-agent-debate-v1": {
        "backend": "sim",
        "spec": _ENVS_ROOT / "multi-agent-debate-v1" / "spec.yaml",
        "tasks": _ENVS_ROOT / "multi-agent-debate-v1" / "tasks.json",
        "description": "Multi-agent review debate loop (LB-MA-1)",
    },
    "loopbench/composed-swarm-v1": {
        "backend": "composed",
        "spec": _ENVS_ROOT / "composed-swarm-v1" / "spec.yaml",
        "tasks": _ENVS_ROOT / "composed-swarm-v1" / "tasks.json",
        "branches": [
            (
                "falsifier",
                _ENVS_ROOT / "research-synthesis-v1" / "spec.yaml",
                _ENVS_ROOT / "research-synthesis-v1" / "tasks.json",
            ),
            (
                "evidence",
                _ENVS_ROOT / "multi-agent-debate-v1" / "spec.yaml",
                _ENVS_ROOT / "multi-agent-debate-v1" / "tasks.json",
            ),
            (
                "operator",
                _ENVS_ROOT / "code-repair-v1" / "spec.yaml",
                _ENVS_ROOT / "code-repair-v1" / "tasks.json",
            ),
        ],
        "description": "Composed parallel swarm rehearsal (LB-COMP-1)",
    },
    "replay/loopnet-v1": {
        "backend": "replay",
        "records_path": _PACKAGE_ROOT.parent / "04-loopnet" / "data" / "v0.2" / "records.jsonl",
        "description": "Replay LoopNet ln/record-v1 trajectories (v0.2 corpus preferred)",
    },
    "sim/mock-llm-v1": {
        "backend": "sim",
        "spec": _PACKAGE_ROOT.parent / "examples" / "minimal-loop.yaml",
        "description": "Generic mock-LLM loop using minimal LSS example",
    },
}


def _normalize_env_id(env_id: str) -> str:
    """Accept lg/ prefix per loop-ids.md."""
    env_id = env_id.strip()
    if env_id.startswith("lg/"):
        return env_id[3:]
    return env_id


def list_envs() -> list[str]:
    """Return registered environment IDs."""
    return sorted(_REGISTRY.keys())


def make(
    env_id: str,
    *,
    spec_path: str | Path | None = None,
    backend: str | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> SimEnv | ComposedSimEnv | ReplayEnv | LiveEnv:
    """Create a loop environment by ID.

    Args:
        env_id: Environment ID (e.g. ``loopbench/code-repair-v1``).
        spec_path: Override bundled LSS spec with a custom YAML path.
        backend: Force backend (``sim``, ``replay``, ``live``).
        seed: RNG seed for reproducible SimEnv trajectories.
        **kwargs: Passed to backend constructor.

    Returns:
        A LoopEnv instance.
    """
    normalized = _normalize_env_id(env_id)
    if normalized not in _REGISTRY:
        available = ", ".join(list_envs())
        raise ValueError(f"Unknown env_id '{env_id}'. Available: {available}")

    entry = _REGISTRY[normalized]
    chosen_backend = backend or entry.get("backend", "sim")

    if chosen_backend == "replay":
        records_path = kwargs.get("records_path") or entry.get("records_path")
        return ReplayEnv(
            normalized,
            trajectory_path=kwargs.get("trajectory_path"),
            records_path=records_path,
        )

    inline_spec = kwargs.get("spec")
    spec_file = Path(spec_path) if spec_path else entry.get("spec")
    if inline_spec is not None:
        spec = inline_spec
        spec_path_resolved = Path(spec_file) if spec_file else None
    else:
        if spec_file is None or not Path(spec_file).exists():
            raise FileNotFoundError(
                f"No LSS spec for '{normalized}'. Provide spec_path= or use a bundled env."
            )
        spec = load_lss_spec(spec_file)
        spec_path_resolved = Path(spec_file)
    tasks_path = entry.get("tasks")

    if chosen_backend == "composed":
        branches_raw = entry.get("branches") or []
        branches: list[tuple[str, Path, Path | None]] = []
        for item in branches_raw:
            if len(item) == 3:
                bid, spath, tpath = item
                branches.append((bid, Path(spath), Path(tpath) if tpath else None))
            else:
                bid, spath = item
                branches.append((bid, Path(spath), Path(spath).parent / "tasks.json"))
        return ComposedSimEnv(
            normalized,
            spec,
            spec_path_resolved or Path(entry["spec"]),
            Path(tasks_path) if tasks_path else None,
            branches,
            seed=seed,
        )

    if chosen_backend == "live":
        return LiveEnv(
            normalized,
            spec,
            spec_path=spec_path_resolved,
            tasks_path=Path(tasks_path) if tasks_path else None,
            seed=seed,
            **{k: v for k, v in kwargs.items() if k not in ("trajectory_path", "spec")},
        )

    return SimEnv(
        normalized,
        spec,
        spec_path=spec_path_resolved,
        tasks_path=Path(tasks_path) if tasks_path and Path(tasks_path).exists() else None,
        seed=seed,
        **{k: v for k, v in kwargs.items() if k not in ("trajectory_path", "spec")},
    )
