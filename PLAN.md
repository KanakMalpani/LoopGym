# 05 — loopgym

## One-line purpose

**LoopGym** — OpenAI Gym equivalent for loops: create, run, benchmark, compare, evolve, visualize.

## Why this repo exists

Adoption follows **runnable code**. LoopGym is the reference runtime that compiles LSS → executable loop environments.

## Scope (in scope)

- `loopgym.make(env_id)` API
- Env backends:
  - **SimEnv** — mock LLM + mock oracles (no API keys)
  - **ReplayEnv** — replay LoopNet trajectories
  - **LiveEnv** — real LLM APIs (optional, user keys)
- LSS compiler → runtime graph
- Evaluator plugins (command, rubric, deterministic)
- Evolution module (optional v0.2): mutate LSS hyperparams
- Integration hooks for LangGraph, generic Python

## Scope (out of scope)

- Official leaderboard hosting → `06-loopbench`
- Dataset curation → `04-loopnet`

## Deliverables v0.1

- [x] `loopgym/` Python package
- [x] `envs/loopbench/` stub envs (3 tasks)
- [x] `examples/quickstart.py`
- [x] `docs/api.md`
- [ ] PyPI publish workflow (optional)

## API sketch

```python
import loopgym as lg

env = lg.make("loopbench/code-repair-v1", spec_path="my-loop.yaml")
obs = env.reset(task_id="cr-001")
while not env.done:
    obs, reward, done, info = env.step(agent.action(obs))
```

## Dependencies

- **01-loop-engineering-core** — LSS schema
- **04-loopnet** — ReplayEnv data (optional v0.1)

## Success criteria

`quickstart.py` runs with MockLLM; same LSS runs on 3 seeds with reproducible trajectories in SimEnv.

## Agent instructions

Extract and generalize `implementations/generic/loop_runtime.py` from `Loop Engineering` repo; don't fork forever — migrate to this repo as canonical runtime.

## Status

✅ v0.1 shipped (2026-06-13)
