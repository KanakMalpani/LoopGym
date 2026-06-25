# LoopGym API Reference (v0.1)

## Installation

```bash
pip install -e .
```

## Core API

### `loopgym.make(env_id, **kwargs)`

Factory for loop environments. Returns a `LoopEnv` instance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `env_id` | `str` | Environment ID (e.g. `loopbench/code-repair-v1`). Accepts optional `lg/` prefix. |
| `spec_path` | `str \| Path` | Override bundled LSS YAML with a custom spec. |
| `backend` | `str` | Force `sim`, `replay`, or `live`. |
| `seed` | `int` | RNG seed for reproducible SimEnv trajectories (default `0`). |

```python
import loopgym as lg

env = lg.make("loopbench/code-repair-v1")
env = lg.make("loopbench/code-repair-v1", spec_path="my-loop.yaml", seed=42)
```

### `loopgym.list_envs()`

Returns sorted list of registered environment IDs.

## LoopEnv interface

Gym-style API for stepping through loop iterations.

### `env.reset(task_id="", seed=None, **kwargs) -> Observation`

Start a new episode. Resolves task input from `tasks.json` when available.

### `env.step(action=None) -> (Observation, reward, done, info)`

Advance one loop iteration:

- `action=None` — built-in MockLLM drives act/evaluate (SimEnv).
- `action={"output": "..."}` — agent overrides worker output for this step.

### `env.done` (property)

Whether the episode has terminated.

### `env.run_episode(task_id="", seed=None, *, trace_path=None) -> dict` (SimEnv / ComposedSimEnv)

Run until termination; returns trajectory summary for benchmarking.

When the episode finishes, the result includes a **`loop_trace`** field — a [Loop Trace 1.0](https://github.com/KanakMalpani/Loop-Engineering/blob/main/standards/LOOP-TRACE-1.0.md) document suitable for `loopctl trace validate` and observed LES scoring.

Pass **`trace_path="path/to/trace.json"`** to write the trace file on disk; the returned dict then includes **`trace_path`**.

## Observation

```python
@dataclass
class Observation:
    task_id: str
    iteration: int
    output: str
    quality_score: float
    objective: str
    done: bool
    info: dict
```

## Environment backends

### SimEnv (`sim`)

Mock LLM + mock oracles. **No API keys.** Default for all `loopbench/*` envs.

Reproducibility: trajectories are deterministic given `(seed, task_id, loop_name)`.

### ReplayEnv (`replay`)

Replay LoopNet `ln/record-v1` trajectories from JSONL (default: sibling `04-loopnet/data/v0.2/records.jsonl`, then seed fallback).

| Parameter | Type | Description |
|-----------|------|-------------|
| `records_path` | `str \| Path` | Path to LoopNet JSONL corpus |
| `trajectory_path` | `str \| Path` | Single JSON file with `steps` or `history` |
| `record_id` | `str` | Passed to `reset(record_id=...)` to select a record |

```python
env = lg.make("replay/loopnet-v1")
obs = env.reset(record_id="ln-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
```

```python
env = lg.make("replay/loopnet-v1")
```

### LiveEnv (`live`)

Real LLM APIs. Requires `OPENAI_API_KEY` and optional `openai` package.

```python
env = lg.make("loopbench/code-repair-v1", backend="live")
```

## Registered environments

| Env ID | Task family | Description |
|--------|-------------|-------------|
| `loopbench/code-repair-v1` | LB-CR-1 | Verify-driven code repair |
| `loopbench/research-synthesis-v1` | LB-RS-1 | Research brief synthesis |
| `loopbench/multi-agent-debate-v1` | LB-MA-1 | Multi-agent debate / review |
| `replay/loopnet-v1` | — | LoopNet trajectory replay (seed + captured v0.2) |
| `sim/mock-llm-v1` | — | Generic mock loop |

Env ID format per [loop-ids.md](https://github.com/loop-engineering/core): `lg/{family}/{name}-v{major}`.

## Runtime (advanced)

```python
from loopgym.runtime import LoopRuntime, MockLLM, load_lss_spec

spec = load_lss_spec("envs/loopbench/code-repair-v1/spec.yaml")
runtime = LoopRuntime(spec, llm=MockLLM(seed="my-seed"))
result = runtime.run(user_input="fix the bug")
```

### LSS compiler

```python
from loopgym.runtime.compiler import compile_lss_file

graph = compile_lss_file("envs/loopbench/code-repair-v1/spec.yaml")
print(graph.worker_ids, graph.max_iterations)
```

## Evaluator plugins

```python
from loopgym.evaluators import run_deterministic, run_rubric
from loopgym.runtime import MockLLM

run_deterministic("evaluators.word_count_max", "hello world", {"max_words": 10})
run_rubric(MockLLM(), "output text", "objective", {"pass_threshold": 0.8, "dimensions": []})
```

## CLI

```bash
loopgym list
loopgym run loopbench/code-repair-v1 --task-id cr-001 --seed 42
```

## Integration hooks

LoopGym v0.1 exposes plain Python objects — wrap in LangGraph nodes or generic agent harnesses by calling `reset`/`step` in your orchestration layer. Full LangGraph adapter ships in v0.2.

## Dependencies

- **LSS schema:** `01-loop-engineering-core` (specs loaded from YAML; no runtime pin yet)
- **LoopNet:** `04-loopnet` optional for ReplayEnv data
