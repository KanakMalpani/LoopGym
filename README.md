<div align="center">

# LoopGym

**Run any loop. Three ways. One API.**

Compile [LSS 1.0](https://github.com/KanakMalpani/Loop-Core-Engineering) YAML into executable environments — simulate for CI, call live models for production eval, or replay [LoopNet](https://github.com/KanakMalpani/loopnet) trajectories without spending a token.

<br>

[![CI](https://github.com/KanakMalpani/LoopGym/actions/workflows/test.yml/badge.svg)](https://github.com/KanakMalpani/LoopGym/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/loopgym.svg)](https://pypi.org/project/loopgym/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LSS 1.0](https://img.shields.io/badge/LSS-1.0.0-green.svg)](https://github.com/KanakMalpani/Loop-Core-Engineering)

<br>

```bash
pip install loopgym
```

<br>

[**Quickstart**](#try-it-in-60-seconds) · [**API docs**](docs/api.md) · [**PyPI**](https://pypi.org/project/loopgym/) · [**LoopBench**](https://github.com/KanakMalpani/LoopBench) · [**Observability**](https://github.com/KanakMalpani/loop-observability)

</div>

---

## The idea in one picture

```mermaid
flowchart TB
  SPEC["Your LSS YAML"]
  MAKE["loopgym.make(env_id)"]

  SIM["SimEnv<br/><i>deterministic · free · CI-safe</i>"]
  LIVE["LiveEnv<br/><i>real models · production eval</i>"]
  REPLAY["ReplayEnv<br/><i>LoopNet trajectories · zero API cost</i>"]

  SPEC --> MAKE
  MAKE --> SIM
  MAKE --> LIVE
  MAKE --> REPLAY
```

**LSS declares the loop. LoopGym runs it.** [LoopBench](https://github.com/KanakMalpani/LoopBench) scores it. Clean separation — like Gym vs. benchmark suites in reinforcement learning.

---

## Three backends, one line of code

```python
import loopgym as lg

env = lg.make("loopbench/code-repair-v1")
obs = env.reset(task_id="cr-001")

while not env.done:
    action = your_agent.policy(obs)
    obs, reward, done, info = env.step(action)
```

| Backend | When to use | API keys? |
|---------|-------------|-----------|
| **SimEnv** | CI, local dev, [LoopBench](https://github.com/KanakMalpani/LoopBench) v0.1 submissions | No |
| **LiveEnv** | Production eval with real LLMs | `OPENAI_API_KEY` (pluggable) |
| **ReplayEnv** | Analyze historical runs from LoopNet | No |

---

## Try it in 60 seconds

```bash
pip install loopgym

python -c "
import loopgym as lg
env = lg.make('loopbench/code-repair-v1')
obs = env.reset(task_id='cr-001')
print('task:', obs.task_id, '| step:', obs.step)
"
```

**Full quickstart:**

```bash
git clone https://github.com/KanakMalpani/LoopGym.git && cd LoopGym
pip install -e ".[dev]"
python examples/quickstart.py
pytest tests/ -q
```

---

## Validate and reproduce

Ran a replay or SimEnv episode? Follow [REPRODUCE.md](https://github.com/KanakMalpani/Loop-Engineering/blob/main/contributions/REPRODUCE.md) and post on [Discussion #10](https://github.com/KanakMalpani/Loop-Engineering/discussions/10). Export trajectories via [loopnet COMMUNITY-SUBMISSION](https://github.com/KanakMalpani/loopnet/blob/main/guides/COMMUNITY-SUBMISSION.md).

---

## Environments (v0.1)

| Env ID | Backend | Stress-tests |
|--------|---------|--------------|
| `loopbench/code-repair-v1` | Sim | Verify-driven repair, iteration limits |
| `loopbench/research-synthesis-v1` | Sim | Multi-step synthesis + rubric |
| `loopbench/multi-agent-debate-v1` | Sim | Role-separated workers + evaluator |
| `loopbench/composed-swarm-v1` | Sim | Composed parallel rehearsal ([scenario-swarm-rehearsal](https://github.com/KanakMalpani/Loop-Engineering/blob/main/loop-library/compositions/scenario-swarm-rehearsal.yaml)) — spec v0.2 |
| `replay/loopnet-v1` | Replay | Full trajectories from [LoopNet v0.2](https://huggingface.co/datasets/KanakMalpani/loopnet-v0.2) |
| `sim/mock-llm-v1` | Sim | Generic sandbox for custom LSS specs |

Bundled specs under [`envs/loopbench/`](envs/loopbench/) — validated against [Loop Core Engineering](https://github.com/KanakMalpani/Loop-Core-Engineering) in CI.

---

## Who this is for

| You want to… | LoopGym gives you… |
|--------------|-------------------|
| **Benchmark your loop design** | Same env IDs [LoopBench](https://github.com/KanakMalpani/LoopBench) uses |
| **Test without burning API budget** | SimEnv + ReplayEnv |
| **Ship production eval pipelines** | LiveEnv with pluggable backends |
| **Replay production-like runs** | ReplayEnv + [LoopNet corpus](https://github.com/KanakMalpani/loopnet) |
| **Trace iterations & LES** | [`loopotel`](https://github.com/KanakMalpani/loop-observability) LTF export |

---

## Observability

Trace loop iterations without raw chat logs ([LTF 0.1](https://github.com/KanakMalpani/loop-observability)):

```bash
pip install loopotel loopgym
python -c "
import loopgym as lg
from loopotel.integrations.loopgym import run_traced_episode
env = lg.make('loopbench/code-repair-v1')
result, trace = run_traced_episode(env, task_id='cr-001', seed=0, enabled=True)
print(result['success'], len(trace['spans']), 'spans')
"
```

Full stack walkthrough: [LoopNet end-to-end tutorial](https://github.com/KanakMalpani/loopnet/blob/main/guides/END-TO-END-TUTORIAL.md).

---

## Ecosystem

| Repo | Role |
|------|------|
| [Loop Core Engineering](https://github.com/KanakMalpani/Loop-Core-Engineering) | LSS / LES authority |
| [LoopNet](https://github.com/KanakMalpani/loopnet) | Trajectory corpus |
| **LoopGym** | Runtime (this repo) |
| [LoopBench](https://github.com/KanakMalpani/LoopBench) | Public scoreboard |
| [loop-observability](https://github.com/KanakMalpani/loop-observability) | LTF traces (`loopotel`) |

Stack map: [ECOSYSTEM.md](https://github.com/KanakMalpani/Loop-Core-Engineering/blob/main/ECOSYSTEM.md)

---

## Citation

```bibtex
@software{loopgym2026,
  title={LoopGym: OpenAI Gym for LSS-Defined Agent Loops},
  author={Malpani, Kanak},
  year={2026},
  url={https://pypi.org/project/loopgym/}
}
```

<div align="center">

<sub>MIT · v0.1 · <a href="CONTRIBUTING.md">Contributing</a> · <a href="SECURITY.md">Security</a> · <a href="PUBLISHING.md">Publishing</a></sub>

</div>
