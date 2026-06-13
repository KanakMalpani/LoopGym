# LoopGym

[![test-loopgym](https://github.com/KanakMalpani/LoopGym/actions/workflows/test.yml/badge.svg)](https://github.com/KanakMalpani/LoopGym/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LSS 1.0.0](https://img.shields.io/badge/LSS-1.0.0-green.svg)](https://github.com/KanakMalpani/Loop-Core-Engineering)

**OpenAI Gym for loops** — create, run, benchmark, compare, and evolve LSS-defined agent loops.

---

## Ecosystem

| Repo | Purpose |
|------|---------|
| [Loop Core Engineering](https://github.com/KanakMalpani/Loop-Core-Engineering) | LSS / LES specs |
| [LoopNet](https://github.com/KanakMalpani/loopnet) | Trajectory dataset for ReplayEnv |
| **LoopGym** (this repo) | Runtime — SimEnv, LiveEnv, ReplayEnv |
| [LoopBench](https://github.com/KanakMalpani/LoopBench) | Benchmark suite (uses LoopGym to run) |

---

## Quick start

**From GitHub (recommended for v0.1):**

```bash
pip install git+https://github.com/KanakMalpani/LoopGym.git
python -c "import loopgym as lg; print(lg.make('loopbench/code-repair-v1'))"
```

**Local development:**

```bash
git clone https://github.com/KanakMalpani/LoopGym.git
cd LoopGym
pip install -e ".[dev]"
python examples/quickstart.py
```

**PyPI** (after first release — see [PUBLISHING.md](PUBLISHING.md)):

```bash
pip install loopgym
```

```python
import loopgym as lg

env = lg.make("loopbench/code-repair-v1")
obs = env.reset(task_id="cr-001")
while not env.done:
    obs, reward, done, info = env.step(agent.action(obs))
```

Optional — clone [LoopNet](https://github.com/KanakMalpani/loopnet) as a sibling for ReplayEnv:

```bash
git clone https://github.com/KanakMalpani/loopnet.git ../loopnet
# or set LOOPNET_SEED_PATH=/path/to/records.jsonl
```

---

## Environments

| Env ID | Backend | Description |
|--------|---------|-------------|
| `loopbench/code-repair-v1` | SimEnv | Verify-driven code repair loop |
| `loopbench/research-synthesis-v1` | SimEnv | Research brief synthesis loop |
| `loopbench/multi-agent-debate-v1` | SimEnv | Multi-agent review / debate loop |
| `replay/loopnet-v1` | ReplayEnv | Replay LoopNet `ln/record-v1` trajectories |
| `sim/mock-llm-v1` | SimEnv | Generic mock-LLM loop env |

---

## Dependencies

- **[Loop Core Engineering](https://github.com/KanakMalpani/Loop-Core-Engineering)** — LSS schema (validate env YAML in CI)
- **[LoopNet](https://github.com/KanakMalpani/loopnet)** — optional sibling for ReplayEnv seed corpus

---

## Layout

| Path | Purpose |
|------|---------|
| `loopgym/` | Python package (registry, envs, runtime, evaluators) |
| `envs/loopbench/` | Bundled LSS specs + task fixtures for LoopBench |
| `examples/quickstart.py` | Smoke test / onboarding |
| `docs/api.md` | API reference |
| `SYNC.md` | Cross-repo sync policy |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Status

v0.1 shipped — see [STATUS.md](STATUS.md).
