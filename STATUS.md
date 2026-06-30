# Status

| Field | Value |
|-------|-------|
| **Phase** | v0.1 shipped |
| **Symbol** | ✅ |
| **Started** | 2026-06-13 |
| **Shipped** | 2026-06-13 |
| **Owner** | — |
| **Blockers** | — |
| **Notes** | Published at https://github.com/KanakMalpani/LoopGym |

## Completion checklist

- [x] `loopgym/` Python package
- [x] `envs/loopbench/` — SimEnvs backing LoopBench micro-tasks (4 env IDs, 19 tasks)
- [x] `examples/quickstart.py`
- [x] `docs/api.md`
- [x] ReplayEnv ↔ LoopNet seed integration
- [x] CI: test workflow + LSS validation
- [x] `SYNC.md` — canonical source policy
- [x] PyPI publish — https://pypi.org/project/loopgym/
- [x] `loopgym capture` → LoopNet export (v0.2 data pipeline)
- [x] ReplayEnv replays captured ln/record-v1 trajectories (v0.2 corpus + fixture tests)

## Links

- Workspace state: [../WORKSPACE_CURRENT_STATE.md](../WORKSPACE_CURRENT_STATE.md)
- Parent map: [../README.md](../README.md)
- Core specs: [../02-loop-core-engineering/](../02-loop-core-engineering/)
- Benchmark tasks: [../07-loopbench/tasks/](../07-loopbench/tasks/)
- Agent brief: [../AGENT_BRIEFS/loopgym.md](../AGENT_BRIEFS/loopgym.md)
