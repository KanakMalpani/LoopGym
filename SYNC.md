# Sync policy — LoopGym

**Runtime layer for Loop Engineering.**

| Artifact | Canonical source | This repo |
|----------|------------------|-----------|
| LSS / LES specs | [Loop Core Engineering](https://github.com/KanakMalpani/Loop-Core-Engineering) | pin `lss@1.0.0`, `les@1.0.0` |
| LoopNet records | [LoopNet](https://github.com/KanakMalpani/loopnet) | ReplayEnv reads seed JSONL |
| Env IDs | Loop Core Engineering `specs/loop-ids.md` | registry in `loopgym/registry.py` |
| Benchmark tasks | [LoopBench](https://github.com/KanakMalpani/LoopBench) | bundled fixtures under `envs/loopbench/` |

**Repository:** https://github.com/KanakMalpani/LoopGym

## Validation before release

```bash
pytest tests/ -q
python examples/quickstart.py
```

Validate bundled specs against Loop Core Engineering in CI.

## Do not duplicate

- LSS JSON Schema — checkout Loop Core Engineering in CI
- LoopBench task YAML — reference env IDs only
- LES observed scoring — belongs in LoopBench

## CI dependency order

```
Loop Core Engineering (schema)
LoopGym (this repo)
LoopBench (downstream)
```
