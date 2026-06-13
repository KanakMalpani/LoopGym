# Contributing to LoopGym

## What belongs here

- Environment implementations (SimEnv, LiveEnv, ReplayEnv)
- LSS compiler / loop runtime
- Env registry and bundled LoopBench env fixtures
- Evaluators and CLI

## What does not belong here

- LSS schema changes — [Loop Core Engineering](https://github.com/KanakMalpani/Loop-Core-Engineering)
- Benchmark task definitions / LES scoring — [LoopBench](https://github.com/KanakMalpani/LoopBench)
- Dataset records — [LoopNet](https://github.com/KanakMalpani/loopnet)

## Before opening a PR

```bash
pip install -e ".[dev]"
pytest tests/ -q
python examples/quickstart.py
```

CI validates bundled LSS specs against [Loop Core Engineering](https://github.com/KanakMalpani/Loop-Core-Engineering).

## License

MIT — see [LICENSE](LICENSE).
