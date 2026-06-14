"""Tests for ReplayEnv and LoopNet corpus replay."""

from __future__ import annotations

from pathlib import Path

import loopgym as lg
from loopgym.envs.replay import (
    find_captured_records,
    is_captured_record,
    load_loopnet_records,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CAPTURED_FIXTURE = FIXTURES / "captured-record.jsonl"


def test_is_captured_record_fixture():
    records = load_loopnet_records(CAPTURED_FIXTURE)
    assert len(records) == 1
    assert is_captured_record(records[0])
    assert records[0]["record_id"] == "ln-37ecfbbf-174b-47d9-8271-2e956f741aac"


def test_replay_captured_record_from_fixture():
    records = load_loopnet_records(CAPTURED_FIXTURE)
    record = records[0]
    env = lg.make("replay/loopnet-v1", records_path=CAPTURED_FIXTURE)
    result = env.run_episode(record_id=record["record_id"])

    assert result["captured"] is True
    assert result["record_id"] == record["record_id"]
    assert result["steps"] == len(record["trajectory"]) - 1
    assert result["success"] is True
    assert result["quality_score"] == record["trajectory"][-1]["goal_score"]
    assert result["les_observed"] == record["les_observed"]["les_normalized"]


def test_replay_captured_record_reset_step():
    records = load_loopnet_records(CAPTURED_FIXTURE)
    record = records[0]
    env = lg.make("replay/loopnet-v1", records_path=CAPTURED_FIXTURE)
    obs = env.reset(record_id=record["record_id"])

    assert obs.info.get("captured") is True
    assert obs.info.get("record_id") == record["record_id"]
    assert obs.info.get("total_steps") == len(record["trajectory"])
    assert obs.info.get("env_id") == "loopbench/code-repair-v1"

    steps = 0
    while not env.done:
        obs, reward, done, info = env.step()
        steps += 1
    assert steps == len(record["trajectory"]) - 1
    assert info.get("success") is True


def test_find_captured_records_in_v02_corpus():
    loopgym_root = Path(__file__).resolve().parents[1]
    v02_path = loopgym_root.parent / "04-loopnet" / "data" / "v0.2" / "records.jsonl"
    if not v02_path.exists():
        return

    records = load_loopnet_records(v02_path)
    captured = find_captured_records(records)
    assert len(captured) >= 1

    env = lg.make("replay/loopnet-v1", records_path=v02_path)
    result = env.run_episode(record_id=captured[0]["record_id"])
    assert result["captured"] is True
    assert result["steps"] > 0
