"""Tests for LoopNet export from LoopGym episodes."""

from __future__ import annotations

import loopgym as lg
from loopgym.export.loopnet import capture_env_episodes, episode_to_record


def test_episode_to_record_schema_fields():
    env = lg.make("loopbench/code-repair-v1")
    episode = env.run_episode(task_id="cr-001", seed=7)
    record = episode_to_record(
        episode,
        spec=env.spec,
        env_id="loopbench/code-repair-v1",
    )

    assert record["schema_version"] == "ln/record-v1"
    assert record["record_id"].startswith("ln-")
    assert record["trajectory"]
    assert "les_observed" in record
    assert record["metadata"]["goal_target"] > 0
    if record["outcome"] == "failure":
        assert record.get("failure_mode")


def test_capture_env_episodes_batch():
    records = capture_env_episodes(
        "loopbench/code-repair-v1",
        task_ids=["cr-001"],
        seeds=[0, 1],
    )
    assert len(records) == 2
    assert records[0]["record_id"] != records[1]["record_id"]
