"""Tests for LoopGym SimEnv and reproducibility."""

from __future__ import annotations

import loopgym as lg


def test_list_envs_includes_loopbench():
    envs = lg.list_envs()
    assert "loopbench/code-repair-v1" in envs
    assert "loopbench/research-synthesis-v1" in envs
    assert "loopbench/multi-agent-debate-v1" in envs
    assert "loopbench/composed-swarm-v1" in envs


def test_composed_swarm_episode():
    env = lg.make("loopbench/composed-swarm-v1")
    result = env.run_episode(task_id="comp-001", seed=7)
    assert result["steps"] > 0
    assert len(result.get("branches", [])) == 3
    assert result["quality_score"] > 0


def test_make_unknown_env_raises():
    try:
        lg.make("unknown/env-v1")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown/env-v1" in str(exc)


def test_episode_includes_loop_trace():
    env = lg.make("loopbench/code-repair-v1")
    result = env.run_episode(task_id="cr-001", seed=42)
    trace = result["loop_trace"]
    assert trace["trace_version"] == "1.0"
    assert trace["loop_name"]
    assert trace["started_at"]
    assert len(trace["iterations"]) >= 1
    assert "quality" in trace["iterations"][-1]["evaluator_scores"]


def test_run_episode_writes_trace_file(tmp_path):
    env = lg.make("loopbench/code-repair-v1")
    out = tmp_path / "episode-trace.json"
    result = env.run_episode(task_id="cr-001", seed=42, trace_path=out)
    assert out.exists()
    assert result["trace_path"] == str(out)
    data = out.read_text(encoding="utf-8")
    assert '"trace_version": "1.0"' in data


def test_quickstart_episode_completes():
    env = lg.make("loopbench/code-repair-v1")
    result = env.run_episode(task_id="cr-001", seed=42)
    assert result["steps"] > 0
    assert result["trajectory"]
    assert result["quality_score"] > 0


def test_reproducible_trajectories_three_seeds():
    env = lg.make("loopbench/code-repair-v1")
    hashes = []
    for seed in (0, 1, 2):
        r1 = env.run_episode(task_id="cr-001", seed=seed)
        r2 = env.run_episode(task_id="cr-001", seed=seed)
        traj = tuple((s["iteration"], s["quality_score"], s["output"][:40]) for s in r1["trajectory"])
        traj2 = tuple((s["iteration"], s["quality_score"], s["output"][:40]) for s in r2["trajectory"])
        assert traj == traj2
        hashes.append(hash(traj))
    assert len(set(hashes)) == 3


def test_reset_step_api():
    env = lg.make("loopbench/research-synthesis-v1", seed=1)
    obs = env.reset(task_id="rs-001")
    assert obs.iteration == 0
    assert not obs.done
    prev_quality = 0.0
    while not env.done:
        obs, reward, done, info = env.step()
        assert obs.quality_score >= prev_quality or done
        prev_quality = obs.quality_score
        if done:
            break
    assert env.done


def test_replay_env_stub():
    env = lg.make("replay/loopnet-v1")
    obs = env.reset(task_id="ln-001")
    steps = 0
    while not env.done:
        obs, reward, done, info = env.step()
        steps += 1
    assert steps >= 1
    assert obs.quality_score > 0


def test_replay_env_loopnet_record():
    from loopgym.envs.replay import _default_loopnet_seed_path, load_loopnet_records

    seed_path = _default_loopnet_seed_path()
    if seed_path is None:
        return

    records = load_loopnet_records(seed_path)
    assert len(records) >= 1
    record_id = records[0]["record_id"]

    env = lg.make("replay/loopnet-v1", records_path=seed_path)
    obs = env.reset(record_id=record_id)
    assert obs.info.get("record_id") == record_id
    assert obs.info.get("total_steps") == len(records[0]["trajectory"])
    steps = 0
    while not env.done:
        obs, reward, done, info = env.step()
        steps += 1
    assert steps == len(records[0]["trajectory"]) - 1
