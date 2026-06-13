#!/usr/bin/env python3
"""LoopGym quickstart — run a loop env with MockLLM (no API keys)."""

from __future__ import annotations

import loopgym as lg


class PassthroughAgent:
    """Agent that lets the env's built-in MockLLM drive each step."""

    def action(self, obs: lg.Observation) -> None:
        return None


def main() -> None:
    print("LoopGym quickstart")
    print("Registered envs:", ", ".join(lg.list_envs()))
    print()

    env = lg.make("loopbench/code-repair-v1")
    agent = PassthroughAgent()

    obs = env.reset(task_id="cr-001", seed=42)
    print(f"reset: task={obs.task_id} objective={obs.objective[:60]}...")
    print()

    step = 0
    while not env.done:
        obs, reward, done, info = env.step(agent.action(obs))
        step += 1
        print(
            f"step {step}: quality={obs.quality_score:.3f} reward={reward:.3f} "
            f"done={done} reason={info.get('termination_reason', '')}"
        )

    print()
    print(f"Finished in {step} steps. Success={info.get('success')}")
    print(f"Final output preview: {obs.output[:120]}...")

    print()
    print("Reproducibility check (3 seeds, same task):")
    for seed in (0, 1, 2):
        result = env.run_episode(task_id="cr-001", seed=seed)
        traj_hash = hash(tuple((s["iteration"], s["quality_score"]) for s in result["trajectory"]))
        print(
            f"  seed={seed}: steps={result['steps']} "
            f"quality={result['quality_score']:.3f} traj_hash={traj_hash}"
        )


if __name__ == "__main__":
    main()
