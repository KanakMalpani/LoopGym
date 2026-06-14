"""LoopGym CLI."""

from __future__ import annotations

import argparse
import json

import loopgym as lg


def main() -> None:
    parser = argparse.ArgumentParser(description="LoopGym — run loop environments")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List registered environments")
    list_cmd.set_defaults(func=_cmd_list)

    run_cmd = sub.add_parser("run", help="Run an environment episode")
    run_cmd.add_argument("env_id", help="Environment ID")
    run_cmd.add_argument("--task-id", default="default")
    run_cmd.add_argument("--seed", type=int, default=0)
    run_cmd.add_argument("--spec-path", default=None)
    run_cmd.set_defaults(func=_cmd_run)

    cap_cmd = sub.add_parser("capture", help="Run episodes and export LoopNet records")
    cap_cmd.add_argument("env_id", help="Environment ID")
    cap_cmd.add_argument("--task-ids", default="cr-001", help="Comma-separated task IDs")
    cap_cmd.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    cap_cmd.add_argument("--output", "-o", required=True, help="Append JSONL output path")
    cap_cmd.add_argument("--spec-path", default=None)
    cap_cmd.add_argument(
        "--source",
        default="case_study",
        choices=["case_study", "community", "production_redacted"],
    )
    cap_cmd.add_argument("--split", default="train", choices=["train", "val", "test"])
    cap_cmd.add_argument(
        "--failure-seeds",
        default="",
        help="Comma-separated seeds to run with max_iterations=1 (failure diversity)",
    )
    cap_cmd.set_defaults(func=_cmd_capture)

    args = parser.parse_args()
    args.func(args)


def _cmd_list(_args: argparse.Namespace) -> None:
    for env_id in lg.list_envs():
        print(env_id)


def _cmd_run(args: argparse.Namespace) -> None:
    env = lg.make(args.env_id, spec_path=args.spec_path, seed=args.seed)
    if hasattr(env, "run_episode"):
        result = env.run_episode(task_id=args.task_id, seed=args.seed)
        print(json.dumps(result, indent=2))
    else:
        obs = env.reset(task_id=args.task_id, seed=args.seed)
        while not env.done:
            obs, reward, done, info = env.step()
            print(f"step reward={reward:.3f} quality={obs.quality_score:.3f} done={done}")


def _cmd_capture(args: argparse.Namespace) -> None:
    from loopgym.export.loopnet import append_jsonl, capture_env_episodes

    task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    failure_seeds = [int(s.strip()) for s in args.failure_seeds.split(",") if s.strip()]
    records = capture_env_episodes(
        args.env_id,
        task_ids=task_ids,
        seeds=seeds,
        spec_path=args.spec_path,
        source=args.source,
        split=args.split,
        failure_seeds=failure_seeds,
    )
    append_jsonl(records, args.output)
    print(json.dumps({"captured": len(records), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
