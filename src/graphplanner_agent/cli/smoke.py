from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec
from graphplanner_agent.env import CodeRepairEnv
from graphplanner_agent.planner import PlannerAction
from graphplanner_agent.repair.cgm_client import MockCgmClient
from graphplanner_agent.runtime import LocalRepoRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal GraphPlanner rebuild smoke flow.")
    parser.add_argument("task_json", type=Path)
    args = parser.parse_args()
    task = TaskSpec.from_dict(json.loads(args.task_json.read_text(encoding="utf-8")))
    config = AgentConfig.from_env()
    env = CodeRepairEnv.create(task, LocalRepoRuntime(task.repo_path), MockCgmClient(), config)
    for action in [
        PlannerAction("run_failed_test", {}),
        PlannerAction("explore_find", {"query": task.issue_title or task.issue_body, "find_type": "any"}),
    ]:
        print(json.dumps(env.step(action), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
