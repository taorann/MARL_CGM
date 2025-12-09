from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build issue-specific code subgraph for a SWE-bench repo."
    )
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--issue-id", type=str, required=True)
    return parser.parse_args()


def build_issue_subgraph(repo: Path, issue_id: str) -> Dict[str, Any]:
    # TODO: replace this placeholder with the real graph builder implementation.
    return {
        "issue_id": issue_id,
        "repo": str(repo),
        "nodes": [],
        "edges": [],
        "meta": {"note": "TODO: replace with real graph builder"},
    }


def main() -> None:
    args = _parse_args()
    repo = Path(args.repo).resolve()
    if not repo.exists():
        raise SystemExit(f"Repository path does not exist: {repo}")

    subgraph = build_issue_subgraph(repo, args.issue_id)
    print(json.dumps(subgraph, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
