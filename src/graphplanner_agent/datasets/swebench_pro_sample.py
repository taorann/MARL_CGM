from __future__ import annotations

import argparse
import ast
import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_PARQUET_URL = (
    "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/resolve/main/"
    "data/test-00000-of-00001.parquet"
)
DEFAULT_RUN_SCRIPT_BASE = "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts"
DEFAULT_IMAGE_PREFIX = "jefzda/sweap-images"


def prepare_swebench_pro_sample(
    output_path: Path,
    *,
    sample_size: int = 10,
    seed: int = 20260604,
    parquet_url: str = DEFAULT_PARQUET_URL,
    run_script_base: str = DEFAULT_RUN_SCRIPT_BASE,
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    instance_ids: Iterable[str] | None = None,
    keep_order: bool = False,
) -> dict[str, object]:
    df = pd.read_parquet(parquet_url)
    records = df.to_dict(orient="records")
    selected_ids = [str(item).strip() for item in (instance_ids or []) if str(item).strip()]
    if selected_ids:
        by_id = {str(record.get("instance_id")): record for record in records}
        missing = [item for item in selected_ids if item not in by_id]
        if missing:
            raise ValueError(f"instance ids not found in SWE-bench Pro parquet: {missing}")
        selected = [by_id[item] for item in selected_ids]
    elif sample_size > 0 and sample_size < len(records):
        selected = records[:sample_size] if keep_order else random.Random(seed).sample(records, sample_size)
    else:
        selected = list(records)

    out: list[dict[str, Any]] = []
    script_errors: list[dict[str, str]] = []
    for row in selected:
        instance_id = str(row["instance_id"])
        try:
            run_script = _fetch_text(f"{run_script_base.rstrip('/')}/{instance_id}/run_script.sh")
            parser = _fetch_text(f"{run_script_base.rstrip('/')}/{instance_id}/parser.py")
        except Exception as exc:
            script_errors.append({"instance_id": instance_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append(_task_record(row, run_script=run_script, parser=parser, image_prefix=image_prefix))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in out:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "parquet_url": parquet_url,
        "output_path": str(output_path),
        "raw_count": len(records),
        "requested_count": len(selected),
        "written_count": len(out),
        "script_error_count": len(script_errors),
        "script_errors": script_errors[:10],
        "seed": seed,
        "keep_order": keep_order,
        "sample_size": sample_size,
    }


def _task_record(row: dict[str, Any], *, run_script: str, parser: str, image_prefix: str) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    repo = str(row.get("repo") or "")
    dockerhub_tag = str(row.get("dockerhub_tag") or "").strip()
    docker_image = f"{image_prefix}:{dockerhub_tag}"
    fail_to_pass = _parse_list(row.get("fail_to_pass"))
    pass_to_pass = _parse_list(row.get("pass_to_pass"))
    selected_tests = _parse_list(row.get("selected_test_files_to_run"))
    issue_body = _problem_statement(row)
    pro_spec = {
        "dataset": "ScaleAI/SWE-bench_Pro",
        "repo": repo,
        "repo_language": str(row.get("repo_language") or ""),
        "dockerhub_tag": dockerhub_tag,
        "selected_test_files_to_run": selected_tests,
        "before_repo_set_cmd": _clean_text(row.get("before_repo_set_cmd")),
        "run_script": run_script,
        "parser": parser,
        "run_script_source": f"{DEFAULT_RUN_SCRIPT_BASE}/{instance_id}/run_script.sh",
        "parser_source": f"{DEFAULT_RUN_SCRIPT_BASE}/{instance_id}/parser.py",
    }
    return {
        "task_id": instance_id,
        "instance_id": instance_id,
        "repo": repo,
        "repo_path": "/app",
        "base_commit": str(row.get("base_commit") or ""),
        "issue_title": instance_id,
        "issue_body": issue_body,
        "problem_statement": issue_body,
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
        "docker_image": docker_image,
        "sandbox": {
            "docker_image": docker_image,
            "repo_path": "/app",
            "workdir": "/app",
            "swebench_pro": pro_spec,
        },
        "metadata": {
            "repo": repo,
            "base_commit": str(row.get("base_commit") or ""),
            "swebench_pro": pro_spec,
        },
    }


def _problem_statement(row: dict[str, Any]) -> str:
    problem = _clean_text(row.get("problem_statement"))
    requirements = _clean_text(row.get("requirements"))
    interface = _clean_text(row.get("interface"))
    return f"{problem}\n\nRequirements:\n{requirements}\n\nNew interfaces introduced:\n{interface}".strip()


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in {"'", '"'} and stripped[-1] == stripped[0]:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except Exception:
                continue
            if isinstance(parsed, str):
                return parsed
    return text


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return [text]


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GraphPlanner tasks from SWE-bench Pro public data.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--keep-order", action="store_true")
    parser.add_argument("--parquet-url", default=DEFAULT_PARQUET_URL)
    parser.add_argument("--run-script-base", default=DEFAULT_RUN_SCRIPT_BASE)
    parser.add_argument("--image-prefix", default=DEFAULT_IMAGE_PREFIX)
    parser.add_argument("--instance-id", action="append", default=[])
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    stats = prepare_swebench_pro_sample(
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        parquet_url=args.parquet_url,
        run_script_base=args.run_script_base,
        image_prefix=args.image_prefix,
        instance_ids=args.instance_id,
        keep_order=args.keep_order,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
