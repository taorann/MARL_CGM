from __future__ import annotations

import json
import os
import base64
from pathlib import Path
import random
import re
import shlex
import subprocess
from typing import Any, Iterable, Mapping


PATCH_DIFF_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
PATCH_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")
TEST_FUNC_RE = re.compile(r"^[ +\-]\s*def\s+(test_[A-Za-z0-9_]+)\s*\(")
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"


def prepare_swebench_sample(
    source_path: Path,
    output_path: Path,
    *,
    sample_size: int = 100,
    seed: int = 20260601,
    ssh_target: str = "",
    ssh_args: str = "",
    require_remote_sif: bool = False,
    remote_sif_dir: str = "",
    keep_order: bool = False,
) -> dict[str, object]:
    records = _load_jsonl(source_path)
    available_images: set[str] | None = None
    if require_remote_sif:
        available_images = _available_sif_stems(ssh_target=ssh_target, ssh_args=ssh_args, remote_sif_dir=remote_sif_dir)
        records = [record for record in records if _record_image_stem(record) in available_images]
    if sample_size > 0 and sample_size < len(records):
        if keep_order:
            selected = records[:sample_size]
        else:
            rng = random.Random(seed)
            selected = rng.sample(records, sample_size)
    else:
        selected = list(records)

    remote_payloads: dict[str, dict[str, Any]] = {}
    if ssh_target:
        remote_paths = _dedupe(_safe_str((record.get("sandbox") or {}).get("r2e_ds_json")) for record in selected if isinstance(record.get("sandbox"), Mapping))
        if remote_paths:
            remote_payloads = _read_jsons_via_ssh(remote_paths, ssh_target=ssh_target, ssh_args=ssh_args)

    enriched: list[dict[str, Any]] = []
    stats = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "raw_input_count": len(_load_jsonl(source_path)),
        "available_input_count": len(records),
        "selected_count": len(selected),
        "with_eval_script": 0,
        "with_selectors": 0,
        "remote_instance_loaded": 0,
        "missing_instance_payload": 0,
        "seed": seed,
        "keep_order": keep_order,
        "require_remote_sif": require_remote_sif,
        "remote_sif_dir": remote_sif_dir,
    }
    for raw in selected:
        ds_path = _safe_str((raw.get("sandbox") or {}).get("r2e_ds_json")) if isinstance(raw.get("sandbox"), Mapping) else ""
        payload_override = remote_payloads.get(ds_path)
        record, meta = enrich_swebench_record(
            raw,
            ssh_target=ssh_target,
            ssh_args=ssh_args,
            instance_payload=payload_override,
            instance_loaded_from="ssh_batch" if payload_override is not None else "",
        )
        if meta.get("eval_script_present"):
            stats["with_eval_script"] += 1
        if int(meta.get("selector_count") or 0) > 0:
            stats["with_selectors"] += 1
        if meta.get("instance_loaded_from") in {"ssh", "ssh_batch"}:
            stats["remote_instance_loaded"] += 1
        if not meta.get("instance_loaded"):
            stats["missing_instance_payload"] += 1
        enriched.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in enriched:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return stats


def _record_image_stem(record: Mapping[str, Any]) -> str:
    sandbox = record.get("sandbox") if isinstance(record.get("sandbox"), Mapping) else {}
    image = _safe_str(
        record.get("docker_image")
        or record.get("image")
        or sandbox.get("docker_image")
        or sandbox.get("image")
        or sandbox.get("sif_path")
        or sandbox.get("sif_name")
    ).strip()
    if not image:
        return ""
    if image.endswith(".sif"):
        return Path(image).stem
    return image.replace("/", "-").replace(":", "-")


def _available_sif_stems(*, ssh_target: str, ssh_args: str, remote_sif_dir: str) -> set[str]:
    if not ssh_target.strip():
        raise ValueError("--require-remote-sif requires --ssh-target")
    directory = remote_sif_dir.strip() or "/appsnew/home/chongbin_pkuhpc/chongbin_cls/sif/sweb"
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6"]
    if ssh_args.strip():
        cmd.extend(shlex.split(ssh_args))
    remote = f"find {shlex.quote(directory)} -maxdepth 1 -name '*.sif' -printf '%f\\n'"
    cmd.extend([ssh_target, remote])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to list remote sif images rc={proc.returncode}: {proc.stderr.strip()[:1000]}")
    stems = {Path(line.strip()).stem for line in proc.stdout.splitlines() if line.strip()}
    if not stems:
        raise RuntimeError(f"no .sif images found in remote directory {directory}")
    return stems


def enrich_swebench_record(
    raw: Mapping[str, Any],
    *,
    ssh_target: str = "",
    ssh_args: str = "",
    instance_payload: Mapping[str, Any] | None = None,
    instance_loaded_from: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = dict(raw)
    sandbox = dict(record.get("sandbox") or {})
    issue = dict(record.get("issue") or {})
    metadata = dict(record.get("metadata") or {})
    repo = _safe_str(record.get("repo") or sandbox.get("repo") or issue.get("repo"))

    if instance_payload is not None:
        payload, loaded_from = dict(instance_payload), instance_loaded_from or "provided"
    else:
        payload, loaded_from = _load_instance_payload(sandbox, issue, ssh_target=ssh_target, ssh_args=ssh_args)
    tests_field = payload.get("tests") if isinstance(payload, Mapping) else None
    test_patch = _safe_str(payload.get("test_patch") if isinstance(payload, Mapping) else "")
    base_commit = _safe_str(payload.get("base_commit") if isinstance(payload, Mapping) else record.get("base_commit"))
    if not repo and isinstance(payload, Mapping):
        repo = _safe_str(payload.get("repo"))

    fail_to_pass = _extract_fail_to_pass_from_tests_field(tests_field)
    pass_to_pass = _extract_pass_to_pass_from_tests_field(tests_field)
    all_tests = _extract_from_tests_field(tests_field)
    patch_function_selectors = _extract_function_selectors_from_test_patch(test_patch)
    patch_path_selectors = _extract_paths_from_test_patch(test_patch)

    source = ""
    selectors: list[str] = []
    if fail_to_pass:
        selectors = fail_to_pass
        source = "instance.tests.fail_to_pass"
    elif all_tests:
        selectors = all_tests
        source = "instance.tests"
    elif patch_function_selectors:
        selectors = patch_function_selectors
        source = "instance.test_patch.function_selectors"
    elif patch_path_selectors:
        selectors = patch_path_selectors
        source = "instance.test_patch.path_selectors"

    selectors = _dedupe(_normalize_selector(s) for s in selectors)
    pass_to_pass = _dedupe(_normalize_selector(s) for s in pass_to_pass)

    spec = dict(metadata.get("swebench_spec") or record.get("swebench_spec") or sandbox.get("swebench_spec") or {})
    if repo:
        spec.setdefault("repo", repo)
    spec.setdefault("language", "py")
    if base_commit:
        record["base_commit"] = base_commit
        metadata.setdefault("base_commit", base_commit)
    if selectors and not spec.get("eval_script_list"):
        spec["eval_script_list"] = _build_eval_script_list(
            selectors,
            repo=repo,
            base_commit=base_commit,
            test_patch=test_patch,
        )
    if spec:
        sandbox["swebench_spec"] = spec
        metadata["swebench_spec"] = spec
        record["swebench_spec"] = spec

    if selectors:
        record["FAIL_TO_PASS"] = selectors
        record["target_fail_to_pass_selectors"] = selectors
        metadata["target_test_selectors"] = selectors
        metadata["target_test_source"] = source
        if patch_function_selectors:
            metadata["target_test_function_selectors"] = patch_function_selectors
    if pass_to_pass:
        record["PASS_TO_PASS"] = pass_to_pass
        metadata["target_pass_to_pass_selectors"] = pass_to_pass
    if repo:
        record["repo"] = repo
        metadata.setdefault("repo", repo)
    if sandbox.get("docker_image") and "docker_image" not in record:
        record["docker_image"] = sandbox["docker_image"]

    metadata["swebench_enrichment"] = {
        "instance_loaded": bool(payload),
        "instance_loaded_from": loaded_from,
        "selector_source": source,
        "selector_count": len(selectors),
        "eval_script_present": bool(spec.get("eval_script_list")),
        "test_patch_present": bool(test_patch.strip()),
    }
    record["metadata"] = metadata
    record["sandbox"] = sandbox
    if issue:
        record["issue"] = issue
    return record, metadata["swebench_enrichment"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            out.append(item)
    return out


def _load_instance_payload(
    sandbox: Mapping[str, Any], issue: Mapping[str, Any], *, ssh_target: str = "", ssh_args: str = ""
) -> tuple[dict[str, Any], str]:
    ds_path = _safe_str(sandbox.get("r2e_ds_json"))
    if ds_path:
        local = _read_json_local(Path(ds_path).expanduser())
        if local is not None:
            return local, "local"
        if ssh_target:
            remote = _read_json_via_ssh(ds_path, ssh_target=ssh_target, ssh_args=ssh_args)
            if remote is not None:
                return remote, "ssh"
    instance_id = _safe_str(issue.get("id") or issue.get("instance_id") or sandbox.get("instance_id"))
    if instance_id:
        local = _find_local_instance_payload(instance_id)
        if local is not None:
            return local, "local_search"
    return {}, ""


def _read_json_local(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _read_json_via_ssh(path: str, *, ssh_target: str, ssh_args: str = "", timeout_s: int = 20) -> dict[str, Any] | None:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6"]
    if ssh_args.strip():
        cmd.extend(shlex.split(ssh_args))
    cmd.extend([ssh_target, f"cat {shlex.quote(path)}"])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _read_jsons_via_ssh(paths: list[str], *, ssh_target: str, ssh_args: str = "", timeout_s: int = 120) -> dict[str, dict[str, Any]]:
    clean_paths = _dedupe(paths)
    if not clean_paths:
        return {}
    code = r'''
import base64, json, pathlib, sys
paths = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
out = {}
for p in paths:
    try:
        payload = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            out[p] = payload
    except Exception:
        pass
print(json.dumps(out, ensure_ascii=False))
'''
    encoded = base64.b64encode(json.dumps(clean_paths).encode("utf-8")).decode("ascii")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6"]
    if ssh_args.strip():
        cmd.extend(shlex.split(ssh_args))
    cmd.extend([ssh_target, "python -c " + shlex.quote(code) + " " + shlex.quote(encoded)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(data, Mapping):
        for path, payload in data.items():
            if isinstance(path, str) and isinstance(payload, Mapping):
                out[path] = dict(payload)
    return out


def _find_local_instance_payload(instance_id: str) -> dict[str, Any] | None:
    roots: list[Path] = []
    for part in _safe_str(os.environ.get("GP_SWEBENCH_INSTANCE_DIRS")).split(":"):
        if part.strip():
            roots.append(Path(part.strip()).expanduser())
    roots.extend(
        [
            Path("../datasets/swebench/instances"),
            Path("datasets/swebench/instances"),
            Path("/root/private_data/MARL_CGM-main/datasets/swebench/instances"),
            Path("/root/private_data/MARL_CGM/datasets/swebench/instances"),
            Path("/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM/datasets/swebench/instances"),
        ]
    )
    for root in roots:
        payload = _read_json_local(root / f"{instance_id}.json")
        if payload is not None:
            return payload
    return None


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = _safe_str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_selector(value: str) -> str:
    s = _safe_str(value).strip().replace("\\", "/")
    for prefix in ("/testbed/", "/repo/"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    if s.startswith("./"):
        s = s[2:]
    if s.startswith("a/") or s.startswith("b/"):
        s = s[2:]
    return s.strip()


def _is_test_selector(selector: str) -> bool:
    s = _normalize_selector(selector).lower()
    if not s:
        return False
    return "::" in s or "/tests/" in s or s.startswith("tests/") or s.endswith("_test.py") or s.endswith("test.py")


def _extract_from_tests_field(tests: Any) -> list[str]:
    out: list[str] = []
    if tests is None:
        return out
    if isinstance(tests, str):
        raw = tests.strip()
        if not raw:
            return out
        if raw.startswith("{") or raw.startswith("["):
            try:
                return _extract_from_tests_field(json.loads(raw))
            except Exception:
                pass
        return _dedupe(_normalize_selector(part) for part in re.split(r"[,\n]+", raw))
    if isinstance(tests, list):
        return _dedupe(_normalize_selector(_safe_str(item)) for item in tests)
    if isinstance(tests, Mapping):
        for key in ("fail_to_pass", "FAIL_TO_PASS", "failing", "failed", "tests"):
            out.extend(_extract_from_tests_field(tests.get(key)))
        return _dedupe(out)
    return out


def _extract_fail_to_pass_from_tests_field(tests: Any) -> list[str]:
    if not isinstance(tests, Mapping):
        return []
    out: list[str] = []
    for key in ("fail_to_pass", "FAIL_TO_PASS", "failing", "failed"):
        out.extend(_extract_from_tests_field(tests.get(key)))
    return _dedupe(out)


def _extract_pass_to_pass_from_tests_field(tests: Any) -> list[str]:
    if not isinstance(tests, Mapping):
        return []
    out: list[str] = []
    for key in ("pass_to_pass", "PASS_TO_PASS", "passing"):
        out.extend(_extract_from_tests_field(tests.get(key)))
    return _dedupe(out)


def _extract_paths_from_test_patch(test_patch: str) -> list[str]:
    paths: list[str] = []
    for raw in _safe_str(test_patch).splitlines():
        line = raw.strip()
        match = PATCH_DIFF_RE.match(line)
        if match:
            candidate = _normalize_selector(match.group(2))
            if _is_test_selector(candidate):
                paths.append(candidate)
            continue
        match = PATCH_FILE_RE.match(line)
        if match:
            candidate = _normalize_selector(match.group(1))
            if _is_test_selector(candidate):
                paths.append(candidate)
    return _dedupe(paths)


def _extract_function_selectors_from_test_patch(test_patch: str) -> list[str]:
    selectors: list[str] = []
    current_path = ""
    current_test = ""
    for raw in _safe_str(test_patch).splitlines():
        line = raw.rstrip("\n")
        match = PATCH_DIFF_RE.match(line.strip())
        if match:
            candidate = _normalize_selector(match.group(2))
            current_path = candidate if _is_test_selector(candidate) else ""
            current_test = ""
            continue
        match = PATCH_FILE_RE.match(line.strip())
        if match:
            candidate = _normalize_selector(match.group(1))
            current_path = candidate if _is_test_selector(candidate) else ""
            current_test = ""
            continue
        if not current_path:
            continue
        if line.startswith("@@"):
            current_test = ""
            continue
        match = TEST_FUNC_RE.match(line)
        if match:
            current_test = match.group(1)
        if current_test and line[:1] in {"+", "-"}:
            selectors.append(f"{current_path}::{current_test}")
    return _dedupe(selectors)


def _repo_test_command(repo: str) -> str:
    if repo == "django/django":
        return "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
    if repo == "astropy/astropy":
        return "pytest -rA -vv -o console_output_style=classic --tb=no"
    if repo == "sympy/sympy":
        return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
    if repo == "sphinx-doc/sphinx":
        return "tox --current-env -epy39 -v --"
    if repo == "mwaskom/seaborn":
        return "pytest --no-header -rA"
    return "pytest -rA"


def _selector_to_repo_directive(selector: str, repo: str) -> str:
    file_part = _normalize_selector(selector).split("::", 1)[0]
    if repo == "django/django":
        if file_part.endswith(".py"):
            file_part = file_part[:-3]
        if file_part.startswith("tests/"):
            file_part = file_part[len("tests/") :]
        return file_part.replace("/", ".").strip(".")
    return file_part


def _build_eval_script_list(selectors: list[str], *, repo: str, base_commit: str, test_patch: str) -> list[str]:
    selected = _dedupe(_selector_to_repo_directive(s, repo) for s in selectors[:32])
    if not selected:
        return []
    command = f"{_repo_test_command(repo)} {' '.join(shlex.quote(s) for s in selected)}".strip()
    lines = [
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
    ]
    test_paths = _extract_paths_from_test_patch(test_patch)
    reset_cmd = ""
    if base_commit and test_paths:
        quoted_paths = " ".join(shlex.quote(p) for p in test_paths)
        reset_cmd = f"git checkout {shlex.quote(base_commit)} -- {quoted_paths} || true"
        lines.append(reset_cmd)
    if test_patch.strip():
        delimiter = "EOF_GRAPHPLANNER_TEST_PATCH"
        lines.extend(
            [
                f"cat >/tmp/graphplanner_test.patch <<'{delimiter}'",
                test_patch,
                delimiter,
                (
                    "if git apply --check /tmp/graphplanner_test.patch >/dev/null 2>&1; then "
                    "git apply --verbose --reject /tmp/graphplanner_test.patch; "
                    "else echo '[graphplanner] test_patch already applied or not applicable; continuing'; fi"
                ),
            ]
        )
    lines.extend(
        [
            "TEST_RC=0",
            f": '{START_TEST_OUTPUT}'",
            f"{command} || TEST_RC=$?",
            f": '{END_TEST_OUTPUT}'",
        ]
    )
    if reset_cmd:
        lines.append(reset_cmd)
    lines.append("exit $TEST_RC")
    return lines
