# Clean Rebuild Runbook

This runbook explains how to start a clean GraphPlanner rebuild from the asset documents while keeping the old repository available as read-only reference.

## Goal

Create a new train-free code-repair agent without inheriting the current repository's legacy structure.

The new implementation should:

- use `docs/agent_assets/` as the primary design source;
- avoid rLLM, Verl, PPO, GRPO, and reward-training dependencies;
- split modules cleanly instead of recreating the monolithic `PlannerEnv`;
- optionally inspect the old repository only when a capability needs implementation detail;
- support both mock CGM and real HTTP CGM backends.

## Create A Clean Root

Recommended new project path:

```bash
mkdir -p /root/private_data/graphplanner-rebuild/docs
cp /root/private_data/MARL_CGM-main/AGENTS.md /root/private_data/graphplanner-rebuild/
cp -r /root/private_data/MARL_CGM-main/docs/agent_assets /root/private_data/graphplanner-rebuild/docs/
```

Create a clean initial source layout:

```bash
mkdir -p /root/private_data/graphplanner-rebuild/{src/graphplanner_agent,tests,scripts,configs,references}
```

Recommended final shape:

```text
graphplanner-rebuild/
  AGENTS.md
  docs/agent_assets/
  src/graphplanner_agent/
  tests/
  scripts/
  configs/
  references/
```

## Add The Old Repo As Read-Only Reference

Use a symlink instead of copying the old repository:

```bash
mkdir -p /root/private_data/graphplanner-rebuild/references
ln -s /root/private_data/MARL_CGM-main /root/private_data/graphplanner-rebuild/references/MARL_CGM-main
```

Check it:

```bash
ls -l /root/private_data/graphplanner-rebuild/references
```

Expected shape:

```text
MARL_CGM-main -> /root/private_data/MARL_CGM-main
```

Meaning:

- `references/MARL_CGM-main` is only a shortcut to the old repo;
- it does not duplicate files;
- Codex can inspect old code when needed;
- the new project remains structurally clean.

## Reference Policy For Codex

Add or keep this rule in the clean project's `AGENTS.md`:

```text
Old repository access policy:
- `references/MARL_CGM-main/` is read-only reference.
- Primary design source is `docs/agent_assets/`.
- When migrating a capability, first check `docs/agent_assets/09_capability_matrix_audit.md`, then inspect the referenced old files only for implementation detail.
- Do not copy old rLLM dependencies, `PlannerEnv` monolith structure, historical rule hacks, or benchmark-test-source repair behavior.
```

## Startup Prompt For Codex

Use this prompt when starting the rebuild:

```text
Please implement a clean train-free GraphPlanner code-repair agent in this repository.

Primary design source:
- `AGENTS.md`
- `docs/agent_assets/README.md`
- `docs/agent_assets/01_overall_design.md`
- `docs/agent_assets/02_action_state_protocol.md`
- `docs/agent_assets/03_module_blueprint.md`
- `docs/agent_assets/07_rebuild_checklist.md`
- `docs/agent_assets/09_capability_matrix_audit.md`

`references/MARL_CGM-main/` is read-only reference only. If you need old implementation details, inspect only the specific files named in `09_capability_matrix_audit.md`.

Hard requirements:
- Do not depend on rLLM, Verl, PPO, GRPO, or reward-training code.
- Do not recreate the old monolithic `PlannerEnv`.
- Split planner client, planner loop, action parser, environment, graph retrieval, memory, repair, runtime, dataset loading, and telemetry into separate modules.
- Do not use benchmark test source as repair evidence. Tests are behavior/fail-to-pass symptoms only.
- Implement both `CGM_BACKEND=mock` and `CGM_BACKEND=http`.
- Make patch apply safe: validate, snapshot, apply, run fail-to-pass, rollback on failure.
- Use `09_capability_matrix_audit.md` as the acceptance matrix before declaring the rebuild complete.
```

## If Only Planner Is Running

If the planner model is available but CGM is not running, work can still proceed.

Use mock CGM:

```bash
export CGM_BACKEND=mock
```

Valid test categories:

```text
1. Unit tests:
   action parser, graph builder, retrieval, memory, patch validator, telemetry.

2. Planner dry-run:
   run_failed_test -> explore_find -> explore_expand/read -> memory_commit.
   Stop before real repair, or route repair through mock CGM.

3. Mock repair chain:
   repair -> mock patch -> validate -> snapshot -> apply -> fail-to-pass -> rollback or verified auto-finish.
```

This lets Codex verify most architecture and control-flow behavior without a loaded CGM model.

## When Real CGM Is Running

Use HTTP CGM:

```bash
export CGM_BACKEND=http
export CGM_ENDPOINT=http://127.0.0.1:30001/generate
```

Default assumption for the current rebuild is that the real CGM service runs in
the same container as the agent. Only override `CGM_ENDPOINT` to a non-local
host when you intentionally deploy CGM elsewhere. Historical traces may still
contain old `172.*:30001` endpoints; those are legacy records, not the current
recommended setup.

Smoke-check the CGM service from the machine that will call it:

```bash
python - <<'PY'
import json
import os
import urllib.request

endpoint = os.environ.get("CGM_ENDPOINT", "http://127.0.0.1:30001/generate")
base = endpoint.rsplit("/", 1)[0]

for method, url, body in [
    ("GET", base + "/healthz", None),
    ("POST", endpoint, {
        "issue": {"title": "smoke", "body": "smoke"},
        "plan": "Return a minimal harmless edit for smoke testing.",
        "graph": {
            "nodes": [{"id": "n1", "type": "function", "path": "a.py", "name": "main", "text": "def main():\\n    pass\\n"}],
            "edges": [],
        },
        "snippets": [{"path": "a.py", "start": 1, "end": 2, "text": "def main():\\n    pass\\n"}],
        "constraints": {"max_edits": 1},
    }),
]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            print(method, url, "->", resp.status)
            print(resp.read(500).decode("utf-8", "ignore"))
    except Exception as exc:
        print(method, url, "-> ERROR:", repr(exc))
PY
```

Only run full end-to-end repair after both checks pass.

## Migration Pattern

Migrate by capability, not by file.

Examples:

```text
Old `PlannerEnv._handle_read`
  -> New `src/graphplanner_agent/env/actions/read.py`

Old `graph_planner/integrations/rllm/swebench_meta.py`
  -> New `src/graphplanner_agent/datasets/swebench_meta.py`

Old `graph_planner/integrations/codefuse_cgm/client.py`
  -> New `src/graphplanner_agent/repair/cgm_client.py`

Old `graph_planner/runtime/sandbox.py`
  -> New `src/graphplanner_agent/runtime/sandbox.py`
```

Before implementing each capability:

1. Read the corresponding row in `09_capability_matrix_audit.md`.
2. Read the relevant design doc in `docs/agent_assets/`.
3. Inspect old reference code only if a behavior detail is unclear.
4. Implement in the new modular structure.
5. Add a unit or integration test that matches the acceptance matrix.

## First Milestones

Recommended build order:

1. Config loader and structured action schema.
2. Planner client with dry-run/mock response option.
3. Action parser and one-step planner loop.
4. Sandbox interface with local fake sandbox.
5. Graph model, graph builder stub, find/expand/read memory path.
6. W/M/T memory manager with hydrated code.
7. Mock CGM backend and safe patch apply/rollback.
8. HTTP CGM backend.
9. Fail-to-pass runner and test-source guard.
10. Telemetry: JSONL plus readable `trace.md`.
11. One real SWE-style end-to-end task.

## Completion Gate

Before treating the rebuild as equivalent, check every row in:

```text
docs/agent_assets/09_capability_matrix_audit.md
```

and every item in:

```text
docs/agent_assets/07_rebuild_checklist.md
```
