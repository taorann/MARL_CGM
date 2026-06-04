# GraphPlanner Agent Rebuild

This directory contains a clean, train-free GraphPlanner rebuild. It keeps new code under
`agent_rebuild/` and treats the rest of the repository as read-only reference.

## Current Entry Points

Run unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Check the stack with mock CGM:

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.check_stack \
  --skip-planner \
  --cgm-backend mock
```

Check a local OpenAI-compatible planner plus mock CGM:

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.check_stack \
  --planner-endpoint http://127.0.0.1:30000/v1/chat/completions \
  --planner-model models/Qwen3-32B \
  --planner-tool-smoke \
  --cgm-backend mock
```

Use `--print-raw` when you need the raw planner message; the default output is
compact and shows whether tool use and visible thinking were parsed.

Run an eval task file:

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.eval \
  --tasks tasks.jsonl \
  --planner-endpoint http://127.0.0.1:30000/v1/chat/completions \
  --planner-model models/Qwen3-32B \
  --planner-tool-calling \
  --cgm-backend mock \
  --results-path runs/results.jsonl \
  --trace-dir runs/traces \
  --progress-path runs/progress.md \
  --verbose
```

Task records are JSON or JSONL objects with at least:

```json
{
  "task_id": "local-issue",
  "repo_path": "/path/to/checkout",
  "issue_title": "short title",
  "issue_body": "problem statement",
  "test_command": "python -m unittest discover -p 'test_*.py'"
}
```

For mock repair, set `CGM_MOCK_RESPONSE` to JSON or `@/path/to/response.json`.

Remote SWE backend wiring is available for the Polaris/Slurm deployment:

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.check_stack \
  --skip-planner \
  --cgm-backend mock \
  --sandbox-backend remote_swe \
  --remote-swe-smoke
```

The backend calls the remote repo's existing `hpc/swe_proxy.py`; it does not copy
or import the legacy agent code locally. When a real image is available, add
`--remote-swe-image <docker-image>` to run `start -> python -V -> stop`.
If `/root/.ssh/id_ed25519_login24` exists, the default Polaris tunnel args are
filled automatically; otherwise set `GP_REMOTE_SWE_SSH_ARGS`.

The remote runners execute Singularity/Apptainer `.sif` files. The CLI and task
loader accept either the historical image key or a `.sif` name/path, for example:

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.check_stack \
  --skip-planner \
  --cgm-backend mock \
  --sandbox-backend remote_swe \
  --remote-swe-smoke \
  --remote-swe-image /appsnew/home/chongbin_pkuhpc/chongbin_cls/sif/sweb/example.sif
```

To exercise every container-backed operation, including read/write, snapshot,
patch application, rollback, fail-to-pass, and repo graph construction:

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.check_stack \
  --skip-planner \
  --cgm-backend mock \
  --sandbox-backend remote_swe \
  --remote-swe-full-smoke \
  --remote-swe-image /appsnew/home/chongbin_pkuhpc/chongbin_cls/sif/sweb/example.sif
```

Remote repo graph payloads are cached locally under `runs/graph_cache` by
default. Set `GRAPHPLANNER_GRAPH_CACHE_DIR` to move the cache, or
`GRAPHPLANNER_DISABLE_GRAPH_CACHE=1` to force fresh remote graph builds.

## Implemented

- Planner action protocol and visible-thinking stripping.
- Local sandbox runtime with read/run/snapshot/rollback/test.
- Python AST graph build/search/expand/read plus filesystem fallback mapped to graph nodes.
- W/M/T memory and hydration before CGM.
- CGM payload builder, mock/http clients, and payload validation.
- JSON edit and simple unified-diff patch normalization.
- Patch validation, syntax check, fail-to-pass retest, rollback on failure.
- JSONL/Markdown telemetry and progress summaries.
- Remote SWE SSH/proxy backend for Slurm/Apptainer runners.
- Direct `.sif` reference normalization for remote SWE tasks.
- Remote SWE full-smoke CLI and local graph payload cache.
- SWE-bench-style task metadata and selector loading.
- Compact terminal output for stack checks, eval progress, and planner step summaries.

## Deployment Notes

- The current real CGM service is expected to run inside this same container by
  default, typically at `http://127.0.0.1:30001/generate`.
- Older traces under `runs/tmp/` may still mention historical remote CGM
  endpoints such as `172.*:30001`; treat those as archival run metadata, not as
  the current deployment recommendation.

## Still To Build

- Full SWE-bench metadata loader and selector enrichment.
- Docker/local Apptainer runtimes and real remote_swe image-level smoke.
- Tree-sitter graph frontend and richer call resolution.
- Full git-diff parser for complex rename/binary/multi-file edge cases.
- Broader real CGM graph-runtime self-check coverage on top of the current
  in-container deployment.
