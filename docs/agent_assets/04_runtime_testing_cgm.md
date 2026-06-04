# Runtime, Testing, And CGM Repair

## Sandbox Contract

Every backend must implement:

- `start(task)`;
- `stop()`;
- `run(cmd, timeout, cwd, env)`;
- `read_file(path, start?, end?)`;
- `write_file(path, content)` or `apply_patch(patch)`;
- `snapshot()` and `rollback(snapshot)`;
- `build_graph(options)`;
- `run_fail_to_pass(task_spec)`.

The agent should not care whether the backend is local checkout, Docker, Apptainer, or remote SSH.

## remote_swe Backend

The current deployment uses a reverse SSH tunnel:

```text
current container -> ssh chongbin_cls@localhost -p 40022 -> remote SWE runner -> Apptainer container
```

Required configuration:

- `sandbox_backend=remote_swe`;
- `sandbox_ssh_target=chongbin_cls@localhost`;
- `GP_REMOTE_SWE_SSH_ARGS="-i /root/.ssh/id_ed25519_login24 -p 40022 ..."`; 
- `sandbox_remote_repo=/appsnew/home/chongbin_pkuhpc/chongbin_cls/MARL_CGM`.

A clean implementation should treat this as one runtime plugin, not as core agent logic.

## Fail-To-Pass Testing

Testing must follow SWE-bench task metadata when possible:

- prefer official `eval_script_list` when available, because it encodes repo/version-specific commands;
- use official fail-to-pass selectors;
- use pass-to-pass selectors when available;
- if selectors are missing, derive minimal selectors from task patches or official harness metadata;
- avoid running full test suites by default because they are slow and can introduce unrelated environment noise.

Do not assume every container uses pytest. SWE-bench official specs may call Django `runtests.py`, tox, SymPy `bin/test`, go test, ctest, make check, or language-specific runners. Evaluation should parse logs with the official repo-specific parser when possible and expose per-test `tests_status` plus `resolved`.

For `remote_swe`, custom repro commands and pytest fallbacks must be wrapped so they execute from `/testbed` with the container/testbed Python. In particular, sanitize `PATH`, unset `PYTHONHOME`, set `PYTHONNOUSERSITE=1`, activate the container `testbed` conda environment when present, and fail as `infra_bug` if `sys.executable` resolves under a host HOME path such as `/home/.../miniconda3`. This prevents shell startup files from making a host Python import the repository and falsely turning environment breakage into patch failure.

The planner should see:

- behavior summary;
- failing exception/message;
- implementation traceback frames;
- target selector names only as test metadata, not as search anchors.

The planner should not see:

- benchmark test source code;
- test helper code;
- assertion helper implementations;
- test-file snippets.

## Failure Semantics

Separate these outcomes:

- `test_failed`: agent patch did not fix fail-to-pass.
- `syntax_failed`: patch made code invalid.
- `patch_rejected`: schema/range/path validation failed.
- `infra_bug`: SSH/container/service/test harness error.
- `timeout`: step or trajectory timed out.

Only `test_failed`, `syntax_failed`, and `patch_rejected` should be fed back as model-repair evidence. `infra_bug` should be counted separately in evaluation.

## CGM Input

CGM payload should contain:

```json
{
  "issue": {
    "title": "...",
    "body": "...",
    "failure_summary": "...",
    "failure_frame": {"path": "...", "line": 123}
  },
  "plan": "short grounded plan",
  "graph": {
    "nodes": [
      {
        "id": "node-id",
        "nodeType": "Function",
        "name": "symbol",
        "path": "pkg/mod.py",
        "start_line": 1,
        "end_line": 20,
        "text": "full code body or high-quality snippet"
      }
    ],
    "edges": [
      {"source": "file::pkg/mod.py", "target": "node-id", "type": "CONTAINS"},
      {"source": "a", "target": "b", "type": "CALLS"}
    ],
    "reponame": "repo",
    "language": "python"
  },
  "snippets": [
    {"path": "pkg/mod.py", "start": 1, "end": 20, "text": "..."}
  ],
  "constraints": {
    "max_edits": 4,
    "implementation_only": true,
    "no_test_changes": true
  },
  "prior_repair_feedback": "optional compact feedback"
}
```

## Graph-Aware CGM Requirement

The target CGM runtime should use:

- node text encoding;
- graph node embeddings;
- adjacency matrix or equivalent edge-aware attention;
- prompt/code embeddings fused with graph embeddings.

Do not confuse a text list of graph nodes with graph-aware input. If the CGM service is only receiving a linearized text prompt, document it as a fallback, not the main path.

## CGM Output

Support two output channels:

1. JSON edits:

```json
{
  "patch": {
    "edits": [
      {"path": "pkg/mod.py", "start": 10, "end": 12, "new_text": "..."}
    ],
    "summary": "..."
  }
}
```

2. Unified diff:

```diff
diff --git a/pkg/mod.py b/pkg/mod.py
...
```

Normalize both into the same internal patch schema.

## Patch Validation

Before applying:

- only implementation files unless explicitly allowed;
- no benchmark test files;
- paths must exist or be intentional new files;
- ranges must be valid;
- `new_text` must preserve final newline where needed;
- reject duplicate patch attempts;
- reject diff markers embedded inside code edits;
- reject broad unrelated file rewrites unless task requires it.

After applying:

- run syntax check for touched Python files;
- run target fail-to-pass;
- optionally run pass-to-pass;
- if failed, rollback.

## Repair Feedback To Planner

Report compactly:

- patch applied or rolled back;
- touched paths;
- syntax/test result;
- failure excerpt;
- whether fail-to-pass changed;
- whether memory had code bodies;
- recommended next evidence direction if repair was blocked.

Do not turn post-repair test names into new search anchors.
