# Clean Module Blueprint

The current code works but too much orchestration lives in a few large files. A clean reimplementation should split modules by responsibility.

## Recommended Package Layout

```text
code_agent/
  cli/
    eval.py
    smoke.py
    check_stack.py
  config/
    schema.py
    defaults.yaml
  planner/
    client.py
    prompt.py
    protocol.py
    response_parser.py
    loop.py
  env/
    stepper.py
    observations.py
    action_handlers.py
    guards.py
  graph/
    build.py
    schema.py
    search.py
    expand.py
    read.py
    rank.py
  memory/
    working.py
    cgm_memory.py
    text_notes.py
    hydration.py
  repair/
    cgm_client.py
    cgm_context.py
    patch_schema.py
    patch_apply.py
    patch_quality.py
    retry_policy.py
  runtime/
    sandbox_base.py
    local_repo.py
    docker_repoenv.py
    apptainer_queue.py
    remote_swe.py
    test_runner.py
  telemetry/
    events.py
    trace_markdown.py
    progress.py
  datasets/
    swebench.py
    task_spec.py
```

## Module Responsibilities

### `planner/`

Owns planner model I/O only:

- chat request construction;
- thinking-mode template options;
- action protocol prompt;
- parsing text/tool-call outputs;
- stripping visible thinking before formal state update;
- retrying malformed model outputs.

It must not execute repository commands or apply patches.

### `env/`

Owns the episode state machine:

- dispatch actions to handlers;
- update W/M/T/test state;
- enforce guards;
- build observations;
- decide terminal status.

This should be thin. It should not contain graph builder, CGM collator, patch parser, SSH runtime, or telemetry formatting logic.

### `graph/`

Owns repository understanding:

- build graph from source;
- support AST/tree-sitter frontends;
- generate file/class/function/method/assignment nodes;
- extract contains/calls/uses/imports/sibling edges;
- search nodes from implementation queries;
- map filesystem fallback spans back to graph nodes;
- read file/class/function windows.

### `memory/`

Owns W/M/T:

- W stores candidates and read snippets.
- M stores CGM evidence subset.
- T stores planner notes.
- Hydration guarantees selected M nodes have full code bodies.

### `repair/`

Owns CGM and patch lifecycle:

- build graph-aware CGM payload;
- normalize graph to official CGM schema;
- call CGM HTTP service;
- parse JSON edits or unified diff;
- validate path/range/newline/schema;
- apply with snapshot;
- run syntax and fail-to-pass tests;
- roll back failures;
- summarize repair feedback.

### `runtime/`

Owns execution:

- checkout/reset task repository;
- run shell commands;
- read files;
- apply patches;
- run target tests;
- manage remote container sessions.

### `telemetry/`

Owns recordkeeping:

- JSONL event stream;
- markdown trace;
- progress dashboard;
- per-issue result summary;
- visible planner thinking as logs only.

## Dependency Direction

Keep dependencies one-way:

```text
cli -> env -> planner/graph/memory/repair/runtime/telemetry
repair -> runtime + telemetry
memory -> graph schema
graph -> runtime file reads only when needed
planner -> no internal dependencies except protocol/config
```

Avoid circular dependencies like `env` importing rLLM wrappers or CGM service importing environment internals.

## Current Code To Split In Future

The current `graph_planner/env/planner_env.py` mixes several responsibilities. In a clean rewrite, split these clusters:

- action dispatch -> `env/action_handlers.py`;
- observation formatting -> `env/observations.py`;
- run_failed_test summarization -> `runtime/test_runner.py`;
- repair guards -> `env/guards.py`;
- CGM payload construction -> `repair/cgm_context.py`;
- issue payload construction -> `datasets/task_spec.py`;
- read/hydration helpers -> `graph/read.py` and `memory/hydration.py`;
- patch quality/syntax retry -> `repair/patch_quality.py`;
- progress entry formatting -> `telemetry/progress.py`.

## rLLM-Free Replacement

Replace rLLM adapter responsibilities with:

- `planner.client.OpenAIPlannerClient` for API calls;
- `planner.loop.PlannerLoop` for multi-step orchestration;
- `env.stepper.CodeRepairEnv` for state transitions;
- `telemetry.TraceWriter` for trajectory records.

No new module should import `rllm`.
