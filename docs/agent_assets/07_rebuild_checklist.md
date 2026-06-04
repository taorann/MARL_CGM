# Rebuild Checklist

Use this checklist when implementing the agent from these assets.

## Phase 1: Minimal Local Agent

- Implement task spec loader with issue text, repo, base commit, fail-to-pass/pass-to-pass selectors.
- Implement local sandbox checkout/reset/read/run/apply/rollback.
- Implement planner client for OpenAI-compatible chat endpoint.
- Implement action parser for JSON/text protocol.
- Implement `run_failed_test`, `explore_find`, `read`, `memory_commit`, and `repair`; verified repair ends the episode.
- Implement telemetry JSONL and markdown trace.
- Run one synthetic repo bug end-to-end.

Acceptance:

- The planner can read implementation code.
- M contains full code for committed nodes.
- Repair is blocked when M is empty.
- Patch failure rolls back.

## Phase 2: Repository Graph

- Build file/class/function/method/assignment graph.
- Add contains/calls/uses/imports/sibling edges.
- Add graph search and expand.
- Add fallback span-to-node mapping.
- Add test-code guard.

Acceptance:

- Searching a symbol returns graph nodes, not raw grep text.
- Fallback text hit maps to covering function/class/assignment.
- Expand from a function shows meaningful callees/siblings when available.

## Phase 3: CGM Integration

- Implement CGM HTTP client.
- Build official graph schema payload.
- Include M nodes, edges, snippets, issue, plan, constraints.
- Parse JSON edits and unified diff.
- Validate/apply/test/rollback patch.

Acceptance:

- `/generate` smoke request succeeds.
- CGM receives nodes with code bodies.
- Graph payload includes edges.
- Invalid patches are rejected before corrupting workspace.

## Phase 4: SWE-bench Runtime

- Implement official SWE-bench eval script execution and fail-to-pass/pass-to-pass result parsing.
- Add Docker/Apptainer/remote_swe runtime plugin as needed.
- Add progress md with pass/not_pass/bug counts.
- Add per-issue result JSON.

Acceptance:

- Missing image is counted as `bug`, not `not_pass`.
- Infra errors do not overwrite issue semantics.
- Every 10 issues progress shows total accuracy and bug-excluded accuracy.

## Phase 5: Hardening

- Add max-step and trajectory timeout.
- Add anti-loop for repeated no-delta actions.
- Add repair memory quality gate.
- Add CGM runtime-mode self-check.
- Add visible thinking logging that is not fed into next step.
- Add compact trace with planner input, action, observation, read snippets, CGM payload summary, patch, test result.

Acceptance:

- A trajectory can be audited from logs without re-running.
- The agent does not search/read/patch benchmark tests.
- Failed repair leads to deeper implementation exploration, not repeated repair.

## Suggested Smoke Tests

1. Local fake repo where bug is in a single function.
2. Local fake repo where bug is in assignment/registry.
3. Local fake repo where public API calls helper.
4. One SWE-bench issue with known image and pytest.
5. One issue where fail-to-pass selector names look like test helper traps.

## Definition Of Done

The rebuilt agent is functionally equivalent when it can:

- run a train-free planner loop;
- build and navigate implementation graph;
- keep W/M/T state correctly;
- feed graph-aware CGM with full code evidence;
- apply and verify patches with rollback;
- produce auditable telemetry;
- run without rLLM imports.
