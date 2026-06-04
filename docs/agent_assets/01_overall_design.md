# Overall Design

## Motivation

The agent is designed for SWE-bench-style code repair where the model must locate the faulty implementation code, collect enough evidence, generate a patch, and verify it against fail-to-pass tests.

The key problem is not just patch generation. The harder problem is giving the repair model the right evidence:

- The planner must not blindly edit from issue text alone.
- The planner must not be pulled into benchmark test code.
- The repair model must receive concrete implementation code, not just file names or vague summaries.
- Failed repairs must be rolled back and treated as evidence that the current hypothesis was shallow.

## Current Effective Architecture

```text
Evaluation driver
  -> Task loader
  -> Sandbox runtime
  -> Repo graph builder
  -> Planner loop
       -> run_failed_test
       -> explore_find / explore_expand
       -> read
       -> memory_commit / memory_delete / memory_commit_note
       -> repair
  -> CGM repair service
  -> Patch validator / applier
  -> Targeted test runner
  -> Telemetry writer
```

## Two-Model Roles

The planner LLM decides what to inspect and when there is enough evidence to repair. It should reason about the issue and implementation graph, but it should not directly write final patches.

The CGM model writes patches. It should see a compact graph-aware repair context:

- original issue summary;
- trusted fail-to-pass behavior summary;
- planner repair plan;
- memory subgraph nodes and edges;
- full code bodies for selected evidence nodes;
- patch constraints and prior failed repair feedback.

## Train-Free Principle

The current system is an inference-time agent. The planner is called through an OpenAI-compatible endpoint; CGM is called through HTTP. The loop is ordinary Python orchestration.

A clean reimplementation should not require:

- rLLM `BaseAgent` / `BaseEnv`;
- Verl;
- Ray;
- reward model training;
- PPO/GRPO datasets;
- rLLM trajectory classes.

The legacy rLLM adapter can be useful as a reference for prompt/action parsing, but it should not be a dependency of the new architecture.

## Core Invariants

- The planner sees benchmark test results as behavior, not source code.
- The planner reads implementation nodes before committing them to CGM memory.
- CGM input is based on memory M, not the entire noisy working graph W.
- Memory M must contain full code bodies for repair-relevant nodes.
- Patch attempts are applied on a reversible snapshot.
- Failed or low-quality patches are rolled back.
- A patch that verifies target tests green ends the episode automatically.
- The next planner step must see that repair failed and why, but not be redirected into test source.
- Telemetry must record enough input/output to reconstruct every decision.

## High-Level Loop

1. Load issue and sandbox.
2. Build or fetch repository graph.
3. Create initial planner state from issue, repo metadata, and optional known fail-to-pass selectors.
4. Ask planner for exactly one action.
5. Execute action in environment.
6. Update W, M, text memory, test summary, and telemetry.
7. Continue until verified repair, max steps, timeout, or unrecoverable backend failure.
8. Save final patch/test result/trajectory.

## What Counts as Success

For a single issue:

- The agent applies a patch.
- Target fail-to-pass tests pass.
- Previously passing targeted pass-to-pass tests do not regress when available.
- The final patch remains in the sandbox result.

For benchmark evaluation:

- `pass`: patch applied and target verification passes.
- `not_pass`: agent finished but did not produce verified patch.
- `bug`: infrastructure failure such as backend timeout, SSH failure, missing image, invalid task metadata, or service outage.
