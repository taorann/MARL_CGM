# GraphPlanner Code Agent Assets

This directory is the durable design asset for the current train-free GraphPlanner code agent.
It intentionally ignores most legacy implementation paths in the repository and records the design
needed to rebuild the same kind of agent from scratch.

## What This Asset Captures

GraphPlanner is a two-model code-repair agent:

1. A planner LLM explores the repository, reads implementation code, selects evidence, and writes a repair plan.
2. A graph-aware CGM repair model receives only the curated evidence subgraph plus issue/test behavior context and produces a patch.
3. The runtime validates, applies, tests, and rolls back patches until fail-to-pass checks pass or the episode ends.

The current system is train-free. It uses an OpenAI-compatible planner endpoint and an HTTP CGM service. A clean reimplementation should not depend on rLLM, Verl, PPO, or reward-training machinery.

## Files

- [01_overall_design.md](01_overall_design.md): system purpose, data flow, and current design principles.
- [02_action_state_protocol.md](02_action_state_protocol.md): planner actions, parameters, state objects, and observation contract.
- [03_module_blueprint.md](03_module_blueprint.md): recommended module split for a clean reimplementation.
- [04_runtime_testing_cgm.md](04_runtime_testing_cgm.md): sandbox, fail-to-pass testing, CGM input/output, patch validation, rollback.
- [05_graph_memory_retrieval.md](05_graph_memory_retrieval.md): repository graph, W/M memory model, search/read/expand behavior.
- [06_legacy_boundaries.md](06_legacy_boundaries.md): old paths and repo-local distractions to avoid.
- [07_rebuild_checklist.md](07_rebuild_checklist.md): implementation checklist and acceptance tests.
- [08_current_code_evidence.md](08_current_code_evidence.md): audited evidence from the current codebase, with notes on what to reuse as design vs. what to decouple.
- [09_capability_matrix_audit.md](09_capability_matrix_audit.md): capability-by-capability audit to reduce the risk of missing hidden runtime responsibilities.
- [10_clean_rebuild_runbook.md](10_clean_rebuild_runbook.md): commands and prompts for starting a clean rebuild while using this repo as read-only reference.

## Non-Goals

- This is not a line-by-line explanation of the current codebase.
- This is not a training design for rLLM/GRPO.
- This is not a CodeFuse-CGM upstream manual.
- This does not preserve historical hacks that were tried during debugging.

## Clean-Room Reimplementation Rule

If a future implementation follows these assets, it should be able to run as:

```text
dataset issue -> sandbox checkout -> planner loop -> graph memory -> CGM patch -> apply/test/rollback -> result
```

without importing rLLM, without depending on old rule-based agents, and without mixing benchmark test source into repair evidence.
