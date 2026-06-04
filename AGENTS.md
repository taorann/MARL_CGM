# GraphPlanner Agent Rebuild Guide

This repository contains several external/reference codebases plus the current GraphPlanner implementation. When working on or rebuilding the code-repair agent, start from the durable assets, not from historical implementation paths.

## Primary Asset Entry Point

Read these first:

1. `docs/agent_assets/README.md`
2. `docs/agent_assets/01_overall_design.md`
3. `docs/agent_assets/02_action_state_protocol.md`
4. `docs/agent_assets/03_module_blueprint.md`
5. `docs/agent_assets/09_capability_matrix_audit.md`

Use the remaining files in `docs/agent_assets/` for runtime, graph/memory, legacy-boundary, checklist, and code-evidence details.

## Rebuild Intent

The intended agent is train-free:

- planner LLM explores, reads implementation code, commits memory, and writes a repair plan;
- graph-aware CGM receives curated memory graph/code plus behavior evidence and generates a patch;
- runtime validates, applies, tests fail-to-pass, and rolls back failed patches.

A clean rebuild should not depend on rLLM, Verl, PPO, GRPO, or reward-training machinery.

## Important Boundaries

- Treat `CodeFuse-CGM/`, `SWE-bench/`, `R2E-Gym/`, and `rllm/` as external or legacy/reference code unless a task explicitly asks to modify them.
- Do not use benchmark test source as repair evidence. Tests provide behavior/fail-to-pass symptoms only.
- Do not copy the current monolithic `PlannerEnv` structure into a rebuild. Split planner loop, environment, graph retrieval, memory, repair, runtime, telemetry, and dataset loading into separate modules.
- Do not preserve old routing hacks as architecture. Prefer general mechanisms: graph search, read hydration, memory curation, repair feedback, patch validation, and rollback.

## Current-Code Audit Notes

`docs/agent_assets/08_current_code_evidence.md` and `docs/agent_assets/09_capability_matrix_audit.md` explain which current files prove each capability exists. They are audit maps, not mandatory module layouts.

High-risk extraction points:

- rLLM wrapper currently contains useful response parsing, visible-thinking logging, and anti-loop handling. Rebuild those as normal planner-loop utilities.
- SWE-bench fail-to-pass selector enrichment currently lives under an rLLM namespace. Extract the capability without keeping the dependency.
- CGM context construction must preserve graph nodes, edges, code snippets, issue context, plan, constraints, and repair feedback.
- Memory nodes used for repair must contain real hydrated code, not just titles or graph IDs.

## Codex Working Style For This Repo

When changing current code, inspect the active path first:

```text
scripts/run_eval_graph_planner.sh
configs/eval/graph_planner_eval_defaults.yaml
scripts/eval_graph_planner_engine.py
graph_planner/agents/common/
graph_planner/env/planner_env.py
graph_planner/memory/
graph_planner/runtime/
graph_planner/integrations/codefuse_cgm/
graph_planner/infra/telemetry.py
```

When creating new architecture, follow `docs/agent_assets/03_module_blueprint.md` instead of mirroring the old file layout.
