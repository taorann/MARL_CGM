# Legacy Boundaries And Anti-Confusion Notes

The repository root includes external projects and historical experiments. A future rebuild should not treat every directory as part of the current agent.

## External Or Reference Code

These are not the new agent core:

- `CodeFuse-CGM/`: upstream CGM source/reference.
- `SWE-bench/`: official benchmark/harness reference.
- `R2E-Gym/`: external environment project/reference.
- `rllm/` if present in a checkout: training framework reference.
- `actor/`: older training/fallback actor experiments.
- `aci/`: utility package; not the hidden runtime `.aci/` cache directory.
- `hpc/`: remote runner/proxy support, useful only for remote_swe deployment.

## Current Agent-Owned Concepts

The current agent-owned design lives conceptually in:

- planner action protocol;
- environment state machine;
- repository graph builder;
- W/M/T memory;
- sandbox runtime abstraction;
- CGM graph-aware repair integration;
- patch validation/rollback;
- telemetry/progress.

When rebuilding, implement those concepts in clean modules rather than preserving current file layout.

## rLLM Boundary

Current implementation still has `graph_planner/integrations/rllm/*` because historical work used rLLM-style agent/env wrappers. New code should not depend on it.

Replace it with:

- plain OpenAI-compatible planner client;
- plain Python planner loop;
- plain environment stepper;
- standalone dataset/eval runner.

The train-free agent should run without importing `rllm`.

## Old Paths To Avoid Reviving

Avoid:

- rule-based hard-coded issue repair paths;
- Planner directly emitting patches as primary mode;
- text-only CGM linearization as the main graph path;
- using benchmark test source as evidence;
- treating all read nodes as CGM input automatically;
- dumping full files into M by default;
- relying on reward signals or RL-only abstractions;
- hidden fallback that silently changes test-derived queries into implementation-looking paths.

## Acceptable General Guards

These are not issue-specific hacks and should remain:

- block benchmark test code as patch/search/read target;
- require memory before repair;
- require full code bodies in CGM memory;
- rollback failed patches;
- deduplicate repeated no-gain actions;
- map grep fallback spans back to graph nodes;
- prefer implementation frames over test frames;
- treat failed repair as shallow-hypothesis evidence.

## Risky Rules To Keep Minimal

Rules based on domain words such as `serializer`, `matrix`, `enum`, `default`, or specific helper names are risky. They may be useful as examples in telemetry analysis, but they should not become hard-coded routing logic.

Prefer structural rules:

- bridge/dispatcher detection;
- registry/map assignment detection;
- sibling helper discovery;
- call/use edge expansion;
- source path and line-span grounding;
- code-body availability checks.

## Documentation Rule

If a behavior is required for the agent to work, document it here or in sibling assets. If a behavior only helped one historical issue, document it as a case study elsewhere, not as core design.
