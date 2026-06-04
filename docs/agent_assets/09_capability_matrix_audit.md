# Capability Matrix Audit

This audit answers one practical question: if the current repository disappeared, what capabilities must a clean train-free GraphPlanner implementation reproduce, and where did we verify that those capabilities exist in the current code?

It is intentionally not a dependency map for preserving the old structure. Use it as a coverage checklist, not as a blueprint for copying files.

## Audit Method

The check was done from both directions:

- Entry-path trace: start at `scripts/run_eval_graph_planner.sh`, follow into `scripts/eval_graph_planner_engine.py`, then into the planner agent wrapper, environment, sandbox, graph, CGM, and telemetry layers.
- Capability search: independently search for the concrete responsibilities a working code-repair agent needs, such as fail-to-pass selection, graph construction, read hydration, memory commit, CGM payload building, patch rollback, and progress logging.

This matters because looking only at the entrypoint would miss cross-cutting capabilities hidden in helper modules, while searching only for modules would overvalue old unused routes.

## Capability Matrix

| Capability | Current Evidence | Rebuild Requirement | Risk If Omitted |
| --- | --- | --- | --- |
| Task loading and issue normalization | `scripts/eval_graph_planner_engine.py::_load_tasks`; `graph_planner/integrations/rllm/env.py`; `graph_planner/integrations/rllm/swebench_meta.py` | Build a standalone dataset loader that produces normalized issue text, sandbox config, fail-to-pass selectors, pass-to-pass selectors, and optional test patch metadata. Do not keep this under an rLLM namespace in a clean rebuild. | The planner sees the wrong issue, the wrong tests, or benchmark metadata leaks into repair context. |
| Runtime configuration and launch defaults | `scripts/run_eval_graph_planner.sh`; `configs/eval/graph_planner_eval_defaults.yaml`; `docs/startup_eval_cgm_runbook.md` | Centralize planner endpoint, planner temperature, thinking/logging flags, CGM endpoint, remote_swe config, timeout, and max-step defaults. | Repeated experiments silently run different models, temperatures, CGM URLs, or old routes. |
| Planner protocol and action schema | `graph_planner/core/actions.py`; `graph_planner/agents/common/contracts.py` | Keep one explicit action schema for `run_failed_test`, `explore_find`, `explore_expand`, `read`, `memory_commit`, `memory_delete`, `memory_commit_note`, and `repair`. Verified repair should end the episode automatically instead of requiring a planner `submit`. | The planner output becomes ambiguous, hard to parse, or impossible to replay. |
| Planner loop shell | Current path uses `GraphPlannerRLLMAgent`, `GraphPlannerRLLMEnv`, and `AgentExecutionEngine` through `scripts/eval_graph_planner_engine.py` | Replace with plain `PlannerClient`, `PlannerLoop`, `ActionParser`, `ObservationBuilder`, and `CodeRepairEnv`. | A future train-free agent remains unnecessarily coupled to rLLM rollout objects. |
| Visible thinking and model response audit | `graph_planner/integrations/rllm/agent.py` emits `planner.visible_thinking`; `graph_planner/infra/telemetry.py` renders it | Log visible thinking or model self-summary as audit-only data, not as hidden state fed into the next step unless intentionally summarized. | Hard to diagnose whether failures are model understanding failures, prompt failures, or retrieval failures. |
| Observation formatting and context compression | `graph_planner/agents/common/chat.py`; current state summaries from `PlannerEnv` | Provide compact state: issue intent, fail-to-pass behavior, W index, M index, last action/result, repair feedback, and allowed next actions. | The planner repeats actions, misses read results, or cannot tell what CGM will see. |
| Fail-to-pass evidence collection | `PlannerEnv._handle_run_failed_test`; selector normalization helpers in `PlannerEnv`; SWE-bench metadata enrichment | Prefer official SWE-bench `eval_script_list` and repo-specific log parsers when metadata is available. Run only authoritative fail-to-pass/pass-to-pass targets, hide benchmark test source, and present failures as behavior symptoms. | The planner chases tests instead of implementation code, or repairs against noisy non-target failures. |
| Test-code guard | Prompt rules in `contracts.py`; observation rules in `chat.py`; guards in current wrapper and env | Enforce at three layers: prompt, action validator, and retrieval/read filter. If the model asks for test code, reject or rewrite to implementation-side intent with explicit feedback. | Test helper names become anchors and derail localization. |
| Repository graph construction | `graph_planner/tools/swe_build_graph.py` | Rebuild graph at file, class, function/method, assignment, import, call/use, and containment granularity. Prefer tree-sitter where available, with AST fallback. | Retrieval returns only file/chunk blobs; CGM receives weak or misleading graph context. |
| Graph retrieval and expand | `PlannerEnv._handle_explore`; `graph_planner/memory/mem_candidates.py`; graph adapters and subgraph store | Implement `find` plus `expand` over graph edges, with filesystem fallback mapped back to graph nodes/spans. | Correct code may be found textually but fail to enter the graph/memory path. |
| Read and code hydration | `PlannerEnv._handle_read`; `_read_node_snippet`; `_read_file_lines`; `_merge_working_nodes_into_memory` | `read` must attach full or sufficient code body to W, and committed memory must hydrate code before CGM. Support file/class/function views. | CGM receives node names without real code, causing structurally bad patches. |
| W/M/T memory lifecycle | `PlannerEnv._handle_memory`; `graph_planner/memory/text_memory.py`; `graph_planner/memory/subgraph_store.py` | Keep W as broad working graph, M as curated CGM evidence, T as planner-only notes. Make repair depend on M, not all W. | CGM input becomes noisy, or important read code never reaches repair. |
| Repair gating and feedback | `PlannerEnv._handle_repair`; fail-to-pass evidence gate; memory quality gates; repair feedback fields | Block repair before fail-to-pass evidence and before memory contains code. After a failed repair, feed back rollback state and failure deltas. | The agent patches too early, repeats the same bad hypothesis, or treats rollback as success. |
| CGM payload construction | `PlannerEnv._build_cgm_issue_payload`; `_build_cgm_repair_prompt`; `_build_cgm_subgraph_linearized`; `graph_planner/integrations/codefuse_cgm/context.py` | Build payload from issue, plan, memory graph, snippets, constraints, previous repair feedback, and normalized graph schema. | CGM receives mismatched plan/code or lacks the evidence needed for a good patch. |
| Graph-aware CGM runtime | `graph_planner/integrations/codefuse_cgm/schema.py`; `service.py`; `graph_inference.py`; `client.py` | Use the official-style graph schema with nodes, edges, and adjacency-aware model input where available. Keep HTTP boundary simple and testable. | The system silently falls back to text-only context and loses the intended CGM advantage. |
| Patch parsing and normalization | `graph_planner/integrations/codefuse_cgm/adapter.py`; `graph_planner/agents/common/text_protocol.py` | Accept JSON edits and unified diff where needed, normalize into one edit schema, then validate. | Good model outputs are rejected, or bad partial patches are applied. |
| Patch validation, apply, rollback, retest | `graph_planner/repair/patch_quality.py`; `SandboxRuntime.snapshot_files`; `restore_files`; `apply_patch_edits`; `PlannerEnv._handle_repair` | Always snapshot touched files, apply patch, syntax-check, run fail-to-pass, record deltas, and rollback on failure unless keeping a known best candidate is intentional. | Bad patches corrupt the sandbox and later tests become meaningless. |
| Sandbox and remote execution | `graph_planner/runtime/sandbox.py`; `graph_planner/runtime/remote_swe_session.py` | Define a backend-neutral sandbox interface: start, exec, read, write/apply, build graph, run tests, snapshot, restore. Remote_swe should be one backend plugin. | Local assumptions break on remote containers or Apptainer/SIF tasks. |
| Progress and result reporting | `graph_planner/infra/telemetry.py`; eval progress logs and markdown reports | Write structured JSONL plus readable markdown. Track pass, not_pass, bug-interrupted, excluded-bug pass rate, and per-task status. | Large evals become impossible to audit or resume safely. |
| Parallel eval orchestration | Eval engine runner options; remote_swe runner preparation; progress writer | Keep parallelism outside the agent logic. Use worker/runners that share planner/CGM services but isolate sandboxes. | Throughput tuning pollutes planner logic, or concurrent tasks corrupt each other. |

## High-Risk Extraction Points

These are the places most likely to be missed or accidentally copied in the wrong form.

- The current rLLM wrapper contains useful response parsing, visible-thinking logging, and anti-loop behavior, but the clean agent should reimplement those as normal planner-loop utilities.
- SWE-bench selector enrichment currently lives under `graph_planner/integrations/rllm`. The capability is needed; the namespace and dependency are not.
- `PlannerEnv` contains too many responsibilities. Split it into action handlers, graph retrieval, memory manager, repair coordinator, test runner, and observation builder.
- CGM input has two concepts that must not be conflated: graph schema/payload construction and actual graph-aware model inference. A rebuild should test both.
- Test-code blocking must be enforced in code, not only in the prompt.
- Read hydration is a correctness requirement: if a node is in M for repair, CGM must receive its real code body, not only the node title.

## Minimal Rebuild Acceptance

A new implementation can be considered feature-equivalent when it passes these checks:

1. It can run one SWE-style issue from issue text to sandbox checkout without importing rLLM.
2. `run_failed_test` shows only fail-to-pass behavior and does not expose benchmark test source.
3. `explore_find`, `explore_expand`, and `read` populate W with graph nodes and code snippets.
4. `memory_commit` produces M with hydrated code bodies and graph edges.
5. `repair` sends issue, plan, graph nodes, graph edges, snippets, and constraints to CGM.
6. Patch apply always has validation, snapshot, rollback, and fail-to-pass retest.
7. Telemetry can reconstruct planner input, model visible reasoning/summary, chosen action, action result, CGM input summary, CGM raw output, patch decision, and final status.
8. Test-code query attempts are rejected or redirected before retrieval.

## Audit Conclusion

The asset set now covers both the architecture and the hidden runtime responsibilities needed to rebuild the agent. The important caveat is that the current codebase mixes active logic with legacy experiments, especially around rLLM wrappers and the monolithic environment. For a new implementation, treat this matrix as the source of required capabilities, and treat the existing files only as examples of one imperfect implementation.
