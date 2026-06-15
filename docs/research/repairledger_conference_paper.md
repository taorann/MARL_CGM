# RepairLedger: Evidence-State Context Compression for Graph-Guided Repository Repair Agents

Draft status: conference-paper draft, implementation-backed, not yet submission-ready until the evaluation is frozen.

Default target style: ACM/ICSE-FSE style systems paper, 8-10 pages plus references.

## Abstract

Repository-level software repair agents are increasingly built as harnessed systems rather than isolated code generators. A repair run combines an issue statement, sandbox execution, repository search, code reading, graph retrieval, planner reasoning, patch generation, validation, rollback, and feedback. This creates a context-compression problem: the harness must repeatedly project a long, heterogeneous repair trajectory into small model inputs. Existing systems improve action interfaces, staged localization and repair, repository graph retrieval, and long-horizon memory. We argue that repository repair adds a stricter invariant: compressed context must preserve the evidence state of each artifact. A search hit is not hydrated code evidence; a graph neighbor is not a trusted patch premise; a planner hypothesis is not a specification; a candidate patch is not a validated fix; and rollback should not erase the assumptions invalidated by a failed edit.

We present RepairLedger, a train-free harness protocol for evidence-state context compression in graph-guided repository repair. RepairLedger maintains typed repair artifacts over a repository graph and runtime trajectory, including working candidates, read code, committed code evidence, planner notes, pending patches, validation outcomes, and failed-patch history. The harness exposes consumer-specific projections: the planner receives stateful working memory and action affordances; the graph-aware patch model receives only curated hydrated code, graph relations, issue and runtime evidence, concise intent, pending-patch context, and recent failure history; the validator consumes explicit pending or generated patches and records pass, fail, syntax, infrastructure, and rollback outcomes. In an implementation in GraphPlanner, this protocol is realized through a planner action interface, W/M memory separation, evidence-gated repair contracts, a propose/revise/submit patch loop, benchmark-hygienic behavior summaries, and rollback-preserving repair history. A pilot trace audit over 20 current-agent SWE-bench tasks shows the protocol active across 320 planner steps, 36 memory commits, 25 patch proposals, 20 patch revisions, 15 submissions, 43 rollback-bearing outcomes, and 16 passed repairs. These preliminary traces do not replace a frozen benchmark evaluation, but they demonstrate that repair success and failure are governed by harness state transitions rather than one-shot patch generation.

## 1. Introduction

Repository-level software repair has become a systems problem. A modern repair agent is not simply asked to write a function. It must inspect a large codebase, identify the code relevant to an issue, understand cross-file behavior, construct a patch, run trusted tests, interpret failures, and decide whether to continue. The model matters, but the harness around the model decides what can be searched, what code is visible, which artifacts are trusted, when a patch can be tested, how rollback is handled, and what state survives into the next model call.

Recent work has clarified several pieces of this harness. SWE-agent shows that the agent-computer interface materially affects autonomous software engineering by shaping how agents navigate, edit, and test repositories. Agentless shows that a simple staged localization-repair-validation pipeline can be competitive without fully unconstrained autonomy. OpenHands generalizes software-development agents as sandboxed platforms that can browse, use command lines, write code, and evaluate tasks. AutoCodeRover and related systems show the value of program-structure-aware search. RepoGraph, GraphCodeAgent/CodeRAG, and Code Graph Model (CGM) show that repository graphs and dependency-aware code context are useful for repository-level generation and repair. MemGPT, ACON, Reflexion, and recent harness-engineering work show that long-horizon agents require explicit memory, feedback, and context management.

These strands are complementary. Together, they imply that repository repair is a long-horizon interaction in which the harness repeatedly compresses searches, reads, graph expansions, runtime output, plans, patches, failures, and rollbacks into model-visible state. However, repository repair imposes a property that is weaker or more implicit in general memory systems: the compressed state must preserve what each artifact is allowed to mean.

Consider a graph-guided repair trajectory. A node returned by search is only a navigation clue. A short code preview helps orientation but is not full implementation evidence. A read function body is stronger evidence, but only for the lines read. A memory commit promotes selected code into the patch model's trusted context. A planner's mechanism analysis is advisory, not a patch recipe. A generated patch is an unvalidated assumption about APIs, types, control flow, and behavior. A failed test is validation evidence, while a syntax error in a generated patch is evidence about the generated edit rather than about the original repository. If these states collapse into undifferentiated prompt text, the harness may send noisy previews to the patch model, treat hypotheses as specifications, retry stale strategies, or forget the meaning of a rollback.

We call this failure mode evidence-state collapse: content survives compression, but its epistemic status, provenance, consumer eligibility, or validation state is lost or upgraded incorrectly. The failure is not just weak retrieval or weak generation. The right code may have been found but not promoted to trusted evidence. A plausible patch may be based on a preview rather than hydrated code. A failed edit may teach the harness something useful, but that information may disappear after files are restored.

This paper introduces RepairLedger, a harness protocol for evidence-state context compression in graph-guided repository repair. RepairLedger is implemented in the current GraphPlanner agent as a train-free two-model system. A planner LLM explores the repository, reads implementation code, commits selected memory, and writes structured repair intent. A graph-aware CGM patch model receives a curated evidence package and generates candidate edits. The runtime validates patches, tests fail-to-pass behavior, tracks pass-to-pass or regression signals where available, and rolls back failed patches. The key contribution is not a new repository graph, a new patch model, or a new training method. It is a repair-specific state protocol that governs how graph artifacts, code evidence, planner hypotheses, patch candidates, and validation outcomes move through the harness.

The current implementation makes this protocol concrete. Working memory W stores exploratory graph context, previews, and read code. Repair memory M stores explicitly committed hydrated code that the patch model may rely on. Text notes T store planner-only hypotheses. Pending patch state P stores a generated candidate that has passed patch-format and syntax checks but has not consumed official test validation. Repair history H stores compact records of failures, error origins, patch previews, rollback state, and CGM insights. The observation builder exposes a current-turn protocol that tells the planner which actions are valid, which blockers exist, which W nodes can be committed, and why repair is disabled or allowed. The CGM context compiler exposes consumer-specific fields rather than the whole transcript.

This framing changes how the method should be compared to current code agents. SWE-agent studies action interfaces; RepairLedger studies evidence-state preservation inside the interface. Agentless studies staged control; RepairLedger turns stage transitions into artifact promotion and demotion rules. Graph-RAG and CGM systems study which code to retrieve or encode; RepairLedger studies when retrieved graph objects become trusted evidence and which consumer may use them. General memory and context-compression systems study how to fit long histories; RepairLedger studies how to preserve repair-specific trust states while compressing those histories.

This paper makes four contributions:

1. It identifies evidence-state collapse as a harness-level failure mode in repository repair agents.
2. It defines RepairLedger, a typed evidence-state protocol for graph-guided repair trajectories.
3. It describes a working GraphPlanner implementation with W/M/T/P/H state, evidence-gated repair contracts, a propose/revise/submit patch loop, benchmark-hygienic runtime summaries, and rollback-preserving history.
4. It provides code-grounded evaluation criteria and a preliminary trace audit showing the protocol operating in current SWE-bench runs.

## 2. Background and Problem Setting

### 2.1 Repository Repair as a Harnessed Interaction

SWE-bench made real GitHub issue repair a standard benchmark setting: a system receives a repository at a base commit and an issue, modifies the code, and is judged by test behavior. The benchmark setting matters because it shifts software repair from isolated code synthesis to environment interaction. A system must decide what to inspect, how to construct a patch, and how to validate it under hidden or benchmark-controlled tests.

SWE-agent argues that the interface between a language model and the computer is itself a design object. The agent-computer interface shapes navigation, editing, and testing. OpenHands takes a broader platform view, providing sandboxed agent infrastructure for coding tasks. Agentless takes the opposite stance from unconstrained autonomy: staged localization, repair, and validation can be competitive and interpretable. These systems make a shared point: performance is shaped by the harness that controls action, state, and feedback.

RepairLedger follows this harness-centered view but focuses on a narrower question: what state semantics must survive when a repair trajectory is compressed for multiple consumers?

### 2.2 Graph-Guided Repository Context

Repository graphs are now a major substrate for code agents. AutoCodeRover uses program-structure-aware search APIs. RepoGraph retrieves repository-level graph context. GraphCodeAgent/CodeRAG retrieves supportive code from code and requirement graphs. CGM integrates code graph structure into a language model and uses graph-RAG style inputs for repository-level tasks.

These systems address an important retrieval question: which code nodes and relations should a model see? RepairLedger asks a different question: after a graph node is retrieved, what is its current evidential state? A node can be a search candidate, a preview, a read implementation fact, a committed patch premise, a pending patch dependency, a validated support, or an invalidated assumption. The graph supplies structure, but the harness must supply lifecycle state.

### 2.3 Memory, Compression, and Feedback

Long-horizon agents need memory. MemGPT frames context management as an operating-system-like memory problem. ACON studies context compression for long-horizon agents. Reflexion shows that retaining feedback in memory can improve future behavior without updating model weights. Recent harness-engineering discussions make the extra-model layer a first-class object: the harness controls what the agent sees, what it can do, how state is carried forward, and how failure is handled.

RepairLedger specializes this memory view to repository repair. It does not merely ask which old observations should fit into context. It asks which artifact states must remain distinct. Compression should preserve not only relevance but also status, provenance, and consumer eligibility.

## 3. Evidence-State Collapse

Let a repair trajectory be a sequence of action-observation pairs:

```text
T = [(a_1, o_1), ..., (a_n, o_n)].
```

Each action can produce artifacts:

```text
A = {issue claims, runtime symptoms, graph nodes, graph edges,
     code previews, read code bodies, relation facts, planner hypotheses,
     candidate edits, validation results, rollback outcomes}.
```

In a repair harness, each artifact has an evidence state:

```text
e(a) = (type, source, status, scope, provenance, consumers, validity).
```

Typical statuses include:

```text
candidate, preview, read, hydrated, committed, hypothesis,
pending, validated, invalidated, superseded.
```

Evidence-state collapse occurs when the compressed projection of a trajectory loses or misstates these fields. Four collapse classes are especially important:

1. Status collapse: a search candidate or preview is treated as read code evidence.
2. Provenance collapse: text is retained but the read action, node id, path, or line scope is lost.
3. Consumer collapse: planner-only hypotheses or noisy W context are exposed to the patch model as trusted repair context.
4. Validation collapse: syntax failures, failed tests, rollback outcomes, or duplicate patches are forgotten or misinterpreted.

The current GraphPlanner implementation includes explicit mechanisms for each class. Search and grep results carry result policies stating that they are navigation context only. The `memory_commit` handler blocks attempts to commit previews that were not explicitly read. The repair contract requires target nodes to be in M and to appear in the evidence chain. Patch generation can be separated into `repair_propose`, `repair_revise`, and `repair_submit`, where proposal stores a pending patch without official tests. Failed patches are rolled back but summarized into repair history and last-repair feedback.

Evidence-state collapse is measurable. A trace can be audited for preview-as-evidence events, unhydrated nodes in the patch payload, target nodes absent from the evidence chain, repeated patch signatures after failure, patch submission without a pending reviewable candidate, and missing rollback feedback after failed tests. These are harness properties, not only model quality properties.

## 4. RepairLedger

RepairLedger is a typed evidence-state ledger over the repository graph and runtime repair trajectory. It is not a separate database in the current implementation; it is an architectural protocol realized through state objects, action guards, observation projections, CGM payload construction, validation results, and telemetry traces.

### 4.1 Artifact Types and States

The implementation tracks five main memory classes:

| Symbol | Implementation object | Meaning | Primary consumer |
| --- | --- | --- | --- |
| W | `WorkingMemory` | Exploratory working context: candidates, previews, and read code | Planner |
| M | `CgmMemory` | Curated hydrated code evidence for CGM | Patch model |
| T | `TextNotes` | Planner-only notes and hypotheses | Planner |
| P | `pending_patch` plus origin | Generated candidate patch not yet officially validated | Planner, patch model, validator |
| H | `repair_attempts`, `repair_history`, `cgm_insights` | Failed patch records, signatures, error origins, rollback state, model insights | Planner and patch model |

The key point is not the labels themselves. It is that each memory class controls which consumers may rely on an artifact.

### 4.2 Graph Artifact Lifecycle

A repository graph node moves through a lifecycle:

```text
graph node
  -> search/grep/expand candidate
  -> orientation preview in W
  -> explicit read code in W
  -> hydrated committed evidence in M
  -> target/evidence-chain dependency for CGM
  -> validated support or invalidated assumption after test
```

The lifecycle is enforced by code rather than convention. `explore_find` can add preview nodes to W, but result policies state that previews are orientation only. `read` hydrates code from the runtime and adds local symbol references, dispatch facts, dispatch relationship context, and value-flow context. `memory_commit` requires explicit selected ids and blocks un-read candidates. `build_cgm_payload` validates that snippets and serialized code contain hydrated memory code before CGM is called.

This gives the graph a state dimension. The graph tells the harness which nodes and edges exist; RepairLedger tells the harness whether a node is only a lead, a read fact, or a committed patch premise.

### 4.3 Consumer-Specific Projections

RepairLedger is a context compiler. It does not produce one transcript summary. It produces different projections for different consumers:

Planner projection:

- issue title and body;
- fail-to-pass runtime summary;
- current-turn protocol with valid and invalid next actions;
- W summary and W code with evidence status;
- M summary and read-not-committed nodes;
- latest action result and blockers;
- pending patch summary;
- repair history, CGM insights, and planner diagnostics;
- trajectory summary.

Patch-model projection:

- cleaned issue context;
- planner-selected targets;
- advisory plan text and confidence;
- hydrated snippets and serialized code from M;
- graph nodes and edges around M;
- structured source facts such as dispatch tables;
- repair history and pending patch when relevant;
- output contract and constraints.

Validator projection:

- a concrete patch object;
- touched paths;
- a runtime snapshot;
- patch-format and syntax decisions;
- fail-to-pass and pass-to-pass status where available;
- rollback result and error origin.

The projections intentionally differ. The planner may see broad W context and notes; CGM should not see arbitrary W previews as trusted code. The validator should not reason over the whole transcript; it should consume the patch and emit validation evidence.

### 4.4 Evidence-Gated Repair Contract

Repair actions require a structured evidence package:

```text
failure_seen: actual issue/runtime behavior
evidence_chain: read implementation node ids plus roles and evidence sentences
target_nodes: committed M nodes or explicit new_file targets
intent_analysis: mechanism analysis, not exact patch text
confidence: numeric self-score
```

The contract blocks common collapse modes. Existing-file targets must be in M and must appear in the evidence chain. Evidence-chain ids must refer to read or committed code nodes, not pseudo-nodes such as `test_behavior`. If a previous patch failed with the same memory, repair is disabled unless M changes or a ready `repair_review` supports the same package. If a previous patch failed because it used an unverified API or signature, the next evidence chain must include code proving that API or signature.

This turns patch generation into a state transition rather than a free-form prompt. The planner cannot simply ask the patch model to "fix it"; it must identify behavior, code evidence, target locus, mechanism, and confidence.

### 4.5 Pending Patch Deliberation

The newest implementation makes pending patches first-class. `repair_propose` asks CGM for a candidate patch, applies it only for validation and syntax checking, rolls it back, and stores it as P. It does not run official fail-to-pass or pass-to-pass tests. The planner then chooses:

- `repair_submit` if the pending patch is ready to test;
- `repair_revise` if the candidate is close but risky or incomplete;
- `discard_pending_patch` if the candidate is wrong or stale;
- search, grep, expand, read, or commit more evidence if the risk cannot be judged.

This separates generation from validation. It also protects the official test budget from weak candidates. A candidate patch becomes a visible, inspectable artifact with summary, touched paths, edit preview, origin, memory node ids, target nodes, and CGM response metadata. A submitted pending patch that fails tests is rolled back and recorded as failure evidence.

### 4.6 Runtime Evidence and Benchmark Hygiene

The runtime layer turns tests into behavior evidence without exposing benchmark test source as repair evidence. `behavior_summary` extracts failed selectors, exception types, actual messages, actual assertion values, implementation traceback frames, parser errors, and safe excerpts. It omits hidden expected values and redacts official evaluation commands that contain benchmark harness setup or test patches.

This matters because repair evidence includes behavior but should not include privileged test implementation. RepairLedger treats tests as symptoms and validation, not as source code for patch synthesis.

### 4.7 Rollback Is Not Forgetting

Rollback restores files but should not erase what the failed patch taught the system. The implementation snapshots touched paths, applies a patch, validates syntax and tests, and rolls back on syntax failure, test failure, infrastructure bugs, or proposal mode. The attempt is still stored with status, tool, summary, touched paths, error origin, source tree state, memory node ids, target nodes, patch preview, and failure feedback.

This creates a distinction between source state and evidence state. The source tree returns to its original state, but the ledger keeps the invalidated assumption.

## 5. Implementation in GraphPlanner

The current implementation is organized around explicit modules rather than a monolithic planner environment.

| Mechanism | Code evidence |
| --- | --- |
| Planner action interface | `src/graphplanner_agent/planner/protocol.py`, `src/graphplanner_agent/planner/prompt.py` |
| Planner loop and anti-malformed-output handling | `src/graphplanner_agent/planner/loop.py`, `src/graphplanner_agent/planner/response_parser.py` |
| Environment state and action gating | `src/graphplanner_agent/env/stepper.py`, `src/graphplanner_agent/env/guards.py` |
| Search/read/expand evidence handling | `src/graphplanner_agent/env/action_handlers.py`, `src/graphplanner_agent/env/evidence.py` |
| Working and repair memory | `src/graphplanner_agent/memory/working.py`, `src/graphplanner_agent/memory/cgm_memory.py` |
| Observation and current-turn protocol | `src/graphplanner_agent/env/observations.py` |
| CGM payload compiler | `src/graphplanner_agent/repair/cgm_context.py` |
| Patch parsing, validation, normalization, and application | `src/graphplanner_agent/repair/patch_schema.py`, `src/graphplanner_agent/repair/patch_apply.py` |
| Runtime and test evidence | `src/graphplanner_agent/runtime/test_runner.py`, `src/graphplanner_agent/runtime/remote_swe.py`, `src/graphplanner_agent/runtime/swebench_official.py`, `src/graphplanner_agent/runtime/swebench_pro.py` |
| Telemetry and evaluation orchestration | `src/graphplanner_agent/cli/eval_parallel.py`, `src/graphplanner_agent/telemetry/progress.py` |

### 5.1 Planner Interface

The planner can call `run_failed_test`, `explore_find`, `grep_code`, `explore_expand`, `read`, `memory_commit`, `memory_delete`, `memory_commit_note`, `repair_review`, `repair_propose`, `repair_revise`, `repair_submit`, `discard_pending_patch`, `repair_chunk`, and `repair`. The action schema itself encodes evidence-state rules. For example, `memory_commit` says that previews cannot be committed directly; `repair_propose` says it stores a candidate patch without tests; `repair_submit` says it submits the pending patch for official verification.

The loop also handles malformed planner output and unavailable actions. If the planner calls a disabled tool, the environment records a diagnostic and asks for exactly one currently available action. This makes harness state enforceable at the planning surface.

### 5.2 Graph and Code Evidence

`explore_find` searches implementation graph nodes and can fall back to runtime file discovery when the graph misses a scoped file. Non-file hits receive small code previews for orientation. File hits list top symbols rather than full text. `grep_code` searches exact text in scoped implementation paths and returns line-level context plus a suggested covering node. `explore_expand` exposes relation candidates: callers, callees, siblings, imports, contains, uses, related, mechanism, and owner_flow. Mechanism modes lazily infer base classes, overrides, composition, pipeline candidates, attribute owners, and symbol consumers from indexed or read code.

`read` is the promotion step from candidate to code evidence. It reads from the runtime, rejects test paths, adds code to W, and extracts local symbol references, dispatch tables, dispatch relationship context, and value-flow context. These relation facts are useful but still have status. Related nodes are added to W as candidates or previews; they must be read before they can become repair evidence.

### 5.3 Observation as a Harness Control Surface

The observation builder emits a `current_turn_protocol` before the issue, memory, and code. This protocol states blockers, valid next actions, invalid next actions, candidate memory commit ids, committed memory ids, repair mechanism requirements, and failure follow-up instructions. It tells the planner that W and M are different, that repair is disabled without behavior and hydrated M, that pending patches must be inspected before new patch generation, and that failed patches require deeper evidence before another repair.

This is a harness-level control surface. Instead of relying on the planner to infer state from a transcript, the environment computes the state and presents it explicitly.

### 5.4 CGM Context Compiler

`build_cgm_payload` compiles M into CGM-visible fields. It includes issue context, selected targets, plan text, graph nodes and edges, hydrated snippets, serialized code, dispatch facts, repair history, pending patch context, CGM insights, and planner decision context. Validation requires graph nodes, subgraph nodes, snippets, serialized code, and numbered text. Context neighbors without paths are skipped; test paths are excluded.

The payload deliberately keeps planner intent advisory. The source snippets and graph node text are authoritative. This prevents a planner hypothesis from overwriting code evidence.

### 5.5 Patch Protocol

CGM outputs are parsed as JSON patch objects or complete diffs, normalized, validated, and applied through runtime file APIs. Patch validation rejects test changes by default, unsafe paths, duplicate ranges, schema artifacts in edit text, diff markers in edit text, suspicious multi-line range collapse, and Python control-flow header removal. Python syntax checks run after application. Internal retry is allowed for format or generated syntax failures, but behavioral failures require new evidence or a changed intent.

This patch protocol helps keep patch-generation errors from being misread as repository behavior.

### 5.6 Remote Harness and SWE-bench Pro

The remote runtime starts isolated SWE containers or SIF-based tasks, builds and caches repository graphs, reads and writes files through encoded payloads, snapshots and rolls back touched paths, and runs official SWE-bench or SWE-bench Pro commands. SWE-bench Pro support runs per-instance scripts and parsers, classifies fail-to-pass and pass-to-pass selectors, and records parser errors. The launcher scripts configure remote roots, SIF directories, runner counts, timeouts, CGM endpoint, and run metadata.

This makes the harness more than a local loop. It is an evaluation scaffold with graph construction, remote execution, validation, and trace logging.

## 6. Comparison to Current Code-Agent Families

| Family | Main contribution | Relation to RepairLedger |
| --- | --- | --- |
| SWE-agent | Agent-computer interface for navigation, editing, and testing | RepairLedger keeps the interface view but adds typed evidence states and consumer-specific context projections. |
| Agentless | Staged localization, repair, and validation without unconstrained autonomy | RepairLedger also values staging, but stages are artifact state transitions rather than fixed pipeline phases only. |
| OpenHands and broad coding platforms | General sandboxed software-development agents and tool infrastructure | RepairLedger is narrower: a repair harness memory protocol for planner/patch/validator coordination. |
| AutoCodeRover | Structured code search and program-aware context extraction | RepairLedger treats structured search results as candidates until read and committed. |
| RepoGraph, GraphCodeAgent/CodeRAG, CGM | Repository graph retrieval or graph-integrated patch generation | RepairLedger is complementary: it controls when graph artifacts become trusted patch evidence and how they are projected to CGM. |
| MemGPT and ACON | General long-horizon memory and context compression | RepairLedger specializes compression to evidence-state preservation under repair actions and validation. |
| Reflexion and self-refinement | Feedback retained across attempts | RepairLedger stores feedback as typed repair history with patch previews, error origins, source tree state, memory ids, and rollback outcomes. |

The strongest way to phrase the novelty is not that prior systems lack memory, graphs, or feedback. They clearly have some of these components. The claim is narrower: repository repair needs a harness invariant that preserves artifact evidence state as graph-guided trajectories are compressed for planner, patch model, and validator. This invariant is under-specified when graph retrieval, memory summarization, and patch feedback are treated as separate ingredients.

## 7. Evaluation Strategy and Pilot Trace Audit

### 7.1 What Should Be Measured

RepairLedger should be evaluated at two levels.

Outcome metrics:

- resolved rate on SWE-bench and SWE-bench Pro;
- bug-excluded accuracy when infrastructure failures are filtered;
- invalid patch rate;
- syntax failure rate;
- patch rejection rate;
- average steps and wall-clock time.

Harness-state metrics:

- preview-as-evidence rate: patches whose evidence chain depends on un-read previews;
- unhydrated-CGM rate: CGM payloads with missing code bodies;
- unsupported-target rate: target nodes not present in evidence chain;
- premature-validation rate: official tests run before a candidate patch is reviewable or before required evidence exists;
- rollback-forgetting rate: failed patches that leave no repair-history or failure-feedback record;
- repeated-strategy rate: duplicate or semantically repeated patch attempts after rollback;
- context compression ratio: raw trace tokens versus planner observation and CGM payload tokens;
- consumer-contamination rate: W-only artifacts exposed to CGM as trusted M evidence.

These metrics are valuable because they test the method's actual thesis. A resolved-rate gain alone would not prove evidence-state preservation. Conversely, a trace-level reduction in collapse events would show the harness property even before all model or CGM weaknesses are solved.

### 7.2 Pilot Trace Audit

The current repository contains a small current-agent run:

```text
runs/tmp/swebench_sample100_current_agent_2026-06-11_08-36-21_UTC/round_01/
```

This run is not treated as a frozen benchmark result. It is useful as pilot evidence that the protocol is exercised in real traces. The run contains 20 result records and corresponding traces. Status counts were:

| Status | Count |
| --- | ---: |
| pass | 16 |
| not_pass | 3 |
| bug | 1 |

Trace-level action counts over the 20 result tasks were:

| Trace signal | Count |
| --- | ---: |
| planner steps | 320 |
| memory commits | 36 |
| direct repairs | 11 |
| repair proposals | 25 |
| repair revisions | 20 |
| repair submissions | 15 |
| discarded pending patches | 4 |
| rollback-bearing results | 43 |
| blocked results | 14 |
| syntax failures | 2 |
| behavioral test failures | 8 |
| patch rejections | 4 |
| passed repair outcomes | 16 |

These numbers support three implementation claims. First, patch generation is frequently mediated by proposal, revision, and submission rather than direct validation. Second, rollback is common enough to be a first-class memory problem, not an exceptional case. Third, the harness actively blocks or redirects planner actions when state requirements are not met.

The same logs also show failure modes that motivate further work. Some not_pass tasks spend many steps revising or resubmitting within the same broad target area. This suggests that RepairLedger should add stronger semantic duplicate detection and better escalation from repeated pending-patch revisions to new evidence collection. Thus the pilot trace audit supports the architecture while also exposing the next research questions.

### 7.3 Unit-Level Evidence

The unit tests in `tests/test_agent_rebuild.py` cover the protocol-level invariants:

- failed patches roll back;
- `repair_propose` stores a pending patch without applying or testing it;
- `repair_submit` tests and clears a pending patch on pass;
- `repair_revise` sends pending patch and history to CGM;
- repair requires a structured evidence package;
- repair targets must appear in the evidence chain;
- W and M are distinguished in observations;
- memory commit does not auto-include related read nodes;
- behavior summaries omit hidden benchmark expected values;
- CGM payloads require hydrated code;
- CGM payloads expose model-visible fields with snippets, serialized code, graph edges, and plan text.

These tests do not establish external performance, but they establish that the evidence-state invariants are implemented as executable checks.

## 8. Discussion

### 8.1 The Harness as the Scientific Object

RepairLedger is best understood as a harness contribution. The method changes neither the planner weights nor the CGM weights. Its main effect is to govern what information means as it moves between model calls and runtime actions. This is why the harness concept is central: the harness decides what the agent can see, which action is available, which state is promoted, and how failure becomes future evidence.

This reframes a common agent-design question. The question is not only "did the model see enough context?" It is "did the model see context whose status matched the decision it was asked to make?" Search previews may be useful for planning but dangerous for patch generation. Planner hypotheses may help CGM understand intent but should not override code. Failed patches should not stay applied, but their assumptions should remain visible.

### 8.2 Why Graphs Need State

Graph context is powerful because it encodes relations among code entities. But graph retrieval alone does not solve evidence trust. A graph node can be close to a target while still being the wrong consumer. A base class can be relevant but not patched. A dispatch table can map a key to a function, but the mapping must be read and committed if it becomes a patch premise. RepairLedger's contribution is to add lifecycle state to graph artifacts.

This distinction keeps the paper from overclaiming graph novelty. The graph is inherited from a strong research line. The novelty is stateful evidence management over graph-guided trajectories.

### 8.3 Why Pending Patches Matter

The pending-patch loop gives the harness a middle state between generation and validation. Many repair systems move directly from generation to tests. That is simple, but it treats each generated patch as if it deserves official validation. In GraphPlanner, a patch can be generated, syntax checked, rolled back, inspected, revised, discarded, or submitted. This converts patch generation from an endpoint into an auditable artifact lifecycle.

Pending patches also make planner-CGM collaboration more concrete. CGM produces a candidate and optional insight summary. The planner can review coverage, risks, and requested changes. CGM can revise against a concrete pending patch and recent history rather than a vague "try again" instruction.

### 8.4 Limits of the Current Implementation

The current implementation still has clear limits. Semantic duplicate detection is mostly patch-signature based, not mechanism based. Repeated pending-patch revisions can consume many steps. Repair reviews are advisory and are not yet used heavily in the latest pilot run. Context compression ratios are not yet automatically reported. The current evaluation traces are promising but not a frozen, statistically sufficient benchmark result. These are engineering and research opportunities rather than reasons to abandon the framing.

## 9. Threats to Validity

Internal validity: Some observed improvements or failures may come from the planner model, CGM model, remote runner stability, or benchmark parser behavior rather than the evidence-state protocol. Ablations are needed to isolate W/M separation, pending patch state, repair history, and action guards.

External validity: The current implementation has been exercised on SWE-bench and SWE-bench Pro style tasks, but the pilot trace audit is small and skewed toward available tasks and infrastructure. Results may differ on JavaScript, Go, Java, or vulnerability-repair benchmarks.

Construct validity: Evidence-state collapse metrics require careful trace annotation. A preview-as-evidence event or unsupported assumption may be ambiguous unless the trace records which artifact caused the planner or CGM decision. Future telemetry should log explicit artifact ids in planner intent and CGM outputs.

Conclusion validity: The pilot run should not be interpreted as a final performance claim. Its purpose is to show that the harness protocol is active and measurable. A conference submission should include a frozen benchmark run, ablations, and confidence intervals or task-level analysis.

## 10. Conclusion

Repository-level repair agents are harnessed systems. They succeed or fail not only because of model quality, graph retrieval, or test feedback, but because the harness compresses a long repair trajectory into state that later model calls can safely use. RepairLedger identifies evidence-state collapse as a repair-specific failure mode and implements a protocol that preserves artifact status, provenance, consumer eligibility, and validation state across graph-guided repair. In GraphPlanner, this appears as W/M/T/P/H memory, evidence-gated repair contracts, consumer-specific CGM payloads, pending patch deliberation, benchmark-hygienic runtime summaries, and rollback-preserving repair history. The broader lesson is that repair harnesses should treat context compression as semantic governance, not only token reduction.

## References

[1] Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? arXiv:2310.06770. https://arxiv.org/abs/2310.06770

[2] Yang et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. arXiv:2405.15793. https://arxiv.org/abs/2405.15793

[3] Xia et al. Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489. https://arxiv.org/abs/2407.01489

[4] Wang et al. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. arXiv:2407.16741. https://arxiv.org/abs/2407.16741

[5] Zhang et al. AutoCodeRover: Autonomous Program Improvement. arXiv:2404.05427. https://arxiv.org/abs/2404.05427

[6] RepoGraph. Repository Graphs for AI Software Engineering. ICLR 2025 proceedings. https://proceedings.iclr.cc/paper_files/paper/2025/hash/4a4a3c197deac042461c677219efd36c-Abstract-Conference.html

[7] Feng et al. GraphCodeAgent: Graph-Based Code Retrieval and Context-Aware Reasoning for Repository-Level Code Generation. arXiv:2504.10046. https://arxiv.org/abs/2504.10046

[8] Code Graph Model: A Graph-Integrated LLM for Repository-Level Software Engineering. arXiv:2505.16901. https://arxiv.org/abs/2505.16901

[9] Packer et al. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560. https://arxiv.org/abs/2310.08560

[10] ACON: Optimizing Context Compression for Long-Horizon LLM Agents. arXiv:2510.00615. https://arxiv.org/abs/2510.00615

[11] Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366. https://arxiv.org/abs/2303.11366

[12] Harness Engineering for Language Agents. Preprints, 2026. https://www.preprints.org/manuscript/202603.1756

## Appendix A. Candidate Research Question and Claims

Research question:

How can a graph-guided code-repair harness preserve the evidential state of repository artifacts as long repair trajectories are compressed for planner, patch model, and validator contexts?

Central thesis:

Repository repair requires evidence-state context compression: the harness must preserve whether each artifact is a clue, code evidence, hypothesis, pending patch assumption, validation result, or invalidated strategy. RepairLedger implements this invariant through typed memory, action-gated promotion, consumer-specific projections, and rollback-preserving failure history.

Core claims:

1. Graph retrieval and memory are insufficient without artifact state.
2. Repair evidence should be promoted by explicit actions, not by presence in context.
3. Patch generation and official validation should be separated by an inspectable pending state when repair risk is high.
4. Rollback should restore files but preserve failed-patch evidence.
5. Benchmark hygiene is an evidence-state rule: test output may be behavior evidence, but benchmark test source should not become repair evidence.

## Appendix B. Suggested Ablations

1. No W/M separation: send all W code and previews to CGM.
2. No pending patch: replace propose/revise/submit with direct repair and test.
3. No repair history: remove failed patch previews, error origins, and CGM insights from later calls.
4. No evidence-chain contract: allow repair without target-in-chain and read-code requirements.
5. No benchmark-hygienic summary: compare against raw test output exposure where allowed by benchmark rules, or against over-redacted summaries.
6. No mechanism/owner_flow expansion: keep only graph neighbors and lexical search.

Expected effects:

- no W/M separation should increase graph evidence contamination;
- no pending patch should increase premature validation and invalid patch tests;
- no repair history should increase repeated failed strategies;
- no evidence-chain contract should increase unsupported target and API assumptions;
- no benchmark-hygienic summary should either leak privileged details or lose actual runtime behavior depending on implementation;
- no mechanism expansion should weaken cross-file and inheritance repairs.
