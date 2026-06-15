# Evidence-Guided Patch Deliberation: Draft Abstract And Introduction

> Status note, 2026-06-11: this draft is now superseded as the paper-level framing by
> `docs/research/patchledger_harness_memory_draft.md`. The useful parts here are the
> state protocol, evidence artifact taxonomy, and patch-deliberation mechanism. The
> top-level story should be harness memory and evidence-preserving context compression,
> not "patch deliberation" alone.

This file contains iterative drafts for a paper-style framing of our current method.
The goal is not to overclaim novelty, but to build a coherent story grounded in observed failures
and related work.

## Paper-Level Outline

1. **Introduction**
   - Repository-level repair is now a central evaluation setting.
   - Existing systems fall into agentic tool-use, staged agentless pipelines, graph/RAG methods, and feedback/refinement loops.
   - Their common bottleneck is not merely finding code or generating code, but deciding when code context and candidate patches are trustworthy enough to drive repair.
   - Our traces show two failure modes: search/issue hints treated as repair evidence, and failed patches not reused as structured evidence.
   - Introduce Evidence-Guided Patch Deliberation.

2. **Background And Motivation**
   - SWE-bench/SWE-Bench Pro complexity.
   - Graph RAG and CGM.
   - Agent interfaces and staged repair.
   - Flipt case study: why missing middleware evidence and thin Go failure output matter.

3. **Method**
   - State model: `G`, `W`, `M`, `P`, `H`.
   - Actions: search, read, commit, propose, revise, submit.
   - Evidence hierarchy: issue/runtime/code/history > planner intent.
   - Benchmark-hygienic runtime feedback.

4. **Implementation**
   - Planner protocol.
   - CGM payload construction.
   - Patch validation, rollback, F2P/P2P.
   - Telemetry.

5. **Evaluation**
   - SWE-bench and SWE-Bench Pro.
   - Ablations:
     - no W/M split;
     - direct repair vs. propose/submit;
     - no failed-patch history;
     - thin vs. enriched runtime feedback;
     - planner-only vs. planner+CGM review.
   - Metrics:
     - resolved rate;
     - patch attempts per issue;
     - invalid patch rate;
     - repeated failed-strategy rate;
     - evidence completeness at first patch.

6. **Analysis**
   - Success and failure traces.
   - When deliberation helps.
   - When model competence or missing runtime feedback dominates.

7. **Related Work**
   - SWE-bench family.
   - SWE-agent/OpenHands/RepairAgent.
   - Agentless/SWE-Fixer.
   - AutoCodeRover/REPOGRAPH/CGM.
   - Reflexion/Self-Refine/REx/SWE-Search/SWE-Gym.

8. **Limitations**
   - Stronger protocols cannot fully compensate for weak patch models.
   - Runtime output can be too sparse.
   - Single-candidate pending patch is cheaper but less exploratory than MCTS/multi-candidate methods.
   - Evaluation must avoid benchmark leakage.

## Abstract Iteration 1

Large language model agents have shown promise on repository-level software repair, but their performance remains limited by poor context selection, noisy tool traces, and premature patch validation. We present a graph-guided repair agent that separates repository exploration from patch generation. A planner explores a repository graph, reads relevant code, and commits a curated memory subgraph to a graph-aware patch model. The patch model proposes fixes, which are then validated against fail-to-pass tests. By combining graph retrieval, planner reasoning, and patch validation, our system improves evidence quality for real-world issue repair.

### Critique

This draft is too generic.

- It could describe SWE-agent + CGM.
- It does not mention the pending-patch deliberation loop.
- It says "improves" without evidence.
- It does not expose the central failure mode: retrieved context and candidate patches are not automatically trustworthy.

## Abstract Iteration 2

Repository-level software repair requires agents to coordinate issue descriptions, runtime failures, cross-file code context, and failed patch attempts. Existing SWE agents can navigate repositories and run tests, while agentless and graph-RAG systems can localize and edit code through structured pipelines. However, both paradigms often collapse distinct forms of evidence into a single prompt: search previews, fully read code, planner hypotheses, candidate patches, and test feedback are treated as comparable context. We introduce Evidence-Guided Patch Deliberation, a train-free repair protocol that separates exploratory context from trusted repair memory and separates candidate patch generation from official test submission. A planner constructs a hydrated memory subgraph for a graph-aware patch model, the patch model produces a pending patch, and the planner decides whether to submit, revise, discard, or gather more evidence. Failed patches and model insights are stored as compact repair history shared by both models. This design targets a practical bottleneck observed in SWE-Bench Pro traces: repairs fail not only because models cannot write code, but because they patch from incomplete evidence and receive weak feedback after failure.

### Critique

This is much stronger, but still has two issues.

- It describes the method but not the expected empirical contribution.
- "both paradigms often collapse distinct forms of evidence" is plausible but broad; the paper should phrase it as a design gap rather than an empirically universal claim.

## Abstract Iteration 3 / Current Best

Repository-level software repair is increasingly framed as an interactive process in which language models inspect code, edit files, and validate changes against tests. Yet recent agentic and agentless systems still struggle when a repair requires coordinating evidence across issue text, runtime behavior, repository structure, and prior failed patches. We argue that a key bottleneck is evidence trust: search hits, code previews, planner hypotheses, generated patches, and test feedback serve different roles, but are often merged into one undifferentiated prompt. We introduce **Evidence-Guided Patch Deliberation**, a train-free protocol for graph-supported software repair. The protocol separates exploratory working context from planner-curated repair memory, feeds only hydrated evidence subgraphs to a graph-aware patch model, and treats generated patches as pending candidates that must be reviewed, revised, discarded, or explicitly submitted for validation. Failed patches and concise model insights are retained as structured cross-model repair memory rather than free-form logs. This design bridges agentic exploration and graph-RAG patch generation: it preserves adaptive codebase navigation while imposing staged evidence gates around patch creation and testing. Our motivating traces on SWE-Bench Pro show that failures frequently arise from unsupported mechanism assumptions, guessed APIs, and thin runtime feedback; the proposed protocol makes these assumptions explicit and reusable, providing a foundation for more reliable repair iteration without training a new model.

## Introduction Iteration 1

Large language models are increasingly evaluated as software engineering agents: given a real issue and a repository, they must locate the relevant implementation, generate a patch, and pass tests. SWE-bench made this setting concrete by collecting real GitHub issues and pull-request-derived tests, showing that repository-level repair requires reasoning across files, functions, and execution feedback rather than generating isolated snippets. SWE-Bench Pro pushes the setting further with longer-horizon, multi-language, multi-file engineering tasks.

Existing systems have explored several design points. SWE-agent and OpenHands emphasize agent-computer interfaces that allow a model to inspect files, edit code, and run tests. RepairAgent and similar systems let an LLM autonomously gather information, generate fixes, and react to feedback. Agentless and SWE-Fixer show that fixed localization-and-repair pipelines can be simpler, cheaper, and competitive. AutoCodeRover, REPOGRAPH, and Code Graph Model demonstrate the value of repository structure and graph-based context retrieval. Reflexion, Self-Refine, and related refinement methods show that language feedback and failed attempts can improve subsequent actions.

Despite these advances, our traces suggest a recurring bottleneck: repair systems often lack a precise protocol for deciding which context is trusted evidence. A search result may only be a preview; a read code node may be complete evidence; a planner statement may be a hypothesis; a generated patch may encode unsupported assumptions; a failed test may reveal a concrete compiler error or only a test name. When these artifacts are all placed into a prompt as undifferentiated text, the patch model can overfit to issue hints, invent missing APIs, or repeat failed strategies.

We propose Evidence-Guided Patch Deliberation. The method maintains a repository graph `G`, a working subgraph `W`, a curated repair memory `M`, a pending patch `P`, and a compact repair history `H`. The planner can search and expand the repository graph, but search previews are treated only as orientation. Concrete code enters the repair memory only after being read and explicitly committed. A graph-aware patch model receives the issue, runtime behavior, the memory subgraph, and a concise planner intent, but not the entire noisy working context. For high-risk or multi-file repairs, the patch model first proposes a pending patch. The planner must then inspect the candidate and choose whether to submit it for official tests, revise it with a focused request, discard it, or gather more code evidence. Failed patches and patch-model insights are preserved as structured history shared by both planner and patch model.

This design is motivated by failures in complex tasks. In a SWE-Bench Pro issue from Flipt, the model correctly identified that authorization engines needed a namespace-evaluation interface, but the first candidate patch omitted middleware context propagation and guessed OPA result types. The official test feedback exposed failed test names but not enough Go runtime detail to quickly diagnose the API and type errors. The failure was not simply a lack of model capacity or a lack of graph retrieval: it was a breakdown in evidence trust and feedback reuse. The system had to distinguish issue requirements from existing mechanisms, verify whether middleware code was actually in memory, and preserve the failed patch assumptions for subsequent repair.

Our contribution is therefore not a new code graph model, a new benchmark, or a fully trained software engineering agent. Instead, we contribute a repair protocol that coordinates existing ingredients: graph-guided exploration, planner-curated memory, graph-aware patch generation, pending-patch review, rollback-based validation, and compact cross-model repair history. The protocol is train-free and can be implemented around existing LLM endpoints and CGM-style patch generators. It also produces structured trajectories that could later support verifier training or policy learning.

### Critique

This introduction is coherent but still reads as a system proposal, not a polished conference introduction.

- It introduces too many systems in one paragraph without grouping them into the argument.
- The Flipt case is useful, but it appears before the reader fully understands the proposed state machine.
- It needs sharper "gap" language: prior work separately studies interface, pipeline, graph retrieval, and feedback, but not the trust boundaries between evidence types.
- It needs a cleaner contribution list.

## Introduction Iteration 2 / Current Best

Software engineering agents are no longer evaluated only on short programming exercises. In SWE-bench, a system receives a real GitHub issue and an entire repository, and must produce a patch that passes issue-specific tests. This setting requires repository-scale reasoning: relevant behavior may span multiple files, the issue description may be incomplete or misleading, and the final answer must integrate with existing APIs and tests. SWE-Bench Pro intensifies this challenge with longer-horizon, multi-file, multi-language tasks drawn from realistic software projects.

The community has responded with several complementary approaches. Agentic systems such as SWE-agent, OpenHands, and RepairAgent give models computer-facing tools for inspecting files, editing code, running commands, and adapting to feedback. Pipeline systems such as Agentless and SWE-Fixer reduce autonomy by decomposing repair into localization and editing stages, often improving interpretability and cost. Repository-context methods such as AutoCodeRover, REPOGRAPH, and CGM use structured search, graphs, or graph-integrated models to retrieve relevant code. Feedback methods such as Reflexion, Self-Refine, SWE-Search, and SWE-Gym show that failed attempts, reflections, search, and verifiers can improve later decisions.

These lines of work reveal a shared lesson: performance depends not only on the base model, but on the form in which repository evidence is exposed. However, they leave open a more specific protocol question: once an agent has many kinds of context, which of them should be trusted by the patch generator, and when should a generated patch be trusted enough to run official tests? In practice, a repair trajectory contains artifacts with very different epistemic status. A search hit is a navigation clue, not full code evidence. A read function body is stronger evidence, but may still miss its caller or consumer. A planner plan is a hypothesis. A generated patch encodes assumptions about imports, APIs, types, and data flow. A failed test may contain a precise compiler error, or only the name of a still-failing test. Treating all of these artifacts as ordinary prompt text makes it easy for a model to patch from stale hypotheses or repeat a failed strategy.

Our traces expose this failure mode. In a SWE-Bench Pro issue from Flipt, the repair required changes to an authorization interface, two policy engines, gRPC middleware, and a namespace endpoint. A first candidate patch identified part of the intended mechanism, but omitted middleware propagation, guessed OPA result types, and relied on context state that had not been established in code evidence. The failure was not simply that no relevant code was retrieved: much of the right code was present, but the system lacked a strict boundary between candidate context, committed evidence, planner assumptions, and patch assumptions. After validation failed, the feedback was also too thin to guide rapid correction. This is representative of a broader long-horizon repair problem: the agent needs not just more context, but a protocol for converting context, patches, and failures into reliable repair evidence.

We introduce **Evidence-Guided Patch Deliberation**, a train-free protocol for graph-supported repository repair. The protocol maintains five typed states. `G` is a read-only repository graph. `W` is the planner's working subgraph, containing candidates and code previews. `M` is a planner-curated repair memory whose nodes must be read and hydrated before they are sent to the patch model. `P` is a pending patch generated by the patch model but not yet submitted to official tests. `H` is compact repair history, including failed patch previews, test outcomes, and patch-model insight summaries. The planner explores `G`, reads code into `W`, and explicitly commits trusted nodes into `M`. A graph-aware patch model receives the issue, runtime behavior, `M`, and a concise planner intent. For high-risk or multi-file repairs, it produces a pending patch. The planner then decides whether to submit, revise, discard, or read more evidence. Only submission applies the patch and runs fail-to-pass and pass-to-pass validation.

The central design principle is evidence priority. Issue text, runtime behavior, hydrated code, and repair history have priority over planner intent. Planner intent is useful as a hypothesis and focusing device, but it must not override code evidence. Likewise, failed patches are not merely discarded; their concrete edits and assumptions become structured history visible to both planner and patch model. This creates a lightweight form of repair-specific reflection without training a new model or storing unbounded chain-of-thought logs.

This paper makes three contributions. First, we formulate evidence trust as a protocol problem in repository-level repair, distinguishing exploratory context, committed repair evidence, candidate patch assumptions, and validation feedback. Second, we present a two-model implementation that combines graph-guided exploration with CGM-style patch generation through explicit `W/M/P/H` state transitions. Third, we analyze failures on long-horizon repair tasks to show why graph retrieval and powerful patch models are insufficient without pre-generation evidence curation and post-generation patch deliberation.

## Related Work Hooks For The Introduction

- Cite SWE-bench/SWE-Bench Pro when motivating repository-level and long-horizon difficulty.
- Cite SWE-agent/OpenHands/RepairAgent for agent-computer interfaces and autonomous tool use.
- Cite Agentless/SWE-Fixer for staged localization/repair and the case against overly complex agents.
- Cite AutoCodeRover/REPOGRAPH/CGM for structured and graph-based repository context.
- Cite Reflexion/Self-Refine/SWE-Search/SWE-Gym for iterative feedback, reflection, search, and verifiers.
- Cite DEVLoRe for the importance of multiple software artifacts, especially issue and runtime/stack traces.

## Current Abstract + Introduction Weaknesses

1. **No evaluation claims yet.** The draft is method/framing-heavy. It needs experimental results before submission.
2. **"Evidence trust" needs operational metrics.** Potential metrics:
   - percentage of repair attempts whose target nodes are all in `M`;
   - percentage of submitted patches whose assumptions are supported by read code;
   - repeated failed-strategy rate;
   - invalid patch or compile-error rate;
   - number of official test submissions per resolved issue.
3. **Flipt case may sound anecdotal.** Need at least 3-5 trace case studies or aggregate statistics.
4. **Need careful benchmark hygiene language.** We should say runtime output is used, test source is excluded.
5. **Need not overpromise train-free.** Train-free is a constraint and advantage, but not necessarily a performance claim.

## Next Revision Targets

- Add a small taxonomy of evidence artifacts:
  - navigation clue;
  - implementation evidence;
  - behavioral evidence;
  - patch assumption;
  - validation evidence.
- Turn the contribution list into paper-ready bullets after experiments.
- Add an ablation plan to show each state boundary matters.
- Add qualitative trace diagrams for successful and failed runs.

## Evidence Artifact Taxonomy

This taxonomy is the conceptual center of the paper. It is more defensible than claiming novelty in
agents, graphs, or feedback alone.

| Artifact type | Example | Failure if misused | Protocol treatment |
| --- | --- | --- | --- |
| Navigation clue | `explore_find` preview, `grep_code` hit, graph neighbor | Planner treats a preview as full code and patches too early | May enter `W`; must be `read` before `memory_commit` |
| Implementation evidence | Full function/class/file window read from implementation code | Evidence is stale, too narrow, or missing caller/consumer | May enter `M` only through explicit `memory_commit` |
| Behavioral evidence | Issue text, runtime output, fail-to-pass status, stack/error messages | Test source leakage or too-thin feedback | Summarize actual runtime output only; exclude benchmark test source |
| Planner hypothesis | `intent_analysis`, evidence-chain roles, confidence | CGM follows a wrong plan over code evidence | Lower priority than issue/code/runtime/history |
| Patch assumption | Generated import, API call, type assertion, context key, data-flow change | Patch compiles poorly or encodes unsupported mechanism | Pending patch must be reviewed before submit |
| Validation evidence | F2P/P2P result, syntax/compile failure, failed patch preview | Failed strategy is forgotten or repeated | Store in `repair_attempts` and share with planner + CGM |

## Abstract Iteration 4 / Polished Version

Repository-level software repair requires language models to transform heterogeneous evidence--issue text, runtime behavior, cross-file code structure, candidate patches, and validation feedback--into a correct code change. Existing agentic systems expose rich tools for repository interaction, while agentless and graph-RAG systems improve localization and context retrieval. However, these approaches often leave implicit which artifacts are trustworthy repair evidence and when a generated patch is ready for validation. We introduce **Evidence-Guided Patch Deliberation**, a train-free protocol for graph-supported software repair that makes these trust boundaries explicit. The protocol separates exploratory working context from planner-curated repair memory, feeds only hydrated code evidence to a graph-aware patch model, and treats generated patches as pending candidates that must be reviewed, revised, discarded, or submitted. Failed patches and concise patch-model insights are retained as structured repair history shared by both the planner and patch generator. This design bridges adaptive agentic exploration with controlled graph-based patch generation: it preserves repository navigation while imposing evidence gates before and after patch creation. Motivated by SWE-Bench Pro traces in which models patched from unsupported mechanism assumptions and thin runtime feedback, our protocol provides an auditable foundation for iterative repair without training a new model.

### Why This Is Better Than Iteration 3

- Starts with heterogeneous evidence, not "agents are hard".
- Clearly distinguishes prior work families before naming the gap.
- Moves "trust boundaries" to the center.
- Avoids claiming empirical improvement before experiments.
- Says "auditable foundation" rather than "more reliable repair" as a guaranteed outcome.

## Introduction Iteration 3 / Polished Version

Real software repair is an evidence-integration problem. Given a natural-language issue and a repository, a repair system must infer the faulty mechanism, locate the relevant implementation, produce an edit compatible with existing APIs, and verify that the change fixes the reported behavior without regressions. SWE-bench established this setting using real GitHub issues and pull-request-derived tests, and SWE-Bench Pro extends it to longer-horizon tasks across larger, multi-language projects. These benchmarks expose a gap between standalone code generation and repository-level engineering: the model must coordinate issue text, runtime feedback, code structure, and validation results across many turns.

Recent systems attack this problem from several directions. Agentic systems such as SWE-agent, OpenHands, and RepairAgent provide model-facing interfaces for inspecting files, editing code, running commands, and adapting to feedback. Pipeline systems such as Agentless and SWE-Fixer reduce autonomy by decomposing repair into localization and patch generation. Repository-context methods such as AutoCodeRover, REPOGRAPH, CodeRAG, and CGM improve the retrieval or representation of relevant code through structured search and graph-based context. Feedback and scaling methods such as Reflexion, Self-Refine, SWE-Search, SWE-Gym, and R2E-Gym study how failed attempts, reflections, candidate search, and verifiers can improve later decisions. Together, these works show that repair performance is shaped not only by model size, but also by interface design, retrieval quality, feedback, and validation.

Yet a key protocol question remains under-specified: after an agent gathers many artifacts, which of them should be trusted by the patch generator, and when should a generated patch be trusted enough to run official validation? A repair trajectory mixes artifacts with very different epistemic roles. A search hit is a navigation clue, not proof of a bug. A full function read is implementation evidence, but may still miss an upstream caller or downstream consumer. An issue requirement describes desired behavior, not necessarily existing code. A planner plan is a hypothesis. A generated patch contains assumptions about imports, APIs, types, context keys, and data flow. A failed test may reveal a precise compiler error, or only the name of a failing selector. When these artifacts are flattened into a single prompt, a model can mistake orientation for evidence, follow a stale plan over code, or repeat a failed patch strategy.

We observed these failures in our own repair traces. In a SWE-Bench Pro issue from Flipt, the expected fix spanned an authorization interface, two policy engines, gRPC middleware, and a namespace-listing endpoint. A candidate patch identified part of the issue mechanism but omitted middleware propagation, guessed OPA result types, and relied on context state that had not been established in the committed evidence. The failure was not simply missing retrieval: several relevant files had been found. Nor was it simply weak generation: the generated patch encoded a plausible but unsupported mechanism. The missing layer was an explicit protocol for converting retrieved code, planner hypotheses, generated edits, and test feedback into typed repair evidence.

We propose **Evidence-Guided Patch Deliberation**, a train-free protocol for graph-supported repository repair. The protocol maintains five states. `G` is a read-only repository graph. `W` is an exploratory working subgraph containing candidates, previews, and read code. `M` is planner-curated repair memory; nodes enter `M` only after code is read and hydrated. `P` is a pending patch generated by the patch model but not yet submitted to official tests. `H` is compact repair history containing failed patch previews, validation outcomes, and patch-model insight summaries. A planner explores `G`, reads code into `W`, and commits trusted evidence into `M`. A graph-aware patch model receives the issue, runtime behavior, `M`, and a concise planner intent. For multi-file or high-risk repairs, it produces `P`; the planner then chooses whether to submit, revise, discard, or gather more evidence. Only submission applies the patch and runs fail-to-pass and pass-to-pass validation.

The method is built around an evidence-priority rule: issue text, actual runtime behavior, hydrated implementation code, and repair history outrank planner intent. Planner intent remains useful as a focusing hypothesis, but the patch model should not treat it as a specification when code or failure history disagrees. Conversely, failed patches are not discarded as noise. Their concrete edits and assumptions become structured history visible to both the planner and the patch model. This gives the system a repair-specific form of reflection without storing long chain-of-thought logs or training a new policy.

Our contribution is not a new benchmark, a new code graph model, or a claim that deliberation alone solves repository repair. Rather, we make three narrower contributions. First, we identify evidence trust as a protocol problem in repository-level repair and distinguish navigation clues, implementation evidence, behavioral evidence, planner hypotheses, patch assumptions, and validation evidence. Second, we instantiate this view in a two-model graph repair system that separates exploratory context from committed repair memory and separates patch proposal from official validation. Third, we show through trace analysis how unsupported mechanism assumptions, guessed APIs, and thin runtime feedback lead to failed repairs, motivating evidence gates and structured cross-model patch history as practical design principles for future SWE agents.

### Remaining Weaknesses After Iteration 3

- It is still a methods/framing introduction. A final paper needs empirical results and ablations.
- It depends on trace evidence; we need a systematic trace analysis section.
- The phrase "evidence trust" must be operationalized in implementation and metrics.
- Some related systems may already have implicit versions of these boundaries; we should claim explicit protocolization, not invention from nothing.
