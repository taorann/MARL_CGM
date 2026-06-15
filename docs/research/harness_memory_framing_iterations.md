# Harness-Memory Framing Iterations

> Status note, 2026-06-11: this file is an intermediate framing step. The latest iteration is
> `docs/research/repairledger_evidence_state_iteration.md`, which broadens the idea from
> patch-centered ledgering to evidence-state context compression over graph-guided repair trajectories.

This file reframes the project from "another graph repair agent" to a harness-level contribution:
typed memory management and evidence-preserving context compression for repository-level repair agents.

The tone here is deliberately critical. Each iteration is evaluated as if by a skeptical top-tier
software engineering / ML systems reviewer.

## Why Reframe Around Harness And Memory?

The previous framing, "Evidence-Guided Patch Deliberation", is directionally correct but still too close
to a standard agent pipeline. It explains what our agent does, but not why the work is scientifically
distinct from SWE-agent + graph retrieval + self-refine.

The stronger angle is:

> In long-horizon repository repair, the harness is responsible for converting an expanding trajectory
> of searches, reads, model hypotheses, candidate patches, tests, and rollbacks into a compact state
> that preserves the epistemic role of each artifact. Existing work often manages context volume;
> our focus is managing repair evidence semantics.

This reframes `W/M/P/H` as a memory and compression design:

- `W`: exploratory working memory, including noisy previews and candidates;
- `M`: committed implementation evidence memory, hydrated and trusted by the patch generator;
- `P`: pending patch memory, an inspectable candidate not yet validated;
- `H`: repair history memory, compact failed-patch and insight records;
- runtime summaries: behavioral memory, benchmark-hygienic and actual-output based.

## Literature Signals Supporting This Angle

### Harness/scaffold is now a first-class object

- Confucius Code Agent argues that real-world coding success depends not only on the underlying LLM,
  but on scaffolding: orchestration, memory structures, and tool abstractions.
- Harness Engineering for Language Agents defines the harness layer as the extra-model layer that governs
  what an agent sees, what it can do, how state is carried forward, and how failures are handled.
- OpenDev frames terminal-native coding agents as systems whose hard problems include scaffolding,
  harness design, context engineering, and safety mechanisms.

### Context management is a known bottleneck, but mostly volume-oriented

- MemGPT introduces virtual context management and memory tiers inspired by operating systems.
- Acon optimizes context compression for long-horizon agents, reducing token usage while preserving performance.
- Context engineering discussions converge on write/select/compress/isolate, file-read caps, tool-result caps,
  compaction, deduplication, and subagent isolation.

These are important, but they are mostly about **which information fits**. In repository repair, the deeper
question is **what status each information item has**: clue, evidence, hypothesis, assumption, or validation.

### SWE repair systems expose but do not isolate repair-memory semantics

- SWE-agent/OpenHands expose tools and observations.
- Agentless/SWE-Fixer stage localization and editing.
- CGM/RepoGraph/CodeRAG retrieve graph-structured repository context.
- Reflexion/Self-Refine/SWE-Search use feedback or search.

But none of these is primarily framed as a harness memory protocol that preserves the semantic status of
repair artifacts across context compression and patch iterations.

## Reviewer-Style Critique Of The Previous Framing

### Previous title-level idea

"Evidence-Guided Patch Deliberation"

### Harsh review

This title sounds plausible but generic. "Evidence-guided" is an overused phrase and does not immediately
distinguish the paper from retrieval-augmented repair, Agentless localization, or test-feedback refinement.
"Patch deliberation" suggests a method-level tweak, not a fundamental systems problem. A reviewer could
reasonably ask: is this just adding a review step before running tests?

### What survives

- Evidence types matter.
- Pending patches as first-class state matter.
- Failed patches should be compressed into reusable memory.

### What must change

The work should be framed around **harness-level memory management**, not around "deliberation" alone.
The pending patch loop becomes one instance of a broader claim: repair harnesses need typed memory states,
because naive transcript context and naive summarization destroy evidence semantics.

## Iteration 1: "Typed Context For Software Repair Agents"

### Candidate story

Long-running repair agents accumulate many forms of context. We introduce typed context states that separate
search results, read code, planner hypotheses, patch candidates, and test feedback. This improves the quality
of the prompt sent to a patch generator.

### Harsh review

Still too vague. "Typed context" is easy to claim but hard to evaluate. It sounds like a prompt-engineering
taxonomy unless grounded in harness behavior. It also does not explain why memory compression is central.

### Keep

- Context artifacts have different roles.
- Mixing them creates errors.

### Revise

Use "memory" and "compression" more concretely: the harness compresses a long interaction history into a
small number of externally stored, typed memory blocks.

## Iteration 2: "Evidence-Preserving Context Compression"

### Candidate story

Existing agent harnesses compress context to fit finite windows, but repository repair requires compression
that preserves evidence semantics. We propose evidence-preserving context compression: a repair harness
maps raw tool traces into typed memories (`W/M/P/H`) so that downstream models receive compact context
without losing whether each artifact is a clue, evidence, hypothesis, patch assumption, or validation result.

### Harsh review

This is much better. It identifies a gap in context compression: most work compresses for size and relevance,
not for epistemic role. However, it may overclaim existing work ignores semantics; some systems have plans,
scratchpads, and memories. The claim should be narrower: existing coding-agent papers rarely make these
repair-specific trust states explicit or evaluate them.

### Keep

- Evidence-preserving compression as the conceptual core.
- Raw trajectory -> typed memory blocks.
- Repair-specific roles.

### Refine

Tie each memory block to a concrete failure mode and an action interface:

- `W` prevents search previews from becoming trusted repair context.
- `M` prevents CGM from seeing noisy or unhydrated code.
- `P` prevents unreviewed patches from consuming official test budget.
- `H` prevents failed patch assumptions from disappearing after rollback.

## Iteration 3: "A Repair Harness With Typed Evidence Memory"

### Candidate story

We present a repair harness that manages the memory hierarchy of a software repair agent. Rather than
append the trajectory or summarize it monolithically, the harness externalizes trajectory artifacts into typed
memory states: exploratory working memory, committed code evidence, pending patch memory, and repair
history. These states serve as evidence-preserving compression boundaries between the planner, patch model,
and validator.

### Harsh review

This is finally distinctive. It names the system object: a repair harness, not a model. It names the mechanism:
typed evidence memory, not just retrieval. It names the compression: preserving evidence status, not merely
shortening text. It can be compared cleanly against SWE-agent/OpenHands/CCA, MemGPT/Acon, Agentless/CGM,
and Reflexion/Self-Refine.

Remaining risk: it could still sound like an engineering pattern unless we define measurable outcomes.

### Metrics needed

- Context compression ratio: raw trajectory tokens vs. exposed planner/CGM state.
- Evidence contamination: preview/unread nodes included in patch-model context.
- Unsupported assumption rate: patch edits invoking APIs/types/symbols not present in `M` or runtime feedback.
- Premature validation rate: official tests run before all patch-critical mechanisms are in `M`.
- Strategy repetition rate: repeated failed patch intents after rollback.
- Resolved rate and invalid patch rate as downstream metrics.

## Current Best Story

### One-sentence claim

Repository-level repair agents need not only larger context windows or better graph retrieval, but a harness
memory system that compresses long repair trajectories into typed evidence states.

### Core thesis

Long-horizon software repair fails when the harness loses the epistemic status of trajectory artifacts.
Search previews become "known code", issue requirements become "existing mechanisms", planner hypotheses
become patch instructions, and failed patches vanish after rollback. These are not model-only failures; they
are memory-management failures in the repair harness.

### Method name candidates

1. **TEMPER**: Typed Evidence Memory for Patch-Generating Repair Agents.
2. **LedgerRepair**: A Repair Harness with Typed Evidence Ledgers.
3. **ECHO**: Evidence-Compressed Harness for Repository Repair.
4. **PatchLedger**: Evidence-Preserving Memory for Repository-Level Repair Agents.
5. **GraphLedger**: Graph-Guided Evidence Memory for Patch Deliberation.

Best current name: **PatchLedger**.

Rationale: "Ledger" suggests an auditable, append/select/compress record of claims, patches, and outcomes.
It is more concrete than "deliberation" and less generic than "memory".

### Final framing paragraph

PatchLedger is a repair harness for repository-level software agents. Its central abstraction is a typed
evidence ledger: an external memory structure that records exploratory code context, committed implementation
evidence, pending patch assumptions, and validation history as separate objects. The ledger acts as a
task-specific context compressor. It reduces the raw interaction history to a compact working set while
preserving whether each artifact is merely a navigation clue, a code fact, a planner hypothesis, an unvalidated
patch assumption, or a test-backed outcome. This allows a planner, a graph-aware patch generator, and a
validator to share repair state without flattening it into an undifferentiated prompt.

## Revised Contribution Claims

### Bad / too broad

- We build a better software engineering agent.
- We improve code graph retrieval.
- We introduce a new self-refinement loop.
- We solve long-horizon repair with memory.

### Better / defensible

- We identify **evidence-status collapse** as a harness-level failure mode in repository repair agents.
- We propose a **typed evidence ledger** that compresses repair trajectories into semantically distinct memory states.
- We instantiate this ledger in a two-model GraphPlanner/CGM harness with `W/M/P/H` state transitions.
- We show how the ledger exposes premature patching, unsupported patch assumptions, and weak runtime feedback in SWE-Bench Pro traces.

## Rewritten Abstract Candidate

Repository-level repair agents operate under severe context pressure: they must inspect large codebases, reason
across multi-file mechanisms, generate patches, and learn from test feedback over long trajectories. Existing
agent scaffolds and context-compression methods manage this pressure largely by truncating, summarizing, or
retrieving relevant content. We argue that software repair requires a stronger invariant: compression must preserve
the evidence status of each artifact. A search hit, a read function body, an issue requirement, a planner hypothesis,
a generated patch, and a failed test are not interchangeable context. Flattening them into a single prompt can turn
navigation clues into false facts, planner guesses into patch instructions, and failed edits into forgotten history.

We present **PatchLedger**, a repair harness that compresses long repair trajectories into a typed evidence ledger.
PatchLedger separates exploratory working context from committed implementation evidence, stores generated
patches as pending assumptions before validation, and records failed patches and runtime outcomes as compact
history shared by the planner and patch generator. Instantiated with a repository graph, a planner LLM, and a
graph-aware patch model, PatchLedger mediates what each model sees and when a patch is eligible for official
fail-to-pass/pass-to-pass testing. Through trace analysis on long-horizon SWE-Bench Pro tasks, we show that
common failures arise from evidence-status collapse--for example, treating issue requirements as existing
middleware or losing failed patch assumptions after rollback--and use these cases to motivate harness-level
memory policies for reliable software repair.

## Harsh Review Of This Abstract

### Strengths

- The object of study is clearer: harness memory, not model or retrieval.
- The term "evidence-status collapse" is crisp and names a real failure.
- The method is described as compression with semantic preservation.
- It connects directly to observed failures.

### Weaknesses

- "Through trace analysis" is weak unless backed by systematic evaluation.
- "Reliable software repair" at the end is too broad.
- It does not explicitly distinguish from MemGPT/Acon, which also compress context.
- It does not state train-free or graph-supported until late.

### Next revision target

Make the abstract less empirical-claim-heavy until experiments exist:

- Replace "we show" with "we analyze".
- Say "provides an auditable substrate" rather than "reliable repair".
- Name the difference from generic context compression: preserving repair roles.

## Abstract Candidate 2 / Current Best

Repository-level repair agents operate under severe context pressure: they must inspect large codebases, reason
across multi-file mechanisms, generate patches, and incorporate validation feedback over long trajectories.
Existing agent scaffolds and context-compression methods typically manage this pressure by selecting, truncating,
summarizing, or retrieving content. We argue that software repair requires a more specific invariant: compression
must preserve the **evidence status** of each artifact. A search hit, a read function body, an issue requirement,
a planner hypothesis, a generated patch, and a failed test are not interchangeable context. Flattening them into
one prompt can turn navigation clues into false facts, planner guesses into patch instructions, and failed edits
into forgotten history.

We present **PatchLedger**, a train-free repair harness that compresses long repair trajectories into a typed
evidence ledger. PatchLedger separates exploratory working context from committed implementation evidence,
stores generated patches as pending assumptions before validation, and records failed patches and runtime
outcomes as compact history shared by the planner and patch generator. Instantiated with a repository graph,
a planner LLM, and a graph-aware patch model, the harness governs what each model sees and when a patch is
eligible for official fail-to-pass/pass-to-pass testing. Trace analysis on SWE-Bench Pro tasks suggests that many
failures are not only generation errors but evidence-status failures, such as treating issue requirements as
existing mechanisms or losing failed patch assumptions after rollback. PatchLedger provides an auditable substrate
for studying and improving memory management in repository-level repair agents.

## Rewritten Introduction Candidate

### Paragraph 1: Problem

Repository-level software repair has become a central testbed for language-agent research. In SWE-bench and
SWE-Bench Pro, a system receives a real issue and a full repository, must discover the relevant implementation,
edit the code, and pass validation tests. These tasks are long-horizon and evidence-rich: an agent may need to
read many files, follow cross-file APIs, inspect runtime failures, generate and roll back patches, and decide
whether a failed attempt invalidates the current hypothesis or only a local edit.

### Paragraph 2: Existing angle

Prior work has improved this loop from several directions. SWE-agent, OpenHands, RepairAgent, and CCA show
that scaffolding, tools, and orchestration substantially affect performance. Agentless and SWE-Fixer show that
more controlled localization-and-editing pipelines can be competitive and cheaper than open-ended agents.
RepoGraph, CodeRAG, AutoCodeRover, and CGM improve repository context through structured retrieval and
code graphs. MemGPT, Acon, and broader context-engineering work address the finite context window through
memory tiers, selection, compression, and isolation. Reflexion, Self-Refine, SWE-Search, SWE-Gym, and R2E-Gym
show that feedback, search, and verifiers can improve later decisions.

### Paragraph 3: Gap

These advances make the harness around the model increasingly important. However, most work treats context
management as a problem of volume and relevance: which files, snippets, messages, or summaries should fit in
the next context window? Repository repair introduces an additional problem: the harness must preserve the
**epistemic role** of each artifact. A grep hit is not a code fact. A file preview is not a hydrated implementation
node. An issue requirement is not evidence that the mechanism exists. A planner plan is not a specification. A
candidate patch is not a verified edit. A failed test name is weaker feedback than a compiler error or assertion
message. When these distinctions are compressed away, the agent may act on stale or unsupported evidence even
when the right files were retrieved.

### Paragraph 4: Failure example

We call this failure mode **evidence-status collapse**. In one SWE-Bench Pro trace from Flipt, the issue required
coordinated changes to an authorization interface, two policy engines, gRPC middleware, and a namespace endpoint.
The agent retrieved several relevant files and produced a plausible patch, but the patch omitted middleware
context propagation, guessed OPA result types, and relied on context state not established by committed code
evidence. After rollback, the official harness exposed only failed test names, giving little help for correcting the
unsupported assumptions. The failure was not simply "bad retrieval" or "bad generation"; it was a harness memory
failure in which requirements, code facts, patch assumptions, and validation evidence were not kept distinct.

### Paragraph 5: Method

We introduce **PatchLedger**, a repair harness that treats software repair as typed memory management. Rather
than append the entire trajectory or summarize it monolithically, PatchLedger stores repair artifacts in an
evidence ledger. `W` contains exploratory working context such as search results, graph neighbors, and code
previews. `M` contains planner-committed implementation evidence whose code has been read and hydrated.
`P` contains a pending patch: a generated candidate that may encode useful assumptions but is not yet trusted.
`H` contains compact repair history, including failed patch previews, validation outcomes, and model insights.
The harness exposes different projections of this ledger to the planner, patch generator, and validator.

### Paragraph 6: Protocol

A planner LLM explores a repository graph, reads implementation code, and explicitly commits evidence into `M`.
A graph-aware patch model receives the issue, runtime behavior, `M`, and a concise planner intent, but not the
entire noisy trajectory. For high-risk repairs, the patch model first produces a pending patch. The planner then
chooses whether to submit it for official tests, revise it with a focused request, discard it, or gather more evidence.
Only submission applies the patch and runs fail-to-pass/pass-to-pass validation. Failed attempts are rolled back
but preserved in `H`, preventing patch assumptions from disappearing with the source snapshot.

### Paragraph 7: Contributions

This paper makes three contributions. First, it identifies evidence-status collapse as a harness-level failure mode
in long-horizon software repair. Second, it proposes typed evidence ledgers as a repair-specific context compression
mechanism that preserves the role of code, hypotheses, patches, and validation outcomes. Third, it instantiates the
idea in a train-free GraphPlanner/CGM harness and uses SWE-Bench Pro traces to analyze how typed memory boundaries
expose premature patching, unsupported API assumptions, and weak runtime feedback.

## Harsh Review Of The Introduction Candidate

### What works

- The first two paragraphs situate the paper in current literature without sounding like a survey dump.
- The gap is sharper: context compression preserves volume/relevance but not epistemic role.
- "Evidence-status collapse" is a memorable failure mode.
- The method naturally follows from the gap.

### What still needs improvement

- It needs citations in the final version, especially after every family claim.
- The Flipt example is useful but may be too detailed for intro; it can be shortened or moved to motivation.
- "Typed evidence ledger" must be formalized enough to not sound like renaming state variables.
- Evaluation still needs to prove this is more than conceptual clarity.

### Next iteration idea

The next version should make the intro more elegant:

- Use "context compression is not neutral" as the core rhetorical sentence.
- Make the failure example shorter.
- Move the full `W/M/P/H` detail to Method, keeping intro at a slightly higher abstraction.
