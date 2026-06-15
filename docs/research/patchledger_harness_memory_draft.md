# PatchLedger: Harness-Level Evidence Memory For Repository Repair Agents

> Status note, 2026-06-11: this draft is now considered too patch-centric. The stronger
> current framing is `docs/research/repairledger_evidence_state_iteration.md`: RepairLedger
> as evidence-state context compression over graph-guided repair trajectories. Patch
> management remains one artifact lifecycle, not the whole paper-level contribution.

This document is the current preferred paper framing. It deliberately moves away from
"we built another graph repair agent" and from the earlier, weaker title "Evidence-Guided
Patch Deliberation." The paper-level object is now a repair harness memory protocol.

Working title:

> PatchLedger: Evidence-Preserving Context Compression for Repository-Level Repair Agents

## 1. Core Reframing

### Old angle, now demoted

The previous framing centered on a patch-generation workflow:

- a planner explores a repository;
- a CGM-like model generates a patch;
- the planner reviews or revises before testing.

This is operationally true, but not novel enough as a paper story. A skeptical reviewer can
reduce it to "SWE-agent plus graph retrieval plus a review step." That is a bad position:
each ingredient has a strong prior art family, and the novelty would look like glue code.

### New angle

Repository repair agents are long-horizon systems whose harness must repeatedly compress
an expanding trajectory into a small prompt state. That trajectory contains heterogeneous
artifacts: search hits, read code, issue claims, planner hypotheses, candidate patches,
syntax failures, test failures, rollbacks, and previous model summaries.

The central claim:

> For repository repair, context compression is not only a token-budget problem. It is an
> evidence-status preservation problem.

When a harness compresses a repair trajectory, it must preserve whether an artifact is a
navigation clue, implementation evidence, behavioral evidence, a hypothesis, a pending
patch assumption, or validation evidence. If those roles collapse into undifferentiated
prompt text, the agent can patch from previews, obey stale plans, invent APIs, forget
failed edits after rollback, or spend official tests on unreviewed candidate patches.

We call this failure mode:

> evidence-status collapse.

PatchLedger is a harness-level response: a typed evidence ledger that compresses repair
trajectories while preserving the role, provenance, and validation status of each artifact.

## 2. Related Work Landscape And Gap

### 2.1 Harness and scaffold work

Recent coding-agent systems increasingly treat scaffold and harness design as central.
Confucius Code Agent argues that real-world coding agents must operate over massive
repositories, sustain long sessions, coordinate toolchains, and use advanced context
management and persistent notes. OpenDev similarly frames terminal coding agents around
scaffolding, harness design, adaptive context compaction, memory, and safety controls.
Harness Engineering for Language Agents explicitly defines the harness as the extra-model
layer controlling what an agent sees, what it can do, how state is carried forward, and how
failures are handled.

Gap:

These works validate the harness as a first-class systems object, but their memory
abstractions are usually broad: hierarchical working memory, notes, adaptive compaction,
or runtime state. They do not isolate the repair-specific problem that code context,
planner claims, generated edits, and validation outcomes have different evidential roles.

### 2.2 General memory and context compression

MemGPT treats context management as a virtual-memory problem over multiple memory tiers.
ACON optimizes context compression for long-horizon agents by learning natural-language
compression guidelines from failures. Surveys of LLM-agent memory identify memory as a
key component for long-term agent-environment interaction.

Gap:

These works optimize access to long histories, but they are mostly task-general. They do
not define the software-repair invariant: a compressed trajectory should preserve whether
a code artifact is only a clue, whether a patch assumption is supported by hydrated code,
and whether a failed patch has invalidated a strategy.

### 2.3 Software repair agents and pipelines

SWE-agent demonstrates that agent-computer interface design can substantially affect
software-repair performance. Agentless shows that a simple localization-repair-validation
pipeline can be competitive and cheaper than complex autonomous agents. CGM, REPOGRAPH,
CodeRAG, and AutoCodeRover show the importance of graph- or structure-aware repository
context.

Gap:

These systems make different points about interaction, staging, and retrieval, but the
status of the retrieved or generated artifacts often remains implicit. A graph subgraph is
not automatically trusted repair evidence. A localization result is not necessarily a
complete mechanism. A generated patch is not necessarily ready for validation. A failed
patch is not merely a bad output; it is evidence about unsupported assumptions.

### 2.4 Where our work fits

PatchLedger does not compete with graph retrieval, agent-computer interfaces, or better
patch models. It is a harness memory layer that sits between them:

- it turns graph search and read results into typed memory;
- it decides which memory is exposed to the patch model;
- it treats patch generation as a candidate-producing step, not an automatic validation step;
- it compresses failed patches into reusable evidence for later planner and patch-model calls.

This is a narrower but more defensible novelty claim than "better repair agent."

## 3. Method Claim

PatchLedger is a train-free repair harness protocol. It manages four repair memories plus
runtime behavior evidence:

| State | Meaning | What it prevents |
| --- | --- | --- |
| `W`, working memory | exploratory graph context, search hits, previews, read snippets | treating every retrieved item as trusted repair context |
| `M`, committed evidence memory | hydrated code evidence selected for patch generation | sending noisy or unread nodes to CGM |
| `P`, pending patch memory | parsed, normalized, syntax-checked candidate patch not yet officially validated | spending official tests before patch review |
| `H`, repair history | compact records of failed patches, assumptions, outcomes, and insights | repeating failed strategies after rollback |
| behavioral memory | issue text and actual runtime/test output summaries, excluding benchmark test source | leaking tests or losing concrete runtime symptoms |

### Context compiler

The harness acts as a context compiler. Its input is a raw trajectory:

```text
issue -> test behavior -> search/read/expand -> planner hypotheses
      -> CGM patch -> patch parser -> syntax/test result -> rollback
```

Its output is not one summary. It is a set of typed projections:

- planner context: issue, recent actions, W/M status, blockers, pending patch summary,
  recent failed attempts, and allowed next decisions;
- CGM context: issue, behavior evidence, M code graph and snippets, concise planner intent,
  pending/revision context, failed patch records, and constraints;
- telemetry context: full raw trace plus compact summaries for audit.

### Compression invariants

PatchLedger compression should preserve four invariants:

1. **Status preservation.** The compressed record says whether an artifact is clue, code
   evidence, hypothesis, patch assumption, or validation evidence.
2. **Provenance preservation.** Code evidence is linked to node id, path, lines, read action,
   and hydration state. Patch assumptions are linked to the generated edit that introduced them.
3. **Consumer-specific projection.** Planner and CGM do not need the same context. CGM should
   receive committed code evidence and repair history, not noisy W previews or long planner
   visible thinking.
4. **Failure retention.** Rollback restores files but should not erase what the failed patch
   taught the harness.

## 4. Harsh Iteration Log

### Iteration A: "A Better Graph Repair Agent"

Claim:

> We combine graph retrieval, planner exploration, and CGM patch generation to improve repair.

Reviewer verdict:

Reject. The contribution is a bundle of known components. Graph retrieval has strong prior
work. Planner exploration has strong prior work. Patch generation and validation are standard
in SWE-bench agents. Without a more precise systems claim, this sounds like engineering glue.

What survives:

The system exposes a real design pain: the planner and patch model keep seeing context whose
trust level is unclear.

### Iteration B: "Evidence-Guided Patch Deliberation"

Claim:

> We separate candidate patch proposal from official validation and let the planner review,
> revise, or submit the patch.

Reviewer verdict:

Borderline at best. This is useful, but it risks sounding like a workflow tweak. A reviewer
will ask why this is not just Self-Refine, Reflexion, SWE-Search-lite, or a verifier stage.
The title explains a mechanism, not the underlying research problem.

What survives:

Pending patches are important, but only as one memory type in a larger repair-harness
compression protocol.

### Iteration C: "Evidence-Preserving Context Compression"

Claim:

> Repository repair needs context compression that preserves the evidential status of each
> artifact.

Reviewer verdict:

Much stronger. This defines a gap relative to general context compression: current methods
select or summarize useful information, but repair needs typed evidence roles. However, this
still risks sounding like a generic memory taxonomy unless the paper shows how the types
control actions, eligibility for patching, and validation.

Revision:

Tie every memory type to a concrete control decision: whether a node can enter CGM input,
whether a patch can be submitted, whether a failed strategy should be retried, and whether
planner intent can override code evidence.

### Iteration D: "PatchLedger"

Claim:

> PatchLedger is a repair harness that compresses long repair trajectories into an auditable
> typed evidence ledger used by the planner, patch generator, and validator.

Reviewer verdict:

This is the best current story. It names the object of study, the mechanism, and the failure
mode. It does not claim graph retrieval or self-refinement as novel. The main remaining risk
is evaluation: without quantitative trace analysis and ablations, the paper will read as a
design essay.

Required evidence:

- show context compression ratios from raw traces to planner/CGM prompts;
- count evidence-status collapse events in failed traces;
- measure premature validation rate before and after propose/submit;
- measure unsupported API/type assumptions in submitted patches;
- measure repeated failed-strategy rate with and without `H`;
- show resolved/invalid-patch changes on SWE-bench/SWE-Bench Pro subsets.

## 5. Polished Abstract, Iteration 1

Repository-level software repair agents operate under a severe context-management constraint:
they must inspect large codebases, reason across multi-file mechanisms, generate patches, and
learn from validation failures over long trajectories. Existing harnesses, coding agents, and
context-compression methods address this pressure by improving tool interfaces, retrieving
relevant code, summarizing older observations, or maintaining long-term memory. We argue that
software repair requires a more specific invariant: compressed context must preserve the
evidence status of each artifact. A search hit, a hydrated function body, an issue requirement,
a planner hypothesis, a generated patch, and a failed test are not interchangeable. When a
repair harness flattens them into ordinary prompt text, navigation clues can become false code
facts, planner guesses can become patch instructions, and failed edits can disappear after
rollback.

We present **PatchLedger**, a train-free repair harness that compresses long repair trajectories
into a typed evidence ledger. PatchLedger separates exploratory working context from committed
implementation evidence, stores generated patches as pending assumptions before validation, and
records failed patches and runtime outcomes as compact history shared by the planner and patch
generator. Instantiated with a repository graph, a planner LLM, and a graph-aware patch model,
PatchLedger controls which artifacts enter patch-generation context and when a candidate patch
is eligible for fail-to-pass and pass-to-pass testing. Our trace analysis motivates evidence-
preserving compression as a practical design principle for repository repair agents: the harness
must manage not only how much context is retained, but what each retained artifact is allowed
to mean.

### Harsh review

Strengths:

- The novelty is now harness memory, not "better agent."
- The first paragraph creates a clear gap relative to existing context compression.
- "What each retained artifact is allowed to mean" is a good closing phrase.

Weaknesses:

- "Our trace analysis motivates" is weak without results. If the paper has only a few traces,
  say "we use trace studies" rather than implying broad empirical proof.
- It still does not mention graph evidence enough; CGM could seem incidental.
- The abstract is slightly abstract. It needs one concrete failure example or metric hook in
  the last sentence if results exist.

## 6. Polished Abstract, Iteration 2 / Current Best

Repository-level software repair agents must compress long trajectories of searches, code reads,
model hypotheses, candidate patches, validation failures, and rollbacks into finite model context.
Existing coding-agent harnesses and context-compression methods mainly address this pressure by
improving tool interfaces, retrieving relevant code, summarizing old observations, or maintaining
long-term memory. We argue that repair requires a stricter compression invariant: the harness must
preserve the evidence status of each artifact. A search hit is a navigation clue; a hydrated function
body is implementation evidence; a planner plan is a hypothesis; a generated edit is an unvalidated
assumption; and a failed test is validation evidence. Collapsing these roles into undifferentiated
prompt text causes agents to patch from previews, follow stale plans over code, invent unsupported
APIs, or forget failed strategies after rollback.

We introduce **PatchLedger**, a train-free repair harness that implements evidence-preserving
context compression for repository-level repair. PatchLedger maintains a typed evidence ledger:
working memory for exploratory graph context, committed memory for hydrated code evidence, pending
patch memory for candidate edits, and repair history for failed patch assumptions and outcomes.
The ledger is projected differently to the planner, graph-aware patch generator, and validator, so
each component receives compact context with explicit evidence status. This protocol separates
retrieval from trust, patch proposal from official validation, and rollback from forgetting. We
position PatchLedger as a harness layer complementary to graph retrieval and stronger patch models,
and propose trace-level metrics for evidence-status collapse, premature validation, unsupported
patch assumptions, and repeated failed strategies.

### Harsh review

This is closer to a top-conference abstract because it does not overclaim benchmark wins before
we have them. It names the scientific object, the failure mode, and the measurement plan. The risk
is that "propose trace-level metrics" sounds incomplete. For a final submission, replace that phrase
with actual measured results.

## 7. Introduction Draft, Harness-Memory Version

Repository-level software repair is increasingly a harness problem. A model is given an issue and a
codebase, but the success of the resulting agent depends on the layer that mediates repository
inspection, action selection, patch generation, validation, rollback, memory, and trace recording.
This layer determines not only which tools are available, but also which information survives across
turns and how old information is compressed into the next model call. As SWE-bench-style tasks become
longer, more multi-file, and more multilingual, this context-management layer becomes a central
determinant of repair reliability rather than an implementation detail.

Recent systems make this shift visible. SWE-agent shows that an agent-computer interface can change
how effectively a language model navigates, edits, and tests code. Agentless shows that a staged
localization-repair-validation pipeline can be competitive with more autonomous loops. CGM, REPOGRAPH,
CodeRAG, and AutoCodeRover improve repository context using structure-aware retrieval or graph models.
Confucius Code Agent and OpenDev make scaffolding, memory, adaptive context compaction, and terminal
harness design explicit parts of coding-agent performance. General memory and compression work such as
MemGPT and ACON treats long-horizon context as a first-class resource. Together, these works suggest a
clear lesson: software-engineering agents are not only models plus prompts; they are models embedded in
memory-bearing, action-mediating harnesses.

However, repository repair exposes a limitation in the usual framing of context management. The problem
is not merely that the trajectory is too long, nor only that the agent must retrieve the right code. A
repair trajectory mixes artifacts with different epistemic roles. A search result may only indicate
where to look. A read function body may be reliable implementation evidence, but only for the lines it
covers. An issue statement describes desired behavior, not necessarily current code. A planner plan is
a hypothesis. A generated patch encodes assumptions about APIs, types, imports, control flow, and data
flow. A failed test can be a precise compiler error or only a selector name. Compressing such a trajectory
into a generic summary can destroy the distinctions the next model call needs most.

We call this failure **evidence-status collapse**. In our repair traces, collapse appears in recurring
forms: search previews are treated as if the code had been read; issue requirements are mistaken for
existing mechanisms; planner hypotheses override implementation evidence; candidate patches are tested
before their API and data-flow assumptions are checked; and failed edits vanish after rollback, causing
later calls to repeat the same strategy. These failures are not fully explained by weak code generation
or poor retrieval. They arise because the harness loses the role and provenance of the artifacts it
compresses.

We introduce **PatchLedger**, a train-free harness protocol for evidence-preserving context compression
in repository repair. PatchLedger stores the repair trajectory in a typed evidence ledger rather than a
flat transcript. The ledger separates exploratory working memory (`W`) from committed code evidence
(`M`), pending patch assumptions (`P`), and repair history (`H`). `W` may contain noisy graph search
results, previews, and read candidates. `M` contains hydrated implementation code selected for patch
generation. `P` contains a parsed candidate patch that has passed lightweight validation but has not
yet consumed official fail-to-pass/pass-to-pass tests. `H` stores compact failed-patch records, runtime
outcomes, and patch-model insight summaries. Runtime behavior evidence is summarized from actual outputs
without exposing benchmark test source.

The ledger acts as a task-specific context compiler. For the planner, it exposes what has been found,
what has been read, what has been committed, what patch is pending, and what failures have invalidated
previous assumptions. For the graph-aware patch generator, it exposes the issue, behavioral evidence,
hydrated memory subgraph, concise planner intent, and recent failed patch records, but not noisy search
previews or long free-form reasoning. For the validator, it makes patch submission an explicit action:
a candidate patch can be revised, discarded, or submitted, and only submission runs official target tests.
Rollback restores the repository but not the memory of what failed.

This design reframes several common repair-agent decisions as memory-status transitions. Retrieval is
not trust: a node found by search enters `W`, but it does not become CGM-facing evidence until read and
committed into `M`. Patch generation is not validation: a generated edit enters `P`, where the planner
can inspect its assumptions before testing. Failure is not deletion: a rejected patch is compressed into
`H`, so both the planner and patch model can avoid repeating the same unsupported mechanism. Planner
intent is not specification: it is passed as a focusing hypothesis whose priority is lower than issue
text, runtime behavior, hydrated code, and repair history.

Our contribution is therefore a harness-level protocol, not a new base model or graph retriever. First,
we identify evidence-status collapse as a recurring failure mode in long-horizon repository repair.
Second, we define a typed evidence ledger that compresses repair trajectories while preserving artifact
status, provenance, and validation state. Third, we instantiate the ledger in a two-model graph repair
agent combining planner exploration, CGM-style patch generation, pending-patch review, rollback, and
fail-to-pass/pass-to-pass validation. Finally, we outline trace-level measurements and ablations for
evaluating whether the harness reduces premature validation, unsupported patch assumptions, and repeated
failed strategies.

### Harsh review of this introduction

What works:

- It starts from harness and memory, not from our implementation.
- The gap is sharper than "agents need evidence."
- It makes PatchLedger complementary to existing graph and agent work.
- It gives concrete failure modes without overclaiming solved performance.

What is still weak:

- It needs actual empirical numbers before submission. Without them, the last contribution sounds like
  a design proposal.
- The introduction is still slightly long. A conference version should cut one paragraph from related
  work and move detailed W/M/P/H explanations to the method section.
- "Evidence-status collapse" must be operationalized in annotation rules; otherwise reviewers may call it
  post-hoc diagnosis.
- The title "PatchLedger" is memorable but could imply only patch history. The subtitle must include
  "context compression" or "typed evidence memory" to avoid narrowing the perceived method.

## 8. Next Revision Plan

To make this submission-grade, the next writing pass should add:

1. A formal definition of evidence-status collapse:
   - input: raw trajectory artifacts;
   - compressed state exposed to planner/CGM;
   - collapse occurs when artifact status is omitted, contradicted, or upgraded without the required action.
2. A small annotation guide for traces:
   - preview-as-evidence;
   - unsupported patch assumption;
   - premature validation;
   - forgotten failed strategy;
   - planner-intent override.
3. A table mapping each PatchLedger state to:
   - producer action;
   - allowed consumers;
   - validity condition;
   - compression rule;
   - failure prevented.
4. An ablation story:
   - no W/M distinction;
   - no pending patch state;
   - no failed patch history;
   - no consumer-specific projection;
   - no behavior-only runtime summaries.
5. A result table once experiments are available:
   - pass rate;
   - invalid patch rate;
   - average official test submissions;
   - repeated strategy count;
   - unsupported assumption count;
   - prompt token budget and compression ratio.

## 9. Source Notes For Later Citation

- Confucius Code Agent, arXiv:2512.10398. Supports the claim that large-scale coding agents need
  advanced context management, persistent notes, modular tools, and scaffold-level design.
- OpenDev, arXiv:2603.05344. Supports terminal-agent framing around scaffolding, harness design,
  adaptive compaction, memory, and reasoning-phase control.
- Harness Engineering for Language Agents, preprint 2026. Supports treating the harness layer as
  the extra-model layer governing context, action, state, feedback, and failure handling.
- MemGPT, arXiv:2310.08560. Supports virtual context management and memory tiers.
- ACON, arXiv:2510.00615. Supports context compression for long-horizon agents and failure-driven
  compression guidelines.
- SWE-agent, NeurIPS 2024. Supports the importance of agent-computer interface design.
- Agentless, arXiv:2407.01489 / FSE 2025. Supports staged repair and the argument against needless
  agentic complexity.
- CGM, arXiv:2505.16901. Supports graph-integrated repository context and clarifies why our novelty
  should not be graph retrieval itself.

## 10. Writing Discipline: Avoid Amateur Framing

The most dangerous writing failure is to describe implementation moves as if they are research claims.
The paper should not read like a development log. Use the following replacements.

| Weak phrasing | Why it sounds weak | Stronger phrasing |
| --- | --- | --- |
| "We give CGM better context." | Sounds like ordinary prompt engineering. | "The harness compiles raw repair trajectories into consumer-specific evidence projections." |
| "The planner should read more before repair." | Sounds like a heuristic rule. | "The protocol distinguishes orientation context from committed implementation evidence." |
| "Failed patches are shown to the model later." | Sounds like logging. | "Rollback is decoupled from forgetting: failed patch assumptions are retained as validation evidence." |
| "We add a review step before testing." | Sounds like Self-Refine or verifier glue. | "Patch proposal and official validation are separated into distinct ledger states." |
| "The model was misled by hints." | Anecdotal and model-blaming. | "The compressed context upgraded a navigation clue into unsupported repair evidence." |
| "The system has memory W/M/P/H." | Implementation-first. | "The harness maintains typed memory boundaries that determine which artifacts may influence patch generation." |

## 11. Final Story Chain

A clean conference version should follow this chain, in this order:

1. Repository repair is a long-horizon harnessed interaction, not a single code-generation call.
2. Long-horizon harnesses must compress trajectories into finite context.
3. Existing coding-agent and compression work mostly optimizes action interfaces, retrieval, relevance,
   token budget, or general memory.
4. Repair adds a stricter requirement: compression must preserve the evidence status of artifacts.
5. Evidence-status collapse explains recurring failures that are not reducible to weak retrieval or weak
   patch generation.
6. PatchLedger implements evidence-preserving compression through typed ledger states and
   consumer-specific projections.
7. Patch deliberation is one consequence of this ledger: generated edits become pending assumptions before
   they are allowed to consume official tests.
8. Evaluation should measure not only pass rate, but also trace-level collapse events, premature validation,
   unsupported assumptions, repeated failed strategies, and context compression.

If a paragraph does not serve one of these steps, cut or move it.

## 12. One-Paragraph Pitch

PatchLedger studies a problem that appears only once software repair agents become long-horizon systems:
the harness must compress a growing trajectory without destroying the evidential role of its artifacts.
Current agents retrieve code, summarize history, run tests, and reflect on failures, but they rarely make
explicit whether a retained artifact is a clue, a code fact, a hypothesis, an unvalidated patch assumption,
or a validation result. PatchLedger turns this distinction into a repair-harness protocol. It stores
exploratory context, committed implementation evidence, pending patches, and failed attempts in separate
ledger states, then projects those states differently to the planner, patch generator, and validator. The
resulting contribution is not another graph retriever or another self-refinement loop, but a concrete
memory and context-compression layer for reliable repository repair.

### Harsh review of the pitch

This pitch is concise and defensible. It still needs empirical backing, but the intellectual object is clear.
The key phrase "without destroying the evidential role" should be retained. The final sentence is strong
because it says what the work is not.
