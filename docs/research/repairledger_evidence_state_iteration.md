# RepairLedger Iteration: From Patch Management To Evidence-State Harnessing

> Status note, 2026-06-11: this file records the transition from PatchLedger to RepairLedger.
> The cleaner current paper draft is `docs/research/repairledger_paper_v2.md`, which should be
> treated as the active abstract/introduction source.

This file supersedes the narrower "PatchLedger" framing as the preferred narrative direction.
The prior draft was useful because it named context compression and evidence-status collapse, but
it over-indexed on patch history. The broader and more defensible object is:

> a harness-level evidence-state memory protocol for graph-guided repository repair.

Working title:

> RepairLedger: Evidence-State Context Compression for Graph-Guided Repository Repair Agents

Alternative title:

> EvidenceLedger: Typed Harness Memory for Repository-Level Software Repair

Current preference: **RepairLedger**. It is broader than PatchLedger but still grounded in repair.

## 1. Why PatchLedger Was Too Narrow

### Harsh self-review

The PatchLedger framing has a serious narrative risk: it makes the reader think the system is mainly
about managing candidate patches. That is only one part of the method. The real problem begins much
earlier, before a patch exists:

- the harness must decide whether a search result is merely a navigation clue or real code evidence;
- it must preserve which code has been read and hydrated;
- it must represent graph relations as evidence structure, not just retrieval hits;
- it must separate issue requirements from existing implementation mechanisms;
- it must expose runtime behavior without leaking benchmark test source;
- it must decide which context projection the planner, patch generator, and validator each receive.

Patch management matters, but it is the final symptom of a broader evidence-state problem.

### What to keep

- evidence-status collapse;
- typed memory states;
- pending patches as unvalidated assumptions;
- failed edits retained after rollback.

### What to demote

- "patch ledger" as the central metaphor;
- patch review as the main novelty;
- repair history as the only memory artifact.

### New central object

RepairLedger is not a ledger of patches. It is a ledger of **repair evidence states** over a repository
graph and runtime trajectory.

Patch is one artifact type. Others include:

- issue claim;
- runtime symptom;
- graph search hit;
- read/hydrated code body;
- local relation fact;
- committed memory node;
- planner hypothesis;
- candidate edit;
- validation result;
- rollback outcome.

## 2. Revised Core Claim

Repository-level repair agents are not only constrained by how much context they can fit. They are
constrained by whether the harness preserves the **state** of each repair artifact as the trajectory is
compressed.

The repository graph gives structure:

```text
file -> class -> method -> call/use/import/contains/override/consumer relations
```

RepairLedger adds state:

```text
candidate -> read -> hydrated -> committed evidence -> patch dependency -> validated/invalidated
```

The contribution is the combination:

> graph-structured evidence plus typed lifecycle state.

This avoids both weak claims:

- not "graphs improve repair" -- already established by RepoGraph, CodeRAG, CGM, and AutoCodeRover;
- not "patch history helps repair" -- adjacent to Reflexion/Self-Refine;
- but "a repair harness should compress graph-guided trajectories into typed evidence states that
  control which artifacts can influence planning, patch generation, and validation."

## 3. Evidence-Backed Gap Analysis

The goal here is not to say prior work is bad. The claim is narrower: their stated contributions
foreground different layers, leaving repair-specific evidence-state management under-specified.

| Prior work family | Evidence from paper / stated focus | What it supports | Gap we can claim without overreach |
| --- | --- | --- | --- |
| SWE-agent | Introduces agent-computer interfaces; custom interface improves ability to create/edit files, navigate repos, and run tests. | Interface design materially affects coding-agent behavior. | ACI designs the action surface, but does not formulate a typed evidence lifecycle separating search clues, hydrated code, planner hypotheses, pending edits, and validation evidence. |
| OpenHands | Platform for agents that write code, use CLI, browse web, run safely in sandboxes, coordinate agents, and evaluate benchmarks. | General agent platform and sandboxed execution matter. | Platform breadth does not by itself define repair-specific memory status transitions or consumer-specific evidence projections for planner/patch model/validator. |
| Agentless | Uses a simple three-phase localization, repair, and patch validation process without letting an LLM decide future actions or use complex tools. | Staged control can beat over-complex autonomy. | Fixed phases improve interpretability, but do not address how post-failure trajectories should be compressed into reusable typed evidence across further exploration. |
| RepoGraph | Repository-level graph offers navigation and boosts methods; existing methods overlook repository-level code understanding. | Graph structure is crucial and can plug into both procedural and agent systems. | Graph retrieval supplies structure and clues, but does not decide whether a retrieved node is trusted evidence, a preview, a committed repair fact, or an invalidated assumption. |
| CodeRAG / graph RAG | Retrieves supportive code through requirement graphs and code graphs; models dependency and semantic relations. | Comprehensive supportive-code retrieval matters for repo-level generation. | Retrieval of supportive code is not the same as maintaining evidence status through repair iterations, failed patches, and validation gates. |
| CGM | Integrates repository code graph structure into LLM attention and combines with agentless graph RAG for SWE-bench Lite. | Strong precedent for graph-aware patch generation. | CGM quality still depends on the harness-provided evidence package; the CGM paper does not make the planner-side lifecycle of clues, committed evidence, patch assumptions, and failed validation the main object. |
| Confucius Code Agent | Large-scale coding agent with advanced context management, persistent notes, modular tools, and orchestrated scaffold. | Harness/scaffold memory is central for industrial code agents. | Its memory/scaffold story is broad; our repair-specific question is what semantic status must survive compression for code repair. |
| OpenDev | Terminal coding agent with safety controls, adaptive context compaction, memory, model routing, and explicit reasoning phases. | Context bloat and reasoning degradation are concrete harness problems. | Adaptive compaction manages old observations generally; repair needs typed compression that preserves evidence status and validation provenance. |
| MemGPT | Uses virtual context management and memory tiers to overcome limited context windows. | Memory hierarchy is a principled way to handle long contexts. | Memory tiers do not by themselves define software-repair-specific artifact roles or validation eligibility. |
| ACON | Compresses observations and histories for long-horizon agents; failure-driven compression guidelines reduce token usage and improve success. | Context compression is an explicit agent optimization problem. | ACON compresses for informativeness; RepairLedger compresses into typed, action-governing evidence states for repair. |
| Reflexion | Maintains reflective text in episodic memory after feedback. | Failed attempts can improve future decisions without weight updates. | Free-form reflection does not guarantee benchmark hygiene, graph provenance, or distinction between implementation evidence and patch assumptions. |

This table is the safer way to discuss "limitations": each prior work is credited for what it actually
targets, and our gap is framed as a different axis rather than a defect.

## 4. Revised Method Shape

RepairLedger maintains a ledger of evidence artifacts. Each artifact has:

- `artifact_type`: issue_claim, runtime_symptom, graph_node, code_body, relation_fact, planner_hypothesis,
  candidate_edit, validation_result, rollback_result;
- `source`: issue, tool result, graph builder, read action, memory commit, CGM output, test runner;
- `status`: clue, read, hydrated, committed, hypothesis, pending, validated, invalidated, superseded;
- `scope`: path, line range, node ids, edge ids, touched files, selectors;
- `consumers`: planner, patch_model, validator, telemetry;
- `compression_rule`: how it should be summarized for each consumer;
- `promotion_rule`: what action can upgrade its status;
- `demotion_rule`: what failure invalidates or weakens it.

### Lifecycle examples

#### Graph node

```text
G node -> W candidate -> W read/hydrated -> M committed evidence -> referenced by candidate edit
       -> validated support or invalidated assumption after tests
```

#### Planner hypothesis

```text
visible reasoning -> compact hypothesis -> supported by M or marked unsupported
                  -> passed to CGM as low-priority intent, not as specification
```

#### Runtime output

```text
test command output -> behavior summary -> allowed evidence
benchmark test source -> blocked evidence
```

#### Candidate patch

```text
CGM output -> parsed candidate edit -> pending assumption -> revise/submit/discard
           -> validation result -> history artifact
```

## 5. Stronger Graph Connection

The graph is not incidental. RepairLedger is best described as **stateful evidence management over a
repository graph**.

RepoGraph/CodeRAG/CGM answer:

> Which code nodes and relations should the model know about?

RepairLedger answers:

> What is the current evidential state of each node/relation/artifact, and which model component is allowed
> to use it for which decision?

This lets us keep graph central without claiming graph retrieval as novel.

Better wording:

> RepairLedger treats repository graphs as evidence-bearing objects rather than retrieval outputs. A graph
> node can be a candidate clue, a hydrated implementation fact, a committed patch-generation premise, or an
> invalidated assumption depending on its trajectory state.

## 6. Revised Abstract Candidate

Repository-level repair agents increasingly combine repository graphs, tool-using planners, patch generators,
and validation harnesses. These systems face a context-management problem: long repair trajectories contain
search hits, graph neighbors, code reads, issue claims, runtime symptoms, planner hypotheses, candidate edits,
test failures, and rollbacks, but only a small compressed state can be shown to the next model call. Existing
coding-agent interfaces, graph-RAG systems, and context-compression methods improve how agents navigate
repositories, retrieve supportive code, or summarize long histories. We argue that repair requires an additional
invariant: compressed context must preserve the evidence state of each artifact. A graph node found by search
is not yet committed implementation evidence; a planner hypothesis is not a specification; a candidate edit is
not a validated fix; and rollback should not erase what the failed edit revealed.

We present **RepairLedger**, a train-free harness protocol for evidence-state context compression in
graph-guided repository repair. RepairLedger stores repair artifacts in a typed ledger over the repository graph
and runtime trajectory. Artifacts are tracked by source, status, scope, allowed consumers, and promotion or
demotion rules. The planner sees exploratory and committed evidence states; the patch generator receives only
hydrated committed code, behavior summaries, concise intent, and relevant failure history; the validator sees
explicit pending candidates and records validation outcomes. This design separates retrieval from trust, planning
from specification, generation from validation, and rollback from forgetting. It positions typed evidence-state
management as a harness layer complementary to stronger repository graphs and stronger patch models.

### Harsh review

This is better than PatchLedger because it no longer starts from patch history. It also ties graph to state.
The remaining weakness is empirical: "evidence-state context compression" must be measured, or it may be
criticized as vocabulary around ordinary state tracking. The final paper needs trace annotations and ablations.

## 7. Revised Introduction Skeleton

1. Repository repair is now a harnessed, long-horizon graph/context problem.
2. Existing work contributes three strong ingredients:
   - action interfaces and sandboxes;
   - graph/retrieval over repository structure;
   - memory/context compression and reflection.
3. These ingredients leave a repair-specific compression question:
   - after many actions, what does each retained artifact mean?
4. Define evidence-state collapse:
   - artifact content survives but its role/provenance/validity is lost or upgraded incorrectly.
5. Show examples across the whole lifecycle, not just patches:
   - search preview treated as read code;
   - issue requirement treated as existing mechanism;
   - uncommitted graph node sent to patch model;
   - planner hypothesis overriding hydrated code;
   - failed patch assumption forgotten after rollback.
6. Introduce RepairLedger as typed evidence-state memory over repository graph.
7. Contributions:
   - formalize evidence-state collapse;
   - define typed evidence ledger and lifecycle transitions;
   - instantiate in GraphPlanner/CGM harness;
   - evaluate with trace-level metrics and ablations.

## 8. What The Paper Should Not Claim

- Do not claim graph retrieval is novel.
- Do not claim context compression is novel.
- Do not claim memory is novel.
- Do not claim patch review is novel.
- Do not claim prior systems do not have any memory or feedback.
- Do not say "existing methods fail because they do not do our exact protocol."

Safer claim:

> Existing systems establish the need for tool interfaces, graph context, staged repair, feedback, and memory.
> RepairLedger studies a different axis: whether the harness preserves the evidence state of artifacts as it
> compresses graph-guided repair trajectories.

## 9. Required Empirical Backing

To avoid sounding like theory-only packaging, the paper needs at least:

### Trace annotation metrics

- `preview_as_evidence`: a search/preview artifact influenced repair before read/hydration.
- `uncommitted_evidence_use`: CGM or repair plan relied on W-only nodes not in M.
- `hypothesis_as_spec`: planner intent contradicted or exceeded committed code evidence.
- `unsupported_edit_assumption`: patch invoked API/type/dataflow not supported by M or runtime output.
- `premature_validation`: official tests ran while patch-critical artifact was still clue/hypothesis.
- `forgotten_failure`: a later repair repeated an assumption invalidated by prior patch/test feedback.

### System metrics

- raw trajectory tokens vs. ledger projection tokens;
- number of W candidates vs. M committed nodes;
- patch attempts per issue;
- syntax/parse failure rate;
- F2P/P2P pass rate;
- repeated strategy rate;
- time/cost overhead.

### Ablations

- no W/M split;
- no consumer-specific projection;
- no pending-candidate state;
- no failed-attempt ledger;
- graph nodes without state labels;
- state labels without graph relations.

## 10. Current Best One-Sentence Claim

Repository repair agents need not only repository graphs or longer context windows, but a harness memory layer
that preserves the evidence state of graph-guided repair artifacts as long trajectories are compressed into
planner, patch-generator, and validator context.
