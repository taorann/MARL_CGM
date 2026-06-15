# Story Iterations: From Broad Agent Framing To Evidence-Guided Patch Deliberation

> Status note, 2026-06-11: this file records earlier story iterations. The current preferred
> framing is in `docs/research/patchledger_harness_memory_draft.md`: PatchLedger as a
> harness-level typed evidence memory and evidence-preserving context compression protocol.
> The older "Evidence-Guided Patch Deliberation" framing should now be treated as a method
> component, not the paper-level novelty claim.

This file records the narrative refinement process. Each iteration contains:

- a candidate story;
- why it is too broad or too weak;
- what should be kept;
- what should be removed or demoted.

## Iteration 0: "A Better SWE Agent With Code Graphs"

### Candidate Story

Repository-level issue repair is hard because agents need to understand large codebases.
We build a SWE-style agent with a code graph and a CGM patch model. The planner searches,
reads, commits memory, and calls CGM to produce patches. The graph improves context, and
tests validate patches.

### Why This Is Weak

This story is true but not distinctive enough.

- SWE-agent and OpenHands already argue for tool-using software agents.
- AutoCodeRover and REPOGRAPH already argue for structure-aware code retrieval.
- CGM already argues that code graphs help repository-level tasks.
- Agentless and SWE-Fixer already show localization + repair pipelines can be effective.

If we stop here, the contribution sounds like "we combined existing things".

### Keep

- Repository-level issue repair needs context beyond one file.
- Code graph structure is useful.
- Planner/patch-generator role separation is meaningful.

### Remove Or Demote

- Do not claim "using code graphs for repair" as the main innovation.
- Do not claim "agent with tools" as the main innovation.
- Do not claim CGM itself as our contribution.

## Iteration 1: "Graph-Curated Memory For Patch Generation"

### Candidate Story

The bottleneck is not just retrieval, but deciding which retrieved code is trusted repair evidence.
Our system separates the planner's noisy working context (`W`) from the curated CGM memory (`M`).
Search and graph expansion return orientation previews; only explicitly read and committed nodes
enter `M`. CGM receives `M`, issue/runtime behavior, and a concise planner intent.

### Why This Is Better

It targets a concrete failure mode we have repeatedly observed:

- search previews are mistaken for full code evidence;
- unrelated graph neighbors confuse the patch model;
- broad file dumps reduce patch quality;
- planner hints can mislead CGM when not backed by code.

It also distinguishes our method from flat RAG and generic agent context windows.

### Remaining Weakness

This still does not explain the pending-patch loop. It also sounds similar to careful RAG.
The story needs a second bottleneck: even with good context, first patches are often wrong.

### Keep

- `W` vs. `M` as an evidence-quality protocol.
- explicit hydration/read-before-commit.
- CGM input prioritizes code/runtime/history over planner plan.

### Demote

- Do not overstate memory curation as sufficient. Flipt shows that even a plausible `M` can miss middleware.

## Iteration 2: "Patch Deliberation: Candidate Patches Become Evidence"

### Candidate Story

Existing systems usually generate a patch and immediately validate it. If it fails, they retry,
often with only a thin failure summary. We instead treat patches as inspectable intermediate
objects. CGM first proposes a pending patch. The planner inspects that candidate with the issue,
code evidence, and known risks, then decides to submit, revise, discard, or read more code.
Failed patches are saved as structured evidence for both planner and CGM.

### Why This Is Stronger

It directly addresses real observed failures:

- planner submitted a Flipt patch before proving the middleware context mechanism;
- CGM guessed OPA result types and imports;
- failed patches were previously too easy to forget or repeat;
- official tests are expensive, so not every syntactically valid patch deserves a full test run.

It connects to Reflexion/Self-Refine but makes the memory repair-specific, compact, and cross-model.

### Remaining Weakness

The story must avoid claiming that candidate patch review is enough. A planner can still review badly
if the runtime feedback is thin or the evidence package is incomplete. Therefore, runtime feedback
quality and evidence gate quality are part of the mechanism.

### Keep

- pending patch as first-class state;
- `repair_propose`, `repair_revise`, `repair_submit`, `discard_pending_patch`;
- structured `repair_attempts` and `cgm_insights`;
- failed patch history is shared across planner and CGM.

### Demote

- Do not say we eliminate testing cost; we only defer and concentrate it.
- Do not claim human-level code review; this is protocol-level deliberation, still model-limited.

## Iteration 3: "Evidence-Guided Patch Deliberation"

### Candidate Story

The central claim becomes:

> Repository-level repair fails when patch generation is driven by incomplete or stale evidence.
> We introduce an evidence-guided patch deliberation protocol that turns graph retrieval,
> planner reasoning, candidate patches, and test feedback into typed repair evidence before
> official validation.

This integrates two insights:

1. **Evidence curation before generation:** `W` is exploratory and noisy; `M` is curated,
   hydrated code evidence.
2. **Evidence revision after generation:** a pending patch is inspected before testing,
   and failed patch history updates both planner and CGM contexts.

### Why This Is The Best Current Story

It is specific enough to be defensible:

- The novelty is not "graphs" alone.
- The novelty is not "agent" alone.
- The novelty is not "self-refine" alone.
- The novelty is the state/protocol that makes evidence explicit before and after patch generation.

It also admits limitations:

- If runtime output is too thin, the feedback channel is weak.
- If the planner accepts unsupported assumptions, pending-patch review can still fail.
- If CGM lacks language/API competence, evidence cannot fully compensate.

### Final Compressed Contribution Claim

We propose **Evidence-Guided Patch Deliberation**, a train-free repair protocol that couples:

- graph-aware repository exploration;
- explicit planner-curated repair memory;
- CGM patch proposal rather than immediate submission;
- structured cross-model history of candidate patches, assumptions, and failures;
- benchmark-hygienic validation feedback.

## Final Story Chain

### Problem

Real issue repair is long-horizon and cross-file. Modern agents can navigate repositories and run tests,
but they often fail because they generate patches from incomplete context, stale hypotheses, or misleading
planner intent. Graph RAG helps retrieve code, but a retrieved subgraph is not automatically reliable repair
evidence.

### Observation From Failures

Our traces show two recurring patterns:

1. **Pre-generation evidence errors:** search previews or issue hints are treated as full mechanism evidence.
2. **Post-generation evidence loss:** failed patches are rolled back but their assumptions and concrete edits are
   not fully used to guide the next repair.

The Flipt SWE-Bench Pro case illustrates both: the first candidate patch implemented interface and engine changes
but omitted middleware context propagation; after failure, only thin test names were available, making recovery slow.

### Method

The method introduces typed repair states:

- `G`: repository graph;
- `W`: working subgraph with candidates and read code;
- `M`: planner-curated CGM memory containing hydrated code;
- `P`: pending patch;
- `H`: compact repair history and CGM insights.

The planner may explore and curate `M`; CGM may propose or revise patches; only `repair_submit` validates against
official tests.

### Position Relative To Prior Work

- Compared with SWE-agent/OpenHands/RepairAgent, we restrict patch generation behind evidence memory and pending-patch review.
- Compared with Agentless/SWE-Fixer, we retain adaptive exploration and post-failure evidence revision.
- Compared with CGM/REPOGRAPH, we use graph context as evidence governed by a repair protocol, not only as retrieval.
- Compared with Reflexion/Self-Refine, we store structured repair artifacts rather than broad self-reflections.

### Contribution

The contribution is a protocol-level bridge between agentic exploration and graph-aware patch generation: a system
where code context, candidate patches, and failed attempts are all represented as inspectable evidence before being
trusted by the patch generator or the validator.

## Claims We Should Avoid

- "We solve repository-level repair." Too broad and unsupported.
- "Our graph is novel." Graph retrieval is already well established.
- "Our planner understands the issue." Sometimes it does; sometimes it accepts unsupported assumptions.
- "Patch deliberation guarantees better patches." It creates an opportunity for review; it still depends on feedback and model quality.
- "We do not need tests." Tests remain the final oracle; we only make test calls more deliberate.

## Claims We Can Defend

- Explicitly separating exploratory context from repair evidence reduces a known source of prompt noise.
- Treating candidate patches as reviewable state exposes wrong assumptions before expensive validation.
- Sharing compact patch history across planner and CGM is a practical form of repair-specific reflective memory.
- Runtime feedback quality is a first-class bottleneck; benchmark-hygienic output extraction is required for reliable iteration.
- The method is train-free but produces structured trajectories that could later support verifier or policy training.
