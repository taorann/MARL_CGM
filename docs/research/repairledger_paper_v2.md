# RepairLedger Paper Draft V2

This is the cleaned-up narrative after the PatchLedger critique. The goal is to make the
story rigorous, research-grounded, and conference-shaped:

- prior work establishes the ingredients;
- the gap is a precise missing axis, not a casual criticism;
- our method is a harness-level state protocol over graph-guided repair trajectories;
- patch management is one consequence, not the centerpiece.

## 1. Current Title Candidates

Best current title:

> RepairLedger: Evidence-State Context Compression for Graph-Guided Repository Repair

More formal but less memorable:

> Typed Evidence-State Memory for Repository-Level Software Repair Agents

More graph-forward:

> Evidence-State Graph Memory for Repository-Level Software Repair

Current preference: **RepairLedger** with the subtitle doing the scientific work.

## 2. The Clean Story In One Page

Repository repair agents are now built from several increasingly well-understood components.
SWE-agent shows that the agent-computer interface matters for repository navigation, editing,
and testing. Agentless shows that staged localization, repair, and validation can be competitive
without unconstrained autonomy. RepoGraph, CodeRAG, AutoCodeRover, and CGM show that repository
structure and graph context are important for locating and editing relevant code. Confucius Code
Agent and OpenDev show that real coding agents require scaffold-level memory, context management,
and adaptive compaction. MemGPT and ACON establish context management and compression as a general
long-horizon agent problem, while Reflexion-style methods show that feedback can be retained as
memory.

These results are not contradictory. They point to the same systems fact: repository repair is a
harnessed, long-horizon interaction in which the model sees only a compressed projection of a much
larger trajectory.

The missing axis is not "more graph" or "more memory." It is **evidence-state preservation**.
During repair, the same code entity can move through several states:

```text
graph node -> search candidate -> read code -> hydrated evidence -> committed premise
           -> patch dependency -> validated support or invalidated assumption
```

If the harness compresses this trajectory into ordinary prompt text, these states can collapse.
A search hit may be treated as code evidence; an issue requirement may be treated as an existing
mechanism; a planner hypothesis may become a patch specification; a candidate patch may be tested
before its assumptions are supported; a failed edit may vanish after rollback.

RepairLedger is a harness protocol that prevents this collapse by storing repair artifacts in a
typed evidence-state ledger over the repository graph and runtime trajectory. The ledger records
the artifact source, status, scope, allowed consumers, and promotion/demotion rules. The planner,
patch generator, and validator receive different projections of this ledger: the planner can see
exploratory and committed evidence; the patch generator receives hydrated committed code, behavior
evidence, concise intent, and failure history; the validator only consumes explicit pending patches
and produces validation outcomes.

The contribution is therefore not a new graph retriever, a new patch model, or a new reflection
loop. It is a repair-specific context compression protocol: graph-guided trajectories are compressed
without losing the evidence state that determines whether an artifact is allowed to guide planning,
patch generation, or validation.

## 3. Formal Core

### Raw trajectory

Let a repair trajectory be:

```text
T = [(a_1, o_1), ..., (a_n, o_n)]
```

where each action-observation pair may produce artifacts:

```text
A = {issue claims, runtime symptoms, graph nodes, graph edges, code bodies,
     relation facts, planner hypotheses, candidate edits, validation results,
     rollback outcomes}
```

### Evidence state

Each artifact has an evidence state:

```text
e(a) = (type, source, status, scope, provenance, consumers, validity)
```

Important statuses include:

```text
clue, read, hydrated, committed, hypothesis, pending, validated, invalidated, superseded
```

### Context compression

A harness cannot expose the full trajectory at every step. It computes projections:

```text
C_planner(T) -> planner context
C_patch(T)   -> patch-model context
C_valid(T)   -> validator context
```

RepairLedger's claim is that these projections should preserve evidence state. Compression is
not only token reduction:

```text
compression should preserve what each artifact is allowed to mean.
```

### Evidence-state collapse

Evidence-state collapse occurs when a compressed projection:

1. omits the artifact's status or provenance;
2. upgrades an artifact without the required action;
3. exposes an artifact to a consumer that should not rely on it;
4. forgets an invalidating validation result.

Examples:

- `search candidate -> committed evidence` without read/hydration;
- `planner hypothesis -> specification` without support from code/runtime evidence;
- `candidate edit -> tested patch` without pending review;
- `rollback -> forgotten failure` without recording invalidated assumptions.

This definition is useful because it can be annotated in traces and ablated in the system.

## 4. Research-Grounded Positioning

### What prior work establishes

1. **Interface matters.** SWE-agent argues that language-model agents are end users of software
   systems and benefit from specially designed interfaces; its ACI improves repository navigation,
   editing, and testing.
2. **Staging matters.** Agentless shows that a simple localization-repair-validation pipeline can be
   strong and interpretable, challenging the assumption that unconstrained autonomous agents are
   always necessary.
3. **Repository structure matters.** RepoGraph, CodeRAG, AutoCodeRover, and CGM show that repository
   graphs, definitions, dependencies, and structural context improve repository-level tasks.
4. **Harness memory and compaction matter.** Confucius Code Agent and OpenDev treat scaffold memory,
   persistent notes, adaptive compaction, and context efficiency as central to practical coding agents.
5. **Long-horizon compression is a general bottleneck.** MemGPT frames context windows as constrained
   memory resources; ACON optimizes compression of observations and histories for long-horizon agents.
6. **Feedback can become memory.** Reflexion and related methods show that retaining feedback can
   improve later behavior without training.

### What remains open

These threads do not by themselves answer a repair-specific question:

> When a graph-guided repair trajectory is compressed, how does the harness preserve whether each
> artifact is a clue, code evidence, hypothesis, pending assumption, or validation result?

This is the narrow gap. It is not a claim that existing systems have no state, no memory, or no feedback.
It is a claim that evidence-state preservation is a distinct harness property worth naming, implementing,
and measuring.

## 5. Elegant Abstract Candidate

Repository-level software repair is no longer a single code-generation problem: modern systems combine
tool-using planners, repository graphs, patch generators, validation harnesses, and memory. These systems
must repeatedly compress long trajectories of searches, graph expansions, code reads, issue claims,
runtime symptoms, planner hypotheses, candidate edits, test failures, and rollbacks into finite model
context. Prior work has improved the action interface, staged the repair pipeline, retrieved repository
structure, and compressed long histories. We argue that repository repair imposes a stricter requirement:
compression must preserve the evidence state of each artifact. A graph node found by search is not yet
committed code evidence; a planner hypothesis is not a specification; a candidate edit is not a validated
fix; and rollback should not erase the assumptions invalidated by a failed patch.

We present **RepairLedger**, a train-free harness protocol for evidence-state context compression in
graph-guided repository repair. RepairLedger stores repair artifacts in a typed ledger over the repository
graph and runtime trajectory, tracking each artifact's source, status, scope, allowed consumers, and
promotion or demotion rules. The planner receives a projection containing exploratory and committed
evidence states; the patch generator receives hydrated committed code, behavior evidence, concise intent,
and relevant failure history; the validator consumes explicit pending candidates and produces validation
evidence. This design separates retrieval from trust, planning from specification, generation from
validation, and rollback from forgetting. RepairLedger positions evidence-state management as a harness
layer complementary to stronger repository graphs, stronger patch models, and general long-context memory.

### Harsh review

This abstract is cleaner than the PatchLedger version. It now has a graceful progression:
modern repair systems -> long trajectories -> finite context -> existing solutions -> stricter repair
invariant -> method. The remaining problem is that the abstract currently promises a protocol, not results.
For a finished paper, the last sentence should be replaced with measured outcomes or trace-analysis findings.

## 6. Introduction Candidate V2

Repository-level software repair has become a systems problem. A repair agent is given an issue and a
codebase, but the final patch depends on more than the language model's isolated code-generation ability.
The harness around the model decides how the repository is searched, how code is read, how runtime behavior
is summarized, how candidate edits are generated, how tests are run, how failures are rolled back, and what
state is carried into the next model call. As tasks grow from single-file bugs to long-horizon repository
issues, this harness becomes a central part of the repair method.

Recent work clarifies several pieces of this harness. SWE-agent shows that agent-computer interface design
affects how effectively models navigate repositories, edit files, and run tests. Agentless shows that a
simple staged process of localization, repair, and validation can be competitive with more complex
autonomous agents. RepoGraph, CodeRAG, AutoCodeRover, and CGM show that repository structure and graph
context are important for repository-level reasoning and patch generation. Confucius Code Agent and OpenDev
make scaffold-level memory, persistent notes, context management, and adaptive compaction explicit parts of
coding-agent design. MemGPT and ACON further establish that long-horizon agents require principled memory
and context compression. Together, these systems show that repair performance is shaped by the interface,
retrieval substrate, memory, and validation protocol surrounding the model.

However, repository repair adds a context-compression requirement that is easy to miss. A repair trajectory
does not contain a homogeneous history. It contains artifacts with different evidence states. A search hit is
a clue. A graph neighbor is a structural lead. A read function body is implementation evidence, but only for
the code actually read. An issue statement is a behavioral requirement, not proof of an existing mechanism.
A planner plan is a hypothesis. A generated edit is an unvalidated assumption about APIs, types, imports, and
data flow. A failed test is validation evidence, but its informativeness varies from a compiler traceback to a
single failed selector. When the harness compresses these artifacts into ordinary prompt text, it can preserve
their content while losing their status.

We call this failure **evidence-state collapse**. Collapse occurs when a compressed context upgrades a clue
into committed evidence, treats a hypothesis as a specification, sends uncommitted graph nodes to the patch
model as trusted premises, tests a candidate edit before its assumptions are supported, or forgets the
assumptions invalidated by a rolled-back patch. These failures are not fully explained by weak retrieval or
weak generation. The right code may have been found but not promoted to trusted evidence; the generated patch
may be plausible but rely on an unsupported mechanism; the failed test may reveal useful information that is
then lost during rollback. The missing layer is a repair-specific protocol for evidence-state preservation.

We introduce **RepairLedger**, a train-free harness protocol for graph-guided repository repair. RepairLedger
stores artifacts from the repair trajectory in a typed ledger over the repository graph and runtime history.
Each artifact records its type, source, status, scope, provenance, allowed consumers, and promotion or demotion
rules. Repository graph nodes can therefore move from search candidates to read code, hydrated evidence,
committed patch-generation premises, and finally validated support or invalidated assumptions. Planner
hypotheses remain low-priority intent unless supported by committed code or behavior evidence. Candidate edits
enter a pending state before official validation. Runtime failures and rollback outcomes become validation
evidence rather than disappearing from the next prompt.

RepairLedger uses this ledger to produce consumer-specific context projections. The planner sees a compact
view of exploratory and committed evidence, pending candidates, recent failures, and available actions. The
graph-aware patch generator receives the issue, behavior evidence, committed hydrated code, relevant graph
relations, concise intent, and repair history, but not noisy search previews or long free-form reasoning. The
validator receives explicit pending candidates and records syntax, fail-to-pass, and pass-to-pass outcomes.
This separation makes four boundaries explicit: retrieval is not trust, planning is not specification,
generation is not validation, and rollback is not forgetting.

This paper makes three contributions. First, it formulates evidence-state collapse as a harness-level failure
mode in repository repair and defines it in terms of artifact status, provenance, consumer eligibility, and
validation state. Second, it presents RepairLedger, a typed evidence-state memory protocol that compresses
graph-guided repair trajectories into consumer-specific contexts. Third, it instantiates the protocol in a
two-model graph repair harness and proposes trace-level measurements and ablations for premature validation,
unsupported patch assumptions, repeated failed strategies, and graph evidence contamination. The broader claim
is not that graphs, memory, staging, or feedback are new, but that reliable repository repair requires a harness
that preserves the evidential state of these artifacts as they move through the repair trajectory.

### Harsh review

Strengths:

- The route is now clear and elegant: harness -> existing ingredients -> compression gap -> evidence-state
  collapse -> RepairLedger.
- The graph is integrated as the substrate whose nodes have lifecycle states.
- The gap is stated as an additional requirement, not as a dismissal of prior work.
- The contribution list is narrow and defensible.

Weaknesses:

- The introduction still has many named systems; a final version should cite compactly and move detail to
  related work.
- "Proposes trace-level measurements" is weaker than reporting them. This wording is acceptable for an
  internal draft, not a submission.
- The term "consumer eligibility" needs examples in the method section.
- The phrase "graph evidence contamination" is useful but needs a precise metric definition.

## 7. Cleaner Related Work Story

The related work section should not be a list of systems. It should be organized by the research premise each
family contributes:

### Agent-computer interfaces and harnesses

SWE-agent, OpenHands, OpenDev, and Harness Engineering establish that the environment-facing layer of an
agent shapes behavior. Our work follows this line but focuses on the memory semantics of that layer during
repository repair.

### Staged repair and agentless pipelines

Agentless and SWE-Fixer show that explicit localization/repair/validation stages can be simpler and strong.
RepairLedger keeps staged control but makes stage transitions evidence-state transitions rather than fixed
pipeline steps.

### Repository graphs and structured code context

RepoGraph, CodeRAG, AutoCodeRover, and CGM show that repository structure improves code understanding and
repair. RepairLedger assumes graph context but adds lifecycle status to graph artifacts.

### Long-horizon memory and context compression

MemGPT, ACON, Confucius Code Agent, and OpenDev show that long trajectories require memory management,
hierarchical state, notes, and compression. RepairLedger specializes this to software repair by preserving
artifact evidence state.

### Feedback and reflection

Reflexion, Self-Refine, SWE-Search, and verifier-based methods show that feedback can improve later attempts.
RepairLedger converts feedback into typed validation evidence tied to graph nodes and candidate assumptions,
rather than unstructured reflection alone.

## 8. Better Vocabulary

Use:

- evidence state
- artifact lifecycle
- consumer-specific projection
- graph-guided trajectory
- promotion/demotion rule
- validation provenance
- retrieval is not trust
- rollback is not forgetting

Avoid:

- "better context"
- "more memory"
- "the model should read more"
- "patch review loop is our key novelty"
- "existing methods ignore X"
- "we solve hallucination"

## 9. Current Best Claim

> RepairLedger studies a repair-specific context-compression problem: how a harness preserves the evidence
> state of graph-guided artifacts as a long repository-repair trajectory is projected into planner, patch-model,
> and validator contexts.

This is currently the cleanest statement. It is broad enough to include graph, memory, patch, validation, and
runtime feedback, but narrow enough to be evaluated.
