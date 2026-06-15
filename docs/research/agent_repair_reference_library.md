# Reference Library: Harness-Memory Repair Agents

This file is a working reference library for framing our current GraphPlanner/CGM repair method.
The preferred paper angle has shifted from "evidence-guided patch deliberation" and the narrower
"patch ledger" idea to a harness-level story: evidence-state context compression for graph-guided
repository repair agents.
It intentionally emphasizes papers that explain a real design pressure in our system, rather than
collecting every adjacent APR or SWE-bench paper.

## Working Method In One Sentence

Our current method is a train-free repair harness that compresses long graph-guided repository-repair
trajectories into a typed evidence-state ledger. A planner explores the repository graph and curates
hydrated code evidence; a graph-aware patch model proposes candidate edits; the harness stores graph
artifacts, planner hypotheses, runtime symptoms, pending patches, and failed attempts with explicit
source/status/scope/consumer metadata before official validation.

## Core Comparison Axes

- **Harness object:** Does the paper report the model only, or the harness layer that controls context,
  action, state, validation, and recovery?
- **Memory semantics:** Does memory merely store more information, or preserve whether an artifact is
  a clue, implementation evidence, behavioral evidence, hypothesis, patch assumption, or validation result?
- **Context compression:** Is compression about token budget/relevance only, or about preserving repair
  evidence status under finite context?
- **Autonomy vs. staged control:** Does the LLM decide arbitrary next actions, or is it constrained to a fixed pipeline?
- **Context construction:** Is repository context retrieved as flat text, graph nodes, executable environment feedback, or a curated memory state?
- **Patch generation protocol:** Is patch generation one-shot, direct test-and-retry, multi-candidate reranking, or explicit propose/review/submit?
- **Failure feedback:** Is failed patch/test information discarded, reflected in natural language, used by a verifier, or shared across model roles?
- **Evidence hygiene:** Does the system separate issue/runtime evidence from benchmark test source and separate orientation context from trusted repair evidence?

## Papers And Systems

### Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases

- Link: https://arxiv.org/abs/2512.10398
- Main idea: A large-scale coding agent built on a scaffold/SDK with advanced context management,
  persistent note-taking, modular tools, and meta-agent configuration refinement.
- Relevant claim: Real-world coding agents need durable memory, context management, and orchestrated
  toolchains, not only stronger base models.
- Relation to us: Strong support for a harness/scaffold framing. Our work can be positioned as a
  repair-specific memory protocol inside this broader scaffold agenda.
- Gap relative to us: CCA's memory is broad scaffold memory. It does not isolate the repair-specific
  invariant that compressed context must preserve the evidential status of search hits, hydrated code,
  candidate patches, and validation failures.

### OpenDev: Building Effective AI Coding Agents for the Terminal

- Link: https://arxiv.org/abs/2603.05344
- Main idea: A terminal-native coding agent emphasizing scaffolding, harness design, safety controls,
  adaptive context compaction, memory, model routing, and explicit reasoning phases.
- Relevant claim: Long-horizon coding agents suffer from context bloat and reasoning degradation; harness
  design and context efficiency are central.
- Relation to us: Confirms that context compaction and harness choices are now first-class in coding-agent
  systems.
- Gap relative to us: Adaptive compaction keeps the agent efficient, but our focus is narrower: repair
  compression must preserve artifact status and validation provenance, not merely reduce old observations.

### Harness Engineering for Language Agents

- Link: https://www.preprints.org/manuscript/202603.1756
- Main idea: Defines the harness as the extra-model layer that determines what an agent sees, what it can
  do, how work unfolds, what feedback is received, and how behavior is constrained and evaluated.
- Relevant claim: Many agent gains and failures are harness-sensitive rather than purely model-driven.
- Relation to us: Provides the cleanest conceptual umbrella for our paper. RepairLedger is a concrete
  repair harness memory design.
- Gap relative to us: The harness literature is broad. We contribute a repair-specific object: a typed
  evidence ledger for context compression across planning, patch generation, validation, and rollback.

### MemGPT: Towards LLMs as Operating Systems

- Link: https://arxiv.org/abs/2310.08560
- Main idea: Virtual context management inspired by operating systems, using memory tiers to make limited
  context windows behave like larger memory.
- Relevant claim: Context windows require explicit memory management rather than unbounded transcripts.
- Relation to us: Supports the memory-hierarchy analogy for long-horizon agents.
- Gap relative to us: MemGPT is task-general. It does not define the software-repair evidence roles that
  must survive compression.

### ACON: Optimizing Context Compression for Long-Horizon LLM Agents

- Link: https://arxiv.org/abs/2510.00615
- Main idea: Optimizes natural-language compression guidelines for long-horizon agents using failure
  analysis; reports substantial peak-token reductions while preserving or improving task performance.
- Relevant claim: Long-horizon agents need explicit compression of observations and history, and bad
  compression can cause failures.
- Relation to us: ACON is the closest context-compression precedent.
- Gap relative to us: ACON compresses history into concise informative representations, while RepairLedger
  proposes typed, consumer-specific repair memory where evidence status controls downstream actions.

### A Survey on the Memory Mechanism of LLM-Based Agents

- Link: https://arxiv.org/abs/2404.13501
- Main idea: Surveys memory mechanisms for LLM agents and argues memory is key for long-term,
  complex agent-environment interactions.
- Relation to us: Provides background that agent memory is an established design dimension.
- Gap relative to us: The survey is broad; our contribution is a repair-harness memory protocol with
  concrete state transitions and validation gates.

### SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

- Link: https://arxiv.org/abs/2310.06770
- Main idea: SWE-bench defines repository-level issue fixing as a benchmark where models must modify codebases based on real GitHub issues and pass tests.
- Relevant claim: Real issues often require coordinating changes across functions, classes, and files, making isolated code generation insufficient.
- Relation to us: SWE-bench motivates our repository-level graph, sandbox validation, fail-to-pass/pass-to-pass protocol, and rollback discipline.
- Gap relative to us: SWE-bench is a benchmark, not an agent architecture. It does not prescribe how to construct evidence or how to deliberate over failed patches.

### SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?

- Link: https://arxiv.org/html/2509.16941v1 and https://openreview.net/forum?id=9R2iUHhVfr
- Main idea: SWE-Bench Pro makes the setting harder with long-horizon, multi-file, multi-language engineering tasks from larger projects.
- Relevant claim: Current systems still struggle with cross-file reasoning and integration in large systems.
- Relation to us: Our recent Flipt failure is exactly a SWE-Bench Pro-style case: interface + engine + middleware + endpoint filter.
- Gap relative to us: The benchmark exposes failures but does not solve the feedback-quality problem. In practice, its per-instance parsers may provide very thin runtime summaries, which our system must compensate for.

### SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering

- Link: https://arxiv.org/abs/2405.15793
- Main idea: The interface between an LM agent and the computer strongly affects repair performance; SWE-agent introduces an agent-computer interface for navigation, editing, and testing.
- Relevant claim: Tool/API design is not incidental; it shapes agent behavior.
- Relation to us: Our `explore_find`, `grep_code`, `explore_expand`, `read`, `memory_commit`, and repair actions are an ACI for repository repair.
- Difference: SWE-agent typically lets the same agent navigate, edit, and test. We separate planner navigation from CGM patch generation and introduce a curated repair memory.
- Gap relative to us: SWE-agent-style loops can still repeatedly chase poor contexts or immediately test weak patches unless the interface explicitly separates orientation, evidence, proposal, and submission.

### OpenHands: An Open Platform for AI Software Developers as Generalist Agents

- Link: https://arxiv.org/abs/2407.16741
- Main idea: OpenHands provides a general platform for developer-like agents that write code, use a command line, browse, and run inside sandboxed environments.
- Relation to us: It motivates safe execution, sandboxing, and composable agent infrastructure.
- Difference: OpenHands is platform-level and broad. Our contribution is narrower: a repair protocol for graph-curated evidence and patch deliberation.
- Gap relative to us: It does not specifically solve CGM evidence curation or pending-patch review.

### RepairAgent: An Autonomous, LLM-Based Agent for Program Repair

- Link: https://arxiv.org/abs/2403.17134
- Main idea: An autonomous LLM agent interleaves information gathering, repair ingredient collection, and validation based on feedback.
- Relevant claim: Repair agents benefit from dynamic prompting, tool use, and feedback from previous attempts.
- Relation to us: We share the iterative repair loop and feedback-driven exploration.
- Difference: RepairAgent centers on a single autonomous agent loop. We use two roles and make failed patch history a structured artifact shared by both planner and CGM.
- Gap relative to us: The abstract design does not distinguish candidate patch proposal from official verification, so test attempts can be spent on patches that were not yet semantically reviewed.

### Agentless: Demystifying LLM-based Software Engineering Agents

- Link: https://arxiv.org/html/2407.01489v1
- Main idea: A simple pipeline can outperform complex agents: localization followed by repair, with candidate patch generation, filtering, and reranking.
- Relevant claim: Fixed, interpretable stages can avoid some unpredictability and cost of autonomous agents.
- Relation to us: We adopt its staged-control instinct: patch generation is not arbitrary; repair memory and pending patch states constrain the process.
- Difference: Agentless avoids autonomous tool decisions; our planner still explores dynamically because real Pro tasks often require discovering cross-file mechanisms not obvious from issue text.
- Gap relative to us: It lacks an interactive mechanism for revising the evidence package after a failed candidate, and it generally treats localization/repair as prearranged phases.

### AutoCodeRover: Autonomous Program Improvement

- Link: https://arxiv.org/abs/2404.05427 and https://www.clioapp.ai/research/autocoderover
- Main idea: Combine LLMs with structured code search APIs over classes/methods, optional SBFL, precise context extraction, patch generation, and validation/retry.
- Relevant claim: Program-structure-aware search gives better context than treating a repository as plain files.
- Relation to us: Our graph search/read/expand tools and scoped `grep_code` are in the same family.
- Difference: We add explicit W/M separation and make evidence curation a planner action. Search results are orientation until read and committed.
- Gap relative to us: AutoCodeRover focuses on context retrieval and retry; it does not articulate a patch-candidate deliberation state shared with a graph-aware patch model.

### Code Graph Model (CGM): A Graph-Integrated LLM for Repository-Level Software Engineering

- Link: https://arxiv.org/pdf/2505.16901
- Main idea: CGM integrates repository code graphs into open-source LLMs and uses an agentless Graph RAG framework with Rewriter, Retriever, Reranker, and Reader.
- Relevant claim: Repository-level tasks need structural and semantic code context; graph retrieval can focus the model on relevant subgraphs.
- Relation to us: CGM is our patch model and the strongest precedent for graph-aware repair.
- Difference: CGM's paper framing is agentless Graph RAG. We wrap a CGM-like patch generator in a planner-driven environment that iteratively curates memory, reviews candidate patches, and shares failure history.
- Gap relative to us: CGM can still generate wrong patches if given an incomplete or misleading subgraph. Our method addresses input evidence quality and patch deliberation around CGM.

### REPOGRAPH

- Link: https://proceedings.iclr.cc/paper_files/paper/2025/file/4a4a3c197deac042461c677219efd36c-Paper-Conference.pdf
- Main idea: A repository graph module retrieves line/file/repository-level ego-graphs to improve AI software engineering.
- Relation to us: Supports the broader claim that graph-based context is becoming a first-class object for coding agents.
- Difference: Our graph is not only a retrieval module; it feeds an explicit state protocol (`W`, `M`, pending patch, repair history).
- Story implication: We should not claim graph retrieval itself as novel; our angle is how graph context becomes trusted repair evidence.

### CodeRAG

- Link: https://arxiv.org/abs/2504.10046
- Main idea: Retrieve supportive code for real-world repository-level generation using requirement graphs and code graphs.
- Relation to us: Reinforces the importance of dependency-aware supportive context.
- Difference: CodeRAG is primarily a retrieval framework for generation. Our focus is repair-state governance: which retrieved code enters memory, how generated patches are reviewed, and how failed attempts are reused.
- Story implication: "More context" is not enough; retrieved supportive code needs an evidence contract.

### SWE-Fixer

- Link: https://aclanthology.org/2025.findings-acl.62.pdf
- Main idea: A trained open-source two-module system: code file retrieval plus code editing, with coarse-to-fine retrieval and efficient patch generation.
- Relevant claim: Splitting retrieval and editing can be effective and efficient.
- Relation to us: We also separate localization/context construction from patch generation.
- Difference: SWE-Fixer trains separate models and aims for few calls. We are train-free and rely on inference-time graph memory curation and deliberation.
- Gap relative to us: Fixed retrieval/editing modules may be less adaptive when a failed patch reveals a missing mechanism.

### SWE-Gym

- Link: https://arxiv.org/abs/2412.21139
- Main idea: A training environment for real-world SWE agents and verifiers; trajectory training and verifier-guided inference improve open-weight agents and reduce loops.
- Relation to us: Our telemetry and structured repair attempts could become training data for a future verifier or policy model.
- Difference: We currently avoid training. Our verifier-like behavior is procedural and prompt/protocol based.
- Gap relative to us: Training helps policy quality but does not remove the need for clean evidence channels and benchmark-safe runtime feedback.

### R2E-Gym / AgentGym

- Link: https://arxiv.org/abs/2504.07164
- Main idea: Procedurally curate executable SWE environments and combine execution-based and execution-free verifiers for test-time scaling.
- Relation to us: Shows the importance of verifiers and executable environments for scaling open-weight SWE agents.
- Difference: R2E-Gym is primarily about data/environment generation, training, and hybrid verifier scaling. Our current method is an inference-time protocol around one issue trajectory.
- Story implication: Our structured traces (`M`, pending patches, repair history) could supply cleaner data for future verifiers, but the paper should not claim trained-verifier benefits unless evaluated.

### SWE-Search

- Link: https://arxiv.org/abs/2410.20285
- Main idea: Use MCTS, value agents, qualitative evaluation, and multi-agent debate to improve software agents.
- Relevant claim: Software repair has exploration/exploitation tension and agents can repeat ineffective actions without strategic evaluation.
- Relation to us: Our pending patch loop is a lightweight local version of strategic evaluation: submit, revise, discard, or read more.
- Difference: We do not run a search tree or multi-agent debate. We keep one active candidate to reduce complexity and cost.
- Gap relative to us: MCTS may improve exploration, but it is expensive and can obscure which evidence caused a patch to be trusted.

### PatchAgent

- Link: https://www.usenix.org/conference/usenixsecurity25/presentation/yu-zheng
- Main idea: An LLM-based APR agent for vulnerability repair that integrates fault localization, patch generation, validation, language-server support, and interaction optimization.
- Relation to us: Strong evidence that language-server/navigation tools and patch verifiers matter in practical repair.
- Difference: PatchAgent targets vulnerability repair and uses a single autonomous repair agent with verifier/tool support. Our focus is repository issue repair with two-model evidence curation and CGM patching.
- Story implication: We should frame our work as complementary to stronger static/language-server validation, not as replacing it.

### Reflexion

- Link: https://arxiv.org/abs/2303.11366
- Main idea: Agents can improve without weight updates by verbalizing feedback into episodic memory.
- Relation to us: `repair_attempts` and `cgm_insights` are a repair-specific, structured analogue of episodic reflection.
- Difference: We avoid long free-form reflections and store compact patch/test/assumption facts shared across roles.
- Gap relative to us: Reflexion is general; it does not define how to keep patch history benchmark-safe or how to connect it to code graph evidence.

### Self-Refine

- Link: https://arxiv.org/abs/2303.17651
- Main idea: Generate, critique, refine iteratively without additional training.
- Relation to us: `repair_revise` is a controlled self-refinement step over a pending patch.
- Difference: We use environment evidence and planner review rather than unconstrained self-feedback.
- Gap relative to us: Self-refinement alone may reinforce wrong assumptions if the code evidence package is incomplete.

### Code Repair with LLMs Gives an Exploration-Exploitation Tradeoff

- Link: https://arxiv.org/abs/2405.17503
- Main idea: Iterative program refinement exposes a tradeoff between exploiting promising candidates and exploring alternative candidates.
- Relation to us: Our pending patch decision (`submit`, `revise`, `discard`, `read more`) is a practical exploitation/exploration decision point.
- Difference: Their work studies multi-candidate search across generated programs; our present system is single-candidate and repository-agent oriented.
- Gap relative to us: It does not solve repository evidence construction or graph memory curation.

### DEVLoRe / Integrating Various Software Artifacts

- Link: https://arxiv.org/abs/2412.03905
- Main idea: Issue content, stack traces, and debugging information complement each other for localization and repair.
- Relation to us: Supports our emphasis on runtime failure summaries and actual output extraction.
- Difference: We must enforce benchmark hygiene: actual runtime output is allowed, benchmark test source is not.
- Gap relative to us: The paper does not address cases where benchmark parsers provide thin summaries, which we saw in SWE-Bench Pro/Go traces.

### Toggle / A Deep Dive into LLMs for Automated Bug Localization and Repair

- Link: https://arxiv.org/abs/2404.11595
- Main idea: Separating bug localization and bug fixing and injecting inductive biases can improve APR.
- Relation to us: Supports role separation and the idea that localization context should be handled differently from patch generation.
- Difference: Toggle is more token/function-level APR. Our setting is repository-level, graph-mediated, and interactive.

## Comparison Matrix

| Work family | Representative systems | What they optimize | What they do not directly solve | Our narrowed angle |
| --- | --- | --- | --- | --- |
| Benchmarks | SWE-bench, SWE-Bench Pro | Realistic issue repair evaluation | Repair protocol design | Long-horizon failures motivate evidence trust |
| Agent interfaces | SWE-agent, OpenHands, RepairAgent | Tool use, autonomy, environment feedback | Clear trust boundaries between search, read, plan, patch, and tests | Typed repair state and evidence priority |
| Agentless pipelines | Agentless, SWE-Fixer | Simplicity, cost, staged localization/editing | Adaptive post-failure evidence revision | Dynamic exploration with staged patch deliberation |
| Graph/context retrieval | CGM, REPOGRAPH, CodeRAG, AutoCodeRover | Better repository context | Whether retrieved context is trusted evidence | Planner-curated `M` and hydrated code evidence |
| Feedback/refinement | Reflexion, Self-Refine, REx, SWE-Search | Learning from failed attempts or candidate search | Benchmark-hygienic, repair-specific patch memory | Structured patch history shared by planner and CGM |
| Training/verifiers | SWE-Gym, R2E-Gym | Policy/verifier improvement through data and scaling | Train-free issue-level protocol | Our traces can become verifier data later |
| APR validation tooling | PatchAgent | Language server + patch verifier + validation | CGM evidence packaging and graph-memory boundaries | Complementary validation layer for propose/submit |

## Synthesis

The closest prior work gives us three ingredients:

1. **Agent interfaces** are crucial (SWE-agent/OpenHands/RepairAgent).
2. **Staged localization + repair** is often simpler and competitive (Agentless/SWE-Fixer).
3. **Graph/context retrieval** improves repository-level repair (CGM/REPOGRAPH/AutoCodeRover).

Our story should not claim any single ingredient is novel. The plausible contribution is the integration:

- a graph-guided but planner-curated evidence memory (`W` vs. `M`);
- a CGM-facing context contract that treats planner intent as secondary to code/runtime/history;
- a pending-patch deliberation loop that separates candidate generation from official test submission;
- compact cross-model repair memory for failed patches and CGM insights.

## Current Reality Check From Flipt Failure

The Flipt SWE-Bench Pro failure gives a useful negative case:

- The planner found much of the issue mechanism but submitted before reading/committing middleware evidence.
- CGM guessed OPA result types and context propagation details.
- The Pro parser exposed failed test names but not enough Go runtime/compile details.

This supports the paper story: the bottleneck is not only "better model" or "more graph"; it is the reliability of evidence packaging, candidate patch review, and runtime feedback extraction.
