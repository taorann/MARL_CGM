# Action And State Protocol

## Planner Output Contract

The planner emits exactly one action per turn. The current robust mode is text protocol with optional visible thinking:

```text
<think>planner-visible reasoning for logs only</think>
```json
{"tool": "explore_find", "params": {"query": "...", "find_type": "function"}}
```
```

The visible thinking is telemetry-only. It is stripped before the formal action is stored and is not fed as hidden state into the next turn.

Official tool-calling can be supported, but the current preferred mode is text actions because Qwen thinking and tool calls can interact poorly in some deployments.

## State Objects

### Repo Graph G

Full read-only repository graph built from source code. It contains nodes and edges:

- file nodes;
- class nodes;
- function/method nodes;
- assignment/config nodes;
- import nodes when useful;
- call/use/contains/sibling/import edges when extractable.

### Working Subgraph W

Planner's growing view of the repository. W contains candidates and read nodes. It can be noisy. W is shown to the planner as a compact index plus recent action results.

### Memory Subgraph M

High-signal subset of W selected by the planner with `memory_commit`. CGM uses M. M must include full code bodies for selected implementation nodes.

### Text Memory T

Planner-only notes. CGM does not read T. Use it for compact hypotheses, failed-repair lessons, and search intent summaries.

### Test State

Stores original fail-to-pass behavior and subsequent repair-test feedback. The planner should see the original failure and patch result summaries, not benchmark test source.

## Actions

### `run_failed_test()`

Purpose: collect trusted fail-to-pass behavior.

Parameters: none.

Output should include:

- selectors or target commands if known;
- compact failure summary;
- non-test implementation traceback frames if present;
- stdout/stderr excerpt trimmed to behavior signal;
- warning if test execution failed due infrastructure.

Rules:

- Do not expose benchmark test source.
- Do not let new noisy test failures overwrite original fail-to-pass semantics.
- If repair was rolled back, report rollback and original failure continuity.

### `explore_find(query, find_type, class_name?)`

Purpose: locate implementation graph nodes.

Parameters:

- `query`: one string, may include ordinary terms and lightweight DSL.
- `find_type`: one of `file`, `class`, `function`, `method`, `assignment`, or `any`.
- `class_name`: optional scoping hint for method search.

Query DSL:

- `+term`: strong constraint; reserve for precise symbols/paths.
- `-term`: negative constraint.
- `symbol:Foo`: symbol constraint.
- `path:pkg/mod.py`: path constraint.
- `"exact phrase"`: phrase constraint.

Rules:

- Search implementation code only.
- Block or rewrite test-derived paths/selectors/helper names.
- Avoid over-constraining broad issue words with `+`.
- If filesystem fallback finds a text span, map it back to graph nodes covering that span before adding to W.
- For small-grain results (`class`, `function`, `method`, `assignment`), return a short line-numbered code preview so the planner can decide whether to read deeper.
- For file-level results, do not return full file text. Return top symbols and require a focused `read` for concrete evidence.
- If a class/name scoping hint yields no hits, retry without that hint and report the relaxation in the result warning.
- A find preview is navigation evidence, not committed CGM evidence; use `read`/`memory_commit` for repair memory.
- Planner-facing node kinds should use the public action taxonomy. Internal graph kinds such as `module_assignment` should be normalized to `assignment`.
- Remote graph builders may encode methods as dotted function names. A method search should tolerate dotted function nodes rather than forcing one internal taxonomy.

### `explore_expand(anchor, expand_mode)`

Purpose: navigate graph relations from an existing W node.

Parameters:

- `anchor`: one node id from candidates or W.
- `expand_mode`: one of `callers`, `callees`, `siblings`, `imports`, `contains`, `uses`, or `related`.

Rules:

- Keep expansions small.
- Prefer several focused expansions over one huge expansion.
- Expand should expose enough neighbor content previews for the planner to choose a next read.

### `read(node_id, view)`

Purpose: put concrete implementation code into W.

Parameters:

- `node_id`: W node id.
- `view`: `body`, `header`, `around_line:N`, or `file_window:start-end`.

Rules:

- Read implementation nodes only.
- Read may target file, class, function, method, or assignment nodes.
- Read output should include full code body when feasible, line numbers, path, node type, and structural facts.
- Read output should also surface local implementation references mentioned by the snippet, such as helper calls, dispatch tables, registry mappings, or sibling functions/classes in the same file.
- Dict assignment dispatch tables should be exposed as key-to-target facts, for example `{"&": "_cstack"}`, rather than only as flat symbol references.
- Local implementation references should be stated as code facts with node ids, not as ordered next-action recommendations.
- Read result must be retained in W for future turns.
- Read result should be available for memory hydration if later committed.

### `memory_commit(select_ids?, keep_ids?, note?, tag?)`

Purpose: choose CGM-facing evidence.

Parameters:

- `select_ids`: ids from W to add to M.
- `keep_ids`: ids already in M to retain.
- `note`: optional planner-only text note.
- `tag`: optional label.

Rules:

- If `select_ids` is omitted, the environment may auto-select a small top-k set from recent high-signal reads.
- If selected nodes directly reference already-read local implementation references, the environment may auto-include those read nodes in M and report them as `auto_included_read_references`.
- Before writing to M, hydrate selected nodes with full code body from W or sandbox.
- M should not include benchmark test nodes.
- Prefer minimal causal subgraphs over broad file dumps.
- Observation should make W/M separation visible: read nodes in W are not CGM evidence until committed to M.

### `memory_delete(delete_ids?, keep_ids?, note?, tag?)`

Purpose: remove stale or misleading evidence from M.

Parameters:

- `delete_ids`: memory ids to remove.
- `keep_ids`: memory ids to retain.
- `note`: optional planner-only reason.
- `tag`: optional label.

### `memory_commit_note(note, tag?)`

Purpose: write planner-only note without changing M.

Parameters:

- `note`: compact hypothesis or lesson.
- `tag`: optional label.

### `repair(plan?)`

Purpose: call CGM and attempt a patch.

Parameters:

- `plan`: concise implementation repair plan grounded in M.

Rules:

- Require at least one memory node.
- If high-signal read nodes exist in W but not M, block repair and tell planner to commit them.
- If M lacks full code body nodes, block or hydrate before CGM.
- Apply patch only after schema and safety validation.
- Run syntax and target tests.
- Roll back failed or low-quality patches.
- If target tests verify green, set the episode final status to pass immediately.
- Report applied/rolled_back, patch summary, failure reason, and rollback/source state facts.

## Observation Contract

Each step observation should include:

- issue summary and implementation-side intent;
- compact W index;
- compact M index with code-body counts;
- latest action result;
- failure/test summary;
- prior failed repair summary;
- explicit runtime facts when action was blocked or rewritten;
- unread implementation symbol references as facts, not recommendations;
- recent executed actions to avoid loops.

The observation must not include benchmark test source code.
