# Graph, Memory, And Retrieval

## Why Graph Memory Exists

Simple keyword search often finds files but misses the real repair locus. The planner needs a graph so it can move from public API to helper, dispatcher, serializer, validator, or assignment node.

The goal is not to hard-code issue keywords. The goal is to expose structural facts so the model can choose the next implementation node.

## Graph Node Granularity

Minimum useful node kinds:

- repository;
- file;
- class;
- function;
- method;
- module assignment;
- class assignment;
- import;
- callsite/usage node when useful.

Function/method and assignment nodes are especially important. Many bugs live in default values, serializer registries, constants, or class attributes rather than ordinary functions.

## Graph Edges

Minimum useful edge kinds:

- `CONTAINS`: repo -> file -> class/function/assignment;
- `CALLS`: function/method -> called function/method when resolvable;
- `USES`: function/method -> symbol/assignment/class referenced;
- `IMPORTS`: file -> imported module/symbol;
- `SIBLING`: same-file helper relation;
- `OVERRIDES` or `INHERITS` when practical;
- `REGISTERED_AS` or `MAPS_TO` for dictionary/registry patterns when extractable.

Tree-sitter helps extract syntax spans and calls. AST helps robust semantic normalization. A practical builder can combine both.

## Search Strategy

`explore_find` should use a layered retrieval plan:

1. Graph symbol/path search.
2. Graph text search over node names/signatures/docstrings/code snippets.
3. Relaxed graph search if strong constraints fail.
4. Filesystem grep fallback.
5. Span-to-graph mapping for fallback hits.

The final result should be graph nodes, not raw grep chunks. If fallback hits a text span, map it to covering function/class/assignment/file nodes and add those to W.

Find results should help navigation without turning search into repair evidence:

- small-grain implementation nodes may include a bounded line-numbered preview;
- file nodes should omit full text and list top symbols instead;
- previews should expose local symbol references so the planner can reason about dispatch tables, helper calls, wrappers, registries, sibling functions, or assignment maps before repair;
- dict assignment dispatch tables should additionally be exposed as structured key-to-target entries, not only as a flat referenced-symbol list;
- W should still distinguish candidate nodes from explicit reads, so CGM-facing M remains based on hydrated `read`/`memory_commit` evidence.

## Test-Code Guard

Queries derived from benchmark tests are dangerous. The system should:

- block `path:tests/...`, `test_*.py`, `*_test.py`, `TestFoo`, pytest selectors, and assertion helper names;
- convert failure summaries into implementation concepts using the issue text and non-test traceback frames;
- avoid exposing test source as snippets.

This is not hiding useful behavior. It is preventing leakage and preventing the planner from patching tests.

## Working Subgraph W

W stores all candidates and read nodes seen by the planner. Each W node should track:

- id;
- kind;
- name;
- path;
- line span;
- score/source;
- whether code body is present;
- last touched/read step;
- neighbor previews;
- relation edges.

W can be noisy. It is for exploration, not direct CGM input.

## Memory Subgraph M

M stores selected evidence for CGM. A good M is small but causal:

- public entry point or failing implementation frame;
- dispatcher/bridge if present;
- concrete downstream implementation;
- relevant assignment/registry/default value;
- sibling helper only if it explains behavior.

M should avoid:

- benchmark tests;
- raw file dumps;
- import-only nodes;
- large unrelated classes;
- stale nodes from a failed hypothesis.

## Hydration

When a W node is committed to M, ensure M has full code body:

1. use read result from W if available;
2. otherwise read exact node span from sandbox;
3. otherwise use the best embedded snippet and mark it partial.

CGM should not receive M nodes with only names and paths unless no code is available.

## Read Behavior

`read` should support:

- body of a function/method/class/assignment;
- header/signature only;
- around a line number;
- small file window.

Read output should include:

- line-numbered code;
- structure facts such as calls and assignments;
- small neighbor previews with names and optionally short bodies;
- whether code was read from sandbox or embedded graph snippet.

## Avoiding Over-Search

After a successful read, the observation should report factual state:

- code text and line numbers for the read node;
- same-file implementation references visible in that code;
- which read nodes are already committed to M and which are not;
- whether prior patch attempts were rejected, rolled back, or test-failed.

The observation should not rank these facts as a recommended next action.

Repeated identical find/read/expand with no delta should be blocked or rewritten into a more productive action.

## Minimal Evidence Is Issue-Dependent

Do not hard-code a universal number of nodes. Use these principles:

- one-node repair is fine if the function body fully explains the bug;
- dispatcher bugs often need dispatcher plus concrete implementation;
- registry/default bugs often need assignment plus serializer/converter;
- operator/composition bugs often need public API plus recursive/helper function;
- runtime shell bugs often need command builder plus callsite.
