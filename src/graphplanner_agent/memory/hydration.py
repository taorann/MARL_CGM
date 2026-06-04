from __future__ import annotations

from pathlib import Path

from graphplanner_agent.graph.read import read_node, read_node_from_runtime
from graphplanner_agent.graph.schema import GraphNode, RepoGraph
from graphplanner_agent.memory.working import WorkingMemory


def hydrate_node(root: Path, graph: RepoGraph, working: WorkingMemory, node_id: str) -> GraphNode:
    node = working.get(node_id) or graph.nodes.get(node_id)
    if not node:
        raise KeyError(f"unknown node id: {node_id}")
    if node.has_code:
        return node
    return read_node(root, node, "body")


def hydrate_node_from_runtime(runtime, graph: RepoGraph, working: WorkingMemory, node_id: str) -> GraphNode:
    node = working.get(node_id) or graph.nodes.get(node_id)
    if not node:
        raise KeyError(f"unknown node id: {node_id}")
    if node.has_code:
        return node
    return read_node_from_runtime(runtime, node, "body")
