from pathlib import Path
import base64
import gzip
import json
import tempfile
import unittest
from unittest.mock import patch

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.datasets import TaskSpec, load_tasks
from graphplanner_agent.env import CodeRepairEnv
from graphplanner_agent.env.action_handlers import handle_action
from graphplanner_agent.graph.build import build_python_graph
from graphplanner_agent.graph.schema import GraphNode, RepoGraph
from graphplanner_agent.graph.search import search_graph
from graphplanner_agent.infra.http import should_bypass_proxy
import graphplanner_agent.integrations.codefuse_cgm.service as cgm_service
from graphplanner_agent.integrations.codefuse_cgm.service import parse_model_output
from graphplanner_agent.integrations.dashscope_cgm_bridge import BridgeConfig, DashScopeCgmBridge, _snippet_section
from graphplanner_agent.memory import CgmMemory, WorkingMemory
from graphplanner_agent.memory.hydration import hydrate_node
from graphplanner_agent.planner import PlannerAction
from graphplanner_agent.planner.client import StaticPlannerClient, _parse_streaming_chat_message
from graphplanner_agent.planner.loop import PlannerLoop
from graphplanner_agent.planner.prompt import build_messages
from graphplanner_agent.planner.response_parser import parse_planner_message, parse_planner_response
from graphplanner_agent.repair.cgm_client import CgmUnavailableError, StaticCgmClient
from graphplanner_agent.repair.cgm_context import build_cgm_payload, validate_cgm_payload
from graphplanner_agent.repair.patch_apply import normalize_patch, validate_patch
from graphplanner_agent.repair.patch_schema import parse_cgm_output
from graphplanner_agent.runtime import LocalRepoRuntime, make_runtime
from graphplanner_agent.cli import eval as eval_cli
from graphplanner_agent.cli import eval_parallel as eval_parallel_cli
from graphplanner_agent.cli import eval_supervisor as eval_supervisor_cli
from graphplanner_agent.cli.remote_preflight import normalize_remote_preflight_mode, run_remote_swe_preflight
from graphplanner_agent.runtime.remote_swe import (
    RemoteSweRuntime,
    decode_repo_graph_payload,
    is_wrong_python_env,
    wrap_testbed_test_command,
)
from graphplanner_agent.runtime.remote_swe_session import (
    RemoteSweError,
    _parse_proxy_response,
    clean_remote_stderr,
    infer_sif_dir_from_ref,
    normalize_sif_image_ref,
)
from graphplanner_agent.runtime.sandbox_base import CommandResult, TestResult
from graphplanner_agent.runtime.swebench_official import official_eval_script, parse_official_report
from graphplanner_agent.runtime import swebench_pro
from graphplanner_agent.runtime.test_runner import behavior_summary
from graphplanner_agent.telemetry.progress import ProgressTracker


class AgentRebuildTests(unittest.TestCase):
    def test_parse_visible_thinking_and_json_action(self):
        parsed = parse_planner_response(
            '<think>look around</think>\n```json\n{"tool":"explore_find","params":{"query":"foo","find_type":"function"}}\n```'
        )
        self.assertEqual(parsed.visible_thinking, "look around")
        self.assertEqual(parsed.action.tool, "explore_find")
        self.assertEqual(parsed.action.params["query"], "foo")

    def test_parse_openai_tool_call_with_reasoning_content(self):
        parsed = parse_planner_message(
            {
                "role": "assistant",
                "content": "<think>content think</think>",
                "reasoning_content": "separate think",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "run_failed_test", "arguments": "{}"},
                    }
                ],
            }
        )

        self.assertEqual(parsed.action.tool, "run_failed_test")
        self.assertIn("separate think", parsed.visible_thinking)
        self.assertIn("content think", parsed.visible_thinking)

    def test_planner_thinking_defaults_enabled_and_can_disable(self):
        with patch.dict("os.environ", {}, clear=True):
            default_config = AgentConfig.from_env()
        with patch.dict("os.environ", {"PLANNER_ENABLE_THINKING": "0"}, clear=True):
            disabled_config = AgentConfig.from_env()

        self.assertIs(default_config.planner_enable_thinking, True)
        self.assertIs(disabled_config.planner_enable_thinking, False)

    def test_dashscope_cgm_bridge_sends_thinking_and_records_reasoning(self):
        seen_bodies = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                events = [
                    {"choices": [{"delta": {"reasoning_content": "thinking "}}]},
                    {"choices": [{"delta": {"reasoning_content": "trace"}}]},
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": json.dumps(
                                        {
                                            "summary": "fix",
                                            "edits": [
                                                {"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}
                                            ],
                                        }
                                    )
                                }
                            }
                        ]
                    },
                ]
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        def fake_urlopen(request, timeout):
            seen_bodies.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        bridge = DashScopeCgmBridge(BridgeConfig(endpoint="http://example", api_key="key", model="qwen3-coder"))
        with patch("graphplanner_agent.integrations.dashscope_cgm_bridge.urllib.request.urlopen", fake_urlopen):
            result = bridge.generate_patch({})

        self.assertTrue(seen_bodies[0]["enable_thinking"])
        self.assertTrue(seen_bodies[0]["stream"])
        self.assertEqual(result["reasoning_content"], "thinking trace")
        self.assertEqual(result["reasoning_chars"], len("thinking trace"))

    def test_dashscope_cgm_bridge_reviews_intent_without_patch(self):
        seen_bodies = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                events = [
                    {"choices": [{"delta": {"reasoning_content": "check evidence "}}]},
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": json.dumps(
                                        {
                                            "verdict": "change_target",
                                            "confidence": 0.31,
                                            "mechanism_assessment": "the consumer is not proven",
                                            "target_assessment": "target is too early",
                                            "evidence_gaps": ["read the output formatter"],
                                            "suggested_next_action": "read formatter",
                                            "adoption_advice": "revise toward the consumer path",
                                        }
                                    )
                                }
                            }
                        ]
                    },
                ]
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        def fake_urlopen(request, timeout):
            seen_bodies.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        bridge = DashScopeCgmBridge(BridgeConfig(endpoint="http://example", api_key="key", model="qwen3-coder"))
        with patch("graphplanner_agent.integrations.dashscope_cgm_bridge.urllib.request.urlopen", fake_urlopen):
            result = bridge.review_intent({"plan_text": "fix target", "snippets": []})

        prompt = seen_bodies[0]["messages"][0]["content"]
        self.assertIn("Do not write a patch", prompt)
        self.assertIn("Benchmark test source is unavailable", prompt)
        self.assertIn("never benchmark tests", prompt)
        self.assertIn("adoption_advice", prompt)
        self.assertIn("Do not output contract-style fields", prompt)
        self.assertEqual(result["review"]["verdict"], "change_target")
        self.assertEqual(result["review"]["evidence_gaps"], ["read the output formatter"])
        self.assertEqual(result["review"]["adoption_advice"], "revise toward the consumer path")
        self.assertEqual(result["reasoning_content"], "check evidence")

    def test_streaming_planner_parser_accumulates_tool_call_and_reasoning(self):
        class FakeResponse:
            def __iter__(self):
                events = [
                    {"choices": [{"delta": {"reasoning_content": "think "}}]},
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_",
                                            "type": "function",
                                            "function": {"name": "run_", "arguments": "{"},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "1",
                                            "function": {"name": "failed_test", "arguments": "}"},
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {"choices": [{"delta": {"reasoning_content": "more"}}]},
                ]
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        message = _parse_streaming_chat_message(FakeResponse())

        self.assertEqual(message["reasoning_content"], "think more")
        self.assertEqual(message["tool_calls"][0]["id"], "call_1")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "run_failed_test")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], "{}")

    def test_tool_calling_prompt_does_not_ask_for_json_action(self):
        messages = build_messages("{}", tool_calling=True)

        self.assertIn("Call exactly one provided tool", messages[0]["content"])
        self.assertNotIn("Emit exactly one JSON action", messages[0]["content"])

    def test_private_cgm_endpoint_bypasses_proxy(self):
        self.assertTrue(should_bypass_proxy("http://172.20.84.101:30001/generate"))
        self.assertTrue(should_bypass_proxy("http://127.0.0.1:30001/generate"))
        self.assertFalse(should_bypass_proxy("https://example.com/generate"))

    def test_planner_loop_can_use_openai_tool_calls(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig(max_steps=1))
            result = PlannerLoop(env, FakeToolPlannerClient(), AgentConfig(max_steps=1, planner_tool_calling=True)).run()

            self.assertEqual(result.status, "not_pass")
            self.assertIsNotNone(env.failure_summary)

    def test_observation_reports_done_after_verified_patch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.verified = True

            observation = json.loads(env.observe())

            self.assertTrue(observation["runtime_facts"]["done_after_verified_repair"])

    def test_graph_search_skips_tests_and_hydrates_memory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_calc.py").write_text("def test_add():\n    assert False\n", encoding="utf-8")

            graph = build_python_graph(root)
            results, warning = search_graph(graph, "add", "function")

            self.assertIsNone(warning)
            self.assertTrue(results)
            self.assertTrue(all("tests/" not in result.node.path for result in results))

            working = WorkingMemory()
            working.add(results[0].node, "test")
            hydrated = hydrate_node(root, graph, working, results[0].node.id)
            memory = CgmMemory()
            memory.commit([hydrated])

            self.assertEqual(hydrated.text, "def add(a, b):\n    return a - b\n")
            self.assertTrue(memory.summary()[0]["has_code"])

    def test_search_allows_assert_term_but_excludes_test_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "checks.py").write_text("def check(value):\n    assert value\n    return value\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_checks.py").write_text("def test_check():\n    assert False\n", encoding="utf-8")
            graph = build_python_graph(root)

            results, warning = search_graph(graph, "assert", "function")

            self.assertIsNone(warning)
            self.assertTrue(results)
            self.assertEqual(results[0].node.path, "pkg/checks.py")

    def test_search_path_glob_scopes_broad_terms(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg" / "ascii").mkdir(parents=True)
            (root / "pkg" / "time").mkdir(parents=True)
            (root / "pkg" / "ascii" / "core.py").write_text(
                "class Writer:\n"
                "    def write(self):\n"
                "        self.formats = {}\n",
                encoding="utf-8",
            )
            (root / "pkg" / "time" / "formats.py").write_text(
                "class TimeFormat:\n"
                "    pass\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)

            results, warning = search_graph(graph, "formats", "any", root=root, path_glob="pkg/ascii/*.py")

            self.assertIsNone(warning)
            self.assertTrue(results)
            self.assertTrue(all(node.node.path.startswith("pkg/ascii/") for node in results))

            recursive_results, recursive_warning = search_graph(
                graph,
                "formats",
                "any",
                root=root,
                path_glob="**/ascii/**/*.py",
            )

            self.assertIsNone(recursive_warning)
            self.assertTrue(recursive_results)
            self.assertTrue(all(node.node.path.startswith("pkg/ascii/") for node in recursive_results))

    def test_expand_excludes_test_paths(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("func:pkg/app.py:do", "function", "do", "pkg/app.py", 1, 3))
        graph.add_node(GraphNode("file:tests/test_app.py", "file", "tests/test_app.py", "tests/test_app.py", 1, 10))
        graph.add_node(GraphNode("func:pkg/next.py:next", "function", "next", "pkg/next.py", 1, 3))
        graph.add_edge("func:pkg/app.py:do", "file:tests/test_app.py", "CALLS")
        graph.add_edge("func:pkg/app.py:do", "func:pkg/next.py:next", "CALLS")
        env = CodeRepairEnv(
            task=TaskSpec("remoteish", Path("."), "title", "body"),
            runtime=FakeFileRuntime(),
            cgm=StaticCgmClient({"patch": {"edits": []}}),
            config=AgentConfig(),
            graph=graph,
        )

        result = env.step(PlannerAction("explore_expand", {"anchor": "func:pkg/app.py:do", "expand_mode": "callees"}))

        self.assertEqual([node["path"] for node in result["results"]], ["pkg/next.py"])

    def test_mechanism_expand_exposes_lazy_inheritance_and_composition_context(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "core.py").write_text(
                "class BaseData:\n"
                "    def write(self, rows):\n"
                "        return rows\n"
                "\n"
                "class BaseReader:\n"
                "    def write(self, table):\n"
                "        return self.data.write(table)\n",
                encoding="utf-8",
            )
            (root / "pkg" / "html.py").write_text(
                "from . import core\n"
                "\n"
                "class HTMLData(core.BaseData):\n"
                "    pass\n"
                "\n"
                "class HTML(core.BaseReader):\n"
                "    data_class = HTMLData\n"
                "    def write(self, table):\n"
                "        return []\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)
            anchor = next(node.id for node in graph.nodes.values() if node.kind == "class" and node.name == "HTML")
            env = CodeRepairEnv(
                task=TaskSpec("lazy", root, "title", "body"),
                runtime=LocalRepoRuntime(root),
                cgm=StaticCgmClient({"patch": {"edits": []}}),
                config=AgentConfig(),
                graph=graph,
            )

            result = env.step(PlannerAction("explore_expand", {"anchor": anchor, "expand_mode": "mechanism"}))

            relations = {(item["relation"], item["name"]) for item in result["results"]}
            self.assertIn(("parent_class", "BaseReader"), relations)
            self.assertIn(("overridden_method", "write"), relations)
            self.assertIn(("composition:data_class", "HTMLData"), relations)
            self.assertIn(("composition_parent", "BaseData"), relations)
            self.assertIn(("pipeline_method", "write"), relations)
            self.assertTrue(any("code" in item for item in result["results"]))

    def test_mechanism_expand_uses_working_code_overlay_after_read(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "basic.py").write_text(
                "class BasicData:\n"
                "    pass\n"
                "\n"
                "class CsvData(BasicData):\n"
                "    pass\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)
            csv_data_id = next(node.id for node in graph.nodes.values() if node.kind == "class" and node.name == "CsvData")
            basic_data_id = next(node.id for node in graph.nodes.values() if node.kind == "class" and node.name == "BasicData")
            graph.nodes[csv_data_id] = GraphNode(
                id=graph.nodes[csv_data_id].id,
                kind=graph.nodes[csv_data_id].kind,
                name=graph.nodes[csv_data_id].name,
                path=graph.nodes[csv_data_id].path,
                start_line=graph.nodes[csv_data_id].start_line,
                end_line=graph.nodes[csv_data_id].end_line,
                text=None,
                preview=graph.nodes[csv_data_id].preview,
                parent_id=graph.nodes[csv_data_id].parent_id,
            )
            env = CodeRepairEnv(
                task=TaskSpec("lazy", root, "title", "body"),
                runtime=LocalRepoRuntime(root),
                cgm=StaticCgmClient({"patch": {"edits": []}}),
                config=AgentConfig(),
                graph=graph,
            )

            before_read = env.step(PlannerAction("explore_expand", {"anchor": csv_data_id, "expand_mode": "mechanism"}))
            env.step(PlannerAction("read", {"node_id": csv_data_id}))
            after_read = env.step(PlannerAction("explore_expand", {"anchor": csv_data_id, "expand_mode": "mechanism"}))

            self.assertFalse(before_read["results"])
            self.assertIn(basic_data_id, [item["id"] for item in after_read["results"]])

    def test_expand_preview_does_not_shrink_later_read_body(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            long_body = "\n".join(f"    value_{idx} = {idx}" for idx in range(40))
            (root / "pkg" / "core.py").write_text(
                "class BaseData:\n"
                f"{long_body}\n"
                "    tail_marker = 'full-body'\n",
                encoding="utf-8",
            )
            (root / "pkg" / "html.py").write_text(
                "from .core import BaseData\n"
                "class HTMLData(BaseData):\n"
                "    pass\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)
            anchor = next(node.id for node in graph.nodes.values() if node.kind == "class" and node.name == "HTMLData")
            base_id = next(node.id for node in graph.nodes.values() if node.kind == "class" and node.name == "BaseData")
            env = CodeRepairEnv(
                task=TaskSpec("lazy", root, "title", "body"),
                runtime=LocalRepoRuntime(root),
                cgm=StaticCgmClient({"patch": {"edits": []}}),
                config=AgentConfig(),
                graph=graph,
            )

            expanded = env.step(PlannerAction("explore_expand", {"anchor": anchor, "expand_mode": "mechanism"}))
            read = env.step(PlannerAction("read", {"node_id": base_id, "view": "body"}))

            self.assertTrue(any(item["id"] == base_id and item.get("code_preview_truncated") for item in expanded["results"]))
            self.assertIn("tail_marker", read["code"])

    def test_mechanism_expand_exposes_inherited_methods_from_base_lineage(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "basic.py").write_text(
                "class BaseData:\n"
                "    def str_vals(self):\n"
                "        return []\n"
                "    def write(self, lines):\n"
                "        return self.str_vals()\n"
                "\n"
                "class BasicData(BaseData):\n"
                "    pass\n"
                "\n"
                "class CsvData(BasicData):\n"
                "    pass\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)
            anchor = next(node.id for node in graph.nodes.values() if node.kind == "class" and node.name == "CsvData")
            env = CodeRepairEnv(
                task=TaskSpec("lazy", root, "title", "body"),
                runtime=LocalRepoRuntime(root),
                cgm=StaticCgmClient({"patch": {"edits": []}}),
                config=AgentConfig(),
                graph=graph,
            )

            result = env.step(PlannerAction("explore_expand", {"anchor": anchor, "expand_mode": "mechanism"}))

            relations = {(item["relation"], item["name"]) for item in result["results"]}
            self.assertIn(("parent_class", "BasicData"), relations)
            self.assertIn(("ancestor_class", "BaseData"), relations)
            self.assertIn(("inherited_method", "write"), relations)
            self.assertIn(("inherited_method", "str_vals"), relations)

    def test_owner_flow_expand_exposes_attribute_owner_and_consumer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "core.py").write_text(
                "class BaseData:\n"
                "    formats = {}\n"
                "    def _set_col_formats(self):\n"
                "        for col in self.cols:\n"
                "            if col.info.name in self.formats:\n"
                "                col.info.format = self.formats[col.info.name]\n"
                "\n"
                "class BaseReader:\n"
                "    pass\n",
                encoding="utf-8",
            )
            (root / "pkg" / "html.py").write_text(
                "from . import core\n"
                "\n"
                "class HTMLData(core.BaseData):\n"
                "    pass\n"
                "\n"
                "class HTML(core.BaseReader):\n"
                "    data_class = HTMLData\n"
                "    def write(self, table):\n"
                "        return []\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)
            anchor = next(node.id for node in graph.nodes.values() if node.kind == "class" and node.name == "HTML")
            env = CodeRepairEnv(
                task=TaskSpec("owner", root, "title", "body"),
                runtime=LocalRepoRuntime(root),
                cgm=StaticCgmClient({"patch": {"edits": []}}),
                config=AgentConfig(),
                graph=graph,
            )

            result = env.step(PlannerAction("explore_expand", {"anchor": anchor, "expand_mode": "owner_flow", "symbol": "formats"}))

            relations = {(item["relation"], item["name"]) for item in result["results"]}
            self.assertIn(("attribute_owner", "formats"), relations)
            self.assertIn(("symbol_consumer", "_set_col_formats"), relations)
            self.assertTrue(all(item.get("suggested_read", {}).get("node_id") for item in result["results"]))

    def test_failed_patch_rolls_back(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "bad edit", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a * b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))

            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))
            node_id = env.latest_result["results"][0]["id"]
            env.step(PlannerAction("read", {"node_id": node_id, "view": "body"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.step(PlannerAction("run_failed_test", {}))
            result = env.step(PlannerAction("repair", self._repair_params(node_id)))

            self.assertEqual(result["status"], "test_failed")
            self.assertIn("return a - b", (root / "pkg" / "calc.py").read_text(encoding="utf-8"))

    def test_repair_chunk_keeps_valid_patch_without_final_verification(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "chunk add implementation", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))

            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))
            node_id = env.latest_result["results"][0]["id"]
            env.step(PlannerAction("read", {"node_id": node_id, "view": "body"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.step(PlannerAction("run_failed_test", {}))
            params = self._repair_params(node_id, "apply the add implementation chunk")
            params["remaining_work"] = "run final verification after this chunk"

            result = env.step(PlannerAction("repair_chunk", params))

            self.assertEqual(result["status"], "chunk_applied")
            self.assertFalse(result["done"])
            self.assertFalse(env.done)
            self.assertIn("return a + b", (root / "pkg" / "calc.py").read_text(encoding="utf-8"))

    def test_repair_propose_saves_pending_patch_without_testing_or_applying(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {
                    "insight_summary": {"bug_mechanism": "add subtracts", "patch_strategy": "replace subtraction"},
                    "patch": {"summary": "candidate add implementation", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}]},
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))

            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))
            node_id = env.latest_result["results"][0]["id"]
            env.step(PlannerAction("read", {"node_id": node_id, "view": "body"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.step(PlannerAction("run_failed_test", {}))
            result = env.step(PlannerAction("repair_propose", self._repair_params(node_id)))

            self.assertEqual(result["status"], "patch_proposed")
            self.assertIsNotNone(env.pending_patch)
            self.assertFalse(env.verified)
            self.assertIn("return a - b", (root / "pkg" / "calc.py").read_text(encoding="utf-8"))
            observation = json.loads(env.observe())
            self.assertTrue(observation["runtime_facts"]["pending_patch_present"])
            self.assertTrue(observation["recent_cgm_insights"])

    def test_repair_submit_tests_pending_patch_and_clears_it_on_pass(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "candidate add implementation", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))

            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))
            node_id = env.latest_result["results"][0]["id"]
            env.step(PlannerAction("read", {"node_id": node_id, "view": "body"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.step(PlannerAction("run_failed_test", {}))
            env.step(PlannerAction("repair_propose", self._repair_params(node_id)))
            result = env.step(PlannerAction("repair_submit", {"decision": "candidate directly fixes the read implementation"}))

            self.assertEqual(result["status"], "passed")
            self.assertTrue(env.verified)
            self.assertIsNone(env.pending_patch)
            self.assertIn("return a + b", (root / "pkg" / "calc.py").read_text(encoding="utf-8"))

    def test_repair_revise_sends_pending_patch_and_history_to_cgm(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "bad candidate", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a * b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))

            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))
            node_id = env.latest_result["results"][0]["id"]
            env.step(PlannerAction("read", {"node_id": node_id, "view": "body"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.step(PlannerAction("run_failed_test", {}))
            env.step(PlannerAction("repair_propose", self._repair_params(node_id)))
            cgm.response = {
                "patch": {"summary": "revised add candidate", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}]}
            }
            params = self._repair_params(node_id, "revise the candidate to add instead of multiply")
            params["revision_focus"] = "pending patch multiplies; the issue requires addition"
            params["pending_patch_review"] = {
                "coverage": "partial",
                "risks": ["wrong operator"],
                "requested_change": "replace multiplication with addition",
            }
            result = env.step(PlannerAction("repair_revise", params))

            self.assertEqual(result["status"], "patch_proposed")
            self.assertIn("return a - b", (root / "pkg" / "calc.py").read_text(encoding="utf-8"))
            revise_payload = cgm.payloads[-1]
            self.assertTrue(revise_payload["pending_patch"])
            self.assertTrue(revise_payload["repair_history"])
            self.assertEqual(revise_payload["planner_decision_context"]["pending_patch_review"]["coverage"], "partial")

    def test_action_guards_reject_empty_find_query(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())

            result = env.step(PlannerAction("explore_find", {"query": "", "find_type": "function"}))

            self.assertTrue(result["blocked"])
            self.assertIn("non-empty", result["reason"])

    def test_read_and_memory_commit_use_runtime_file_api(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("func:pkg/calc.py:add", "function", "add", "pkg/calc.py", 1, 2))
        env = CodeRepairEnv(
            task=TaskSpec("remoteish", Path("."), "title", "body"),
            runtime=FakeFileRuntime(),
            cgm=StaticCgmClient({"patch": {"edits": []}}),
            config=AgentConfig(),
            graph=graph,
        )

        read = env.step(PlannerAction("read", {"node_id": "func:pkg/calc.py:add", "view": "body"}))
        commit = env.step(PlannerAction("memory_commit", {"select_ids": ["func:pkg/calc.py:add"]}))

        self.assertIn("return a + b", read["code"])
        self.assertEqual(commit["committed"], ["func:pkg/calc.py:add"])
        self.assertTrue(env.memory.nodes["func:pkg/calc.py:add"].has_code)

    def test_read_view_defaults_to_body(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("func:pkg/calc.py:add", "function", "add", "pkg/calc.py", 1, 2))
        env = CodeRepairEnv(
            task=TaskSpec("remoteish", Path("."), "title", "body"),
            runtime=FakeFileRuntime(),
            cgm=StaticCgmClient({"patch": {"edits": []}}),
            config=AgentConfig(),
            graph=graph,
        )

        result = env.step(PlannerAction("read", {"node_id": "func:pkg/calc.py:add"}))

        self.assertFalse(result.get("blocked", False))
        self.assertIn("return a + b", result["code"])

    def test_find_function_returns_code_preview_and_symbol_references(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def outer(value):\n"
                "    return inner(value)\n\n"
                "def inner(value):\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())

            result = env.step(PlannerAction("explore_find", {"query": "outer", "find_type": "function"}))
            first = result["results"][0]

            self.assertIn("return inner(value)", first["code"])
            self.assertTrue(any(reference["name"] == "inner" for reference in first["unread_local_symbol_references"]))
            self.assertTrue(env.working.get(first["id"]).has_code)
            self.assertIn("return inner(value)", env.working.get(first["id"]).text)
            self.assertTrue(env.working.entries[first["id"]].source.startswith("find_preview:"))

    def test_grep_code_returns_covering_node_and_suggested_read(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "class Writer:\n"
                "    def write(self, value):\n"
                "        formats = {'value': str}\n"
                "        return formats['value'](value)\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())

            result = env.step(
                PlannerAction(
                    "grep_code",
                    {"pattern": "formats", "path_glob": "pkg/*.py", "context_lines": 1, "limit": 5},
                )
            )

            self.assertEqual(result["tool"], "grep_code")
            self.assertEqual(len(result["hits"]), 2)
            hit = result["hits"][0]
            self.assertEqual(hit["path"], "pkg/flow.py")
            self.assertIn("formats", hit["context"])
            self.assertIn("suggested_read", hit)
            self.assertEqual(hit["suggested_read"]["view"], f"around_line:{hit['line']}")
            self.assertIn(hit["covering_node"]["id"], env.working.entries)
            self.assertFalse(env.working.get(hit["covering_node"]["id"]).has_code)

    def test_grep_code_path_glob_matches_direct_child_for_recursive_pattern(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg" / "ascii").mkdir(parents=True)
            (root / "pkg" / "time").mkdir(parents=True)
            (root / "pkg" / "ascii" / "core.py").write_text(
                "def write():\n"
                "    formats = {'value': str}\n"
                "    return formats\n",
                encoding="utf-8",
            )
            (root / "pkg" / "time" / "formats.py").write_text(
                "def unrelated():\n"
                "    formats = {}\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())

            result = env.step(
                PlannerAction(
                    "grep_code",
                    {"pattern": "formats", "path_glob": "**/ascii/**/*.py", "context_lines": 1, "limit": 5},
                )
            )

            self.assertEqual(result["tool"], "grep_code")
            self.assertTrue(result["hits"])
            self.assertTrue(all(hit["path"].startswith("pkg/ascii/") for hit in result["hits"]))

    def test_find_puts_preview_not_full_code_into_working_memory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            body = "\n".join(f"    value += {idx}" for idx in range(45))
            (root / "pkg" / "flow.py").write_text(
                "def long_func(value):\n"
                f"{body}\n"
                "    return value\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())

            result = env.step(PlannerAction("explore_find", {"query": "long_func", "find_type": "function"}))
            first = result["results"][0]
            working = env.working.get(first["id"])

            self.assertTrue(first["code_preview_truncated"])
            self.assertTrue(working.has_code)
            self.assertNotIn("value += 44", working.text)
            self.assertTrue(env.working.entries[first["id"]].source.startswith("find_preview:"))

            commit = env.step(PlannerAction("memory_commit", {"select_ids": [first["id"]]}))
            self.assertTrue(commit["blocked"])
            self.assertIn("requires an explicit read", commit["reason"])

            env.step(PlannerAction("read", {"node_id": first["id"], "view": "body"}))
            commit = env.step(PlannerAction("memory_commit", {"select_ids": [first["id"]]}))
            self.assertFalse(commit.get("blocked", False))
            self.assertIn("value += 44", env.memory.nodes[first["id"]].text)

    def test_read_assignment_reports_dispatch_table_facts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def _cstack(left, right):\n"
                "    return left\n\n"
                "def _cdot(left, right):\n"
                "    return right\n\n"
                "_operators = {'&': _cstack, '|': _cdot}\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())

            result = env.step(PlannerAction("read", {"node_id": "_operators"}))

            self.assertEqual(result["dispatch_tables"][0]["name"], "_operators")
            entries = result["dispatch_tables"][0]["entries"]
            self.assertEqual([(entry["key"], entry["target"]) for entry in entries], [("&", "_cstack"), ("|", "_cdot")])
            self.assertEqual(entries[0]["name"], "_cstack")
            self.assertNotIn("next_action", entries[0])
            references = result["local_symbol_references"]
            self.assertEqual([reference["name"] for reference in references[:2]], ["_cstack", "_cdot"])

            observation = json.loads(env.observe())
            self.assertEqual(observation["dispatch_tables"][0]["entries"][0]["target"], "_cstack")

    def test_find_file_result_omits_code_and_lists_top_symbols(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def outer(value):\n"
                "    return inner(value)\n\n"
                "def inner(value):\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())

            result = env.step(PlannerAction("explore_find", {"query": "pkg/flow.py", "find_type": "file"}))
            first = result["results"][0]

            self.assertNotIn("code", first)
            self.assertEqual([symbol["name"] for symbol in first["top_symbols"]], ["outer", "inner"])

    def test_find_class_name_fallback_when_scope_is_too_strict(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "separable.py").write_text(
                "def separability_matrix(model):\n"
                "    return model\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)

            results, warning = search_graph(graph, "separability_matrix", "function", class_name="WrongClass", root=root)

            self.assertTrue(results)
            self.assertIn("retried without", warning)
            self.assertEqual(results[0].node.name, "separability_matrix")

    def test_assignment_find_matches_remote_module_assignment_kind(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("module_assignment:pkg/core.py:BINARY_OPERATORS:1", "module_assignment", "BINARY_OPERATORS", "pkg/core.py", 1, 2))
        graph.add_node(GraphNode("module_assignment:pkg/flow.py:_operators:10", "module_assignment", "_operators", "pkg/flow.py", 10, 11))

        results, warning = search_graph(graph, "_operators", "assignment")

        self.assertIsNone(warning)
        self.assertEqual(results[0].node.kind, "module_assignment")
        self.assertEqual(results[0].node.name, "_operators")

    def test_explore_find_accepts_internal_assignment_alias_but_returns_public_kind(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(
            GraphNode(
                "module_assignment:pkg/flow.py:_operators:10",
                "module_assignment",
                "_operators",
                "pkg/flow.py",
                10,
                11,
                text="_operators = {'&': _cstack}\n",
            )
        )
        env = CodeRepairEnv(
            task=TaskSpec("alias", Path("."), "title", "body"),
            runtime=FakeFileRuntime(),
            cgm=StaticCgmClient({"patch": {"edits": []}}),
            config=AgentConfig(),
            graph=graph,
        )

        result = env.step(PlannerAction("explore_find", {"query": "_operators", "find_type": "module_assignment"}))

        self.assertFalse(result.get("blocked", False))
        self.assertEqual(result["results"][0]["kind"], "assignment")

    def test_method_find_matches_dotted_function_nodes_from_remote_graph(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("func:pkg/core.py:Model.compute:10", "function", "Model.compute", "pkg/core.py", 10, 12))

        results, warning = search_graph(graph, "Model.compute", "method")

        self.assertIsNone(warning)
        self.assertEqual(results[0].node.name, "Model.compute")

    def test_read_short_node_ref_resolves_unique_candidate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            task = TaskSpec("short", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))

            result = env.step(PlannerAction("read", {"node_id": "add"}))

            self.assertFalse(result.get("blocked", False))
            self.assertEqual(result["node"]["name"], "add")
            self.assertIn("return a + b", result["code"])

    def test_ambiguous_short_node_ref_returns_candidates(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("func:pkg/a.py:add", "function", "add", "pkg/a.py", 1, 2))
        graph.add_node(GraphNode("func:pkg/b.py:add", "function", "add", "pkg/b.py", 1, 2))
        env = CodeRepairEnv(
            task=TaskSpec("remoteish", Path("."), "title", "body"),
            runtime=FakeFileRuntime(),
            cgm=StaticCgmClient({"patch": {"edits": []}}),
            config=AgentConfig(),
            graph=graph,
        )

        result = env.step(PlannerAction("read", {"node_id": "add"}))

        self.assertTrue(result["blocked"])
        self.assertIn("ambiguous", result["reason"])
        self.assertEqual(len(result["candidates"]), 2)

    def test_observation_reports_unread_symbol_references(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def outer(value):\n"
                "    return inner(value)\n\n"
                "def inner(value):\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body", test_command="python -m unittest")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "outer"}))

            observation = json.loads(env.observe())

            self.assertEqual(observation["unread_symbol_references"][0]["name"], "inner")
            self.assertFalse(observation["evidence_status"]["fail_to_pass_behavior_present"])

    def test_observation_keeps_trajectory_and_working_code(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))

            observation = json.loads(env.observe())

            self.assertEqual([item["tool"] for item in observation["trajectory_summary"]], ["read", "memory_commit"])
            self.assertEqual(observation["working_code_W"][0]["name"], "add")
            self.assertIn("return a - b", observation["working_code_W"][0]["code"])
            self.assertTrue(observation["working_code_W"][0]["in_repair_memory_M"])
            self.assertEqual(observation["evidence_status"]["working_code_node_count"], 1)

    def test_observation_reports_working_code_truncation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            huge_body = "def huge():\n" + "".join(f"    value_{i} = {i}\n" for i in range(900)) + "    return value_0\n"
            (root / "pkg" / "huge.py").write_text(huge_body, encoding="utf-8")
            task = TaskSpec("huge", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "huge"}))

            observation = json.loads(env.observe())

            report = observation["input_truncation_report"]
            self.assertTrue(report["truncated"])
            working_report = report["fields"][0]
            self.assertEqual(working_report["field"], "working_code_W")
            self.assertEqual(working_report["truncated_nodes"][0]["name"], "huge")
            self.assertGreater(working_report["truncated_nodes"][0]["omitted_chars"], 0)
            self.assertTrue(observation["working_code_W"][0]["truncated"])
            self.assertTrue(observation["evidence_status"]["input_truncated"])

    def test_observation_compacts_large_latest_test_command(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.latest_result = {
                "tool": "repair",
                "status": "test_failed",
                "test_summary": {
                    "status": "failed",
                    "returncode": 1,
                    "command": "python - <<'PY'\n" + ("x" * 5000) + "\nPY",
                    "excerpt": "failed",
                },
            }

            observation = json.loads(env.observe())

            self.assertFalse(observation["input_truncation_report"]["truncated"])
            self.assertNotIn("test_summary", observation["latest_action_result"])
            self.assertIn("failure_feedback", observation["latest_action_result"])
            self.assertIn("failed", observation["latest_action_result"]["failure_feedback"]["error_summary"])

    def test_behavior_summary_keeps_actual_output_without_hidden_expected(self):
        output = """
FAILED astropy/timeseries/tests/test_sampled.py::test_required_columns - AssertionError: assert "TimeSeries object is invalid - expected 'time' as the first columns but found 'time'" == "hidden benchmark expected message"
E   AssertionError: assert "TimeSeries object is invalid - expected 'time' as the first columns but found 'time'" == "hidden benchmark expected message"
        """
        summary = behavior_summary(TestResult("failed", "python -m pytest", output, "", 1))
        serialized = json.dumps(summary, ensure_ascii=False)

        self.assertIn("test_required_columns", serialized)
        self.assertIn("expected 'time' as the first columns but found 'time'", serialized)
        self.assertNotIn("hidden benchmark expected message", serialized)
        self.assertTrue(summary["runtime_observations"]["omitted_hidden_expected_values"])

    def test_behavior_summary_omits_official_eval_command_payload(self):
        command = "python - <<'PY'\nimport base64\nbase64.b64decode('hidden-test-patch')\nPYTESTPATCH\n"
        summary = behavior_summary(TestResult("failed", command, "FAILED tests/test_x.py::test_y", "", 1))
        serialized = json.dumps(summary, ensure_ascii=False)

        self.assertTrue(summary["command_omitted_for_benchmark_hygiene"])
        self.assertNotIn("hidden-test-patch", serialized)

    def test_behavior_summary_extracts_plain_traceback_exception(self):
        output = """
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'erfa'
        """
        summary = behavior_summary(TestResult("infra_bug", "python - <<'PY'", "", output, 1))

        self.assertIn("ModuleNotFoundError", summary["runtime_observations"]["exception_types"])
        self.assertIn("No module named 'erfa'", json.dumps(summary, ensure_ascii=False))

    def test_observation_distinguishes_read_w_from_committed_m(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def outer(value):\n"
                "    return inner(value)\n\n"
                "def inner(value):\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.failure_summary = {"status": "failed"}
            env.step(PlannerAction("read", {"node_id": "outer"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::flow.py::outer::1"]}))
            env.step(PlannerAction("read", {"node_id": "inner"}))

            observation = json.loads(env.observe())

            pending = observation["working_vs_memory"]["read_not_committed_to_M"]
            self.assertEqual(pending[0]["name"], "inner")
            self.assertNotIn("next_action", pending[0])
            self.assertEqual(observation["evidence_status"]["read_not_committed_count"], 1)
            self.assertEqual(observation["current_turn_protocol"]["candidate_memory_commit_nodes"][0]["name"], "inner")

    def test_text_observation_puts_repair_blocker_first(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            config = AgentConfig(observation_mode="text")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), config)
            env.failure_summary = {"status": "failed"}
            env.step(PlannerAction("read", {"node_id": "add"}))

            observation = env.observe()

            self.assertTrue(observation.startswith("CURRENT TURN PROTOCOL"))
            first_screen = observation[:900]
            self.assertIn("repair_memory_M has no hydrated code", first_screen)
            self.assertIn("memory_commit", first_screen)
            self.assertIn("function::pkg::calc.py::add::1", first_screen)
            self.assertIn("WORKING CODE W", observation)

    def test_memory_commit_does_not_auto_include_read_local_references(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def outer(value):\n"
                "    return inner(value)\n\n"
                "def inner(value):\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "outer"}))
            env.step(PlannerAction("read", {"node_id": "inner"}))

            result = env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::flow.py::outer::1"]}))

            self.assertEqual(result["committed"], ["function::pkg::flow.py::outer::1"])
            self.assertNotIn("function::pkg::flow.py::inner::4", [item["id"] for item in result["memory"]])
            self.assertNotIn("auto_included_read_references", result)
            self.assertTrue(result["memory_changed"])

    def test_memory_commit_reports_no_change_for_duplicate_commit(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))

            result = env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))

            self.assertFalse(result["memory_changed"])
            self.assertEqual(result["already_present_ids"], ["function::pkg::calc.py::add::1"])
            self.assertIn("No new repair evidence", result["note_to_planner"])

    def test_repair_requires_structured_evidence_package(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", {"plan": "fix add"}))

            self.assertTrue(result["blocked"])
            self.assertIn("failure_seen", result["reason"])
            self.assertIn("evidence_chain", result["reason"])

    def test_repair_does_not_force_unrelated_w_nodes_into_memory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def outer(value):\n"
                "    return inner(value)\n\n"
                "def inner(value):\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            task = TaskSpec("flow", root, "title", "body")
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "outer"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::flow.py::outer::1"]}))
            env.step(PlannerAction("read", {"node_id": "inner"}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", self._repair_params("function::pkg::flow.py::outer::1", "fix outer")))

            self.assertFalse(result.get("blocked", False))
            self.assertNotIn("considered_alternatives", result.get("reason", ""))

    def test_repair_target_must_appear_in_evidence_chain(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))
            env.failure_summary = {"status": "failed"}
            params = self._repair_params("function::pkg::calc.py::add::1")
            params["evidence_chain"] = [
                {
                    "node_id": "function::pkg::calc.py::missing::99",
                    "role": "decision",
                    "evidence": "not actually read",
                }
            ]

            result = env.step(PlannerAction("repair", params))

            self.assertTrue(result["blocked"])
            self.assertIn("evidence_chain", result["reason"])

    def test_repair_converts_cgm_generation_error_to_patch_rejected(self):
        graph = RepoGraph(root="/testbed")
        node = GraphNode("func:pkg/calc.py:add", "function", "add", "pkg/calc.py", 1, 2, text="def add(a, b):\n    return a - b\n")
        graph.add_node(node)
        env = CodeRepairEnv(
            task=TaskSpec("fake", Path("."), "title", "body"),
            runtime=FakeFileRuntime(),
            cgm=FailingCgmClient(),
            config=AgentConfig(),
            graph=graph,
        )
        env.memory.commit([node])
        env.failure_summary = {"status": "failed"}

        result = env.step(PlannerAction("repair", self._repair_params("func:pkg/calc.py:add")))

        self.assertEqual(result["status"], "patch_rejected")
        self.assertIn("CGM generation failed", result["reason"])
        self.assertIn("CGM generation failed", env.repair_feedback)

    def test_repair_converts_cgm_unavailable_to_retryable_infra_result(self):
        graph = RepoGraph(root="/testbed")
        node = GraphNode("func:pkg/calc.py:add", "function", "add", "pkg/calc.py", 1, 2, text="def add(a, b):\n    return a - b\n")
        graph.add_node(node)
        env = CodeRepairEnv(
            task=TaskSpec("fake", Path("."), "title", "body"),
            runtime=FakeFileRuntime(),
            cgm=UnavailableCgmClient(),
            config=AgentConfig(),
            graph=graph,
        )
        env.memory.commit([node])
        env.failure_summary = {"status": "failed"}

        result = env.step(PlannerAction("repair", self._repair_params("func:pkg/calc.py:add")))

        self.assertEqual(result["status"], "infra_retryable")
        self.assertTrue(result["retryable"])
        self.assertFalse(result["done"])
        self.assertFalse(env.done)
        self.assertEqual(env.status, "not_pass")
        self.assertEqual(result["error_origin"], "cgm_unavailable")
        self.assertIn("CGM unavailable", result["reason"])
        self.assertEqual(result["source_tree_state"], "unchanged")

    def test_syntax_failed_feedback_says_generated_patch_was_rolled_back(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "bad syntax", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "-    return a + b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", self._repair_params("function::pkg::calc.py::add::1")))

            self.assertEqual(result["status"], "syntax_failed")
            self.assertIn("generated patch was syntactically invalid", result["reason"])
            self.assertIn("original source remains unchanged", result["reason"])
            self.assertEqual(result["error_origin"], "generated_patch")
            observation = json.loads(env.observe())
            self.assertNotIn("prior_repair_feedback", observation)
            self.assertEqual(observation["last_repair_attempt"]["error_origin"], "generated_patch")
            self.assertEqual(observation["runtime_facts"]["last_repair_error_origin"], "generated_patch")
            feedback = observation["last_repair_attempt"]["failure_feedback"]
            self.assertIn("failed_patch", feedback)
            self.assertIn("-    return a + b", feedback["failed_patch"]["edits"][0]["new_text"])
            self.assertIn("generated patch was syntactically invalid", feedback["error_summary"])

    def test_failed_repair_observation_requires_deeper_followup(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "same behavior", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a - b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", self._repair_params("function::pkg::calc.py::add::1")))
            observation = json.loads(env.observe())

            self.assertEqual(result["status"], "test_failed")
            self.assertIn("prior intent_analysis", env.repair_feedback)
            followup = observation["current_turn_protocol"]["repair_failure_followup"]
            self.assertEqual(followup["instruction"], "deepen_before_next_repair")
            self.assertIn("failed_patch", followup["required_before_next_repair"][1])
            self.assertIn("fallback_after_failed_patch", observation["current_turn_protocol"]["repair_mechanism_requirements"])
            self.assertIn("repair_review", observation["current_turn_protocol"]["repair_mechanism_requirements"])

    def test_repair_review_allowed_after_failed_same_memory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {
                    "review": {
                        "verdict": "needs_more_evidence",
                        "confidence": 0.2,
                        "mechanism_assessment": "the observed behavior is still not explained",
                        "target_assessment": "same target may be incomplete",
                        "evidence_gaps": ["read caller"],
                        "adoption_advice": "commit downstream evidence",
                        "suggested_next_action": "read caller before repair",
                    }
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}
            env.repair_history.record_outcome("test_failed", [node_id], "generated_patch_behavior")

            result = env.step(PlannerAction("repair_review", self._repair_params(node_id, "review before retry")))
            observation = json.loads(env.observe())

            self.assertEqual(result["status"], "reviewed")
            self.assertEqual(result["review"]["verdict"], "needs_more_evidence")
            self.assertEqual(result["cgm_payload"]["plan_targets"], ["pkg/calc.py:1-2"])
            self.assertIn("repair_review", observation["text_notes_T"][-1]["tag"])
            self.assertTrue(
                any("latest repair_review did not endorse immediate patching" in item for item in observation["current_turn_protocol"]["blockers"])
            )
            blocked = env.step(PlannerAction("repair", self._repair_params(node_id, "same evidence after review")))
            self.assertTrue(blocked["blocked"])
            self.assertIn("latest repair_review verdict=needs_more_evidence", blocked["reason"])
            self.assertEqual(observation["runtime_facts"]["last_repair_review_verdict"], "needs_more_evidence")

    def test_ready_repair_review_unblocks_same_memory_repair_package(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {"patch": {"summary": "reviewed retry", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a - b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}
            env.repair_history.record_outcome("test_failed", [node_id], "generated_patch_behavior")

            review = env.step(PlannerAction("repair_review", self._repair_params(node_id, "review same package")))
            observation = json.loads(env.observe())
            result = env.step(PlannerAction("repair", self._repair_params(node_id, "review same package")))

            self.assertEqual(review["review"]["verdict"], "ready")
            self.assertIsNone(observation["current_turn_protocol"]["repair_disabled_reason"])
            self.assertFalse(result.get("blocked", False), result.get("reason"))

    def test_ready_review_with_evidence_gaps_can_still_be_adopted(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {
                    "patch": {
                        "summary": "adopt ready review with optional gaps",
                        "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a - b\n"}],
                    },
                    "review": {
                        "verdict": "ready",
                        "confidence": 0.8,
                        "mechanism_assessment": "target is plausible",
                        "evidence_gaps": ["read the formatter API definition"],
                        "suggested_next_action": "read formatter API before patching",
                        "adoption_advice": "revise after reading the formatter API",
                    }
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            review = env.step(PlannerAction("repair_review", self._repair_params(node_id, "review incomplete contract")))
            observation = json.loads(env.observe())
            result = env.step(PlannerAction("repair", self._repair_params(node_id, "accept incomplete contract")))

            self.assertEqual(review["review"]["verdict"], "ready")
            self.assertIn("adoption_caveat", review["review"])
            self.assertFalse(
                any("evidence_gaps" in item for item in observation["current_turn_protocol"]["blockers"])
            )
            self.assertFalse(result.get("blocked", False), result.get("reason"))

    def test_ready_review_without_evidence_gaps_can_be_adopted(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {
                    "review": {
                        "verdict": "ready",
                        "confidence": 0.9,
                        "mechanism_assessment": "target is plausible",
                        "evidence_gaps": [],
                        "suggested_next_action": "repair now",
                        "adoption_advice": "adopt the current target if planner agrees",
                    }
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            review = env.step(PlannerAction("repair_review", self._repair_params(node_id, "review ready advice")))
            result = env.step(PlannerAction("repair", self._repair_params(node_id, "adopt ready advice")))

            self.assertEqual(review["review"]["verdict"], "ready")
            self.assertFalse(result.get("blocked", False), result.get("reason"))

    def test_ready_review_that_suggests_more_inspection_stays_advisory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {
                    "patch": {
                        "summary": "adopt ready review with suggested inspection",
                        "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a - b\n"}],
                    },
                    "review": {
                        "verdict": "ready",
                        "confidence": 0.95,
                        "mechanism_assessment": "target is plausible",
                        "evidence_gaps": [],
                        "suggested_next_action": "Inspect implementation lines after 421 before repair",
                        "adoption_advice": "revise after inspection",
                    }
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            review = env.step(PlannerAction("repair_review", self._repair_params(node_id, "review asks for inspection")))
            result = env.step(PlannerAction("repair", self._repair_params(node_id, "accept inspection-needed contract")))

            self.assertEqual(review["review"]["verdict"], "ready")
            self.assertIn("adoption_caveat", review["review"])
            self.assertFalse(result.get("blocked", False), result.get("reason"))

    def test_ready_repair_review_is_advisory_not_selected_contract(self):
        class ReviewAdviceCgmClient:
            def __init__(self):
                self.patch_payload = None

            def review_intent(self, payload):
                return {
                    "review": {
                        "verdict": "ready",
                        "confidence": 0.9,
                        "mechanism_assessment": "message compares the wrong column facts",
                        "target_assessment": "target is plausible",
                        "evidence_gaps": [],
                        "suggested_next_action": "repair if planner adopts this critique",
                        "adoption_advice": "adopt if visible code confirms message construction happens here",
                    }
                }

            def generate_patch(self, payload):
                self.patch_payload = payload
                return {
                    "patch": {
                        "summary": "contracted retry",
                        "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a - b\n"}],
                    }
                }

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = ReviewAdviceCgmClient()
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}
            env.repair_history.record_outcome("test_failed", [node_id], "generated_patch_behavior")

            env.step(PlannerAction("repair_review", self._repair_params(node_id, "review advice")))
            result = env.step(PlannerAction("repair", self._repair_params(node_id, "adopt advice")))

            self.assertEqual(result["status"], "test_failed")
            self.assertIsNotNone(cgm.patch_payload)
            self.assertNotIn("selected_fix_contract", cgm.patch_payload)
            self.assertFalse(result["cgm_payload"]["selected_fix_contract_present"])

    def test_api_signature_failure_blocks_repair_until_api_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {"patch": {"summary": "retry", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}
            env.last_repair_attempt = {
                "status": "test_failed",
                "error_origin": "generated_patch_behavior",
                "failure_feedback": {
                    "failed_patch": {},
                    "failed_tests": [],
                    "error_summary": "exception_types: TypeError\nactual_messages: get_str_vals() takes 1 positional argument but 2 were given",
                },
            }

            observation = json.loads(env.observe())
            blocked = env.step(PlannerAction("repair", self._repair_params(node_id, "retry without reading get_str_vals")))

            hint = observation["runtime_facts"]["last_patch_api_signature_failure"]
            self.assertEqual(hint["api_symbol"], "get_str_vals")
            self.assertTrue(
                any("get_str_vals" in item for item in observation["current_turn_protocol"]["blockers"])
            )
            self.assertTrue(blocked["blocked"])
            self.assertIn("get_str_vals", blocked["reason"])

    def test_repair_review_sanitizes_test_source_suggestions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {
                    "review": {
                        "verdict": "needs_more_evidence",
                        "confidence": 0.4,
                        "evidence_gaps": ["read test_required_columns", "read caller implementation"],
                        "suggested_next_action": "read test_sampled.py before repair",
                    }
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair_review", self._repair_params(node_id, "review target")))

            self.assertEqual(result["status"], "reviewed")
            self.assertIn("read caller implementation", result["review"]["evidence_gaps"])
            self.assertNotIn("read test_required_columns", result["review"]["evidence_gaps"])
            self.assertIn("Do not read benchmark test source", result["review"]["evidence_gaps"][-1])
            self.assertIn("runtime output summaries", result["review"]["suggested_next_action"])
            self.assertEqual(result["review"]["removed_benchmark_test_source_requests"][0], "read test_required_columns")

    def test_repair_allows_format_retry_after_generated_syntax_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "bad syntax", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "-    return a + b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": ["function::pkg::calc.py::add::1"]}))
            env.failure_summary = {"status": "failed"}
            first = env.step(PlannerAction("repair", self._repair_params("function::pkg::calc.py::add::1", "fix add")))

            second = env.step(PlannerAction("repair", self._repair_params("function::pkg::calc.py::add::1", "try again")))

            self.assertEqual(first["status"], "syntax_failed")
            self.assertFalse(second.get("blocked", False))
            self.assertEqual(second["status"], "patch_rejected")
            self.assertEqual(second["error_origin"], "duplicate_patch")

    def test_planner_loop_console_status_does_not_override_final_status(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig(max_steps=1))
            planner = StaticPlannerClient(['{"tool":"memory_commit_note","params":{"note":"still exploring"}}'])

            result = PlannerLoop(env, planner, AgentConfig(max_steps=1), console=True).run()

            self.assertEqual(result.status, "not_pass")

    def test_planner_malformed_response_records_raw_diagnostic(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig(max_steps=1))
            planner = MalformedThenToolPlannerClient()

            result = PlannerLoop(
                env,
                planner,
                AgentConfig(max_steps=1, planner_tool_calling=True, planner_max_parse_retries=1),
            ).run()

            self.assertEqual(result.status, "not_pass")
            self.assertEqual(env.planner_diagnostics[0]["step"], 1)
            self.assertIn("Extra data", env.planner_diagnostics[0]["error"])
            self.assertIn("content", env.planner_diagnostics[0]["raw_response"])
            retry_message = planner.seen_messages[1][-1]["content"]
            self.assertIn("tool_calls", retry_message)
            self.assertNotIn("JSON action", retry_message)

    def test_progress_tracker_writes_realtime_task_steps(self):
        with tempfile.TemporaryDirectory() as raw:
            progress_path = Path(raw) / "run" / "progress.md"
            progress = ProgressTracker(progress_path)

            progress.start_task("task-1", {"backend": "remote_swe"})
            self.assertTrue(progress_path.exists())
            self.assertIn("phase: starting", progress_path.read_text(encoding="utf-8"))

            progress.record_step("task-1", 1, "run_failed_test", "failed", 1.2, "test=failed")
            text = progress_path.read_text(encoding="utf-8")
            self.assertIn("| 1 | run_failed_test | failed | 1.2s | test=failed |", text)

            progress.record("not_pass", task_id="task-1", reason="max_steps")
            text = progress_path.read_text(encoding="utf-8")
            self.assertIn("final_status: not_pass", text)
            self.assertIn("reason: max_steps", text)

    def test_progress_tracker_can_seed_resume_baseline_counts(self):
        with tempfile.TemporaryDirectory() as raw:
            progress_path = Path(raw) / "run" / "progress.md"
            progress = ProgressTracker(progress_path)

            progress.seed_counts({"pass": 7, "not_pass": 2, "bug": 0}, label="previous results")
            progress.record("pass", task_id="task-3", reason="env_done")

            summary = progress.summary()
            self.assertEqual(summary["total"], 10)
            self.assertEqual(summary["pass"], 8)
            self.assertEqual(summary["not_pass"], 2)
            self.assertEqual(summary["current_run_total"], 1)
            self.assertEqual(summary["baseline_total"], 9)
            text = progress_path.read_text(encoding="utf-8")
            self.assertIn("- baseline_total: 9", text)
            self.assertIn("baseline_label: previous results", text)

    def test_eval_parallel_detects_remote_runner_pool_start_bug(self):
        record = {
            "status": "bug",
            "reason": (
                "RemoteSweError: remote swe_proxy failed rc=1 op='start' runtime=302.1s "
                "stdout={\"ok\": false, \"error\": \"RuntimeError(\\\"Timed out waiting for an idle runner. "
                "rid=0: stale age=100.0s current_run_id='gp-x'\\\")\"}"
            ),
        }

        self.assertTrue(eval_parallel_cli._is_remote_runner_pool_bug(record))
        self.assertFalse(eval_parallel_cli._is_planner_network_bug(record))

    def test_eval_parallel_marks_remote_sandbox_infra_contamination(self):
        with tempfile.TemporaryDirectory() as raw:
            trace_path = Path(raw) / "trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "kind": "planner_step",
                        "payload": {
                            "result": {
                                "reason": "remote read_file failed: no active instance on this runner",
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            record = {"task_id": "x", "status": "not_pass", "reason": "max_steps"}

            eval_parallel_cli._annotate_infra_contamination(record, trace_path)

            self.assertTrue(record["infra_contaminated"])
            self.assertIn("no_active_instance", record["infra_contamination_reasons"])
            self.assertTrue(eval_parallel_cli._is_remote_sandbox_invalid(record))

    def test_eval_parallel_marks_remote_sandbox_invalid_as_bug_when_continuing(self):
        record = {
            "task_id": "x",
            "status": "not_pass",
            "reason": "max_steps",
            "infra_contaminated": True,
            "infra_contamination_reasons": ["no_active_instance"],
        }

        eval_parallel_cli._mark_remote_sandbox_invalid_bug(record)

        self.assertEqual(record["status"], "bug")
        self.assertEqual(record["original_status"], "not_pass")
        self.assertIn("skipping this issue", record["reason"])

    def test_eval_parallel_baseline_counts_clean_latest_results(self):
        with tempfile.TemporaryDirectory() as raw:
            first = Path(raw) / "first.jsonl"
            second = Path(raw) / "second.jsonl"
            first.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {"task_id": "a", "status": "pass"},
                        {"task_id": "b", "status": "not_pass"},
                        {"task_id": "c", "status": "bug"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {"task_id": "b", "status": "pass"},
                        {"task_id": "d", "status": "not_pass", "infra_contaminated": True},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            counts, info = eval_parallel_cli._baseline_counts_from_results([first, second])

            self.assertEqual(counts, {"pass": 2, "not_pass": 0, "bug": 0})
            self.assertEqual(info["unique_task_count"], 4)
            self.assertEqual(info["skipped_bug"], 1)
            self.assertEqual(info["skipped_infra_contaminated"], 1)

    def test_eval_supervisor_remaining_tasks_excludes_clean_results_only(self):
        with tempfile.TemporaryDirectory() as raw:
            tasks = Path(raw) / "tasks.jsonl"
            results = Path(raw) / "results.jsonl"
            tasks.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {"task_id": "a", "issue_body": "a"},
                        {"task_id": "b", "issue_body": "b"},
                        {"task_id": "c", "issue_body": "c"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            results.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        {"task_id": "a", "status": "pass"},
                        {"task_id": "b", "status": "not_pass", "infra_contaminated": True},
                        {"task_id": "c", "status": "bug"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            clean = eval_supervisor_cli._clean_result_records([results])
            remaining = eval_supervisor_cli._remaining_task_records(tasks, clean)

            self.assertEqual(sorted(clean), ["a"])
            self.assertEqual([record["task_id"] for record in remaining], ["b", "c"])

    def test_good_patch_auto_finishes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "fix add", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))

            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))
            node_id = env.latest_result["results"][0]["id"]
            env.step(PlannerAction("read", {"node_id": node_id, "view": "body"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.step(PlannerAction("run_failed_test", {}))
            repair = env.step(PlannerAction("repair", self._repair_params(node_id)))

            self.assertEqual(repair["status"], "passed")
            self.assertTrue(repair["done"])
            self.assertTrue(env.done)
            self.assertEqual(env.status, "pass")

    def test_repair_internally_retries_rejected_patch_format(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = SequentialCgmClient(
                [
                    {
                        "patch": {
                            "summary": "polluted",
                            "edits": [
                                {
                                    "path": "pkg/calc.py",
                                    "start": 2,
                                    "end": 2,
                                    "new_text": "    return a + b('',) {'patch': {'edits': []}}\n",
                                }
                            ],
                        }
                    },
                    {
                        "patch": {
                            "summary": "clean",
                            "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}],
                        }
                    },
                ]
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", self._repair_params(node_id)))

            self.assertEqual(result["status"], "passed")
            self.assertEqual(cgm.calls, 2)
            self.assertIn("Previous generated patch was rejected", cgm.payloads[1]["plan_text"])
            self.assertIn("internal_retry_from", result["patch_preview"])

    def test_repair_internally_retries_generated_syntax_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = SequentialCgmClient(
                [
                    {
                        "patch": {
                            "summary": "bad syntax",
                            "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "-    return a + b\n"}],
                        }
                    },
                    {
                        "patch": {
                            "summary": "clean",
                            "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}],
                        }
                    },
                ]
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", self._repair_params(node_id)))

            self.assertEqual(result["status"], "passed")
            self.assertEqual(cgm.calls, 2)
            self.assertIn("Python syntax checking", cgm.payloads[1]["plan_text"])
            self.assertIn("internal_retry_from", result["patch_preview"])

    def test_repair_accepts_complete_cgm_diff_fallback_protocol(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {
                    "patch": {
                        "summary": "codefuse-cgm-diff",
                        "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}],
                    },
                    "summary": "codefuse-cgm-diff",
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", self._repair_params(node_id)))

            self.assertEqual(result["status"], "passed")
            self.assertIn("codefuse-cgm-diff", result["patch_preview"]["summary"])

    def test_repair_rejects_cgm_partial_fallback_protocol(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            node_id = "function::pkg::calc.py::add::1"
            cgm = StaticCgmClient(
                {
                    "patch": {
                        "summary": "codefuse-cgm-partial",
                        "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}],
                    },
                    "summary": "codefuse-cgm-partial",
                }
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_patch_edits=4))
            env.step(PlannerAction("read", {"node_id": "add"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))
            env.failure_summary = {"status": "failed"}

            result = env.step(PlannerAction("repair", self._repair_params(node_id)))

            self.assertEqual(result["status"], "patch_rejected")
            self.assertEqual(result["error_origin"], "cgm_output_protocol")
            self.assertIn("complete JSON patch object", result["reason"])
            self.assertIn("codefuse-cgm-partial", result["cgm_output"]["summary"])

    def test_unified_diff_normalizes_to_edit(self):
        patch = parse_cgm_output(
            """diff --git a/pkg/calc.py b/pkg/calc.py
--- a/pkg/calc.py
+++ b/pkg/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""
        )

        self.assertEqual(patch.edits[0].path, "pkg/calc.py")
        self.assertEqual(patch.edits[0].start, 1)
        self.assertEqual(patch.edits[0].end, 2)
        self.assertIn("return a + b", patch.edits[0].new_text)

    def test_new_cgm_service_parser_prefers_complete_diff(self):
        parsed = parse_model_output(
            """Some text that should be ignored.
diff --git a/pkg/calc.py b/pkg/calc.py
--- a/pkg/calc.py
+++ b/pkg/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.parser, "unified_diff")
        self.assertEqual(parsed.patch["summary"], "codefuse-cgm-diff")
        self.assertEqual(parsed.patch["edits"][0]["path"], "pkg/calc.py")
        self.assertEqual(parsed.patch["edits"][0]["new_text"], "def add(a, b):\n    return a + b\n")

    def test_new_cgm_service_parser_rejects_partial_by_default(self):
        raw = (
            '{"patch":{"edits":[{"path":"pkg/calc.py","start":2,"end":2,'
            '"new_text":"    return a + b\\n'
        )

        self.assertIsNone(parse_model_output(raw))
        parsed = parse_model_output(raw, allow_partial=True)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.parser, "partial_fallback")

    def test_new_cgm_service_json_parser_preserves_indentation(self):
        parsed = parse_model_output(
            '{"patch":{"edits":[{"path":"pkg/calc.py","start":2,"end":2,"new_text":"    return a + b\\n"}]}}'
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.parser, "json_patch")
        self.assertEqual(parsed.patch["edits"][0]["new_text"], "    return a + b\n")

    def test_json_patch_parser_accepts_double_escaped_multiline_new_text(self):
        patch = parse_cgm_output(
            {
                "patch": {
                    "edits": [
                        {
                            "path": "pkg/rst.py",
                            "start": 60,
                            "end": 61,
                            "new_text": "    def __init__(self, **kwargs):\\n        super().__init__(**kwargs)",
                        }
                    ]
                }
            }
        )

        self.assertEqual(
            patch.edits[0].new_text,
            "    def __init__(self, **kwargs):\n        super().__init__(**kwargs)",
        )

    def test_new_cgm_service_parser_rejects_repeated_empty_tuple_artifact(self):
        parsed = parse_model_output(
            """diff --git a/pkg/calc.py b/pkg/calc.py
--- a/pkg/calc.py
+++ b/pkg/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b('',)('',)('',)
"""
        )

        self.assertIsNone(parsed)

    def test_new_cgm_service_parser_rejects_single_tuple_signature_artifact(self):
        parsed = parse_model_output(
            """diff --git a/pkg/calc.py b/pkg/calc.py
--- a/pkg/calc.py
+++ b/pkg/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b('',) ) Editted by Clouseau
"""
        )

        self.assertIsNone(parsed)

    def test_patch_validation_rejects_cgm_schema_artifacts_in_new_text(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            patch = parse_cgm_output(
                {
                    "patch": {
                        "edits": [
                            {
                                "path": "pkg/calc.py",
                                "start": 2,
                                "end": 2,
                                "new_text": "    return a + b('',) {'patch': {'edits': []}}\n",
                            }
                        ]
                    }
                }
            )

            decision = validate_patch(root, patch)

            self.assertFalse(decision.ok)
            self.assertIn("schema", decision.reason)

    def test_patch_validation_reports_artifact_before_edit_count(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            patch = parse_cgm_output(
                {
                    "patch": {
                        "edits": [
                            {
                                "path": "pkg/calc.py",
                                "start": 2,
                                "end": 2,
                                "new_text": "    return a + b('',) {'patch': {'edits': []}}\n",
                            },
                            *[
                                {
                                    "path": "pkg/calc.py",
                                    "start": 2,
                                    "end": 2,
                                    "new_text": "    return a + b\n",
                                }
                                for _ in range(5)
                            ],
                        ]
                    }
                }
            )

            decision = validate_patch(root, patch, max_edits=4)

            self.assertFalse(decision.ok)
            self.assertIn("schema", decision.reason)

    def test_patch_validation_rejects_standalone_diff_plus_in_new_text(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            patch = parse_cgm_output(
                {
                    "patch": {
                        "edits": [
                            {
                                "path": "pkg/calc.py",
                                "start": 2,
                                "end": 2,
                                "new_text": "    return a + b\n+\n",
                            }
                        ]
                    }
                }
            )

            decision = validate_patch(root, patch)

            self.assertFalse(decision.ok)
            self.assertIn("plus marker", decision.reason)

    def test_patch_normalization_shrinks_single_line_range_and_aligns_indent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "flow.py").write_text(
                "def combine(right):\n"
                "    else_block = True\n"
                "    cright = np.zeros((noutp, right.shape[1]))\n"
                "    cright[-right.shape[0]:, -right.shape[1]:] = 1\n"
                "    return cright\n",
                encoding="utf-8",
            )
            patch = parse_cgm_output(
                {
                    "patch": {
                        "edits": [
                            {
                                "path": "pkg/flow.py",
                                "start": 3,
                                "end": 5,
                                "new_text": "cright[-right.shape[0]:, -right.shape[1]:] = right",
                            }
                        ]
                    }
                }
            )

            normalized, notes = normalize_patch(root, patch)

            self.assertEqual(normalized.edits[0].start, 4)
            self.assertEqual(normalized.edits[0].end, 4)
            self.assertEqual(normalized.edits[0].new_text, "    cright[-right.shape[0]:, -right.shape[1]:] = right")
            self.assertTrue(any("single-line edit range" in note for note in notes))
            self.assertTrue(any("aligned edit indentation" in note for note in notes))

    def test_patch_validation_rejects_removed_control_flow_header(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "branch.py").write_text(
                "def choose(right):\n"
                "    if right:\n"
                "        return 1\n"
                "    else:\n"
                "        return 0\n",
                encoding="utf-8",
            )
            patch = parse_cgm_output(
                {
                    "patch": {
                        "edits": [
                            {
                                "path": "pkg/branch.py",
                                "start": 4,
                                "end": 4,
                                "new_text": "        return right\n",
                            }
                        ]
                    }
                }
            )

            decision = validate_patch(root, patch)

            self.assertFalse(decision.ok)
            self.assertIn("control-flow header", decision.reason)

    def test_remote_repo_graph_payload_decodes_to_repo_graph(self):
        jsonl = "\n".join(
            [
                json.dumps({"type": "node", "id": "file:pkg/calc.py", "kind": "file", "name": "pkg/calc.py", "path": "pkg/calc.py", "span": [1, 2]}),
                json.dumps(
                    {
                        "type": "node",
                        "id": "func:pkg/calc.py:add",
                        "kind": "function",
                        "name": "add",
                        "path": "pkg/calc.py",
                        "span": [1, 2],
                        "snippet_lines": ["def add(a, b):", "    return a + b"],
                    }
                ),
                json.dumps({"type": "edge", "src": "file:pkg/calc.py", "dst": "func:pkg/calc.py:add", "kind": "CONTAINS"}),
            ]
        )
        payload = base64.b64encode(gzip.compress(jsonl.encode("utf-8"))).decode("ascii")

        graph = decode_repo_graph_payload(payload)

        self.assertIn("func:pkg/calc.py:add", graph.nodes)
        self.assertTrue(graph.nodes["func:pkg/calc.py:add"].has_code)
        self.assertEqual(graph.edges_between({"file:pkg/calc.py", "func:pkg/calc.py:add"})[0].type, "CONTAINS")

    def test_remote_repo_graph_payload_accepts_span_dict_and_dedupes_edges(self):
        jsonl = "\n".join(
            [
                json.dumps({"id": "file:pkg/calc.py", "kind": "file", "name": "pkg/calc.py", "path": "pkg/calc.py", "span": {"start": 3, "end": 9}}),
                json.dumps({"type": "edge", "src": "repo", "dst": "file:pkg/calc.py", "kind": "CONTAINS"}),
                json.dumps({"type": "edge", "src": "repo", "dst": "file:pkg/calc.py", "kind": "CONTAINS"}),
            ]
        )
        payload = base64.b64encode(gzip.compress(jsonl.encode("utf-8"))).decode("ascii")

        graph = decode_repo_graph_payload(payload)

        self.assertEqual(graph.nodes["file:pkg/calc.py"].start_line, 3)
        self.assertEqual(graph.nodes["file:pkg/calc.py"].end_line, 9)
        self.assertEqual(len(graph.edges_between({"repo", "file:pkg/calc.py"})), 1)

    def test_remote_swe_runtime_converts_session_results(self):
        session = FakeRemoteSession()
        task = TaskSpec(
            task_id="remote",
            repo_path=Path("."),
            issue_title="remote issue",
            issue_body="body",
            docker_image="example/image:latest",
            test_command="python -m unittest",
        )
        runtime = RemoteSweRuntime(AgentConfig(sandbox_backend="remote_swe", command_timeout=30), session=session)

        runtime.start(task)
        command = runtime.run("echo hello", timeout=10)
        test = runtime.run_fail_to_pass(task)
        runtime.stop()

        self.assertEqual(command.returncode, 0)
        self.assertIn("echo hello", session.exec_commands)
        self.assertEqual(test.status, "passed")
        self.assertTrue(session.started)
        self.assertTrue(session.cleaned)
        self.assertTrue(session.stopped)

    def test_remote_swe_wraps_custom_tests_with_testbed_python_guard(self):
        cmd = wrap_testbed_test_command("python - <<'PY'\nprint('ok')\nPY")

        self.assertIn("cd /testbed", cmd)
        self.assertIn("PYTHONNOUSERSITE", cmd)
        self.assertIn("INFRA_WRONG_PYTHON_ENV", cmd)
        self.assertIn("python - <<'PY'\nprint('ok')\nPY", cmd)

    def test_remote_swe_runtime_marks_wrong_python_as_infra_bug(self):
        class WrongPythonSession(FakeRemoteSession):
            def exec(self, cmd, *, cwd=None, env=None, timeout=None):
                self.exec_commands.append(cmd)
                return {
                    "ok": False,
                    "returncode": 97,
                    "stdout": "",
                    "stderr": "INFRA_WRONG_PYTHON_ENV: python resolves to /home/user/miniconda3/bin/python\n",
                }

        session = WrongPythonSession()
        task = TaskSpec(
            task_id="remote",
            repo_path=Path("."),
            issue_title="remote issue",
            issue_body="body",
            docker_image="example/image:latest",
            test_command="python -c 'print(1)'",
        )
        runtime = RemoteSweRuntime(AgentConfig(sandbox_backend="remote_swe", command_timeout=30), session=session)

        runtime.start(task)
        test = runtime.run_fail_to_pass(task)

        self.assertEqual(test.status, "infra_bug")
        self.assertEqual(test.parser_error, "wrong_python_env")
        self.assertTrue(is_wrong_python_env(CommandResult(test.command, test.returncode, test.stdout, test.stderr)))

    def test_read_surfaces_issue_bound_dispatch_consumer_context(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg" / "table").mkdir(parents=True)
            (root / "pkg" / "io" / "ascii").mkdir(parents=True)
            (root / "pkg" / "io" / "votable" / "validator").mkdir(parents=True)
            (root / "pkg" / "table" / "connect.py").write_text(
                "class TableWrite:\n"
                "    def __init__(self, registry, instance):\n"
                "        self.registry = registry\n"
                "        self._instance = instance\n"
                "\n"
                "    def __call__(self, *args, **kwargs):\n"
                "        instance = self._instance\n"
                "        self.registry.write(instance, *args, **kwargs)\n",
                encoding="utf-8",
            )
            (root / "pkg" / "io" / "ascii" / "html.py").write_text(
                "class HTML:\n"
                "    def write(self, table):\n"
                "        rows = []\n"
                "        return rows\n",
                encoding="utf-8",
            )
            (root / "pkg" / "io" / "votable" / "validator" / "html.py").write_text(
                "def write_table(table):\n"
                "    return 'validator only'\n",
                encoding="utf-8",
            )
            (root / "pkg" / "table" / "row.py").write_text(
                "def _repr_html_(row):\n"
                "    return '<tr></tr>'\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)
            node_id = next(node.id for node in graph.nodes.values() if node.name == "__call__" and node.path.endswith("connect.py"))
            task = TaskSpec(
                "fake-html",
                root,
                "HTML writer ignores formats",
                "Running t.write(out, format='html', formats={'a': lambda x: x}) ignores formats.",
            )
            env = CodeRepairEnv(
                task=task,
                runtime=LocalRepoRuntime(root),
                cgm=StaticCgmClient({"patch": {"edits": []}}),
                config=AgentConfig(),
                graph=graph,
            )

            result = handle_action(env, PlannerAction("read", {"node_id": node_id, "view": "body"}))

            context = result["dispatch_relationship_context"]
            self.assertTrue(context)
            fact = context[0]
            self.assertEqual(fact["dispatcher_status"], "dispatcher_wrapper_registry_call")
            self.assertEqual(fact["dispatch_key_candidates"][0]["key"], "format")
            self.assertEqual(fact["dispatch_key_candidates"][0]["values"], ["html"])
            candidate_ids = [item["id"] for item in fact["consumer_candidates"]]
            html_write_id = next(node.id for node in graph.nodes.values() if node.name == "write" and node.path.endswith("html.py"))
            self.assertIn(html_write_id, candidate_ids)
            html_candidate = next(item for item in fact["consumer_candidates"] if item["id"] == html_write_id)
            self.assertIn("code", html_candidate)
            self.assertIn("def write", html_candidate["code"])
            self.assertIn(html_write_id, env.working.entries)
            self.assertEqual(env.working.entries[html_write_id].source, "relation_context:consumer_candidate_preview")
            self.assertNotIn(html_write_id, [item["id"] for item in result["unread_local_symbol_references"]])

    def test_remote_swe_runtime_can_disable_cleanup_before_start(self):
        session = FakeRemoteSession()
        task = TaskSpec("remote", Path("."), "title", "body", docker_image="example/image:latest")
        runtime = RemoteSweRuntime(AgentConfig(sandbox_backend="remote_swe", sandbox_cleanup_pool_before_start=False), session=session)

        runtime.start(task)

        self.assertTrue(session.started)
        self.assertFalse(session.cleaned)

    def test_remote_swe_runtime_build_graph_uses_base64_gzip_jsonl(self):
        session = FakeRemoteSession()
        task = TaskSpec("remote", Path("."), "title", "body", docker_image="example/image:latest")
        runtime = RemoteSweRuntime(AgentConfig(sandbox_backend="remote_swe"), session=session)
        runtime.start(task)

        graph = runtime.build_graph()

        self.assertIn("func:pkg/calc.py:add", graph.nodes)

    def test_remote_swe_runtime_caches_repo_graph_payload(self):
        with tempfile.TemporaryDirectory() as raw:
            session = FakeRemoteSession()
            task = TaskSpec("remote", Path("."), "title", "body", docker_image="example/image:latest", base_commit="abc")
            config = AgentConfig(sandbox_backend="remote_swe", sandbox_graph_cache_dir=str(Path(raw) / "graph-cache"))
            runtime = RemoteSweRuntime(config, session=session)
            runtime.start(task)

            first = runtime.build_graph()
            second = runtime.build_graph()

            self.assertIn("func:pkg/calc.py:add", first.nodes)
            self.assertIn("func:pkg/calc.py:add", second.nodes)
            self.assertEqual(session.build_graph_calls, 1)
            self.assertTrue(runtime.last_graph_cache_hit)

    def test_remote_swe_snapshot_and_rollback_use_encoded_payloads(self):
        session = FakeRemoteSession()
        runtime = RemoteSweRuntime(AgentConfig(sandbox_backend="remote_swe"), session=session)
        runtime.start(TaskSpec("remote", Path("."), "title", "body", docker_image="example/image:latest"))

        snapshot = runtime.snapshot(["/testbed/pkg/calc.py"])
        runtime.rollback(snapshot)

        self.assertEqual(snapshot, {"/testbed/pkg/calc.py": "plain-body"})
        joined_commands = "\n".join(session.exec_commands)
        self.assertIn("base64.b64decode", joined_commands)
        self.assertNotIn("plain-body", joined_commands)

    def test_remote_swe_proxy_non_json_is_infra_error(self):
        with self.assertRaises(RemoteSweError):
            _parse_proxy_response("not json")

    def test_remote_swe_default_ssh_args_when_key_exists(self):
        with patch("graphplanner_agent.config.schema.Path.exists", return_value=True):
            with patch.dict("os.environ", {"GP_SANDBOX_BACKEND": "remote_swe"}, clear=True):
                config = AgentConfig.from_env()

        self.assertIn("-p 40022", config.sandbox_ssh_args)
        self.assertIn("id_ed25519_login24", config.sandbox_ssh_args)

    def test_remote_swe_default_ssh_args_after_cli_style_backend_override(self):
        with patch("graphplanner_agent.config.schema.Path.exists", return_value=True):
            with patch.dict("os.environ", {}, clear=True):
                config = AgentConfig.from_env()
                config.sandbox_backend = "remote_swe"
                config.finalize()

        self.assertIn("-p 40022", config.sandbox_ssh_args)

    def test_remote_swe_accepts_sif_references(self):
        ref = "/remote/sif/slimshetty-swebench-verified-demo.sif"

        self.assertEqual(normalize_sif_image_ref(ref), "slimshetty-swebench-verified-demo")
        self.assertEqual(infer_sif_dir_from_ref(ref), "/remote/sif")

    def test_remote_stderr_filters_known_hosts_warning(self):
        cleaned = clean_remote_stderr(
            "Warning: Permanently added '[localhost]:40022' (ED25519) to the list of known hosts.\nreal error\n"
        )

        self.assertEqual(cleaned, "real error")

    def test_remote_preflight_auto_and_cleanup_mode(self):
        class FakePreflightSession:
            last = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.calls: list[str] = []
                FakePreflightSession.last = self

            def check_remote_layout(self, timeout=30.0):
                self.calls.append("layout")
                return {"ok": True, "stdout": "remote_repo=/repo\nswe_proxy_ok\nrunner_manager_ok\nPython 3.10\n", "stderr": ""}

            def cleanup_pool(self, timeout=None, cwd="/testbed"):
                self.calls.append("cleanup_pool")
                return {"ok": True, "returncode": 0}

            def ensure_remote_runners(self, timeout=180.0):
                self.calls.append("ensure_runners")

            def start(self, timeout=None, cwd="/testbed"):
                self.calls.append("start")
                return {"ok": True, "returncode": 0}

            def stop(self, timeout=None):
                self.calls.append("stop")
                return {"ok": True, "returncode": 0}

        task = TaskSpec("remote", Path("."), "title", "body", docker_image="example/image:latest")
        config = AgentConfig(sandbox_backend="remote_swe", sandbox_remote_repo="/repo", sandbox_num_runners=2)

        self.assertEqual(normalize_remote_preflight_mode("auto", backend="remote_swe"), "cleanup")
        with patch("graphplanner_agent.cli.remote_preflight.RemoteSweSession", FakePreflightSession):
            result = run_remote_swe_preflight(config, task, mode="auto")

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "cleanup")
        self.assertEqual(FakePreflightSession.last.calls, ["layout", "cleanup_pool", "ensure_runners"])
        self.assertEqual(FakePreflightSession.last.kwargs["num_runners"], 2)

    def test_remote_preflight_full_smoke_start_stop(self):
        class FakePreflightSession:
            last = None

            def __init__(self, **kwargs):
                self.calls: list[str] = []
                FakePreflightSession.last = self

            def check_remote_layout(self, timeout=30.0):
                self.calls.append("layout")
                return {"ok": True, "stdout": "ok", "stderr": ""}

            def cleanup_pool(self, timeout=None, cwd="/testbed"):
                self.calls.append("cleanup_pool")
                return {"ok": True}

            def ensure_remote_runners(self, timeout=180.0):
                self.calls.append("ensure_runners")

            def start(self, timeout=None, cwd="/testbed"):
                self.calls.append("start")
                return {"ok": True, "returncode": 0}

            def stop(self, timeout=None):
                self.calls.append("stop")
                return {"ok": True, "returncode": 0}

        task = TaskSpec("remote", Path("."), "title", "body", docker_image="example/image:latest")
        config = AgentConfig(sandbox_backend="remote_swe")

        with patch("graphplanner_agent.cli.remote_preflight.RemoteSweSession", FakePreflightSession):
            result = run_remote_swe_preflight(config, task, mode="full")

        self.assertTrue(result["ok"])
        self.assertEqual(FakePreflightSession.last.calls, ["layout", "cleanup_pool", "ensure_runners", "start", "stop"])

    def test_runtime_factory_selects_remote_swe(self):
        task = TaskSpec("remote", Path("."), "title", "body", docker_image="example/image:latest")
        config = AgentConfig(sandbox_backend="remote_swe")

        runtime = make_runtime(task, config)

        self.assertIsInstance(runtime, RemoteSweRuntime)

    def test_run_label_directory_uses_readable_utc_timestamp(self):
        task = TaskSpec("owner__repo-1", Path("."), "title", "body")

        with patch("graphplanner_agent.cli.eval.datetime") as fake_datetime:
            fake_datetime.now.return_value.strftime.return_value = "2026-05-14_12-23-03_UTC"
            run_dir = eval_cli._make_run_dir([task], Path("runs/tmp"), "evidence package")

        self.assertEqual(str(run_dir), "runs/tmp/owner__repo-1__evidence_package__2026-05-14_12-23-03_UTC")

    def test_filesystem_fallback_maps_text_to_node(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "pkg").mkdir()
            (root / "pkg" / "calc.py").write_text(
                "def add(a, b):\n    sentinel_value = a - b\n    return sentinel_value\n",
                encoding="utf-8",
            )
            graph = build_python_graph(root)

            results, warning = search_graph(graph, "+missing sentinel_value", "function", root=root)

            self.assertTrue(results)
            self.assertIn("filesystem fallback", warning)
            self.assertEqual(results[0].node.name, "add")

    def test_task_loader_accepts_jsonl_and_string_selectors(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tasks.jsonl"
            path.write_text(
                '{"task_id":"one","repo_path":"/tmp/repo","issue_title":"t","issue_body":"b","fail_to_pass":"tests/test_one.py::test_a"}\n',
                encoding="utf-8",
            )

            tasks = load_tasks(path)

            self.assertEqual(tasks[0].fail_to_pass, ["tests/test_one.py::test_a"])

    def test_task_loader_accepts_swebench_style_metadata(self):
        task = TaskSpec.from_dict(
            {
                "instance_id": "django__django-13837",
                "repo": "django/django",
                "problem_statement": "problem",
                "base_commit": "abc123",
                "FAIL_TO_PASS": '["tests/admin_views/test.py::Case::test_a"]',
                "PASS_TO_PASS": '["tests/admin_views/test.py::Case::test_b"]',
                "image_name": "slimshetty/swebench-verified:sweb.eval.x86_64.django__django-13837",
                "eval_script_list": ["python -m pytest tests/admin_views/test.py::Case::test_a"],
            }
        )

        self.assertEqual(task.task_id, "django__django-13837")
        self.assertEqual(task.repo_path, Path("/testbed"))
        self.assertEqual(task.fail_to_pass, ["tests/admin_views/test.py::Case::test_a"])
        self.assertEqual(task.pass_to_pass, ["tests/admin_views/test.py::Case::test_b"])
        self.assertEqual(task.base_commit, "abc123")
        self.assertEqual(task.metadata["repo"], "django/django")
        self.assertEqual(task.metadata["swebench_spec"]["eval_script_list"][0], "python -m pytest tests/admin_views/test.py::Case::test_a")

    def test_swebench_pro_without_explicit_p2p_does_not_infer_regression(self):
        task = TaskSpec(
            "pro",
            Path("/testbed"),
            "title",
            "body",
            fail_to_pass=["TestScanner"],
            pass_to_pass=[],
            metadata={"swebench_pro": {"run_script": "unused", "parser": "unused", "selected_test_files_to_run": ["TestScanner"]}},
        )
        report = {
            "tests": [
                {"name": "TestScanner", "status": "PASSED"},
                {"name": "UnrelatedRegression", "status": "PASSED"},
            ],
            "runs": [{"label": "selected", "run_script_returncode": 0, "parser_returncode": 0, "test_count": 2}],
        }
        stdout = f"{swebench_pro.START_PRO_JSON}\n{json.dumps(report)}\n{swebench_pro.END_PRO_JSON}\n"

        result = swebench_pro.result_from_run(task, CommandResult("cmd", 0, stdout, ""))

        self.assertTrue(result.resolved)
        self.assertFalse(result.tests_status["PASS_TO_PASS"]["required"])
        self.assertEqual(result.tests_status["PASS_TO_PASS"]["source"], "not_provided")

    def test_eval_rejects_incomplete_swebench_instance(self):
        task = TaskSpec.from_dict(
            {
                "instance_id": "astropy__astropy-12907",
                "repo": "astropy/astropy",
                "docker_image": "slimshetty/swebench-verified:sweb.eval.x86_64.astropy__astropy-12907",
                "base_commit": "abc123",
                "patch": "diff --git a/a.py b/a.py\n",
            }
        )

        with self.assertRaisesRegex(ValueError, "missing issue body"):
            eval_cli._validate_task_inputs([task], Path("bad.instance.json"))

    def test_cgm_payload_skips_pathless_context_neighbors(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("func:pkg/calc.py:add", "function", "add", "pkg/calc.py", 1, 2, text="def add(a, b):\n    return a - b\n"))
        graph.add_node(GraphNode("repo::pkg", "repo", "pkg", "", 1, 1))
        graph.add_edge("func:pkg/calc.py:add", "repo::pkg", "RELATED")
        memory = CgmMemory()
        memory.commit([graph.nodes["func:pkg/calc.py:add"]])
        task = TaskSpec("fake", Path("/testbed"), "title", "body", fail_to_pass=["tests/test_calc.py::test_add"])

        payload = build_cgm_payload(task, graph, memory, "fix add", 0.75, None, None, 4)

        self.assertEqual(validate_cgm_payload(payload), [])
        self.assertEqual({node["path"] for node in payload["graph"]["nodes"]}, {"pkg/calc.py"})

    def test_official_eval_script_keeps_swebench_harness_shape(self):
        task = TaskSpec.from_dict(
            {
                "instance_id": "django__django-13837",
                "repo": "django/django",
                "version": "3.2",
                "base_commit": "abc123",
                "FAIL_TO_PASS": '["tests/admin_views/test.py::Case::test_a"]',
                "eval_script_list": [
                    "cd /testbed",
                    ": '>>>>> Start Test Output'",
                    "./tests/runtests.py --verbosity 2 admin_views",
                    ": '>>>>> End Test Output'",
                ],
            }
        )

        script = official_eval_script(task)

        self.assertIn("set -uxo pipefail", script)
        self.assertNotIn("set -e", script)
        self.assertIn("./tests/runtests.py --verbosity 2 admin_views", script)

    def test_swebench_parser_falls_back_without_version_for_pytest_summary(self):
        task = TaskSpec.from_dict(
            {
                "instance_id": "astropy__astropy-12907",
                "repo": "astropy/astropy",
                "problem_statement": "problem",
                "base_commit": "abc123",
                "FAIL_TO_PASS": '["astropy/modeling/tests/test_separable.py::test_separable"]',
                "eval_script_list": [
                    "cd /testbed",
                    ": '>>>>> Start Test Output'",
                    "python -m pytest -rA -vv -o console_output_style=classic --tb=no astropy/modeling/tests/test_separable.py",
                    ": '>>>>> End Test Output'",
                ],
            }
        )
        output = """
>>>>> Start Test Output
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model8-result8]
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]
============================== 2 passed in 0.30s ==============================
>>>>> End Test Output
"""

        report, error = parse_official_report(task, output)

        self.assertIsNone(error)
        self.assertIsNotNone(report)
        self.assertTrue(report["resolved"])
        self.assertEqual(
            report["tests_status"]["FAIL_TO_PASS"]["success"],
            ["astropy/modeling/tests/test_separable.py::test_separable"],
        )

    def test_swebench_parser_fallback_detects_failed_parameterized_selector(self):
        task = TaskSpec.from_dict(
            {
                "instance_id": "astropy__astropy-12907",
                "repo": "astropy/astropy",
                "problem_statement": "problem",
                "FAIL_TO_PASS": '["astropy/modeling/tests/test_separable.py::test_separable"]',
                "eval_script_list": [
                    ": '>>>>> Start Test Output'",
                    "python -m pytest -rA astropy/modeling/tests/test_separable.py",
                    ": '>>>>> End Test Output'",
                ],
            }
        )
        output = """
>>>>> Start Test Output
PASSED astropy/modeling/tests/test_separable.py::test_separable[compound_model8-result8]
FAILED astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]
>>>>> End Test Output
"""

        report, error = parse_official_report(task, output)

        self.assertIsNone(error)
        self.assertFalse(report["resolved"])
        self.assertEqual(
            report["tests_status"]["FAIL_TO_PASS"]["failure"],
            ["astropy/modeling/tests/test_separable.py::test_separable"],
        )


    def test_cgm_payload_requires_hydrated_code(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), StaticCgmClient({"patch": {"edits": []}}), AgentConfig())
            env.step(PlannerAction("explore_find", {"query": "add", "find_type": "function"}))
            node_id = env.latest_result["results"][0]["id"]
            env.step(PlannerAction("read", {"node_id": node_id, "view": "body"}))
            env.step(PlannerAction("memory_commit", {"select_ids": [node_id]}))

            payload = build_cgm_payload(task, env.graph, env.memory, "plan", 0.75, None, None, 4)

            self.assertEqual(validate_cgm_payload(payload), [])
            self.assertTrue(payload["graph"]["nodes"][0]["text"])

    def test_cgm_payload_uses_model_visible_fields(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(GraphNode("func:pkg/calc.py:add", "function", "add", "pkg/calc.py", 1, 2, text="def add(a, b):\n    return a - b\n"))
        graph.add_node(GraphNode("func:pkg/util.py:helper", "function", "helper", "pkg/util.py", 1, 1))
        graph.add_edge("func:pkg/calc.py:add", "func:pkg/util.py:helper", "CALLS")
        memory = CgmMemory()
        memory.commit([graph.nodes["func:pkg/calc.py:add"]], note="right implementation evidence")
        task = TaskSpec(
            "fake",
            Path("/testbed"),
            "add bug",
            "add subtracts",
            base_commit="abc",
            fail_to_pass=["test_calc.py::test_add"],
        )

        payload = build_cgm_payload(
            task,
            graph,
            memory,
            "change subtraction to addition",
            0.75,
            {"status": "failed", "command": "python -m pytest", "returncode": 1, "excerpt": "assert -1 == 3"},
            "previous edit changed the wrong line",
            4,
        )

        self.assertIn("prompt", payload)
        self.assertIn("Return exactly one complete unified diff", payload["prompt"])
        self.assertEqual(payload["metadata"]["output_format"], "unified_diff")
        self.assertEqual(payload["metadata"]["constraints"]["output_format"], "unified_diff")
        self.assertNotIn("Plan context", payload["prompt"])
        self.assertIn("Planner intent analysis", payload["plan_text"])
        self.assertIn("Planner confidence in this intent analysis: 0.75", payload["plan_text"])
        self.assertIn("advisory; issue text and source snippets are authoritative", payload["plan_text"])
        self.assertIn("change subtraction to addition", payload["plan_text"])
        self.assertIn("Suggested starting point", payload["plan_text"])
        self.assertEqual(payload["answer"], "")
        self.assertEqual(payload["task"], "issue_fix")
        self.assertEqual(payload["subgraph"], payload["graph"]["nodes"])
        self.assertNotIn("prior_repair_feedback", payload)
        self.assertNotIn("constraints", payload)
        self.assertNotIn("Observed failing behavior", payload["issue"]["body"])
        self.assertNotIn("Current failing behavior summary", payload["plan_text"])
        self.assertIn("previous edit changed the wrong line", payload["plan_text"])
        self.assertIn("right implementation evidence", payload["plan_text"])
        self.assertEqual(payload["snippets"][0]["lines"], ["def add(a, b):", "    return a - b"])
        self.assertIn("    1: def add", payload["snippets"][0]["numbered_text"])
        self.assertEqual(payload["serialized_code"][0]["id"], "func:pkg/calc.py:add")
        self.assertEqual(validate_cgm_payload(payload), [])
        self.assertEqual(payload["graph"]["edges"][0]["type"], "CALLS")
        self.assertIn("func:pkg/calc.py:add", payload["graph"]["adjacency_list"])

    def test_cgm_payload_keeps_context_code_visible_when_target_is_selected(self):
        graph = RepoGraph(root="/testbed")
        target_id = "func:pkg/writer.py:write"
        context_id = "func:pkg/writer.py:set_formats"
        graph.add_node(
            GraphNode(
                context_id,
                "function",
                "set_formats",
                "pkg/writer.py",
                1,
                2,
                text="def set_formats(formatters):\n    return dict(formatters)\n",
            )
        )
        graph.add_node(
            GraphNode(
                target_id,
                "function",
                "write",
                "pkg/writer.py",
                4,
                5,
                text="def write(table):\n    return table\n",
            )
        )
        graph.add_edge(target_id, context_id, "CALLS")
        memory = CgmMemory()
        memory.commit([graph.nodes[context_id], graph.nodes[target_id]])
        task = TaskSpec("fake", Path("/testbed"), "format bug", "writer ignores formatters", base_commit="abc")

        payload = build_cgm_payload(task, graph, memory, "fix writer", 0.75, None, None, 4, target_node_ids=[target_id])

        self.assertEqual([snippet["id"] for snippet in payload["snippets"]], [target_id, context_id])
        self.assertEqual([snippet["role"] for snippet in payload["snippets"]], ["target", "context"])
        self.assertEqual(payload["metadata"]["snippet_target_count"], 1)
        self.assertEqual(payload["metadata"]["snippet_context_count"], 1)
        self.assertEqual(payload["plan"]["targets"][0]["id"], target_id)
        self.assertNotIn("confidence", payload["plan"]["targets"][0])
        self.assertEqual(payload["plan"]["planner_confidence"], 0.75)
        self.assertEqual(payload["metadata"]["planner_confidence"], 0.75)
        self.assertEqual({item["id"] for item in payload["serialized_code"]}, {target_id, context_id})
        rendered = _snippet_section(payload["snippets"])
        self.assertIn("[Target code: pkg/writer.py:4-5]", rendered)
        self.assertIn("[Context code: pkg/writer.py:1-2]", rendered)
        self.assertIn("def set_formats", rendered)

    def test_cgm_service_normalize_graph_matches_official_shape(self):
        graph = cgm_service.normalize_graph(
            {
                "reponame": "astropy",
                "language": "python",
                "nodes": [
                    {
                        "id": "module_assignment:astropy/modeling/separable.py:_operators:316",
                        "kind": "assignment",
                        "nodeType": "Function",
                        "name": "_operators",
                        "path": "astropy/modeling/separable.py",
                        "start_line": 316,
                        "end_line": 317,
                        "text": "_operators = {'&': _cstack}\n",
                    }
                ],
                "edges": [],
            },
            issue={"repo": "astropy/astropy", "language": "python"},
        )

        nodes = {node["id"]: node for node in graph["nodes"]}
        op_node = nodes["module_assignment:astropy/modeling/separable.py:_operators:316"]
        self.assertEqual(op_node["nodeType"], "Attribute")
        self.assertEqual(op_node["attributeType"], "assignment")
        self.assertIn("file::astropy/modeling/separable.py", nodes)
        self.assertIn("repo::astropy", nodes)
        self.assertIn(
            {"source": "file::astropy/modeling/separable.py", "target": "module_assignment:astropy/modeling/separable.py:_operators:316", "type": "CONTAINS", "edgeType": "CONTAINS"},
            graph["edges"],
        )

    def test_cgm_service_node_sentence_follows_official_python_format(self):
        sentence = cgm_service._node_sentence(
            {
                "nodeType": "Function",
                "name": "_cstack",
                "header": "def _cstack(left, right):",
                "comment": "Function corresponding to '&' operation.",
                "text": "def _cstack(left, right):\n    return np.hstack([cleft, cright])\n",
            },
            repo_name="astropy",
        )

        self.assertIn("def _cstack(left, right): _cstack", sentence)
        self.assertIn("Function corresponding to '&' operation.", sentence)
        self.assertIn("return np.hstack", sentence)

    def test_cgm_payload_cleans_issue_and_strips_display_line_numbers(self):
        graph = RepoGraph(root="/testbed")
        graph.add_node(
            GraphNode(
                "func:pkg/calc.py:add",
                "function",
                "add",
                "pkg/calc.py",
                10,
                11,
                text="  10: def add(a, b):\n  11:     return a - b\n",
            )
        )
        memory = CgmMemory()
        memory.commit([graph.nodes["func:pkg/calc.py:add"]])
        task = TaskSpec(
            "fake",
            Path("/testbed"),
            "add bug",
            "<!-- hidden template -->\n### Description\nVisible bug.\n\n### System Details\nPython 3.10\n\nBase commit: abcdef1234567890",
            base_commit="abcdef1234567890",
        )

        payload = build_cgm_payload(task, graph, memory, "plan", 0.75, None, None, 4)

        self.assertIn("Visible bug", payload["issue"]["body"])
        self.assertNotIn("hidden template", payload["issue"]["body"])
        self.assertNotIn("System Details", payload["issue"]["body"])
        self.assertNotIn("Base commit", payload["issue"]["body"])
        self.assertEqual(payload["snippets"][0]["text"], "def add(a, b):\n    return a - b\n")
        self.assertEqual(payload["snippets"][0]["lines"], ["def add(a, b):", "    return a - b"])
        self.assertEqual(payload["graph"]["nodes"][0]["text"], "def add(a, b):\n    return a - b\n")

    def test_planner_loop_with_static_actions_and_static_cgm(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = self._make_repo(root)
            cgm = StaticCgmClient(
                {"patch": {"summary": "fix add", "edits": [{"path": "pkg/calc.py", "start": 2, "end": 2, "new_text": "    return a + b\n"}]}}
            )
            env = CodeRepairEnv.create(task, LocalRepoRuntime(root), cgm, AgentConfig(max_steps=8))
            planner = StaticPlannerClient(
                [
                    '{"tool":"run_failed_test","params":{}}',
                    '{"tool":"explore_find","params":{"query":"add","find_type":"function"}}',
                    # Filled by exact node id would be model-selected in real runs; direct env tests cover read/memory/repair.
                    '{"tool":"memory_commit_note","params":{"note":"need exact node id before repair"}}',
                ]
            )
            result = PlannerLoop(env, planner, AgentConfig(max_steps=3)).run()

            self.assertEqual(result.status, "not_pass")

    def _repair_params(self, node_id: str, plan: str = "fix add") -> dict[str, object]:
        return {
            "failure_seen": "The fail-to-pass run reports that add returns the wrong value.",
            "evidence_chain": [
                {
                    "node_id": node_id,
                    "role": "target",
                    "evidence": "The read code contains the implementation expression selected for repair.",
                }
            ],
            "target_nodes": [node_id],
            "intent_analysis": plan,
            "confidence": 0.8,
        }

    def _make_repo(self, root: Path) -> TaskSpec:
        (root / "pkg").mkdir()
        (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (root / "test_calc.py").write_text(
            "import unittest\nfrom pkg.calc import add\n\n\nclass CalcTests(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        return TaskSpec(
            task_id="fake",
            repo_path=root,
            issue_title="add returns the wrong value",
            issue_body="add should add two numbers",
            test_command="python -m unittest discover -p 'test_*.py'",
        )


class FakeRemoteSession:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.cleaned = False
        self.exec_commands: list[str] = []
        self.build_graph_calls = 0

    def cleanup_pool(self, timeout=None, cwd="/testbed"):
        self.cleaned = True
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": "no active instance; stop is no-op"}

    def start(self, timeout=None, cwd="/testbed"):
        self.started = True
        return {"ok": True, "returncode": 0, "stdout": "started", "stderr": ""}

    def exec(self, cmd, *, cwd=None, env=None, timeout=None):
        self.exec_commands.append(cmd)
        if "snap = {}" in cmd:
            return {"ok": True, "returncode": 0, "stdout": '{"/testbed/pkg/calc.py": "plain-body"}\n', "stderr": ""}
        if 'print("ok")' in cmd:
            return {"ok": True, "returncode": 0, "stdout": "ok\n", "stderr": ""}
        if "unittest" in cmd:
            return {"ok": True, "returncode": 0, "stdout": "OK\n", "stderr": ""}
        return {"ok": True, "returncode": 0, "stdout": "hello\n", "stderr": ""}

    def stop(self, timeout=None):
        self.stopped = True
        return {"ok": True, "returncode": 0, "stdout": "stopped", "stderr": ""}

    def build_repo_graph(self, repo_id="", timeout=1200, *, cwd=None, repo=None):
        self.build_graph_calls += 1
        jsonl = "\n".join(
            [
                json.dumps({"type": "node", "id": "file:pkg/calc.py", "kind": "file", "name": "pkg/calc.py", "path": "pkg/calc.py", "span": [1, 2]}),
                json.dumps(
                    {
                        "type": "node",
                        "id": "func:pkg/calc.py:add",
                        "kind": "function",
                        "name": "add",
                        "path": "pkg/calc.py",
                        "span": [1, 2],
                        "snippet_lines": ["def add(a, b):", "    return a + b"],
                    }
                ),
            ]
        )
        return base64.b64encode(gzip.compress(jsonl.encode("utf-8"))).decode("ascii")


class FakeToolPlannerClient:
    def complete_message(self, messages, tools=None, tool_choice=None):
        return {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "run_failed_test", "arguments": "{}"}}],
        }


class MalformedThenToolPlannerClient:
    def __init__(self):
        self.calls = 0
        self.seen_messages = []

    def complete_message(self, messages, tools=None, tool_choice=None):
        self.calls += 1
        self.seen_messages.append(messages)
        if self.calls == 1:
            return {"role": "assistant", "content": '{"tool":"run_failed_test","params":{}} {"tool":"read","params":{"node_id":"x"}}'}
        return {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "run_failed_test", "arguments": "{}"}}],
        }


class FailingCgmClient:
    def generate_patch(self, payload):
        raise RuntimeError("service unavailable")


class UnavailableCgmClient:
    def generate_patch(self, payload):
        raise CgmUnavailableError("CGM request failed: <urlopen error [Errno 111] Connection refused>")


class SequentialCgmClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []
        self.calls = 0

    def generate_patch(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        if not self.responses:
            return {"patch": {"edits": []}}
        return self.responses.pop(0)


class FakeFileRuntime:
    root = Path("/testbed")

    def read_file(self, path, start=None, end=None):
        self.last_read = (path, start, end)
        return "def add(a, b):\n    return a + b\n"


if __name__ == "__main__":
    unittest.main()
