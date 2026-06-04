# Planner Action And Input Audit

本文记录新 agent 中 planner 输入、动作协议、tool calling 开关，以及容易误导模型行动的边界。

## `PLANNER_TOOL_CALLING=1` 的含义

`PLANNER_TOOL_CALLING=1` 不是把 `tool_choice` 设置成 `required`，也不是强制模型一定 tool call。它的含义是：

1. `AgentConfig.planner_tool_calling=True`。
2. `PlannerLoop` 调用 `OpenAIPlannerClient.complete_message(...)`，而不是只取普通文本 `content`。
3. 请求体中带上 `tools=PLANNER_TOOL_SCHEMAS`。
4. 请求体中设置 `tool_choice="auto"`。
5. 返回后通过 `parse_planner_message()` 同时解析 `tool_calls` 和 `reasoning_content` / `<think>...</think>`。

所以你记得的方向是对的：对当前 vLLM/Qwen 配置来说，想在同一回答里观察到 thinking 和 tool use，关键是请求端实际传 `tool_choice="auto"`，服务端也需要已用 `--enable-auto-tool-choice --tool-call-parser hermes` 启动。

当前代码路径：

- 配置开关：`src/graphplanner_agent/config/schema.py`
- eval CLI 参数：`src/graphplanner_agent/cli/eval.py --planner-tool-calling`
- 请求发起：`src/graphplanner_agent/planner/loop.py`
- OpenAI-compatible body：`src/graphplanner_agent/planner/client.py`
- tool response parser：`src/graphplanner_agent/planner/response_parser.py`

## Tool Calling 模式下的 Prompt

tool-calling 模式下，system prompt 不能再要求 `Emit exactly one JSON action`。那会和 `tool_choice="auto"` 冲突，让模型在工具调用和 JSON 文本之间摇摆。

现在 prompt 分成两种：

- 文本 JSON 模式：要求输出一个 JSON action。
- tool calling 模式：要求 `Call exactly one provided tool. Do not answer in prose.`

这样 `PLANNER_TOOL_CALLING=1` 时，planner 的系统指令和 OpenAI tool request 是一致的。

## Test Source 边界

不应该靠 query 词硬挡 test source。比如 `assert`、`pytest`、`unittest` 可能出现在实现代码、错误消息或正常 API 名称里，直接 block 会误伤 planner。

当前边界改成语料级约束：

- graph build 本地路径默认跳过 tests。
- search 只返回非 test path 节点。
- filesystem fallback 只扫非 test path。
- expand 过滤 test path 邻居。
- read 遇到 test path 会 blocked。
- memory_commit 遇到 test path 会 blocked。

也就是说：planner 可以搜索 `assert` 这样的实现概念，但检索面不会暴露 benchmark test source。Observation 和 prompt 会明确告诉模型：测试只作为 behavior/fail-to-pass 症状，test file selector 不适合作为搜索目标，因为 tests 不在可读证据面里。

## SWE-bench 官方测试方式

不要把 SWE-bench 容器统一当成 pytest 容器。官方 harness 的方法是：

- 由 `make_test_spec()` 根据 `repo/version/test_patch/FAIL_TO_PASS/PASS_TO_PASS` 生成 `TestSpec`。
- `TestSpec.eval_script` 写成 `/eval.sh`，脚本头是 `set -uxo pipefail`，不是 `set -e`。
- `eval_script_list` 会重置/应用 benchmark test patch，运行 repo/version 对应的 `test_cmd`，并用 `>>>>> Start Test Output` / `>>>>> End Test Output` 包住测试输出。
- `test_cmd` 来自 `MAP_REPO_VERSION_TO_SPECS`，可能是 pytest、Django `runtests.py`、tox、SymPy `bin/test`、go test、ctest、make check 等。
- 结果不应该只看进程 returncode，而应该用 SWE-bench repo-specific log parser 解析输出，再通过 `FAIL_TO_PASS` / `PASS_TO_PASS` 计算 `tests_status` 和 `resolved`。

新 agent 的 runtime 规则是：有 `swebench_spec.eval_script_list` 时优先按官方 eval script 执行，并尝试用 `swebench.harness.grading.get_eval_report(..., include_tests_status=True)` 解析；只有缺少官方 metadata 的本地/ad-hoc Python 任务才 fallback 到 pytest selector。

## 动作参数校验为何收紧

原来 `explore_find` 的空 query 会通过。由于 search 层对空 query 会返回一批任意高层节点，planner 可能误以为这是一次有效探索，后续 read/memory_commit 会被带偏。

现在 guard 层会阻止：

- 空 `explore_find.query`
- 非法 `find_type`
- 空 `explore_expand.anchor`
- 非法 `expand_mode`
- 空或非法 `read.view`
- 非 list 的 memory id 参数
- 空 `memory_commit_note.note`

这个收紧一般不会误导模型，反而把无效动作变成明确 blocked feedback。`read.view` 是例外：真实 tool-calling planner 偶尔会只传 `node_id`，所以现在省略 `view` 时默认读取 `body`，只有显式给出非法 view 才会被挡住。

## Planner Observation 的字段

Observation 现在包含：

- `task`: task id、base commit、sandbox backend、image、repo path
- `graph_summary`: node/edge 数
- `retrieval_scope`: 检索/读取只暴露实现代码，不暴露 benchmark tests
- `working_subgraph_W`
- `working_code_W`
- `repair_memory_M`
- `text_notes_T`
- `latest_action_result`
- `trajectory_summary`
- `planner_diagnostics`
- `unread_symbol_references`
- `evidence_status`
- `last_repair_attempt`
- `recent_actions`
- `recent_action_signatures`
- `verified`
- `runtime_facts`

这些字段的目标是让 planner 看到事实状态，而不是让环境替 planner 暗示下一步。

`working_subgraph_W` 是节点索引摘要；`working_code_W` 是 W 中已经 read/hydrate 的代码包，带行号、来源、是否已进入 M。为了避免 observation 无限膨胀，它有总字符预算和单节点预算，但在常见 SWE-bench 轨迹里会持续保留已读函数/类/assignment 的代码内容，减少模型重复 read 或凭记忆复述。

`trajectory_summary` 是从第 1 步开始的完整行动摘要，不再只给最近几个 action。它记录每步 tool、参数摘要、状态、结果摘要和 block/reject 原因，用来让 planner 明确知道哪些 read/search/repair 已经发生过。

`planner_diagnostics` 记录 planner 输出解析失败时的错误和截断后的原始响应。这样可以区分“模型没按 tool schema 输出”和“环境/动作执行失败”。

## Find/Read 的证据粒度

`explore_find` 现在不是“只返回名字”的动作了。对 `class/function/method/assignment` 这种小粒度实现节点，它会返回一小段带行号的代码预览，并从预览中抽取同文件的实现线索，例如 helper 调用、dispatch table、registry、wrapper、assignment map、兄弟函数或类。

但这不等于自动把 find 当成 repair 证据：

- file 级结果不返回全文，只返回 `top_symbols`，避免把大文件塞进 observation。
- find 返回的代码预览只是导航证据，不进入 CGM repair memory。
- W 里仍把 find/expand 结果作为候选节点；真正要进 CGM 的 M，仍应由 `read -> memory_commit` 形成 hydrated evidence。
- 如果 planner 用短 symbol 做 `read`/`expand` 且存在多个同名节点，环境会 blocked 并返回 candidates，避免静默选错。
- `memory_commit` 会补上“已经读过、且由所选节点直接引用”的本地实现引用，并在结果中写入 `auto_included_read_references`。这不是替模型探索未知代码，而是避免模型已经读了关键 helper/dispatch table 却漏选进 M。
- 对 planner 暴露的是公共 kind：`file/class/function/method/assignment`。内部图里的 `module_assignment` 会归一化成 `assignment`；如果 planner 仍传了 `find_type=module_assignment`，环境也会按 `assignment` 兼容处理。
- 远端图里部分类方法会以 dotted function 形态出现，例如 `Model.foo` 的 kind 是 `function`。`find_type=method` 会兼容这种 dotted function，避免因为图构建器 taxonomy 差异搜不到。

`read` 的结果会带 `local_symbol_references` / `unread_local_symbol_references`。这些字段只表达“当前代码文本引用了哪些同文件实现节点”，不表达优先级，也不附带 `next_action`。

如果当前代码里存在 dict assignment 形式的表驱动分派，`read` / 小粒度 `find` 还会带 `dispatch_tables`，例如：

```json
{"name": "_operators", "entries": [{"key": "&", "target": "_cstack"}, {"key": "|", "target": "_cdot"}]}
```

这同样是源码事实，不是下一步建议。它保留的是 key 到 target 的关系，避免把 dispatch table 退化成普通 `_cstack/_cdot` 符号列表。普通 `local_symbol_references` 现在按代码出现顺序展示，而不是按距离或函数位置排序。

旧版曾把这些引用写成 lead 并附带：

```json
{"tool": "read", "params": {"node_id": "module_assignment:pkg/mod.py:_operators:316"}}
```

这容易被 planner 当成系统推荐。现在只保留 node id、kind、name、path、line span、symbol、read_status、relation/source 等事实字段。

`evidence_status` 不给出 `ready/caution/revise` 这类等级，只列事实：是否已有 fail-to-pass 行为、M 里有多少 hydrated code、是否有 unhydrated memory、最近结果里有多少 unread symbol reference、W 中有多少 read-but-not-committed 代码、W 中有多少 visible code、上一轮 patch status 是什么。

Observation 还会包含 `working_vs_memory.read_not_committed_to_M`，明确列出“已经 read 到 W，但还没有进入 M”的代码。因为 CGM repair 只看 M，不会自动看到整个 W。该字段不再附带 `next_action`。

`repair` 现在不是普通探索动作，而是需要提交结构化证据包：

- `hypothesis`: 具体因果诊断。
- `behavior_evidence`: 对应的 fail-to-pass 行为。
- `causal_chain`: 从已读代码节点到 patch target 的因果链。
- `target_nodes`: 已进入 M 的修复目标节点。
- `why_these_targets`: 为什么这些节点是修改位置。
- `known_uncertainties`: 仍然不确定的点。
- `why_ready_to_patch`: 为什么这些不确定点不阻止当前修复。
- `considered_alternatives`: 当 W 中存在已读但未进 M 的代码时，说明至少一个被考虑但不作为 target 的节点。
- `plan`: 给 CGM 的具体修复计划。

环境只做通用一致性校验：target 必须在 M 中、causal chain 必须引用已读/已提交代码、target 必须出现在 causal chain 中、如果上一次失败后 M 没变则阻止重复 repair。这些检查不按 helper、wrapper、dispatcher 等具体函数类型做判断。

## `runtime_facts` 是否比模型更正确

不是。它不是“比模型更聪明”的决策器，只列硬 runtime 状态。

例如：

- `verified=True` 是环境事实，表示上一轮 repair 已经通过 fail-to-pass 验证。
- 在这个状态下，runtime 会直接设置 `done=True`、`status=pass` 并结束轨迹。

真正的硬约束仍在环境里：

- repair 通过验证时环境自动结束。
- 没有 fail-to-pass behavior 或 hydrated memory 时 repair 会被 blocked。
- test path read/memory_commit 会被 blocked。

当 repair 返回 `syntax_failed` 时，它表示“生成的 patch 语法不合法并且已经 rollback”，不是原始源码本身有语法错误。Observation 通过 `last_repair_attempt.error_origin=generated_patch` 和 `runtime_facts.source_after_failed_patch=rolled_back_or_unchanged` 表达这个事实。

失败 repair 会在 `last_repair_attempt.patch_preview` 中保留 patch 摘要、路径、行号和新文本截断片段；`test_summary` 也会保留截断摘要。这样后续 planner 能看到“刚才到底改了什么以及为什么没过”，而不是只知道失败。

## 可能误导模型的剩余风险

1. 符号引用列表仍有展示顺序。该顺序只能视为代码出现顺序，不能视为相关性排序；dispatch table 已单独用 `dispatch_tables` 结构化事实表达。

2. test selector query 不会被硬挡，但大概率找不到结果。Observation 只说明 tests 不在 indexed code 中；如果模型反复搜 test selector，repeat guard 和 latest blocked/no-result feedback 会把它拉回来。

## 当前建议

正式用 Qwen/vLLM planner 跑 eval 时建议：

```bash
PYTHONPATH=src python -m graphplanner_agent.cli.eval \
  --tasks tasks.jsonl \
  --planner-endpoint http://127.0.0.1:30000/v1/chat/completions \
  --planner-model models/Qwen3-32B \
  --planner-tool-calling \
  --cgm-backend mock \
  --sandbox-backend remote_swe \
  --verbose
```

对应环境变量写法：

```bash
export PLANNER_TOOL_CALLING=1
```

这会让请求端使用 `tool_choice="auto"`，同时保留 reasoning/thinking 解析。

## CGM 输入输出协议审计

当前默认的真实 CGM 部署假设已经改为“本容器内本地服务”，通常通过
`http://127.0.0.1:30001/generate` 调用。旧的 `172.*:30001` 记录只代表
历史实验，不再作为当前部署说明。

当前真实 CGM 服务协议与旧 `graph_planner.integrations.codefuse_cgm.service`
保持兼容。该服务接受的关键字段是：

- `issue`: issue 标题、正文、repo、language。
- `plan`: 结构化 target 列表。
- `plan_text`: planner 的诊断、证据和修复计划。
- `prompt`: 给 CGM reader 的显式指令。
- `subgraph`: 旧 client 兼容字段，等价于 graph nodes。
- `graph`: `{nodes, edges, reponame, language}`。
- `snippets`: 可编辑候选代码片段，服务端会用 `SnippetFormatter` 转成带行号文本。
- `metadata.constraints`: `max_edits`、只改 implementation、不改 tests 等约束。

旧 client 会同时发送 `prompt`、`answer`、`task`、`subgraph`。新 agent 之前只发送了 `plan_text`、`graph`、`snippets`，服务虽然能 fallback，但协议对齐不完整。现在 payload 已补齐旧字段，并把 `prompt` 改成专门的输出契约，而不是把 planner plan 原样当系统指令。

官方 CodeFuse-CGM 的公开链路更接近：

```text
issue -> Rewriter -> Retriever subgraph -> Reranker selected files -> Reader patch
```

其中 Reader 的图输入不是把邻接矩阵写进 prompt，而是通过 graph nodes/edges 编码成 graph prefix/adjacency。官方 Rewriter/Reranker prompt 里有 `Instructions`，但那是检索/排序阶段的任务说明；当前旧服务里的 `[Instruction]` 是 `payload.prompt`，主要用途应当是输出契约。

因此当前 CGM 输入约定调整为：

- `prompt` 只保留短输出契约：JSON patch schema、精确行号、不要 markdown/diff、不要改 tests、源码片段优先。
- `plan_text` 降权为非权威 repair context，只保留 target code、fail-to-pass selector、结构化源码事实、prior feedback；不再把 `hypothesis/why_ready/alternatives` 或局部 repair intent 压到 snippets 前面。
- `issue + graph + snippets` 是修复证据主体；planner 的 repair plan 继续用于环境校验、trace 和 telemetry，但默认不作为 CGM 的自然语言修复指令。

2026-05-17 直连 CGM 对比 `astropy__astropy-12907`：

- 无自然语言修复 plan：CGM 约 10s 返回单行语义正确 patch，把 `_cstack` 中 `= 1` 改为 `= right`。
- 带自然语言 repair intent：CGM 约 109s 返回 `codefuse-cgm-partial`，把 `else` 块折叠成 `cright = right`，不可靠。
- 只保留 target/source facts、不带 repair intent：CGM 约 9s 返回单行语义正确 patch。

结论：CGM 对源码片段和结构化事实已经能定位这类问题；自然语言修复 plan 会提高误修风险，应从 CGM prompt 中移除。

CGM 的 graph-aware 路径不是把邻接矩阵作为纯文本发进 prompt。服务端会调用 `normalize_cgm_graph` 标准化节点/边，再由 `encode_graph` 把每个 node sentence 编成 CodeT5 embedding，并根据 `graph.edges` 生成 adjacency tensor。也就是说，client 需要稳定提供 nodes 和 edges；dense adjacency matrix 由服务端构造。新 payload 额外保留 `adjacency_list` / `adjacency_edges` 用于审计，但真正生效的仍是 `graph.edges`。

代码正文有两条输入路径：

- `graph.nodes[].text`: 用于图节点 embedding。
- `snippets[].lines`: 用于 CGM prompt 中的 `[Snippets]` 带行号文本。

新 payload 额外加入 `serialized_code[].numbered_text` 和 `snippets[].numbered_text`，用于 trace/debug，并让 prompt 明确说明：snippets 和 graph node text 是权威当前源码；如果 planner plan 提到当前片段里不存在的表达式，CGM 应忽略该计划片段，按实际源码生成最小 patch。

最近真实 astropy 轨迹里的 CGM 报错含义：

```text
CGM HTTP 500: Local CGM output cannot be parsed into patch schema.
first_output=--- a/astropy/modeling/separable.py ...
```

这不是 patch apply 失败，而是 CGM 服务内部没能把模型原始输出解析成 patch schema。`first_output` 显示模型输出了 malformed unified diff，并混入了 `('',)`、`/dev/null`、不存在于当前 snippet 的源码行。预期输出应该是：

```json
{"patch":{"edits":[{"path":"...","start":123,"end":123,"new_text":"...\\n"}],"summary":"..."}}
```

或者至少是服务端可安全解析的单文件 unified diff。现在的优化重点是让输入协议更贴近服务端预期，并降低 planner 错误假设对 CGM 的覆盖力：源码片段优先，plan 是可被代码反驳的假设。

## Planner Observation 模式

现在 planner observation 支持两种模式：

- `json`: 默认模式，保留完整结构化 JSON，便于回放和程序化审计。
- `text`: 实验模式，把 `CURRENT TURN PROTOCOL` 放在输入第一屏，用自然语言突出当前阻断、合法动作、`W != M` 规则和可提交的已读节点；后面仍保留必要结构化状态和完整 working code。

切换方式：

```bash
export GRAPHPLANNER_OBSERVATION_MODE=text
```

或在 eval 命令中加：

```bash
--observation-mode text
```

设计意图不是把所有 JSON 删除，而是把最容易被模型忽略的协议状态移到显眼位置：

- `repair_memory_M` 为空时，明确说 `repair` 当前无效；
- `working_code_W` 里已有代码但未提交时，明确列出 `memory_commit` 候选 id；
- 强调 CGM repair 只看 M，不看 W；
- 后续仍保留 `trajectory_summary`、`evidence_status`、`runtime_facts` 和 `working_code_W`，避免丢失可审计性。

2026-05-24 对 `astropy__astropy-12907` 的一次验证：

- `json` 模式旧轨迹：读到 `_separable` 后没有提交 M，最后 `repair` 被拦截为 `repair requires at least one committed memory node with code`。
- `text` 模式新轨迹：第 3 步读 `separability_matrix` 后，第 5 步主动 `memory_commit`；第 8 步读 `_separable` 后，第 9 步再次 `memory_commit`；第 10 步进入 CGM repair。

这说明文本第一屏对“提交记忆”这个协议行为有效。该轮最终未通过，是后续 patch 质量/定位深度问题：planner 只提交了 `separability_matrix` 和 `_separable`，CGM 修改 `_separable` 后导致更多 separable 测例失败；它还需要继续读取并提交 `_operators`、`_cstack`、`_coord_matrix` 等真正组合矩阵的实现节点。

## Thinking 默认值

2026-05-24 起，planner 与 DashScope CGM bridge 默认都开启 thinking：

- planner: `AgentConfig.planner_enable_thinking=True`，请求体默认带 `enable_thinking: true`。
- CGM bridge: `BridgeConfig.enable_thinking=True`，DashScope OpenAI-compatible 请求体默认带 `enable_thinking: true`。

仍可显式关闭：

```bash
export PLANNER_ENABLE_THINKING=0
export CGM_DASHSCOPE_ENABLE_THINKING=0
```

或 CGM bridge 启动时加：

```bash
--disable-thinking
```

CGM bridge 会把 DashScope 返回的 `reasoning_content` 记录为：

- `reasoning_content`
- `reasoning_preview`
- `reasoning_chars`
- `thinking_enabled`

agent 的 repair 结果里会保留 compact 后的 `cgm_response`，用于判断 CGM 是否真的返回了 reasoning 内容。
