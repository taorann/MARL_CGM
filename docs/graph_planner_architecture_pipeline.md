# Graph Planner 架构与运行全景

> **Summary (English)**  
> Graph Planner is a two-model workflow for code repair: a **Planner** decides what to explore / remember / repair, and a **CGM** model generates patches. This document summarizes the **current** module boundaries, data flow, container backends (including `remote_swe`), and the eval/training entrypoints that exist **in this repository version**.

---

## 1. 顶层结构 / Repository layering

```
scripts/
  run_eval_graph_planner.sh         # 评测入口（Shell）
  eval_graph_planner_engine.py      # 评测主程序（Python）
  prepare_datasets.py               # 数据集转换（R2E-Gym / SWE-bench -> jsonl）
  register_graphplanner_dataset.py  # 注册为 rLLM 可消费的数据集（parquet）
  validate_contracts.py             # 契约冒烟检查
configs/
  eval/graph_planner_eval_defaults.yaml
  eval_planner_grpo.yaml            # 训练相关配置（但仓库内未提供直接可跑的训练主程序）
graph_planner/
  agents/common/                    # 契约、prompt、text-protocol（核心 SSOT）
  core/                             # Action/Patch 等结构化数据模型
  env/                              # PlannerEnv：动作编排与奖励/信息回传
  memory/                           # 子图/候选/文本记忆：Explore & Memory 底座
  runtime/                          # sandbox/runtime：repoenv/docker/remote_swe 后端
  integrations/                     # codefuse_cgm / local_llm / rllm 适配层
  infra/                            # config / telemetry /（可选）wandb metrics
aci/                                # 补丁/工具/guard 的 python 包（注意：不是 .aci 运行时目录）
actor/                              # rLLM/verl 训练侧 CGM fallback（本地，无 RPC 递归）
tests/
  test_reward_manager_loading.py
rllm/                               # vendored rLLM 子模块（训练底座）
```

> 说明：仓库里“运行时目录”是 **`.aci/`**（点开头），用于子图缓存/本地覆盖配置；它通常 **不提交进 git**，首次运行会自动创建。

---

## 2. 核心模块职责（只列当前仓库确实存在的实现）

| 模块 | 关键内容 | 一句话职责 |
|---|---|---|
| `graph_planner/agents/common/contracts.py` | `SYSTEM_PROMPT`、`parse_action_block`、`validate_planner_action` | Planner/CGM 协议的唯一事实来源（SSOT） |
| `graph_planner/agents/common/text_protocol.py` | repair 文本轨迹 → CGM payload → patch → apply/test | “Planner 只给计划，环境负责调用 CGM 并落补丁”的闭环 |
| `graph_planner/core/actions.py` | `ExploreAction/MemoryAction/RepairAction/SubmitAction` | 结构化动作模型（env / rLLM 共享） |
| `graph_planner/env/planner_env.py` | `PlannerEnv.reset/step` | 环境主循环：执行 explore/memory/repair 并产出 observation/info |
| `graph_planner/memory/*` | `subgraph_store`、`mem_candidates`、`graph_adapter`、`text_memory` | working/memory 子图维护 + 候选生成 + 文本记忆 |
| `graph_planner/runtime/sandbox.py` | `SandboxRuntime` | 统一封装容器后端：repoenv/docker/apptainer_queue/remote_swe |
| `graph_planner/runtime/remote_swe_session.py` | `RemoteSweSession` | 通过 ssh 调用 login 节点 `hpc/swe_proxy.py` 实现跨集群容器交互 |
| `graph_planner/integrations/codefuse_cgm/*` | data/formatting/inference/training | CGM 的数据编排、线性化、推理与（监督）训练组件 |
| `graph_planner/integrations/local_llm/hf.py` | `HuggingFaceChatClient` | 本地 HF 模型的聊天式 Planner 推理接口 |
| `graph_planner/integrations/rllm/*` | `GraphPlannerRLLMAgent/Env` 等 | rLLM agent/env 适配层（训练入口需自行接入） |
| `graph_planner/infra/telemetry.py` | `log_event`、`log_test_result` | 轻量遥测：写 `logs/events.jsonl` 与 `logs/test_runs.jsonl` |
| `graph_planner/infra/metrics.py` | wandb helpers | 可选：把指标打到 W&B（不是必须） |

---

## 3. Planner 动作协议（现状）

Planner 输出通过 `<function=...>` 区块解析（`contracts.py`）并映射为 `core/actions.py` 中的动作：

- `explore`: `find / expand / read`
- `memory`: `commit / delete`
- `repair`: Planner 给出 `subplan`（文本计划），环境通过 `text_protocol` 调用 CGM 生成补丁并可选应用/测试
- `submit`: 终止 episode
- `noop`: 显式跳过

### 3.1 关于 scope（对齐当前实现）

- `MemoryAction` **不再暴露 `scope` 字段**（模型不需要知道 turn/session）
- `PlannerEnv` 内部调用 `text_memory` 时固定使用 `scope="session"`
- `text_memory` 仍保留 scope 参数作为实现细节，但不对模型暴露

---

## 4. 记忆系统：working 子图 / memory 子图 / 文本记忆

### 4.1 子图存储（`.aci/subgraphs`）

- `graph_planner/memory/subgraph_store.py` 将子图缓存到 `.aci/subgraphs/<issue>.json`
- `.aci` 是运行时目录：默认仓库里可能没有，但运行时会自动创建

### 4.2 候选生成（1-hop 扩展）

- `mem_candidates.build_mem_candidates` + `graph_adapter.one_hop_expand`
- 优先读取任务自带的 repo 图（若提供），否则按需构建轻量图（文件/函数/类/导入边）

### 4.3 文本记忆（text memory）

- `text_memory.memory_commit / memory_delete` 维护“可被模型回忆的 note”
- 当前版本：**recall/find 记忆**如果要做，需要显式落到 `explore.find(query=...)` 的实现逻辑里（或新增 `explore.recall`），属于扩展点（文档不假设已实现）。

---

## 5. Repair 闭环：Planner 给计划，环境调 CGM 落补丁

主闭环如下：

1) Planner 输出 `repair(subplan=...)`  
2) `text_protocol.handle_planner_repair(...)` 构造 CGM payload（Issue + Plan + Subgraph + Snippets）  
3) 调用 CGM（本地 HF 或远端服务）生成 patch  
4) `SandboxRuntime.apply_patch` + 可选测试/命令执行  
5) 结果写回 observation/info，并落遥测日志（events / test_runs）

> 设计要点：协议收敛在 `common/contracts.py + common/text_protocol.py`，避免 Planner 直接输出 diff 造成不可控与难训练。

---

## 6. 容器与运行时：本地 / RepoEnv / remote_swe（曙光-北极星）

### 6.1 SandboxRuntime 的后端选择

`graph_planner/runtime/sandbox.py` 支持：
- `repoenv` / `r2e` / `docker`（本地 Docker/RepoEnv）
- `apptainer_queue`（本地队列式 apptainer runtime）
- `remote_swe`（跨机 ssh 到 login 节点，通过 `hpc/swe_proxy.py` 控制远端容器）

### 6.2 remote_swe（对齐当前代码）

核心对象是 `RemoteSweSession`：

- 可选调用 `hpc/ensure_runners.py --target N` 确保足够 runner
- 然后把 `start/exec/stop/build_graph` JSON 通过 ssh stdin/stdout 转发给 `hpc/swe_proxy.py`
- 通过 `GP_NUM_RUNNERS` 控制并发 runner 数量

---

## 7. 日志与遥测（现状实现）

当前仓库**明确实现**的文件：

- `logs/events.jsonl`：事件流（动作、阶段、错误等）
- `logs/test_runs.jsonl`：测试/执行结果详情（含时间戳）

`metrics.py`：
- 主要是 W&B 初始化与上报 helper
- 是否保留取决于你是否需要 W&B；如果删除，需确保不再被脚本/模块 import

> 注：本文档不再声称存在 `steps.jsonl / metrics.jsonl` 等文件（当前代码中没有对应写入实现）。如果你要升级为完整 trajectory 三流，建议保持调用点不变，只扩展 telemetry 后端。

---

## 8. 运行方式（仓库内真实存在的入口）

### 8.1 评测（eval）

```bash
bash scripts/run_eval_graph_planner.sh   --config configs/eval/graph_planner_eval_defaults.yaml   --dataset datasets/xxx.jsonl   --parallel 1
```

### 8.2 数据准备

```bash
PYTHONPATH=. python scripts/prepare_datasets.py   --r2e-dataset R2E-Gym/R2E-Gym-Lite   --swebench-dataset princeton-nlp/SWE-bench_Verified
```

### 8.3 dataset 注册（供 rLLM 消费）

```bash
PYTHONPATH=. python scripts/register_graphplanner_dataset.py   --name graph_planner_repoenv   --split val   --jsonl datasets/r2e_gym/val.jsonl
```

---

## 9. 当前版本的“已移除/待恢复”能力（避免误导）

- `graph_planner/agents/rule_based/`、`graph_planner/agents/model_based/`：当前仓库版本已不存在  
- `GraphPlannerRLLMAgent.use_rule_fallback`：默认关闭；若打开可能因缺少 `agents/rule_based` 导致导入失败（属于待恢复/待清理项）  
- “GRPO/PPO 训练脚本入口”：仓库内已有 rLLM 适配层与训练配置文件，但缺少一个“直接可跑”的训练主程序（需要你补脚本或直接调用 rLLM trainer）

---



## 10. 曙光—北极星容器控制流（remote_swe）

```
曙光命令行
   └── scripts/run_eval_graph_planner.sh
         └── eval_graph_planner_engine.py（解析 --parallel）
               └── sandbox_overrides["num_runners"] = parallel
                     └── SandboxRuntime(backend="remote_swe")
                           └── RemoteSweSession(num_runners)
                                 ├── ssh 调 hpc/ensure_runners.py --target=num_runners  (可选)
                                 └── ssh 调 hpc/swe_proxy.py（stdin/stdout JSON，带 env: GP_NUM_RUNNERS）
                                      └── ApptainerQueueRuntime(num_runners)
                                            └── 将不同 run_id 分配给 runner-0..runner-(num_runners-1)
                                                  └── 每个 runner 维护一个长期容器 instance（gp-XX）
```

---

## Update note

本文档是对旧版“Graph Planner 架构与训练运行全景”文档的**对齐与收敛版本**，主要变化：

- 移除对 `agents/rule_based`、`agents/model_based` 的引用（当前仓库不存在）
- 训练/GRPO 部分降级为“配置存在但缺少训练入口脚本”
- 遥测只保留现有实现：`events.jsonl` + `test_runs.jsonl`
- 去掉所有行号级引用，避免文档随重构快速失效
