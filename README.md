# Graph Planner

Graph Planner 是一个面向代码修复任务的双智能体系统，通过**规划模型**与**CGM 补丁模型**协作，在容器化沙箱中定位问题并生成修复补丁。项目的主要目标如下：

- 复现论文 CGM 架构（Planner + CGM）在真实代码仓库上的工作流，并提供规则策略作为回退。
- 基于图检索的记忆维护与观察压缩，让模型能够在大规模仓库上进行局部推理。
- 借助 [R2E-Gym](R2E-Gym/README.md) 的 RepoEnv / DockerRuntime 运行环境完成端到端强化学习训练。
- 保留本地部署接口，使 Planner 模型与 CGM 可以在离线环境下无缝接入。



## 目录结构概览

核心代码已经收敛到单一的 `graph_planner/` 包，其他目录仅保留必要的配置、脚本与文档：

| 子目录 | 主要职责 |
| --- | --- |
| `graph_planner/agents/` | 规则策略 `PlannerAgent`、本地 LLM 策略及共享对话协议，驱动 Explore→Memory→Repair→Submit 的动作循环。 |
| `graph_planner/core/` | 统一的 Planner 动作模型与协议校验，训练、评测与集成脚本共享相同的数据结构。 |
| `graph_planner/datasets/` | 将 SWE-bench / R2E-Gym 等外部任务转换成 Graph Planner JSONL/manifest，供训练与评测直接消费。 |
| `graph_planner/env/` | `PlannerEnv` 负责解析动作、计算奖励并调用容器运行时，是训练与评测的核心环境实现。 |
| `graph_planner/infra/` | 配置加载、并行度控制、遥测日志目录等运行期开关。 |
| `graph_planner/integrations/` | 本地 LLM 客户端、CodeFuse CGM 接入以及 rLLM/VERL 训练封装。 |
| `graph_planner/memory/` | 图检索、候选节点与文本记忆管理，为 Explore/Memory 动作提供状态维护能力。 |
| `graph_planner/models/` | 预留的模型适配命名空间（当前未内置示例模型，加载逻辑统一走 `integrations/`）。 |
| `graph_planner/runtime/` | `SandboxRuntime` 抽象容器后端，封装 RepoEnv / R2E / Docker 等执行路径。 |
| `scripts/` | 数据准备、数据注册、评测与自动化工具脚本。 |
| `tests/` | rLLM 生态相关的单元/集成测试，例如奖励管理器与数据注册流程。 |


## 快速上手

1. **安装依赖**
   ```bash
   cd rllm/
   pip install -e ./verl  #rllm需要安装verl           
   pip install -e .       #安装rllm
   cd ..
   pip install -e .
   ```


2. **准备训练/评测数据集**
   ```bash
   PYTHONPATH=. python scripts/prepare_datasets.py \
     --r2e-dataset R2E-Gym/R2E-Gym-Lite \
     --swebench-dataset princeton-nlp/SWE-bench_Verified
   ```
  `prepare_datasets.py` 会把 Hugging Face 上的 R2E-Gym / SWE-bench 数据集转换成 Graph Planner 所需的 JSON/JSONL 结构，并在 `datasets/` 下生成对应的任务文件、`instances/*.json` 以及 docker manifest。脚本同时支持 `--skip-r2e`、`--skip-swebench`、`--prepull-*` 等参数，便于按需刷新或预拉容器。

3. **注册数据集以复用 Parquet 索引（可选）**
    ```bash
    PYTHONPATH=. python scripts/register_graphplanner_dataset.py \
      --name graph_planner_repoenv \
      --split val \
      --jsonl datasets/r2e_gym/val.jsonl
    ```
    该脚本会把 JSONL 与 `instances/*.json` 注册到 rLLM 的本地数据集仓库，写出 `rllm/rllm/data/datasets/<name>/<split>_verl.parquet`。训练或评测前可以直接通过 `DatasetRegistry.get(name)` 复用索引，避免重复解析任务描述。

4. **运行 Graph Planner 评测**
    ```bash

    ```


5. **强化学习训练**






