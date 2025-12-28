# graph_planner/core/actions.py
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, RootModel, conint

# A) Explore（定位/阅读/扩展）
class ExploreAction(BaseModel):
    # 兼容旧版本字段：允许 extra 参数但忽略
    model_config = {"extra": "ignore"}

    type: Literal["explore"] = "explore"
    op: Literal["find", "expand"] = "find"

    # 两种用法：
    # 1) anchors -> 在 repo 图里做 find/expand
    # 2) query   -> 在记忆（text_memory / memory_subgraph）里做 recall
    anchors: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: List[str] = Field(default_factory=list)
    query: Optional[Union[str, List[str]]] = None

    # 图扩展半径 & 数量预算
    hop: conint(ge=0, le=2) = 1
    limit: conint(ge=1, le=100) = 50  # 兼容旧字段
    max_per_anchor: Optional[conint(ge=1, le=200)] = None
    total_limit: Optional[conint(ge=1, le=300)] = None
    dir_diversity_k: Optional[conint(ge=0, le=50)] = None

    schema_version: int = 2

# B) Memory（记忆维护，外部只给策略信号）
class MemoryAction(BaseModel):
    # 兼容旧 payload（可能还带 scope）：允许 extra 参数但忽略
    model_config = {"extra": "ignore"}

    type: Literal["memory"] = "memory"
    target: Literal["explore", "observation"] = "explore"
    intent: Literal["commit", "delete"] = "commit"

    # selector 允许：
    # - str: "latest" / tag / note_id
    # - dict: {"nodes":[...], "note":"...", "tag":"..."} 等
    selector: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    schema_version: int = 2

# C) Repair（是否打补丁；仅 apply=True 需要 plan）
class RepairAction(BaseModel):
    type: Literal["repair"] = "repair"
    apply: bool
    issue: Dict[str, Any]
    plan: Optional[str] = None  # 仅 apply=True 时需要，用于 Collater→CGM
    plan_targets: List[Dict[str, Any]] = Field(default_factory=list)
    patch: Optional[Dict[str, Any]] = None
    schema_version: int = 1

# D) Submit（终局评测）
class SubmitAction(BaseModel):
    type: Literal["submit"] = "submit"
    schema_version: int = 1

class NoopAction(BaseModel):
    type: Literal["noop"] = "noop"
    schema_version: int = 1


ActionUnion = Union[ExploreAction, MemoryAction, RepairAction, SubmitAction, NoopAction]


class _ActionSchema(RootModel[ActionUnion]):
    pass


def export_action_schema() -> Dict[str, Any]:
    """Return the serialisable schema for planner actions."""

    return _ActionSchema.model_json_schema()
