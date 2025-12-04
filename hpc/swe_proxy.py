#!/usr/bin/env python
"""
swe_proxy.py

一个很薄的本地代理：

- 从 stdin 读取一条 JSON 请求；
- 通过 ApptainerQueueRuntime 把请求发到指定的 runner；
- runner 在计算节点上使用 singularity/apptainer 执行命令；
- 最后把结果封装成 JSON 打印到 stdout。

支持两种模式：

1) 一次性执行（旧用法，兼容）：
   输入 JSON 不带 "op" 字段时：
     {
       "run_id": "local-test",
       "image": "<docker image>",
       "cmd": "cd /testbed && python -V",
       "timeout": 600,
       "env": {...},     # 可选
       "cwd": "/home"    # 可选，默认 "/home"
     }
   会在对应 SIF 上执行一次性的 `apptainer exec`。

2) 长期 instance（轨迹模式）：
   需要显式带上 "op" 字段：

   (a) 启动轨迹容器：
       {"op": "start", "run_id": "ep-1", "image": "<docker image>", "timeout": 600}

   (b) 在该轨迹对应的容器里执行命令：
       {
         "op": "exec", "run_id": "ep-1",
         "image": "<docker image>",
         "cmd": "cd /testbed && python -V",
         "timeout": 600,
         "env": {...}, "cwd": "/home"
       }

   (c) 结束轨迹、停止容器：
       {"op": "stop", "run_id": "ep-1", "timeout": 600}

其中 run_id 用来标识一条轨迹；ApptainerQueueRuntime 会把同一个 run_id
稳定映射到某一个 runner（例如 gp-00 / gp-01 / ...）。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

from graph_planner.runtime.apptainer_queue_runtime import ApptainerQueueRuntime
from graph_planner.runtime.queue_protocol import QueueRequest, QueueResponse


# ---------------------------------------------------------------------
# 小工具：把 QueueResponse / ExecResult 转成统一的 dict
# ---------------------------------------------------------------------


def _resp_to_dict(resp: QueueResponse) -> Dict[str, Any]:
    return {
        "ok": bool(resp.ok),
        "returncode": int(resp.returncode or 0),
        "stdout": resp.stdout or "",
        "stderr": resp.stderr or "",
        "runtime_sec": float(resp.runtime_sec or 0.0),
        "error": resp.error,
    }


def _exec_result_to_dict(res: Any) -> Dict[str, Any]:
    # ExecResult 定义在 queue_protocol 里，这里不强依赖类型
    return {
        "ok": bool(res.returncode == 0),
        "returncode": int(res.returncode),
        "stdout": getattr(res, "stdout", "") or "",
        "stderr": getattr(res, "stderr", "") or "",
        "runtime_sec": float(getattr(res, "runtime_sec", 0.0) or 0.0),
        "error": getattr(res, "error", None),
    }


# ---------------------------------------------------------------------
# 通过 ApptainerQueueRuntime 发送 instance_* 请求（start/exec/stop）
# 注意：这里会调用 ApptainerQueueRuntime 的“内部方法”，但在我们自己的项目中是 OK 的。
# ---------------------------------------------------------------------


def _instance_roundtrip(
    aq: ApptainerQueueRuntime,
    *,
    op_type: str,          # "instance_start" | "instance_exec" | "instance_stop"
    run_id: str,
    image: Optional[str],
    cmd: Optional[str],
    timeout: float,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> QueueResponse:
    """构造一个 QueueRequest 发给 runner，走长期 instance 逻辑。"""
    # 选一个 runner（内部是稳定 hash 到 [0, num_runners)）
    runner_id = aq._choose_runner(run_id)  # type: ignore[attr-defined]

    # 映射 docker image → SIF 路径（start / exec 需要，stop 可以忽略）
    sif_path: Optional[Path] = None
    if image:
        sif_path = aq._image_to_sif(image)  # type: ignore[attr-defined]

    # cmd: 对 exec 场景封装成 ["bash", "-lc", "<shell cmd>"]，其他 op 则为空
    if cmd is None:
        cmd_list: List[str] = []
    else:
        cmd_list = ["bash", "-lc", cmd]

    workdir = cwd or Path("/home")

    # 构造请求
    req = QueueRequest(
        req_id=aq._new_req_id(),  # type: ignore[attr-defined]
        runner_id=runner_id,
        run_id=run_id,
        type=op_type,
        image=image,
        sif_path=str(sif_path) if sif_path is not None else None,
        cmd=cmd_list,
        cwd=str(workdir),
        env=dict(env or {}),
        timeout_sec=float(timeout or aq.default_timeout_sec),
        src=None,
        dst=None,
        meta={},
    )

    # 发给对应 runner，并等待响应
    resp = aq._roundtrip(req)  # type: ignore[attr-defined]
    return resp


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"ok": False, "error": "empty stdin"}))
        return 1

    try:
        req = json.loads(raw)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"invalid json: {e}"}))
        return 1

    # 公共字段
    run_id = str(req.get("run_id") or "__default__")
    image = req.get("image")
    cmd = req.get("cmd")
    timeout = float(req.get("timeout") or 600.0)
    env = req.get("env") or {}
    cwd = Path(req.get("cwd") or "/home")

    # 多 runner 数量（与 Slurm 启动的 runner 数量保持一致）
    num_runners = int(os.environ.get("GP_NUM_RUNNERS", "1"))

    queue_root = Path.home() / "gp_queue"
    sif_dir = Path.home() / "sif" / "sweb"

    aq = ApptainerQueueRuntime(
        queue_root=queue_root,
        sif_dir=sif_dir,
        num_runners=num_runners,
        default_timeout_sec=timeout,
    )

    op = str(req.get("op") or "").strip().lower()

    # ---------------------------------------------------------------
    # 1) 长期 instance 模式：显式带 op = start / exec / stop
    # ---------------------------------------------------------------
    if op in {"start", "exec", "stop"}:
        if op in {"start", "exec"} and not image:
            print(json.dumps({"ok": False, "error": "image is required for op=start/exec"}))
            return 1

        if op == "start":
            resp = _instance_roundtrip(
                aq,
                op_type="instance_start",
                run_id=run_id,
                image=str(image),
                cmd=None,
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
        elif op == "exec":
            if not isinstance(cmd, str) or not cmd:
                print(json.dumps({"ok": False, "error": "cmd is required for op=exec"}))
                return 1
            resp = _instance_roundtrip(
                aq,
                op_type="instance_exec",
                run_id=run_id,
                image=str(image),
                cmd=str(cmd),
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
        else:  # op == "stop"
            resp = _instance_roundtrip(
                aq,
                op_type="instance_stop",
                run_id=run_id,
                image=None,
                cmd=None,
                timeout=timeout,
                env=None,
                cwd=cwd,
            )

        out = _resp_to_dict(resp)
        print(json.dumps(out, ensure_ascii=False))
        return 0

    # ---------------------------------------------------------------
    # 2) 兼容旧用法：不带 op → 一次性 exec（非长期 instance）
    # ---------------------------------------------------------------
    if not image:
        print(json.dumps({"ok": False, "error": "image is required when op is omitted"}))
        return 1

    if not isinstance(cmd, str) or not cmd:
        print(json.dumps({"ok": False, "error": "cmd is required when op is omitted"}))
        return 1

    exec_cmd = ["bash", "-lc", cmd]

    try:
        res = aq.exec(
            run_id=run_id,
            docker_image=str(image),
            cmd=exec_cmd,
            cwd=cwd,
            env=env,
            timeout_sec=timeout,
            meta={"src": "swe_proxy", "op": "exec_oneoff"},
        )
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"aq.exec failed: {e}"}))
        return 1

    out = _exec_result_to_dict(res)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
