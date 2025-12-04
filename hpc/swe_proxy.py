#!/usr/bin/env python
"""
简单代理：
  读取一条 JSON 请求，从 ApptainerQueueRuntime 发给 runner，
  再把执行结果 JSON 打印到 stdout。

支持四种 op：
  - "start": 启动一个 Apptainer instance（轨迹开始）
  - "exec" : 在已有 instance 中执行命令（轨迹中一步）
  - "stop" : 停止 instance（轨迹结束）
  - 其他/缺省: 一次性 exec（老行为）
"""

import os
import sys
import json
from pathlib import Path

from graph_planner.runtime.apptainer_queue_runtime import ApptainerQueueRuntime


def main() -> int:
    # 1. 从 stdin 读取整段 JSON
    data = sys.stdin.read()
    if not data.strip():
        print(json.dumps({"ok": False, "error": "empty stdin"}))
        return 1

    # 2. 解析 JSON
    try:
        req = json.loads(data)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"invalid json: {e}"}))
        return 1

    op = str(req.get("op") or "once")  # 默认保留一次性 exec 行为

    run_id = str(req.get("run_id") or "__default__")
    image = str(req.get("image") or "")
    cmd = str(req.get("cmd") or "echo hello")
    timeout = float(req.get("timeout") or 600.0)
    env = req.get("env") or {}
    workdir = Path(req.get("cwd") or "/home")

    if not image and op in ("start", "exec", "once"):
        print(json.dumps({"ok": False, "error": "missing 'image' in request"}))
        return 1

    # 3. 构造 ApptainerQueueRuntime
    queue_root = Path.home() / "gp_queue"
    sif_dir = Path.home() / "sif/sweb"

    num_runners_env = os.environ.get("GP_NUM_RUNNERS", "1") or "1"
    try:
        num_runners = int(num_runners_env)
    except ValueError:
        num_runners = 1

    aq = ApptainerQueueRuntime(
        queue_root=queue_root,
        sif_dir=sif_dir,
        num_runners=num_runners,
        default_timeout_sec=timeout,
    )

    try:
        # -------------------------
        # 4. 根据 op 选择调用模式
        # -------------------------
        if op == "start":
            # 启动 instance（只需 image / run_id / cwd / env）
            res = aq.start_instance(
                run_id=run_id,
                docker_image=image,
                cwd=workdir,
                env=env,
                timeout_sec=timeout,
                meta={"src": "swe_proxy", "op": "instance_start"},
            )

        elif op == "exec":
            # 在已有 instance 中执行命令
            exec_cmd = ["bash", "-lc", cmd]
            res = aq.exec_in_instance(
                run_id=run_id,
                docker_image=image,
                cmd=exec_cmd,
                cwd=workdir,
                env=env,
                timeout_sec=timeout,
                meta={"src": "swe_proxy", "op": "instance_exec"},
            )

        elif op == "stop":
            # 停止 instance（image 不再使用）
            res = aq.stop_instance(
                run_id=run_id,
                cwd=workdir,
                env=env,
                timeout_sec=timeout,
                meta={"src": "swe_proxy", "op": "instance_stop"},
            )

        else:
            # 老行为：一次性 exec 容器（无 instance 会话）
            exec_cmd = ["bash", "-lc", cmd]
            res = aq.exec(
                run_id=run_id,
                docker_image=image,
                cmd=exec_cmd,
                cwd=workdir,
                env=env,
                timeout_sec=timeout,
                meta={"src": "swe_proxy", "op": "exec_once"},
            )

    except Exception as e:
        print(json.dumps({"ok": False, "error": f"aq call failed: {e}"}))
        return 1

    # 5. 把结果封成 JSON 打印出来
    out = {
        "ok": res.returncode == 0,
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "runtime_sec": res.runtime_sec,
        "error": res.error,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

