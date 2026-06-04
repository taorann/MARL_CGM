from __future__ import annotations

from graphplanner_agent.repair.patch_schema import Patch
from graphplanner_agent.runtime.sandbox_base import SandboxRuntime, TestResult


def syntax_check_python(runtime: SandboxRuntime, patch: Patch) -> TestResult | None:
    py_paths = [p for p in patch.touched_paths if p.endswith(".py")]
    if not py_paths:
        return None
    cmd = "python -m py_compile " + " ".join(py_paths)
    result = runtime.run(cmd)
    status = "passed" if result.returncode == 0 else "syntax_failed"
    return TestResult(status, cmd, result.stdout, result.stderr, result.returncode)
