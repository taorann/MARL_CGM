from pathlib import Path
from graph_planner.runtime.apptainer_queue_runtime import ApptainerQueueRuntime

queue_root = Path.home() / "gp_queue"
sif_dir    = Path.home() / "sif/sweb"

aq = ApptainerQueueRuntime(
    queue_root=queue_root,
    sif_dir=sif_dir,
    num_runners=1,
    default_timeout_sec=600,
)

# 使用某个已存在的 SWE 镜像；如果已经有 .sif，就不会重新拉
docker_image = "ghcr.1ms.run/epoch-research/swe-bench.eval.x86_64.astropy__astropy-12907:latest"

result = aq.exec(
    run_id="local-test",
    docker_image=docker_image,
    cmd=["bash", "-lc", "cd /testbed && pwd && ls | head"],
    cwd=Path("/home"),   # runner 宿主侧工作目录，只要存在即可
    env={},
    timeout_sec=300.0,
)

print("returncode:", result.returncode)
print("stdout:\n", result.stdout)
print("stderr:\n", result.stderr)
