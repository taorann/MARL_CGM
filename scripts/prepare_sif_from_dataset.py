#!/usr/bin/env python
"""
Scan Graph Planner JSONL datasets for docker_image fields and ensure corresponding
Apptainer/Singularity .sif images exist under a target directory.

用法（在 login24 上）示例：

    cd $HOME/MARL_CGM
    python scripts/prepare_sif_from_dataset.py \
        --dataset datasets/swebench/test.jsonl \
        --sif-dir $HOME/sif/sweb \
        --apptainer-bin singularity

可以多次传 --dataset 参数（多个 jsonl 或目录），脚本会自动去重 docker 镜像。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Set, Dict, Any, List


# ==========================
# 1. 数据集路径解析
# ==========================

def _iter_dataset_paths(raw_paths: Iterable[str]) -> Iterable[Path]:
    """
    接受一组字符串路径：
    - 如果是文件：直接返回；
    - 如果是目录：枚举目录下所有 *.jsonl 文件；
    - 否则报错。
    """
    for item in raw_paths:
        p = Path(item).expanduser().resolve()
        if p.is_file():
            yield p
        elif p.is_dir():
            # 简单约定：目录下的所有 JSONL 都视为数据集文件
            for jsonl in sorted(p.glob("*.jsonl")):
                yield jsonl
        else:
            raise FileNotFoundError(f"Dataset path does not exist: {p}")


# ==========================
# 2. 从 JSON 记录中提取 docker_image
# ==========================

def _extract_docker_image(record: Dict[str, Any]) -> str | None:
    """
    尝试从一条记录中提取 docker 镜像字符串。

    优先支持 Graph Planner 转换后的格式：
        {"sandbox": {"docker_image": "...."}}

    然后回退支持几种常见的 SWE-bench / 其它字段命名：
        - docker_image
        - image_name
        - metadata.docker_image / metadata.image_name
        - environment.docker_image / environment.image
    """
    # 1) Graph Planner 格式：sandbox.docker_image
    sbx = record.get("sandbox")
    if isinstance(sbx, dict):
        img = sbx.get("docker_image")
        if isinstance(img, str) and img.strip():
            return img.strip()

    # 2) 一些候选路径：平铺的和嵌套在 metadata / environment 里
    candidate_keys: List[List[str]] = [
        ["docker_image"],
        ["image_name"],
        ["metadata", "docker_image"],
        ["metadata", "image_name"],
        ["environment", "docker_image"],
        ["environment", "image"],
    ]
    for path in candidate_keys:
        cur: Any = record
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()

    return None


def _collect_docker_images(datasets: Iterable[Path]) -> Set[str]:
    """
    扫描一组 JSONL 数据集文件，收集所有出现过的 docker 镜像名（去重）。
    """
    images: Set[str] = set()
    for ds in datasets:
        with ds.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # 坏行直接跳过
                    continue
                img = _extract_docker_image(rec)
                if img:
                    images.add(img)
    return images


# ==========================
# 3. docker_image → .sif 文件名
# ==========================

def _normalize_sif_name(docker_image: str) -> str:
    """
    根据 docker 镜像名得到对应的 .sif 文件名。

    规则必须和 ApptainerQueueRuntime._image_to_sif 保持一致：
        - 把 / : @ 全部替换成 -
        - 后缀加上 .sif
    """
    normalized = (
        docker_image.replace("/", "-")
        .replace(":", "-")
        .replace("@", "-")
    )
    return f"{normalized}.sif"


# ==========================
# 4. 调用 Apptainer / Singularity 构建 .sif
# ==========================

def _build_sif(
    apptainer_bin: str,
    docker_image: str,
    sif_path: Path,
    dry_run: bool = False,
) -> None:
    """
    从 docker 镜像构建一个 .sif 文件。

    为了在北极星上避开 registry-1.docker.io 直连，这里内置 Docker Hub 镜像逻辑：
    - 优先用环境变量 DKMR0 / DKMR1；
    - 如果都没设，默认使用 docker.1ms.run；
    - 只对类似 "namespace/repo:tag" 这种「未显式带 registry 域名」的镜像加前缀，
      防止误改 ghcr.io / 自建 registry 等带域名的情况。
    """
    sif_path = sif_path.expanduser().resolve()
    sif_path.parent.mkdir(parents=True, exist_ok=True)

    # 选择 Docker 镜像站
    mirror = os.environ.get("DKMR0") or os.environ.get("DKMR1") or "docker.1ms.run/"
    mirror = mirror.rstrip("/")

    docker_ref = docker_image
    first = docker_image.split("/", 1)[0]
    # 如果镜像名最前面这一段不含 "." 或 ":"，基本是 docker.io 风格，给它挂镜像前缀
    if "." not in first and ":" not in first:
        docker_ref = f"{mirror}/{docker_image}"

    cmd = [
        apptainer_bin,
        "build",
        str(sif_path),
        f"docker://{docker_ref}",
    ]
    if dry_run:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return

    print(f"[BUILD] {docker_image} (via {docker_ref}) -> {sif_path}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(
            f"Apptainer/Singularity build failed for {docker_image} (rc={proc.returncode})"
        )


# ==========================
# 5. 主逻辑：扫描 JSONL、比对现有 SIF、构建缺失的
# ==========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare .sif images for Graph Planner SWE-bench runs."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Path to a Graph Planner JSONL dataset (or a directory containing JSONL files). "
            "Repeat this argument to scan multiple files / directories."
        ),
    )
    parser.add_argument(
        "--sif-dir",
        type=str,
        default=str(Path.home() / "sif" / "sweb"),
        help="Directory to store .sif images (default: ~/sif/sweb).",
    )
    parser.add_argument(
        "--apptainer-bin",
        type=str,
        default="singularity",
        help="Apptainer/Singularity binary to use (default: 'singularity').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which images would be built, without actually building.",
    )

    args = parser.parse_args()

    # 1) 解析数据集路径
    ds_paths = list(_iter_dataset_paths(args.dataset))
    if not ds_paths:
        raise SystemExit("No dataset files found.")

    print(f"[INFO] Scanning {len(ds_paths)} dataset file(s) for docker_image fields...")

    # 2) 收集所有 docker 镜像
    images = _collect_docker_images(ds_paths)
    if not images:
        print("[INFO] No docker_image fields found; nothing to do.")
        return

    print(f"[INFO] Found {len(images)} unique docker images.")

    # 3) 基于已存在的 SIF 文件，筛选出缺失的
    sif_root = Path(args.sif_dir).expanduser().resolve()
    sif_root.mkdir(parents=True, exist_ok=True)

    missing: List[tuple[str, Path]] = []
    for img in sorted(images):
        sif_name = _normalize_sif_name(img)
        sif_path = sif_root / sif_name
        if sif_path.exists():
            print(f"[SKIP] {img} -> {sif_path} (already exists)")
            continue
        missing.append((img, sif_path))

    if not missing:
        print("[INFO] All required .sif images already exist.")
        return

    print(f"[INFO] {len(missing)} image(s) missing; building...")

    # 4) 逐个调用 apptainer/singularity build
    for img, sif_path in missing:
        _build_sif(
            apptainer_bin=args.apptainer_bin,
            docker_image=img,
            sif_path=sif_path,
            dry_run=args.dry_run,
        )

    print("[DONE] SIF preparation complete.")


if __name__ == "__main__":
    main()

