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

SWE-bench Pro 原始数据可直接用 dockerhub_tag 字段；脚本会将它转换为：

    jefzda/sweap-images:{dockerhub_tag}

可以多次传 --dataset 参数（多个 jsonl 或目录），脚本会自动去重 docker 镜像。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
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

def _nested_get(record: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = record
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _pro_tag_to_image(tag: str, image_prefix: str) -> str:
    tag = tag.strip()
    prefix = image_prefix.strip().rstrip(":")
    if not tag:
        return ""
    if tag.startswith("docker://"):
        return tag[len("docker://") :].strip()
    if "/" in tag and ":" in tag:
        return tag
    return f"{prefix}:{tag}"


def _extract_docker_image(record: Dict[str, Any], pro_image_prefix: str = "jefzda/sweap-images") -> str | None:
    """
    尝试从一条记录中提取 docker 镜像字符串。

    优先支持 Graph Planner 转换后的格式：
        {"sandbox": {"docker_image": "...."}}

    然后回退支持几种常见的 SWE-bench / 其它字段命名：
        - docker_image
        - image_name
        - dockerhub_tag (SWE-bench Pro)
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
        cur = _nested_get(record, path)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()

    pro_tag_paths: List[List[str]] = [
        ["dockerhub_tag"],
        ["metadata", "dockerhub_tag"],
        ["sandbox", "dockerhub_tag"],
    ]
    for path in pro_tag_paths:
        cur = _nested_get(record, path)
        if isinstance(cur, str) and cur.strip():
            image = _pro_tag_to_image(cur, pro_image_prefix)
            if image:
                return image

    return None


def _collect_docker_images(datasets: Iterable[Path], pro_image_prefix: str = "jefzda/sweap-images") -> Set[str]:
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
                img = _extract_docker_image(rec, pro_image_prefix=pro_image_prefix)
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

def _mirror_candidates(cli_mirrors: Iterable[str] | None = None) -> List[str]:
    """Return Docker Hub mirror candidates in the order they should be tried."""

    candidates: List[str] = []
    for raw in list(cli_mirrors or []) + [
        os.environ.get("DKMR0", ""),
        os.environ.get("DKMR1", ""),
        # The PKU login nodes currently reach this mirror more reliably than
        # docker.1ms.run. Keep 1ms as a later fallback for other networks.
        "docker.1panel.live",
        "docker.m.daocloud.io",
        "docker.1ms.run",
        "dockerproxy.com",
    ]:
        value = str(raw or "").strip().rstrip("/")
        if not value or value in candidates:
            continue
        candidates.append(value)
    return candidates


def _docker_refs_for_image(docker_image: str, mirrors: Iterable[str]) -> List[str]:
    """Build concrete docker refs, adding mirrors only for Docker Hub-style refs."""

    docker_image = docker_image.strip()
    first = docker_image.split("/", 1)[0]
    if "." in first or ":" in first:
        return [docker_image]
    refs: List[str] = []
    for mirror in mirrors:
        ref = f"{mirror.rstrip('/')}/{docker_image}"
        if ref not in refs:
            refs.append(ref)
    if docker_image not in refs:
        refs.append(docker_image)
    return refs


def _build_sif(
    apptainer_bin: str,
    docker_image: str,
    sif_path: Path,
    apptainer_tmpdir: str | None = None,
    apptainer_cachedir: str | None = None,
    mirrors: Iterable[str] | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
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

    env = os.environ.copy()
    if apptainer_tmpdir:
        tmpdir = str(Path(apptainer_tmpdir).expanduser().resolve())
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        env["APPTAINER_TMPDIR"] = tmpdir
        env["SINGULARITY_TMPDIR"] = tmpdir
    if apptainer_cachedir:
        cachedir = str(Path(apptainer_cachedir).expanduser().resolve())
        Path(cachedir).mkdir(parents=True, exist_ok=True)
        env["APPTAINER_CACHEDIR"] = cachedir
        env["SINGULARITY_CACHEDIR"] = cachedir
    refs = _docker_refs_for_image(docker_image, mirrors or _mirror_candidates())
    attempts: List[Dict[str, Any]] = []
    part_path = sif_path.with_name(f"{sif_path.name}.part-{os.getpid()}")
    if part_path.exists() and not dry_run:
        part_path.unlink()

    for docker_ref in refs:
        cmd = [
            apptainer_bin,
            "build",
            str(part_path),
            f"docker://{docker_ref}",
        ]
        if dry_run:
            print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
            attempts.append({"docker_ref": docker_ref, "returncode": 0})
            continue

        if part_path.exists():
            part_path.unlink()
        print(f"[BUILD] {docker_image} (via {docker_ref}) -> {sif_path}", flush=True)
        proc = subprocess.run(cmd, env=env)
        attempts.append({"docker_ref": docker_ref, "returncode": proc.returncode})
        if proc.returncode == 0 and part_path.exists():
            part_path.replace(sif_path)
            return {"returncode": 0, "docker_ref": docker_ref, "attempts": attempts}

    if part_path.exists() and not dry_run:
        part_path.unlink()
    rc = attempts[-1]["returncode"] if attempts else 1
    return {
        "returncode": int(rc),
        "docker_ref": attempts[-1]["docker_ref"] if attempts else "",
        "attempts": attempts,
    }


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
        "--apptainer-tmpdir",
        type=str,
        help="Set APPTAINER_TMPDIR/SINGULARITY_TMPDIR for builds. Use a large filesystem, not $HOME.",
    )
    parser.add_argument(
        "--apptainer-cachedir",
        type=str,
        help="Set APPTAINER_CACHEDIR/SINGULARITY_CACHEDIR for builds. Use a large filesystem, not $HOME.",
    )
    parser.add_argument(
        "--mirror",
        action="append",
        default=[],
        help=(
            "Docker Hub mirror to try before env/default mirrors. Repeat to set fallback order, "
            "for example --mirror docker.1panel.live --mirror docker.m.daocloud.io."
        ),
    )
    parser.add_argument(
        "--min-existing-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Treat existing .sif files smaller than this as partial/corrupt and rebuild them.",
    )
    parser.add_argument(
        "--pro-image-prefix",
        type=str,
        default="jefzda/sweap-images",
        help="Docker image prefix used when a record has SWE-bench Pro dockerhub_tag (default: jefzda/sweap-images).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which images would be built, without actually building.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep building remaining images if one image fails.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Optional JSONL manifest path recording skip/build/failure status for each image.",
    )

    args = parser.parse_args()

    # 1) 解析数据集路径
    ds_paths = list(_iter_dataset_paths(args.dataset))
    if not ds_paths:
        raise SystemExit("No dataset files found.")

    print(f"[INFO] Scanning {len(ds_paths)} dataset file(s) for docker_image fields...")

    # 2) 收集所有 docker 镜像
    images = _collect_docker_images(ds_paths, pro_image_prefix=args.pro_image_prefix)
    if not images:
        print("[INFO] No docker_image fields found; nothing to do.")
        return

    print(f"[INFO] Found {len(images)} unique docker images.")

    # 3) 基于已存在的 SIF 文件，筛选出缺失的
    sif_root = Path(args.sif_dir).expanduser().resolve()
    sif_root.mkdir(parents=True, exist_ok=True)

    missing: List[tuple[str, Path]] = []
    manifest_records: List[Dict[str, Any]] = []
    mirrors = _mirror_candidates(args.mirror)
    print(f"[INFO] Docker Hub mirror order: {', '.join(mirrors)}")

    for img in sorted(images):
        sif_name = _normalize_sif_name(img)
        sif_path = sif_root / sif_name
        if sif_path.exists():
            size = sif_path.stat().st_size
            if size < int(args.min_existing_bytes):
                print(f"[STALE] {img} -> {sif_path} ({size} bytes); removing partial file")
                if not args.dry_run:
                    sif_path.unlink()
                missing.append((img, sif_path))
                manifest_records.append(
                    {
                        "image": img,
                        "sif_path": str(sif_path),
                        "status": "stale_removed",
                        "bytes": size,
                    }
                )
                continue
            print(f"[SKIP] {img} -> {sif_path} (already exists)")
            manifest_records.append(
                {
                    "image": img,
                    "sif_path": str(sif_path),
                    "status": "exists",
                    "bytes": size,
                }
            )
            continue
        missing.append((img, sif_path))

    if not missing:
        _write_manifest(args.manifest, manifest_records)
        print("[INFO] All required .sif images already exist.")
        return

    print(f"[INFO] {len(missing)} image(s) missing; building...")

    # 4) 逐个调用 apptainer/singularity build
    for img, sif_path in missing:
        started = time.time()
        build_result = _build_sif(
            apptainer_bin=args.apptainer_bin,
            docker_image=img,
            sif_path=sif_path,
            apptainer_tmpdir=args.apptainer_tmpdir,
            apptainer_cachedir=args.apptainer_cachedir,
            mirrors=mirrors,
            dry_run=args.dry_run,
        )
        rc = int(build_result.get("returncode", 1))
        record: Dict[str, Any] = {
            "image": img,
            "sif_path": str(sif_path),
            "returncode": rc,
            "elapsed_seconds": round(time.time() - started, 3),
            "status": "dry_run" if args.dry_run else ("built" if rc == 0 and sif_path.exists() else "failed"),
            "docker_ref": build_result.get("docker_ref", ""),
            "attempts": build_result.get("attempts", []),
        }
        if sif_path.exists():
            record["bytes"] = sif_path.stat().st_size
        manifest_records.append(record)
        _write_manifest(args.manifest, manifest_records)
        if rc != 0 and not args.continue_on_error:
            raise SystemExit(f"Apptainer/Singularity build failed for {img} (rc={rc})")

    failed = [record for record in manifest_records if record.get("status") == "failed"]
    if failed:
        print(f"[DONE] SIF preparation finished with {len(failed)} failed image(s).")
    else:
        print("[DONE] SIF preparation complete.")


def _write_manifest(path: str | None, records: List[Dict[str, Any]]) -> None:
    if not path:
        return
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
