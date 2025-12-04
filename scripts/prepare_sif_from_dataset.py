#!/usr/bin/env python
"""
Scan Graph Planner JSONL datasets for docker_image fields and ensure corresponding
Apptainer/Singularity .sif images exist under a target directory.

Usage (on login24):

    cd $HOME/MARL_CGM
    python scripts/prepare_sif_from_dataset.py \
        --dataset datasets/swebench/test.jsonl \
        --sif-dir $HOME/sif/sweb \
        --apptainer-bin singularity

You can pass multiple --dataset arguments if needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable, Set, Dict, Any, List


def _iter_dataset_paths(raw_paths: Iterable[str]) -> Iterable[Path]:
    for item in raw_paths:
        p = Path(item).expanduser().resolve()
        if p.is_file():
            yield p
        elif p.is_dir():
            # Heuristic: pick *.jsonl files in the directory
            for jsonl in sorted(p.glob("*.jsonl")):
                yield jsonl
        else:
            raise FileNotFoundError(f"Dataset path does not exist: {p}")


def _extract_docker_image(record: Dict[str, Any]) -> str | None:
    # Prefer sandbox.docker_image when present (Graph Planner format)
    sbx = record.get("sandbox")
    if isinstance(sbx, dict):
        img = sbx.get("docker_image")
        if isinstance(img, str) and img.strip():
            return img.strip()

    # Fallbacks for raw SWE-bench style entries
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
                    continue
                img = _extract_docker_image(rec)
                if img:
                    images.add(img)
    return images


def _normalize_sif_name(docker_image: str) -> str:
    # Must match ApptainerQueueRuntime._image_to_sif
    normalized = (
        docker_image.replace("/", "-")
        .replace(":", "-")
        .replace("@", "-")
    )
    return f"{normalized}.sif"


def _build_sif(apptainer_bin: str, docker_image: str, sif_path: Path, dry_run: bool = False) -> None:
    sif_path = sif_path.expanduser().resolve()
    sif_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        apptainer_bin,
        "build",
        str(sif_path),
        f"docker://{docker_image}",
    ]
    if dry_run:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return

    print(f"[BUILD] {docker_image} -> {sif_path}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Apptainer/Singularity build failed for {docker_image} (rc={proc.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare .sif images for Graph Planner SWE-bench runs.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Path to a Graph Planner JSONL dataset (or a directory containing JSONL files). "
             "Repeatable.",
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
    ds_paths = list(_iter_dataset_paths(args.dataset))
    if not ds_paths:
        raise SystemExit("No dataset files found.")

    print(f"[INFO] Scanning {len(ds_paths)} dataset file(s) for docker_image fields...")
    images = _collect_docker_images(ds_paths)
    if not images:
        print("[INFO] No docker_image fields found; nothing to do.")
        return

    print(f"[INFO] Found {len(images)} unique docker images.")
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
    for img, sif_path in missing:
        _build_sif(args.apptainer_bin, img, sif_path, dry_run=args.dry_run)

    print("[DONE] SIF preparation complete.")


if __name__ == "__main__":
    main()
