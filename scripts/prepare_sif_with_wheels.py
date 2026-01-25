#!/usr/bin/env python
"""
Prepare updated .sif images by downloading wheel-based packages (optional),
installing them into a sandbox, and rebundling the sandbox into a new .sif image.

This script mirrors the flow in scripts/prepare_sif_from_dataset.py (docker -> sif),
but operates on existing .sif images:
  1) Build a writable sandbox from the source .sif
  2) (Optional) Download wheels into a wheel directory
  3) Install packages offline from a wheel directory
  4) Rebuild a new .sif from the sandbox

Example:

    python scripts/prepare_sif_with_wheels.py \
        --sif-dir $HOME/sif/sweb \
        --in-place \
        --work-dir /local_scratch/$USER/sif_work \
        --wheel-dir /appsnew/home/chongbin_pkuhpc/chongbin_cls/wheels \
        --download-wheels \
        --packages pytest hypothesis \
        --limit 10 \
        --show-existing
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List


def _iter_sif_paths(sif_dir: Path) -> List[Path]:
    return sorted(p for p in sif_dir.glob("*.sif") if p.is_file())


def _load_requirements(path: Path) -> List[str]:
    reqs: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


def _check_existing_packages(
    apptainer_bin: str,
    sif_path: Path,
    packages: Iterable[str],
    env: dict[str, str],
) -> None:
    pkg_list = list(packages)
    if not pkg_list:
        return
    py = """\
import importlib.metadata as m
import sys

pkgs = sys.argv[1:]
for name in pkgs:
    try:
        ver = m.version(name)
        print(f"{name}=={ver}")
    except Exception:
        print(f"{name}==MISSING")
"""
    cmd = [
        apptainer_bin,
        "exec",
        str(sif_path),
        "python",
        "-c",
        py,
        *pkg_list,
    ]
    subprocess.run(cmd, check=False, env=env)


def _install_wheels(
    apptainer_bin: str,
    sandbox_dir: Path,
    wheel_dir: Path,
    packages: Iterable[str],
    env: dict[str, str],
) -> None:
    pkg_list = list(packages)
    if not pkg_list:
        return
    cmd = [
        apptainer_bin,
        "exec",
        "--bind",
        f"{wheel_dir}:/mnt/wheels",
        str(sandbox_dir),
        "python",
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links=/mnt/wheels",
        *pkg_list,
    ]
    subprocess.run(cmd, check=True, env=env)


def _download_wheels(
    wheel_dir: Path,
    packages: Iterable[str],
    env: dict[str, str],
) -> None:
    pkg_list = list(packages)
    if not pkg_list:
        return
    wheel_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "pip",
        "download",
        "-d",
        str(wheel_dir),
        *pkg_list,
    ]
    subprocess.run(cmd, check=True, env=env)


def _build_sandbox(
    apptainer_bin: str,
    src_sif: Path,
    sandbox_dir: Path,
    env: dict[str, str],
) -> None:
    cmd = [apptainer_bin, "build", "--sandbox", str(sandbox_dir), str(src_sif)]
    subprocess.run(cmd, check=True, env=env)


def _build_sif(
    apptainer_bin: str,
    src_sandbox: Path,
    out_sif: Path,
    env: dict[str, str],
) -> None:
    cmd = [apptainer_bin, "build", str(out_sif), str(src_sandbox)]
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install wheel-based packages into existing .sif images and rebuild them."
    )
    parser.add_argument(
        "--sif-dir",
        type=str,
        required=True,
        help="Directory containing source .sif images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store rebuilt .sif images (required unless --in-place).",
    )
    parser.add_argument(
        "--wheel-dir",
        type=str,
        required=True,
        help="Directory containing offline wheel files.",
    )
    parser.add_argument(
        "--download-wheels",
        action="store_true",
        help="Download wheels into --wheel-dir before rebuilding .sif images.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace .sif files in --sif-dir (ignores --output-dir).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for sandboxes and temp outputs (recommended on local scratch).",
    )
    parser.add_argument(
        "--apptainer-tmpdir",
        type=str,
        default=None,
        help="Override APPTAINER_TMPDIR/SINGULARITY_TMPDIR for builds.",
    )
    parser.add_argument(
        "--apptainer-cachedir",
        type=str,
        default=None,
        help="Override APPTAINER_CACHEDIR/SINGULARITY_CACHEDIR for builds.",
    )
    parser.add_argument(
        "--packages",
        nargs="*",
        default=["pytest", "hypothesis"],
        help="Packages to install from wheels (default: pytest hypothesis).",
    )
    parser.add_argument(
        "--requirements-file",
        type=str,
        default=None,
        help="Optional requirements file (one package per line). Overrides --packages.",
    )
    parser.add_argument(
        "--apptainer-bin",
        type=str,
        default="singularity",
        help="Apptainer/Singularity binary to use (default: singularity).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of .sif images to process (0 means all).",
    )
    parser.add_argument(
        "--show-existing",
        action="store_true",
        help="Print current versions of requested packages inside each .sif.",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="Keep the intermediate sandbox directories (default: remove).",
    )
    args = parser.parse_args()

    sif_dir = Path(args.sif_dir).expanduser().resolve()
    if args.in_place:
        output_dir = sif_dir
    elif args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        raise SystemExit("--output-dir is required unless --in-place is set.")
    wheel_dir = Path(args.wheel_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = output_dir

    env = os.environ.copy()
    if args.apptainer_tmpdir:
        env["APPTAINER_TMPDIR"] = args.apptainer_tmpdir
        env["SINGULARITY_TMPDIR"] = args.apptainer_tmpdir
    if args.apptainer_cachedir:
        env["APPTAINER_CACHEDIR"] = args.apptainer_cachedir
        env["SINGULARITY_CACHEDIR"] = args.apptainer_cachedir

    if not sif_dir.is_dir():
        raise SystemExit(f"SIF dir not found: {sif_dir}")
    if not wheel_dir.is_dir():
        if args.download_wheels:
            wheel_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise SystemExit(f"Wheel dir not found: {wheel_dir}")

    packages = args.packages
    if args.requirements_file:
        packages = _load_requirements(Path(args.requirements_file).expanduser().resolve())

    sif_paths = _iter_sif_paths(sif_dir)
    if not sif_paths:
        raise SystemExit(f"No .sif files found under {sif_dir}")

    if args.limit and args.limit > 0:
        sif_paths = sif_paths[: args.limit]

    if args.download_wheels:
        _download_wheels(wheel_dir, packages, env)

    for sif_path in sif_paths:
        out_sif = output_dir / sif_path.name
        temp_out_sif = out_sif
        if args.in_place:
            temp_out_sif = work_dir / f"{sif_path.stem}.tmp.sif"
        sandbox_dir = work_dir / f"{sif_path.stem}.sandbox"

        print(f"[INFO] Processing {sif_path.name}")
        if args.show_existing:
            _check_existing_packages(args.apptainer_bin, sif_path, packages, env)

        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir)

        _build_sandbox(args.apptainer_bin, sif_path, sandbox_dir, env)
        _install_wheels(args.apptainer_bin, sandbox_dir, wheel_dir, packages, env)
        _build_sif(args.apptainer_bin, sandbox_dir, temp_out_sif, env)
        if args.in_place and temp_out_sif != out_sif:
            os.replace(temp_out_sif, out_sif)

        if args.show_existing:
            _check_existing_packages(args.apptainer_bin, out_sif, packages, env)

        if not args.keep_sandbox:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

        print(f"[DONE] {sif_path.name} -> {out_sif}")


if __name__ == "__main__":
    main()
