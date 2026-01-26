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

Cache files (sandboxes, temporary outputs, and Apptainer temp/cache dirs) are
created under a ".sif-cache" subdirectory alongside the source .sif images and
cleaned up after each image is processed.

Example:

    python scripts/prepare_sif_with_wheels.py \
        --sif-dir $HOME/sif/sweb \
        --in-place \
        --wheel-dir /appsnew/home/chongbin_pkuhpc/chongbin_cls/wheels \
        --output-dir /home/chongbin_pkuhpc/chongbin_cls/lustre1 \
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


_STALE_NFS_MARKERS = ("stale NFS file handle",)
_DEFAULT_OUTPUT_ROOT = Path("/home/chongbin_pkuhpc/chongbin_cls/lustre1/sif/sweb")


def _is_stale_nfs_error(output: str) -> bool:
    haystack = output.lower()
    return any(marker in haystack for marker in _STALE_NFS_MARKERS)


def _run_with_retry(cmd: List[str], env: dict[str, str], retries: int) -> None:
    attempt = 0
    while True:
        attempt += 1
        result = subprocess.run(
            cmd,
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", flush=True)
            return
        combined = (result.stdout or "") + (result.stderr or "")
        if _is_stale_nfs_error(combined) and attempt <= retries:
            print(
                f"[WARN] Detected stale NFS handle (attempt {attempt}/{retries}), retrying..."
            )
            continue
        if combined:
            print(combined, end="")
        raise subprocess.CalledProcessError(result.returncode, cmd)


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
    _run_with_retry(cmd, env, retries=0)


def _install_wheels(
    apptainer_bin: str,
    sandbox_dir: Path,
    wheel_dir: Path,
    packages: Iterable[str],
    env: dict[str, str],
    retries: int,
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
    _run_with_retry(cmd, env, retries=retries)


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
    _run_with_retry(cmd, env, retries=0)


def _build_sandbox(
    apptainer_bin: str,
    src_sif: Path,
    sandbox_dir: Path,
    env: dict[str, str],
    retries: int,
) -> None:
    cmd = [apptainer_bin, "build", "--sandbox", str(sandbox_dir), str(src_sif)]
    _run_with_retry(cmd, env, retries=retries)


def _build_sif(
    apptainer_bin: str,
    src_sandbox: Path,
    out_sif: Path,
    env: dict[str, str],
    retries: int,
) -> None:
    cmd = [apptainer_bin, "build", str(out_sif), str(src_sandbox)]
    _run_with_retry(cmd, env, retries=retries)


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
        help=(
            "Directory to store rebuilt .sif images (default: "
            f"{_DEFAULT_OUTPUT_ROOT})."
        ),
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
        help="Replace .sif files in --output-dir when rebuilding.",
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
        "--nfs-retries",
        type=int,
        default=2,
        help="Retries per step when stale NFS handle is detected (default: 2).",
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
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _DEFAULT_OUTPUT_ROOT
    )
    wheel_dir = Path(args.wheel_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / ".sif-cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

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
        cache_dir = cache_root / sif_path.stem
        sandbox_dir = cache_dir / "sandbox"
        apptainer_tmpdir = cache_dir / "apptainer_tmp"
        apptainer_cachedir = cache_dir / "apptainer_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        apptainer_tmpdir.mkdir(parents=True, exist_ok=True)
        apptainer_cachedir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["APPTAINER_TMPDIR"] = args.apptainer_tmpdir or str(apptainer_tmpdir)
        env["SINGULARITY_TMPDIR"] = env["APPTAINER_TMPDIR"]
        env["APPTAINER_CACHEDIR"] = args.apptainer_cachedir or str(apptainer_cachedir)
        env["SINGULARITY_CACHEDIR"] = env["APPTAINER_CACHEDIR"]

        out_sif = output_dir / sif_path.name
        temp_out_sif = out_sif
        if args.in_place:
            temp_out_sif = cache_dir / f"{sif_path.stem}.tmp.sif"

        print(f"[INFO] Processing {sif_path.name}")
        if args.show_existing:
            _check_existing_packages(args.apptainer_bin, sif_path, packages, env)

        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir)

        try:
            _build_sandbox(args.apptainer_bin, sif_path, sandbox_dir, env, args.nfs_retries)
            _install_wheels(
                args.apptainer_bin,
                sandbox_dir,
                wheel_dir,
                packages,
                env,
                args.nfs_retries,
            )
            _build_sif(args.apptainer_bin, sandbox_dir, temp_out_sif, env, args.nfs_retries)
            if args.in_place and temp_out_sif != out_sif:
                os.replace(temp_out_sif, out_sif)

            if args.show_existing:
                _check_existing_packages(args.apptainer_bin, out_sif, packages, env)

        finally:
            if not args.keep_sandbox:
                shutil.rmtree(cache_dir, ignore_errors=True)
            else:
                shutil.rmtree(apptainer_tmpdir, ignore_errors=True)
                shutil.rmtree(apptainer_cachedir, ignore_errors=True)
                if temp_out_sif.exists() and temp_out_sif != out_sif:
                    temp_out_sif.unlink()

        print(f"[DONE] {sif_path.name} -> {out_sif}")


if __name__ == "__main__":
    main()
