#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict verification of a BEHAVIOR-1K install in the ASPIRE sim workspace.

Runs the checks that actually distinguish a working BEHAVIOR environment from a
plausible-looking one:

  * GPU, driver, and the validated dependency pins
  * Isaac Sim / OmniGibson imports
  * curobo CUDA extension import (the compiled kernels, not just the package)
  * dataset presence, including the custom r1pro_ik.urdf overlay
  * SAM3 and Contact-GraspNet server startup on real sockets
  * one exact soda-can oracle seed, end to end, with video

Exits non-zero if any check fails, and writes an environment manifest.

Usage:
    python scripts/behavior/verify_behavior.py [--gpu-id 0] [--quick]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

SIM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SIM_ROOT.parents[1]
B1K_ROOT = SIM_ROOT / "cap" / "third_party" / "b1k"

ORACLE_CONFIG = SIM_ROOT / "env_configs" / "r1pro" / "r1pro_pick_up_trash_oracle.yaml"

# Validated stack; see docs/behavior-tasks.md § Tested configuration.
EXPECTED_TORCH = "2.6.0+cu124"
EXPECTED_TORCH_CUDA_MAJOR = "12"

GREEN, RED, YELLOW, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"


class Results:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}

    def record(self, name: str, ok: bool, detail: str, warn: bool = False) -> None:
        self.checks.append({"name": name, "ok": ok, "detail": detail, "warn": warn})
        if ok:
            tag = f"{YELLOW}WARN{RESET}" if warn else f"{GREEN}PASS{RESET}"
        else:
            tag = f"{RED}FAIL{RESET}"
        print(f"  [{tag}] {name}: {detail}", flush=True)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["ok"]]


def section(title: str) -> None:
    print(f"\n{BOLD}== {title}{RESET}", flush=True)


def run_check(res: Results, name: str, fn: Callable[[], str]) -> bool:
    """Run fn; it returns a detail string on success or raises on failure."""
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - verification reports every failure
        res.record(name, False, f"{type(exc).__name__}: {exc}")
        return False
    res.record(name, True, detail)
    return True


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_gpu(res: Results) -> None:
    section("GPU and driver")

    def _driver() -> str:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        gpus = [line.strip() for line in out.splitlines() if line.strip()]
        if not gpus:
            raise RuntimeError("nvidia-smi reported no GPUs")
        res.manifest["gpus"] = gpus
        res.manifest["driver_version"] = gpus[0].split(",")[-1].strip()
        return f"{len(gpus)} GPU(s); driver {res.manifest['driver_version']}"

    run_check(res, "nvidia-smi", _driver)

    def _torch_cuda() -> str:
        import torch

        res.manifest["torch"] = torch.__version__
        res.manifest["torch_cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        if torch.__version__ != EXPECTED_TORCH:
            raise RuntimeError(
                f"torch {torch.__version__}, expected {EXPECTED_TORCH}. "
                "A cu13 torch cannot build curobo against a CUDA 12 toolkit."
            )
        if not (torch.version.cuda or "").startswith(EXPECTED_TORCH_CUDA_MAJOR):
            raise RuntimeError(f"torch CUDA {torch.version.cuda}, expected 12.x")
        return f"torch {torch.__version__} (CUDA {torch.version.cuda}), {torch.cuda.device_count()} device(s)"

    run_check(res, "torch + CUDA", _torch_cuda)


def check_pins(res: Results) -> None:
    section("Dependency pins")

    def _numpy() -> str:
        import numpy

        res.manifest["numpy"] = numpy.__version__
        if int(numpy.__version__.split(".")[0]) >= 2:
            raise RuntimeError(f"numpy {numpy.__version__}; OmniGibson requires numpy<2")
        return numpy.__version__

    def _setuptools() -> str:
        import setuptools

        res.manifest["setuptools"] = setuptools.__version__
        if int(setuptools.__version__.split(".")[0]) >= 81:
            raise RuntimeError(
                f"setuptools {setuptools.__version__} removed pkg_resources, which sam3 imports"
            )
        return setuptools.__version__

    def _cv2() -> str:
        import cv2

        return f"OpenCV {cv2.__version__}"

    run_check(res, "numpy<2", _numpy)
    run_check(res, "setuptools<81", _setuptools)
    run_check(res, "cv2 importable", _cv2)


def check_imports(res: Results) -> None:
    section("Simulator and perception imports")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    def _isaac() -> str:
        import isaacsim  # noqa: F401
        import omnigibson

        res.manifest["omnigibson"] = getattr(omnigibson, "__version__", "unknown")
        return f"omnigibson {res.manifest['omnigibson']} + isaacsim"

    def _curobo() -> str:
        # The compiled extension, not just the Python package: this is what a
        # torch/CUDA-toolkit mismatch actually breaks.
        from curobo.curobolib import geom_cu  # noqa: F401
        from curobo.curobolib import lbfgs_step_cu  # noqa: F401

        return "curobo CUDA extensions import"

    def _sam3() -> str:
        import sam3  # noqa: F401

        return "sam3"

    def _graspnet() -> str:
        import contact_graspnet_pytorch  # noqa: F401

        return "contact_graspnet_pytorch"

    def _aspire() -> str:
        import aspire.sim.cap.envs.launch_b1k  # noqa: F401
        import aspire.sim.cap.serving.launch_contact_graspnet_server  # noqa: F401
        import aspire.sim.cap.serving.launch_sam3_server  # noqa: F401

        return "aspire.sim.cap entrypoints"

    run_check(res, "isaacsim + omnigibson", _isaac)
    run_check(res, "curobo CUDA extensions", _curobo)
    run_check(res, "sam3", _sam3)
    run_check(res, "contact_graspnet_pytorch", _graspnet)
    run_check(res, "aspire entrypoints", _aspire)


def check_datasets(res: Results) -> None:
    section("Datasets")

    def _data_path() -> str:
        from omnigibson.macros import gm

        data_path = Path(gm.DATA_PATH)
        res.manifest["data_path"] = str(data_path)
        required = {
            "behavior-1k-assets": data_path / "behavior-1k-assets",
            "omnigibson-robot-assets": data_path / "omnigibson-robot-assets",
            "2025-challenge-task-instances": data_path / "2025-challenge-task-instances",
        }
        missing = [name for name, path in required.items() if not path.is_dir()]
        if missing:
            raise RuntimeError(
                f"missing dataset(s): {', '.join(missing)}. "
                "Re-run scripts/behavior/setup_behavior.sh --accept-dataset-license"
            )

        # Directory existence is not enough for omnigibson-robot-assets: the b1k
        # submodule git-tracks the r1pro_ik.urdf overlay inside it, so the tree
        # exists on a fresh clone even when the download never ran. Require the
        # R1Pro USD the simulator actually loads.
        r1pro_usd = required["omnigibson-robot-assets"] / "models" / "r1pro" / "usd" / "r1pro.usda"
        if not r1pro_usd.is_file():
            raise RuntimeError(
                f"{r1pro_usd} missing: omnigibson-robot-assets exists but holds no downloaded "
                "payload (the git-tracked r1pro_ik.urdf overlay creates the directory). "
                "Re-run scripts/behavior/setup_behavior.sh --accept-dataset-license"
            )
        return ", ".join(sorted(required))

    def _ik_urdf() -> str:
        from omnigibson.macros import gm

        urdf = Path(gm.DATA_PATH) / "omnigibson-robot-assets" / "models" / "r1pro" / "urdf" / "r1pro_ik.urdf"
        if not urdf.is_file():
            raise RuntimeError(
                f"{urdf} missing; R1Pro IK will not work. It is easy to lose because "
                "the upstream installer deletes the asset tree before re-downloading."
            )
        return f"{urdf.name} ({urdf.stat().st_size} bytes)"

    run_check(res, "dataset directories", _data_path)
    run_check(res, "r1pro_ik.urdf overlay", _ik_urdf)


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        with contextlib.closing(socket.socket()) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2.0)
    raise TimeoutError(f"port {port} did not open within {timeout:.0f}s")


def _check_server(res: Results, name: str, module: str, timeout: float) -> None:
    def _run() -> str:
        port = _free_port()
        cmd = [sys.executable, "-m", module, "--port", str(port), "--host", "127.0.0.1", "--device", "cuda"]
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        try:
            started = time.time()
            _wait_for_port(port, proc, timeout)
            return f"listening on 127.0.0.1:{port} after {time.time() - started:.0f}s"
        except Exception:
            proc.kill()
            output = (proc.stdout.read() if proc.stdout else "") or ""
            raise RuntimeError(f"startup failed; last output:\n{output[-1500:]}") from None
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()

    run_check(res, name, _run)


def check_servers(res: Results) -> None:
    section("Perception servers")
    # SAM3 loads several GB of weights on first start.
    _check_server(res, "SAM3 server", "aspire.sim.cap.serving.launch_sam3_server", timeout=600)
    _check_server(res, "Contact-GraspNet server",
                  "aspire.sim.cap.serving.launch_contact_graspnet_server", timeout=600)


def check_oracle_seed(res: Results, gpu_id: int, output_dir: Path, attempts: int) -> None:
    section("Soda-can oracle seed (end to end)")

    def _run() -> str:
        last_error = ""
        for attempt in range(1, attempts + 1):
            run_dir = output_dir / f"attempt_{attempt}"
            cmd = [
                sys.executable, "-m", "aspire.sim.cap.envs.launch_b1k",
                "--config-path", str(ORACLE_CONFIG),
                "--trial-ids", "1",
                "--output-dir", str(run_dir),
                "--record-video", "True",
            ]
            env = dict(
                os.environ,
                OMNI_KIT_ACCEPT_EULA="YES",
                OMNIGIBSON_HEADLESS="1",
                OMNIGIBSON_GPU_ID=str(gpu_id),
            )
            print(f"  attempt {attempt}/{attempts}: launching trial 1 on GPU {gpu_id} "
                  f"(several minutes)...", flush=True)
            proc = subprocess.run(cmd, cwd=str(SIM_ROOT), env=env,
                                  capture_output=True, text=True)

            # The launcher writes to <output-dir parent>/<model-name>/<basename>/,
            # interposing a model directory, so globbing under run_dir itself finds
            # nothing. Verified identical in the CaP-X control at 53e9966d.
            trial_dirs = sorted(
                d for d in output_dir.glob(f"**/{run_dir.name}/**/trial_01_*") if d.is_dir()
            )
            if not trial_dirs:
                last_error = (f"no trial directory produced (exit {proc.returncode}); "
                              f"stderr tail:\n{proc.stderr[-1500:]}")
                continue
            trial_dir = trial_dirs[0]
            videos = sorted(trial_dir.glob("video_*.mp4"))
            nonempty = [v for v in videos if v.stat().st_size > 0]
            completed = "taskcompleted_1" in trial_dir.name

            # Task success and clean process exit are separate facts. The launcher
            # segfaults at teardown (139 / rc -11) on this node *after* writing
            # artifacts; the pinned CaP-X control does the same, so it is inherited
            # upstream, not an ASPIRE regression. Report it, never silently pass it.
            exit_note = ""
            if proc.returncode != 0:
                exit_note = f"; launcher exit {proc.returncode} (teardown, artifacts intact)"
                res.manifest.setdefault("launcher_nonzero_exits", []).append(
                    {"attempt": attempt, "returncode": proc.returncode}
                )

            if not nonempty:
                last_error = f"{trial_dir.name}: no non-empty video written"
                continue
            if not completed:
                # Cold-start perception flake; reproduced on the FIRST oracle run of
                # the pinned CaP-X control too, so it is inherited, not ASPIRE-side.
                # See docs/behavior-tasks.md § Known issues.
                last_error = f"{trial_dir.name}: task not completed{exit_note}"
                print(f"  {YELLOW}retrying{RESET}: {last_error}", flush=True)
                continue

            res.manifest["oracle_trial"] = {
                "dir": str(trial_dir.relative_to(SIM_ROOT)),
                "videos": [v.name for v in nonempty],
                "attempt": attempt,
                "launcher_returncode": proc.returncode,
            }
            return (f"{trial_dir.name} on attempt {attempt}; "
                    f"{len(nonempty)} video(s){exit_note}")
        raise RuntimeError(last_error or "oracle seed failed")

    run_check(res, "oracle trial 1", _run)


def write_manifest(res: Results, path: Path) -> None:
    def _git_rev(repo: Path) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    res.manifest["revisions"] = {
        "aspire": _git_rev(REPO_ROOT),
        "b1k": _git_rev(B1K_ROOT),
        "sam3": _git_rev(SIM_ROOT / "cap" / "third_party" / "sam3"),
        "contact_graspnet_pytorch": _git_rev(SIM_ROOT / "cap" / "third_party" / "contact_graspnet_pytorch"),
    }
    res.manifest["python"] = sys.version.split()[0]
    res.manifest["venv"] = sys.prefix
    res.manifest["checks"] = res.checks
    res.manifest["verified"] = not res.failed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res.manifest, indent=2) + "\n")
    print(f"\nManifest: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-id", type=int, default=int(os.environ.get("OMNIGIBSON_GPU_ID", "0")),
                        help="GPU used by Isaac Sim for the oracle seed")
    parser.add_argument("--quick", action="store_true",
                        help="Skip the perception servers and the oracle seed")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip only the oracle seed")
    parser.add_argument("--attempts", type=int, default=2,
                        help="Oracle seed attempts before failing (perception can flake)")
    parser.add_argument("--output-dir", type=Path,
                        default=SIM_ROOT / "outputs" / "behavior" / "verify",
                        help="Where the oracle verification trial is written")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest path (default: <output-dir>/environment_manifest.json)")
    args = parser.parse_args()

    print(f"{BOLD}BEHAVIOR-1K environment verification{RESET}")
    print(f"sim root: {SIM_ROOT}")
    print(f"python:   {sys.executable}")

    res = Results()
    check_gpu(res)
    check_pins(res)
    check_imports(res)
    check_datasets(res)

    if args.quick:
        print(f"\n{YELLOW}--quick: skipping perception servers and oracle seed{RESET}")
    else:
        check_servers(res)
        if args.skip_oracle:
            print(f"\n{YELLOW}--skip-oracle: skipping the oracle seed{RESET}")
        else:
            check_oracle_seed(res, args.gpu_id, args.output_dir, max(1, args.attempts))

    manifest_path = args.manifest or (args.output_dir / "environment_manifest.json")
    write_manifest(res, manifest_path)

    section("Result")
    if res.failed:
        for check in res.failed:
            print(f"  {RED}FAILED{RESET} {check['name']}: {check['detail']}")
        print(f"\n{RED}Verification FAILED ({len(res.failed)} check(s)).{RESET}")
        return 1
    scope = "quick" if args.quick else ("no oracle seed" if args.skip_oracle else "full")
    print(f"{GREEN}All checks passed ({scope}).{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
