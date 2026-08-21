#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an immutable, resumable LIBERO Fix Loop held-out validation.

A run identity hashes the code, config, suite, task, and requested seeds. Results
from different identities are never mixed. ``--resume`` continues only the
matching manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRIAL_RE = re.compile(r"trial_(\d+)_sandboxrc_(\d+)_reward_([\d.]+)_taskcompleted_(\d+)")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_identity(*, suite: str, task: str, fix_code: Path, config: Path, seeds: list[int]) -> dict:
    return {
        "suite": suite,
        "task": task,
        "code_sha256": sha256_file(fix_code),
        "config_sha256": sha256_file(config),
        "seeds": sorted(set(seeds)),
    }


def run_id_for_identity(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def is_full_heldout(identity: dict) -> bool:
    return identity.get("seeds") == list(range(1, 51))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--fix-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/libero_fix_loop_eval"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 51)))
    parser.add_argument("--resume", action="store_true", help="Continue this exact run identity")
    parser.add_argument("--config", type=Path, default=Path("env_configs/libero/franka_libero_traced.yaml"))
    return parser.parse_args()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def latest_trial(task_output: Path, seed: int) -> dict | None:
    matches: list[tuple[float, Path, re.Match[str]]] = []
    if not task_output.exists():
        return None
    for path in task_output.rglob(f"trial_{seed:02d}_sandboxrc_*_reward_*_taskcompleted_*"):
        match = TRIAL_RE.fullmatch(path.name)
        if path.is_dir() and match:
            matches.append((path.stat().st_mtime, path, match))
    if not matches:
        return None
    _, path, match = max(matches, key=lambda item: item[0])
    return {
        "seed": seed,
        "sandbox_rc": int(match.group(2)),
        "reward": float(match.group(3)),
        "task_completed": int(match.group(4)),
        "trial_dir": str(path),
    }


def update_stage1_validation(fix_code: Path, manifest: dict, manifest_path: Path) -> None:
    stage1_path = fix_code.parent / "stage1_result.json"
    validation_path = fix_code.parent / "validation_result.json"
    summary = {
        "schema_version": 1,
        "state": "validated",
        "run_id": manifest["run_id"],
        "manifest": str(manifest_path),
        "code_sha256": manifest["identity"]["code_sha256"],
        "seeds": manifest["identity"]["seeds"],
        "passes": manifest["passes"],
        "trials": manifest["trials"],
        "pass_rate": manifest["pass_rate"],
        "validated_at": manifest["updated_at"],
    }
    write_json_atomic(validation_path, summary)
    if not stage1_path.exists():
        return
    stage1 = json.loads(stage1_path.read_text())
    if stage1.get("code_sha256") != summary["code_sha256"]:
        return
    stage1["validation_state"] = "validated"
    stage1["validation"] = summary
    stage1["updated_at"] = now()
    write_json_atomic(stage1_path, stage1)


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    fix_code = args.fix_code.resolve()
    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    seeds = sorted(set(args.seeds))

    if not fix_code.is_file():
        raise SystemExit(f"fix code does not exist: {fix_code}")
    if not config.is_file():
        raise SystemExit(f"config does not exist: {config}")
    invalid = [seed for seed in seeds if not 1 <= seed <= 50]
    if invalid:
        raise SystemExit(f"held-out seeds must be in 1..50: {invalid}")

    identity = build_identity(
        suite=args.suite, task=args.task, fix_code=fix_code, config=config, seeds=seeds
    )
    run_id = run_id_for_identity(identity)
    run_dir = output_dir / args.suite / args.task / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    results_base = run_dir / "results"
    task_output = results_base / args.suite / args.task
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise SystemExit(f"manifest identity mismatch: {manifest_path}")
        if not args.resume:
            raise SystemExit(f"immutable run already exists; pass --resume: {manifest_path}")
    else:
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "identity": identity,
            "fix_code_path": str(fix_code),
            "config_path": str(config),
            "git_commit": git_commit(),
            "created_at": now(),
            "updated_at": now(),
            "status": "running",
            "evidence_scope": (
                "heldout_full" if is_full_heldout(identity) else "heldout_subset"
            ),
            "results": {},
            "passes": 0,
            "trials": 0,
            "pass_rate": 0.0,
        }
        write_json_atomic(manifest_path, manifest)

    env = os.environ.copy()
    env.update(
        MUJOCO_GL="egl",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    missing_artifacts = 0
    for seed in seeds:
        key = str(seed)
        if key in manifest["results"]:
            print(f"seed {seed:02d}: SKIP matching manifest")
            continue
        log_path = logs / f"seed_{seed:02d}.log"
        command = [
            str(ROOT / ".venv-libero/bin/python3"),
            "scripts/libero/replay_trial.py",
            "--args.suite", args.suite,
            "--args.task", args.task,
            "--args.trial", str(seed),
            "--args.replay-code", str(fix_code),
            "--args.config", str(config),
            "--args.output-dir", str(results_base),
        ]
        print(f"seed {seed:02d}: RUN -> {log_path}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
            )
        trial = latest_trial(task_output, seed)
        if trial is None:
            missing_artifacts += 1
            print(f"seed {seed:02d}: NO ARTIFACT exit={process.returncode}", flush=True)
            continue
        trial["process_exit_code"] = process.returncode
        trial["recorded_at"] = now()
        manifest["results"][key] = trial
        manifest["trials"] = len(manifest["results"])
        manifest["passes"] = sum(
            int(row["task_completed"]) for row in manifest["results"].values()
        )
        manifest["pass_rate"] = round(manifest["passes"] / manifest["trials"], 6)
        manifest["updated_at"] = now()
        write_json_atomic(manifest_path, manifest)
        print(
            f"seed {seed:02d}: reward={trial['reward']:.3f} "
            f"completed={trial['task_completed']} exit={process.returncode}",
            flush=True,
        )

    manifest["trials"] = len(manifest["results"])
    manifest["passes"] = sum(
        int(row["task_completed"]) for row in manifest["results"].values()
    )
    manifest["pass_rate"] = round(
        manifest["passes"] / manifest["trials"], 6
    ) if manifest["trials"] else 0.0
    manifest["status"] = "complete" if manifest["trials"] == len(seeds) else "partial"
    manifest["updated_at"] = now()
    write_json_atomic(manifest_path, manifest)

    if (
        manifest["status"] == "complete"
        and is_full_heldout(manifest["identity"])
    ):
        update_stage1_validation(fix_code, manifest, manifest_path)
    subprocess.run(
        [str(ROOT / ".venv-libero/bin/python3"), "scripts/libero/gen_progress.py"],
        check=False,
    )
    print(
        f"run={run_id} status={manifest['status']} passes={manifest['passes']}/"
        f"{manifest['trials']} manifest={manifest_path}"
    )
    return 1 if missing_artifacts or manifest["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
