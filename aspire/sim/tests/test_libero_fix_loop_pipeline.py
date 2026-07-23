# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "libero"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_identity_changes_with_code_or_config(tmp_path: Path):
    validation = load_script("run_fix_loop_validation")
    code = tmp_path / "fix.py"
    config = tmp_path / "config.yaml"
    code.write_text("get_observation()\n")
    config.write_text("env: {}\n")

    identity = validation.build_identity(
        suite="suite", task="task", fix_code=code, config=config, seeds=[2, 1, 2]
    )
    first = validation.run_id_for_identity(identity)
    assert identity["seeds"] == [1, 2]

    code.write_text("get_observation()\nopen_gripper()\n")
    changed_code = validation.build_identity(
        suite="suite", task="task", fix_code=code, config=config, seeds=[1, 2]
    )
    assert validation.run_id_for_identity(changed_code) != first

    code.write_text("get_observation()\n")
    config.write_text("env: {changed: true}\n")
    changed_config = validation.build_identity(
        suite="suite", task="task", fix_code=code, config=config, seeds=[1, 2]
    )
    assert validation.run_id_for_identity(changed_config) != first


def test_progress_counts_one_manifest_only(tmp_path: Path):
    progress = load_script("gen_progress")
    old = tmp_path / "runs" / "old"
    new = tmp_path / "runs" / "new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "manifest.json").write_text(
        '{"run_id":"old","updated_at":"2026-01-01",'
        '"results":{"1":{"task_completed":1},"2":{"task_completed":1}}}'
    )
    (new / "manifest.json").write_text(
        '{"run_id":"new","updated_at":"2026-02-01",'
        '"results":{"1":{"task_completed":0},"2":{"task_completed":1},'
        '"51":{"task_completed":1}}}'
    )
    manifest = progress.latest_validation_manifest(tmp_path)
    assert manifest and manifest["run_id"] == "new"
    assert progress.manifest_counts(manifest) == (2, 1)


def test_progress_filters_identity_and_prefers_full_run(tmp_path: Path):
    progress = load_script("gen_progress")
    subset = tmp_path / "runs" / "subset"
    full = tmp_path / "runs" / "full"
    for path in (subset, full):
        path.mkdir(parents=True)
    identity = {"code_sha256": "code", "config_sha256": "config"}
    (subset / "manifest.json").write_text(json.dumps({
        "run_id": "subset", "updated_at": "2026-03-01",
        "identity": {**identity, "seeds": [1, 2]}, "results": {}
    }))
    (full / "manifest.json").write_text(json.dumps({
        "run_id": "full", "updated_at": "2026-02-01",
        "identity": {**identity, "seeds": list(range(1, 51))}, "results": {}
    }))
    manifest = progress.latest_validation_manifest(
        tmp_path, code_sha256="code", config_sha256="config"
    )
    assert manifest and manifest["run_id"] == "full"


def test_only_full_partition_validates_stage1_evidence():
    validation = load_script("run_fix_loop_validation")
    assert validation.is_full_heldout({"seeds": list(range(1, 51))})
    assert not validation.is_full_heldout({"seeds": [1, 2]})


def test_selected_fix_requires_all_50_results(tmp_path: Path):
    progress = load_script("gen_progress")
    fix = tmp_path / "fix_code.py"
    fix.write_text("get_observation()")
    assert progress.get_status(None, 0) == "pending"
    assert progress.get_status(fix, 49) == "stage1-done"
    assert progress.get_status(fix, 50) == "done"
