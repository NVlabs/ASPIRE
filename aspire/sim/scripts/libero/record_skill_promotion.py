#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record auditable, per-task updates to the shared LIBERO skill library.

The coordinator calls ``begin`` before editing ``.claude/libero/skills`` and
``finish`` afterwards. Each completed promotion preserves the before snapshot,
an exact patch, file hashes, and an append-only JSONL record. ``verify`` is the
dispatch gate used before the task's GPU is assigned to another Stage 1 worker.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
SKILLS_REL = Path(".claude/libero/skills")
BUILD_REL = Path("outputs/libero_fix_loop")
SCHEMA_VERSION = 1
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_name(value: str, label: str) -> str:
    if not NAME_RE.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, '.', '_' or '-': {value}")
    return value


def relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def skill_files(root: Path) -> list[Path]:
    skills = root / SKILLS_REL
    if not skills.is_dir():
        raise FileNotFoundError(f"skill library does not exist: {skills}")
    files = sorted(path for path in skills.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"skill library is empty: {skills}")
    return files


def skill_hashes(root: Path) -> dict[str, str]:
    return {
        relative_path(root, path): sha256_file(path)
        for path in skill_files(root)
    }


def library_sha256(hashes: dict[str, str]) -> str:
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def campaign_dir(root: Path, suite: str) -> Path:
    return root / BUILD_REL / suite


def promotions_dir(root: Path, suite: str) -> Path:
    return campaign_dir(root, suite) / "skill_promotions"


def ledger_path(root: Path, suite: str) -> Path:
    return campaign_dir(root, suite) / "skill_promotions.jsonl"


def read_ledger(root: Path, suite: str) -> list[dict]:
    ledger = ledger_path(root, suite)
    if not ledger.exists():
        return []
    entries = []
    for line_number, line in enumerate(ledger.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {ledger}:{line_number}: {error}") from error
    return entries


def promotion_directories(root: Path, suite: str) -> list[Path]:
    base = promotions_dir(root, suite)
    return sorted(path for path in base.glob("*") if path.is_dir()) if base.exists() else []


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def completed_record(root: Path, suite: str, task: str) -> tuple[Path, dict] | None:
    for directory in promotion_directories(root, suite):
        record_path = directory / "record.json"
        if record_path.exists():
            record = load_json(record_path)
            if record.get("task") == task:
                return directory, record
    return None


def pending_promotions(root: Path, suite: str) -> list[tuple[Path, dict]]:
    pending = []
    for directory in promotion_directories(root, suite):
        begin_path = directory / "begin.json"
        if begin_path.exists() and not (directory / "record.json").exists():
            pending.append((directory, load_json(begin_path)))
    return pending


def next_sequence(root: Path, suite: str) -> int:
    sequences = [
        int(entry["sequence"])
        for entry in read_ledger(root, suite)
        if isinstance(entry.get("sequence"), int)
    ]
    for directory in promotion_directories(root, suite):
        for filename in ("begin.json", "record.json"):
            path = directory / filename
            if path.exists():
                record = load_json(path)
                if isinstance(record.get("sequence"), int):
                    sequences.append(int(record["sequence"]))
    return max(sequences, default=0) + 1


def append_ledger_entry(root: Path, suite: str, record: dict) -> None:
    ledger = ledger_path(root, suite)
    existing_ids = {entry.get("promotion_id") for entry in read_ledger(root, suite)}
    if record["promotion_id"] in existing_ids:
        return
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as destination:
        destination.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def begin_promotion(
    root: Path,
    *,
    suite: str,
    task: str,
    timestamp: Callable[[], str] = now,
) -> dict:
    suite = validate_name(suite, "suite")
    task = validate_name(task, "task")
    if completed_record(root, suite, task):
        raise ValueError(f"promotion already completed for {suite}/{task}")
    pending = pending_promotions(root, suite)
    if pending:
        _, record = pending[0]
        raise ValueError(
            "another promotion is unfinished: "
            f"{record.get('suite')}/{record.get('task')} ({record.get('promotion_id')})"
        )

    sequence = next_sequence(root, suite)
    promotion_id = f"{sequence:04d}_{task}"
    directory = promotions_dir(root, suite) / promotion_id
    before_dir = directory / "before"
    before_hashes = skill_hashes(root)
    for relative in before_hashes:
        source = root / relative
        destination = before_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    findings = campaign_dir(root, suite) / task / "findings.md"
    record = {
        "schema_version": SCHEMA_VERSION,
        "promotion_id": promotion_id,
        "sequence": sequence,
        "suite": suite,
        "task": task,
        "began_at": timestamp(),
        "findings_path": relative_path(root, findings),
        "findings_sha256": sha256_file(findings) if findings.is_file() else None,
        "library_before_sha256": library_sha256(before_hashes),
        "skill_sha256_before": before_hashes,
    }
    write_json_atomic(directory / "begin.json", record)
    return record


def build_patch(root: Path, directory: Path, before: dict[str, str], after: dict[str, str]) -> str:
    chunks: list[str] = []
    for relative in sorted(set(before) | set(after)):
        before_path = directory / "before" / relative
        after_path = root / relative
        before_lines = (
            before_path.read_text(errors="replace").splitlines(keepends=True)
            if before_path.is_file()
            else []
        )
        after_lines = (
            after_path.read_text(errors="replace").splitlines(keepends=True)
            if after_path.is_file()
            else []
        )
        chunks.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{relative}" if before_path.exists() else "/dev/null",
                tofile=f"b/{relative}" if after_path.exists() else "/dev/null",
            )
        )
    return "".join(chunks)


def finish_promotion(
    root: Path,
    *,
    suite: str,
    task: str,
    reason: str | None = None,
    timestamp: Callable[[], str] = now,
) -> dict:
    suite = validate_name(suite, "suite")
    task = validate_name(task, "task")
    completed = completed_record(root, suite, task)
    if completed:
        _, record = completed
        append_ledger_entry(root, suite, record)
        return record

    matches = [
        (directory, begin)
        for directory, begin in pending_promotions(root, suite)
        if begin.get("task") == task
    ]
    if not matches:
        raise ValueError(f"no unfinished promotion for {suite}/{task}; run begin first")
    directory, begin = matches[0]
    before_hashes = begin["skill_sha256_before"]
    after_hashes = skill_hashes(root)
    changed = sorted(
        relative
        for relative in set(before_hashes) | set(after_hashes)
        if before_hashes.get(relative) != after_hashes.get(relative)
    )
    if not changed and not reason:
        raise ValueError("no skill files changed; pass --reason to record an intentional no-op")

    patch_path = directory / "changes.patch"
    patch_path.write_text(build_patch(root, directory, before_hashes, after_hashes))
    record = {
        **begin,
        "completed_at": timestamp(),
        "changed_skill_files": changed,
        "library_after_sha256": library_sha256(after_hashes),
        "skill_sha256_after": after_hashes,
        "patch_path": relative_path(root, patch_path),
        "no_op": not changed,
        "reason": reason,
    }
    write_json_atomic(directory / "record.json", record)
    append_ledger_entry(root, suite, record)
    return record


def verify_promotion(root: Path, *, suite: str, task: str) -> dict:
    suite = validate_name(suite, "suite")
    task = validate_name(task, "task")
    completed = completed_record(root, suite, task)
    if not completed:
        raise ValueError(f"promotion is not complete for {suite}/{task}")
    _, record = completed
    append_ledger_entry(root, suite, record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("begin", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--suite", required=True)
        subparser.add_argument("--task", required=True)
    finish = subparsers.add_parser("finish")
    finish.add_argument("--suite", required=True)
    finish.add_argument("--task", required=True)
    finish.add_argument("--reason", help="Required when no shared skill file changed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    try:
        if args.command == "begin":
            record = begin_promotion(root, suite=args.suite, task=args.task)
            print(
                f"promotion started: {record['promotion_id']} "
                f"(library {record['library_before_sha256']})"
            )
        elif args.command == "finish":
            record = finish_promotion(
                root, suite=args.suite, task=args.task, reason=args.reason
            )
            print(
                f"promotion recorded: {record['promotion_id']} "
                f"({len(record['changed_skill_files'])} skill files changed)"
            )
        else:
            record = verify_promotion(root, suite=args.suite, task=args.task)
            print(
                f"promotion verified: {record['promotion_id']} "
                f"(library {record['library_after_sha256']})"
            )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
