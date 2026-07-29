#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_ROOT="$SIM_ROOT/cap/third_party/contact_graspnet_pytorch"
PATCH_FILE="$SIM_ROOT/patches/contact_graspnet_pytorch-compat.patch"
EXPECTED_REVISION="2d71da4e50a04aa353352d1cae99f20f7022145b"

if [[ ! -d "$VENDOR_ROOT/.git" && ! -f "$VENDOR_ROOT/.git" ]]; then
  echo "ERROR: Contact-GraspNet submodule is not initialized."
  echo "Run: git submodule update --init aspire/sim/cap/third_party/contact_graspnet_pytorch"
  exit 1
fi

actual_revision="$(git -C "$VENDOR_ROOT" rev-parse HEAD)"
if [[ "$actual_revision" != "$EXPECTED_REVISION" ]]; then
  echo "ERROR: unexpected Contact-GraspNet revision: $actual_revision"
  echo "Expected: $EXPECTED_REVISION"
  exit 1
fi

if git -C "$VENDOR_ROOT" apply --unidiff-zero --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Contact-GraspNet compatibility patch is already applied."
elif git -C "$VENDOR_ROOT" apply --unidiff-zero --check "$PATCH_FILE"; then
  git -C "$VENDOR_ROOT" apply --unidiff-zero "$PATCH_FILE"
  echo "Applied Contact-GraspNet compatibility patch."
else
  echo "ERROR: Contact-GraspNet compatibility patch does not apply cleanly."
  exit 1
fi

expected_paths=(
  "contact_graspnet_pytorch/checkpoints.py"
  "contact_graspnet_pytorch/contact_graspnet.py"
  "contact_graspnet_pytorch/visualization_utils_o3d.py"
)
expected_blob_hashes=(
  "ba7a97b0a96d84104cf9a373faf953537f9af9be"
  "9e73c2bc729f45fcf9f9121f966405925b00cee0"
  "8ab829e5541c3835ef45a240dd8d864db9e71cf1"
)

expected_path_list="$(printf '%s\n' "${expected_paths[@]}" | sort)"
changed_path_list="$(
  git -C "$VENDOR_ROOT" diff --name-only -- "${expected_paths[@]}" | sort
)"
all_changed_path_list="$(git -C "$VENDOR_ROOT" diff HEAD --name-only | sort)"

if [[ "$changed_path_list" != "$expected_path_list" ]] ||
   [[ "$all_changed_path_list" != "$expected_path_list" ]]; then
  echo "ERROR: Contact-GraspNet submodule contains changes outside the expected patch."
  git -C "$VENDOR_ROOT" status --short
  exit 1
fi

for index in "${!expected_paths[@]}"; do
  actual_blob="$(git -C "$VENDOR_ROOT" hash-object "${expected_paths[$index]}")"
  if [[ "$actual_blob" != "${expected_blob_hashes[$index]}" ]]; then
    echo "ERROR: patched Contact-GraspNet blob does not match the tested fork."
    echo "Path: ${expected_paths[$index]}"
    echo "Expected: ${expected_blob_hashes[$index]}"
    echo "Actual: $actual_blob"
    exit 1
  fi
done

git -C "$VENDOR_ROOT" diff --check
echo "Contact-GraspNet revision and three-file compatibility patch verified byte-for-byte."
