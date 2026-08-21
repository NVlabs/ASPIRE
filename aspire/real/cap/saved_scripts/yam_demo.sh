#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S="${OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S:-8.0}"
export OPENFORGE_PREVIEW_RECORDER_BACKEND="${OPENFORGE_PREVIEW_RECORDER_BACKEND:-python}"
export OPENFORGE_PREVIEW_RECORDER_REENCODE_H264="${OPENFORGE_PREVIEW_RECORDER_REENCODE_H264:-1}"
cd "$ROOT"
source "$ROOT/cap/saved_scripts/shell_scripts/speed_scale.sh"

usage() {
  cat <<'USAGE'
Usage: yam_demo.sh <command>

Commands:
  full              Run full demo sequence including bottle-rack
  bowls             Run white-dish then orange-on-white
  white-dish        Place white dish on bottom shelf
  orange-on-white   Put orange dish on white dish in bottom rack
  can-trash         Put can in trash
  kitkat            Run KitKat handover/trash
  bottle-rack       Put bottle on rack
  drawer            Observe, pull, push, and re-observe top drawer
  drawer-close      Close an already-open top drawer
  drawer-candy      Open drawer, place yellow candy bar inside, close drawer
  home              Open grippers and go home
USAGE
}

full_sequence_usage() {
  cat <<'USAGE'
Usage: yam_demo.sh full

Runs, in order:
  1. drawer-close, home, white-dish, orange-on-white, can-trash, KitKat
  2. yam_demo.sh bottle-rack

Speed:
  YAM_FULL_DEMO_SPEED_SCALE=1.0 keeps validated defaults.
  Larger values speed up planned motion and shorten direct step durations.

The dish/bowl/can/KitKat stage is placed under:
  logs/yam_demo_full_<timestamp>/full_demo

Bottle-rack child logs are copied under:
  logs/yam_demo_full_<timestamp>/bottle_rack
USAGE
}

safe_label() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

quote_concat_path() {
  printf '%s' "$1" | sed "s/'/'\\\\''/g"
}

full_demo_collect_new_logs() {
  local label="$1"
  local marker="$2"
  local status="$3"
  local step_dir
  local safe
  local src
  local new_dirs=()

  safe="$(safe_label "$label")"
  step_dir="$FULL_DEMO_RUN_DIR/steps/$(printf '%02d_%s' "$step_index" "$safe")"
  mkdir -p "$step_dir"

  mapfile -t new_dirs < <(
    find "$ROOT/logs" -mindepth 1 -maxdepth 1 -type d \
      -newer "$marker" \
      ! -path "$FULL_DEMO_RUN_DIR" \
      -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-
  )

  if [ "${#new_dirs[@]}" -eq 0 ]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$step_index" "$label" "$status" "(none)" "$step_dir" >> "$MANIFEST"
    return
  fi

  for src in "${new_dirs[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\n' "$step_index" "$label" "$status" "$src" "$step_dir/$(basename "$src")" >> "$MANIFEST"
  done
}

full_demo_copy_recorded_logs() {
  local source_dir
  local copied_dir

  if [ "${FULL_DEMO_SKIP_LOG_COPY:-0}" = "1" ]; then
    printf 'child log folder copy skipped because FULL_DEMO_SKIP_LOG_COPY=1\n' >> "$MERGE_SUMMARY"
    return
  fi

  tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r _step _label _status source_dir copied_dir; do
    if [ "$source_dir" = "(none)" ] || [ -z "$source_dir" ] || [ -z "$copied_dir" ]; then
      continue
    fi
    if [ -e "$copied_dir" ]; then
      continue
    fi
    mkdir -p "$(dirname "$copied_dir")"
    cp -a "$source_dir" "$copied_dir"
  done
}

full_demo_merge_one_camera() {
  local camera="$1"
  local list_file="$FULL_DEMO_RUN_DIR/concat/${camera}.txt"
  local output="$FULL_DEMO_RUN_DIR/${camera}.mp4"
  local log_file="$FULL_DEMO_RUN_DIR/concat/${camera}.ffmpeg.log"
  local videos=()
  local video

  mapfile -t videos < <(
    tail -n +2 "$MANIFEST" |
      awk -F '\t' '$4 != "(none)" && $4 != "" { print $4 "/" cam ".mp4" }' cam="$camera" |
      while IFS= read -r video; do
        [ -f "$video" ] && printf '%s\n' "$video"
      done
  )
  if [ "${#videos[@]}" -eq 0 ]; then
    printf '%s: no videos found\n' "$camera" >> "$MERGE_SUMMARY"
    return
  fi

  : > "$list_file"
  for video in "${videos[@]}"; do
    printf "file '%s'\n" "$(quote_concat_path "$video")" >> "$list_file"
  done

  if ffmpeg -hide_banner -loglevel warning -y -f concat -safe 0 -i "$list_file" -c copy "$output" > "$log_file" 2>&1; then
    printf '%s: merged %s videos -> %s\n' "$camera" "${#videos[@]}" "$output" >> "$MERGE_SUMMARY"
    return
  fi

  if ffmpeg -hide_banner -loglevel warning -y -f concat -safe 0 -i "$list_file" -c:v libx264 -preset veryfast -crf 20 -an "$output" >> "$log_file" 2>&1; then
    printf '%s: re-encoded %s videos -> %s\n' "$camera" "${#videos[@]}" "$output" >> "$MERGE_SUMMARY"
    return
  fi

  printf '%s: merge failed, see %s\n' "$camera" "$log_file" >> "$MERGE_SUMMARY"
}

full_demo_merge_quad_video() {
  local output="$FULL_DEMO_RUN_DIR/full_demo.mp4"
  local log_file="$FULL_DEMO_RUN_DIR/concat/full_demo.ffmpeg.log"

  for camera in top left right bottom; do
    if [ ! -f "$FULL_DEMO_RUN_DIR/${camera}.mp4" ]; then
      printf 'full_demo.mp4: skipped because %s.mp4 is missing\n' "$camera" >> "$MERGE_SUMMARY"
      return
    fi
  done

  if ffmpeg -hide_banner -loglevel warning -y \
    -i "$FULL_DEMO_RUN_DIR/top.mp4" \
    -i "$FULL_DEMO_RUN_DIR/left.mp4" \
    -i "$FULL_DEMO_RUN_DIR/right.mp4" \
    -i "$FULL_DEMO_RUN_DIR/bottom.mp4" \
    -filter_complex "[0:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[top];[1:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[left];[2:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[right];[3:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[bottom];[top][left]hstack=inputs=2[row0];[right][bottom]hstack=inputs=2[row1];[row0][row1]vstack=inputs=2[v]" \
    -map "[v]" -shortest -an -c:v libx264 -preset veryfast -crf 20 "$output" > "$log_file" 2>&1; then
    printf 'full_demo.mp4: created 2x2 merged video -> %s\n' "$output" >> "$MERGE_SUMMARY"
  else
    printf 'full_demo.mp4: merge failed, see %s\n' "$log_file" >> "$MERGE_SUMMARY"
  fi
}

full_demo_merge_videos() {
  if [ "$merge_done" = "1" ]; then
    return
  fi
  merge_done=1

  if [ "${FULL_DEMO_SKIP_MERGE:-0}" = "1" ]; then
    printf 'merged-video generation skipped because FULL_DEMO_SKIP_MERGE=1\n' >> "$MERGE_SUMMARY"
    return
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'ffmpeg not found; copied logs are available but merged videos were not generated\n' >> "$MERGE_SUMMARY"
    return
  fi

  mkdir -p "$FULL_DEMO_RUN_DIR/concat"
  : > "$MERGE_SUMMARY"
  full_demo_merge_one_camera top
  full_demo_merge_one_camera left
  full_demo_merge_one_camera right
  full_demo_merge_one_camera bottom
  full_demo_merge_quad_video
}

full_demo_on_exit() {
  local status="$?"
  full_demo_copy_recorded_logs || true
  full_demo_merge_videos || true
  echo "=== full_demo: consolidated logs: $FULL_DEMO_RUN_DIR ==="
  exit "$status"
}

full_demo_run_step() {
  local label="$1"
  local marker
  local status
  shift

  marker="$FULL_DEMO_RUN_DIR/.marker_${step_index}_$(safe_label "$label")"
  : > "$marker"
  echo "=== full_demo: running $label ==="
  set +e
  "$@"
  status="$?"
  set -e
  full_demo_collect_new_logs "$label" "$marker" "$status"
  step_index=$((step_index + 1))
  if [ "$status" -ne 0 ]; then
    echo "=== full_demo: step failed: $label (status $status) ===" >&2
    exit "$status"
  fi
}

full_demo_run_initial_home() {
  env "${COMMON_ENV[@]}" \
    OPENFORGE_OPEN_HOME_SPEED_RAD_S="${OPENFORGE_OPEN_HOME_SPEED_RAD_S:-$(yam_speed_mul 0.5)}" \
    OPENFORGE_OPEN_HOME_SETTLE_S="${OPENFORGE_OPEN_HOME_SETTLE_S:-1.0}" \
    uv run python run_script.py \
      script_file=cap/saved_scripts/open_grippers_return_home.py \
      skill_library_path=cap/saved_scripts/skill_library \
      env.name=yam-real \
      robot=real_yam \
      robot.dashboard=true \
      robot.await_exit=false \
      robot.go_home_on_exit=false \
      runtime.no_cameras=true \
      recording.enabled=true \
      debug_ui.enabled=true \
      debug_ui.auto_open=false \
      debug_ui.auto_exit_on_run_end=false \
      debug_ui.host=0.0.0.0 \
      debug_ui.port="${OPENFORGE_RESET_DEBUG_PORT:-8788}"
}

run_full_demo() {
  DEMO_RUNNER="$ROOT/cap/saved_scripts/shell_scripts/dish_bowl_rack.sh"
  KITKAT_RUNNER="$ROOT/cap/saved_scripts/shell_scripts/run_kitkat_handover.sh"
  DRAWER_CLOSE_RUNNER="$ROOT/cap/saved_scripts/shell_scripts/drawer_push_close_x.sh"
  FULL_DEMO_RUN_DIR="${FULL_DEMO_RUN_DIR:-$ROOT/logs/full_demo_$(date +%Y%m%dT%H%M%S)}"

  if [[ "$FULL_DEMO_RUN_DIR" != /* ]]; then
    FULL_DEMO_RUN_DIR="$ROOT/$FULL_DEMO_RUN_DIR"
  fi

  MANIFEST="$FULL_DEMO_RUN_DIR/manifest.tsv"
  MERGE_SUMMARY="$FULL_DEMO_RUN_DIR/merge_summary.txt"
  merge_done=0
  step_index=0

  if [ ! -r "$DEMO_RUNNER" ]; then
    echo "Missing readable demo runner: $DEMO_RUNNER" >&2
    exit 1
  fi

  if [ ! -r "$KITKAT_RUNNER" ]; then
    echo "Missing readable KitKat runner: $KITKAT_RUNNER" >&2
    exit 1
  fi

  if [ "${FULL_DEMO_INCLUDE_DRAWER_CLOSE:-0}" = "1" ] && [ "${FULL_DEMO_SKIP_DRAWER_CLOSE:-0}" != "1" ] && [ ! -r "$DRAWER_CLOSE_RUNNER" ]; then
    echo "Missing readable drawer-close runner: $DRAWER_CLOSE_RUNNER" >&2
    exit 1
  fi

  cd "$ROOT"
  source .forge_env
  mkdir -p "$FULL_DEMO_RUN_DIR/steps" "$FULL_DEMO_RUN_DIR/concat"
  printf 'step\tlabel\tstatus\tsource_dir\tcopied_dir\n' > "$MANIFEST"
  printf 'full_demo run dir: %s\n' "$FULL_DEMO_RUN_DIR" > "$MERGE_SUMMARY"
  trap full_demo_on_exit EXIT

  COMMON_ENV=(
    YAM_STATION_CALIBRATED_XML="${YAM_STATION_CALIBRATED_XML:?Set YAM_STATION_CALIBRATED_XML in .forge_env}"
    CAP_TOP_CAMERA_BACKEND=realsense
    CAP_TOP_CAMERA_FRAME=top_camera_d405
    CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP=0
    OPENFORGE_ALLOW_PHYSICAL_MOTION=1
  )

  drawer_close_step_ran=0
  if [ "${FULL_DEMO_INCLUDE_DRAWER_CLOSE:-0}" = "1" ] && [ "${FULL_DEMO_SKIP_DRAWER_CLOSE:-0}" != "1" ]; then
    full_demo_run_step "drawer_close_home" bash "$DRAWER_CLOSE_RUNNER"
    drawer_close_step_ran=1
  fi

  if [ "${FULL_DEMO_SKIP_INITIAL_HOME:-0}" = "1" ]; then
    echo "=== full_demo: skipping initial home because FULL_DEMO_SKIP_INITIAL_HOME=1 ==="
  elif [ "$drawer_close_step_ran" = "1" ] && [ "${OPENFORGE_DRAWER_CLOSE_SKIP_HOME:-0}" != "1" ]; then
    echo "=== full_demo: skipping initial home because drawer_close_home already ran home ==="
  else
    full_demo_run_step "initial_home" full_demo_run_initial_home
  fi

  if [ -n "${FULL_DEMO_TASKS:-}" ]; then
    read -r -a demo_tasks <<< "$FULL_DEMO_TASKS"
  else
    demo_tasks=(white-dish orange-on-white can-trash)
  fi

  for task in "${demo_tasks[@]}"; do
    full_demo_run_step "$task" bash "$DEMO_RUNNER" "$task"
  done

  if [ "${FULL_DEMO_SKIP_KITKAT:-0}" = "1" ]; then
    echo "=== full_demo: skipping KitKat handover because FULL_DEMO_SKIP_KITKAT=1 ==="
    exit 0
  fi

  full_demo_run_step "kitkat_handover_trash" bash "$KITKAT_RUNNER"
}

sequence_copy_bottle_logs() {
  local copied_dir
  local source_dir
  local bottle_logs=()

  if [ -z "$marker" ] || [ ! -e "$marker" ]; then
    return
  fi

  mkdir -p "$SEQUENTIAL_RUN_DIR/bottle_rack"
  mapfile -t bottle_logs < <(
    find "$ROOT/logs" -mindepth 1 -maxdepth 1 -type d \
      -newer "$marker" \
      ! -path "$SEQUENTIAL_RUN_DIR" \
      ! -path "$SEQUENTIAL_RUN_DIR/*" \
      -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-
  )

  if [ "${#bottle_logs[@]}" -eq 0 ]; then
    printf 'bottle_rack_logs: none found\n' >> "$SUMMARY"
    return
  fi

  printf 'bottle_rack_logs:\n' >> "$SUMMARY"
  for source_dir in "${bottle_logs[@]}"; do
    copied_dir="$SEQUENTIAL_RUN_DIR/bottle_rack/$(basename "$source_dir")"
    if [ ! -e "$copied_dir" ]; then
      cp -a "$source_dir" "$copied_dir"
    fi
    printf '  %s -> %s\n' "$source_dir" "$copied_dir" >> "$SUMMARY"
  done
}

run_full_sequence() {
  case "${1:-}" in
    -h|--help|help)
      full_sequence_usage
      exit 0
      ;;
    "")
      ;;
    *)
      echo "Unknown full argument: $1" >&2
      full_sequence_usage >&2
      exit 2
      ;;
  esac

  BOTTLE_RACK_SCRIPT="$ROOT/cap/saved_scripts/shell_scripts/bottle_rack.sh"
  SEQUENTIAL_RUN_DIR="${YAM_DEMO_FULL_RUN_DIR:-${FULL_DEMO_SEQUENTIAL_RUN_DIR:-$ROOT/logs/yam_demo_full_$(date +%Y%m%dT%H%M%S)}}"

  if [[ "$SEQUENTIAL_RUN_DIR" != /* ]]; then
    SEQUENTIAL_RUN_DIR="$ROOT/$SEQUENTIAL_RUN_DIR"
  fi

  SUMMARY="$SEQUENTIAL_RUN_DIR/summary.txt"
  marker=""

  if [ ! -r "$BOTTLE_RACK_SCRIPT" ]; then
    echo "Missing readable bottle rack script: $BOTTLE_RACK_SCRIPT" >&2
    exit 1
  fi

  cd "$ROOT"
  source .forge_env
  mkdir -p "$SEQUENTIAL_RUN_DIR"
  {
    printf 'yam_demo full run dir: %s\n' "$SEQUENTIAL_RUN_DIR"
    printf 'started_at: %s\n' "$(date --iso-8601=seconds)"
  } > "$SUMMARY"

  echo "=== yam_demo full: run dir: $SEQUENTIAL_RUN_DIR ==="

  echo "=== yam_demo full: running dish/bowl/can/KitKat stage ==="
  set +e
  (
    FULL_DEMO_RUN_DIR="${FULL_DEMO_RUN_DIR:-$SEQUENTIAL_RUN_DIR/full_demo}"
    export FULL_DEMO_INCLUDE_DRAWER_CLOSE="${FULL_DEMO_INCLUDE_DRAWER_CLOSE:-1}"
    run_full_demo
  )
  demo_stage_status="$?"
  set -e
  printf 'dish_bowl_can_kitkat_status: %s\n' "$demo_stage_status" >> "$SUMMARY"
  if [ "$demo_stage_status" -ne 0 ]; then
    printf 'finished_at: %s\n' "$(date --iso-8601=seconds)" >> "$SUMMARY"
    echo "=== yam_demo full: dish/bowl/can/KitKat stage failed (status $demo_stage_status) ===" >&2
    echo "=== yam_demo full: consolidated logs: $SEQUENTIAL_RUN_DIR ==="
    exit "$demo_stage_status"
  fi

  marker="$SEQUENTIAL_RUN_DIR/.before_bottle_rack"
  : > "$marker"

  echo "=== yam_demo full: running bottle-rack ==="
  set +e
  bash "$BOTTLE_RACK_SCRIPT"
  bottle_rack_status="$?"
  set -e
  printf 'bottle_rack_status: %s\n' "$bottle_rack_status" >> "$SUMMARY"
  sequence_copy_bottle_logs

  if [ "$bottle_rack_status" -ne 0 ]; then
    printf 'finished_at: %s\n' "$(date --iso-8601=seconds)" >> "$SUMMARY"
    echo "=== yam_demo full: bottle-rack failed (status $bottle_rack_status) ===" >&2
    echo "=== yam_demo full: consolidated logs: $SEQUENTIAL_RUN_DIR ==="
    exit "$bottle_rack_status"
  fi

  printf 'finished_at: %s\n' "$(date --iso-8601=seconds)" >> "$SUMMARY"
  echo "=== yam_demo full: consolidated logs: $SEQUENTIAL_RUN_DIR ==="
}

command="${1:-}"
shift || true
case "$command" in
  full|sequential)
    ( run_full_sequence "$@" )
    ;;
  bowls)
    case "${1:-}" in
      -h|--help|help)
        usage
        exit 0
        ;;
      "")
        ;;
      *)
        echo "Unknown bowls argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    (
      export FULL_DEMO_TASKS="${FULL_DEMO_TASKS:-white-dish orange-on-white}"
      export FULL_DEMO_SKIP_KITKAT=1
      run_full_demo
    )
    ;;
  white-dish|place-white-dish|white-dish-bottom-shelf)
    exec bash cap/saved_scripts/shell_scripts/dish_bowl_rack.sh white-dish
    ;;
  orange-on-white|orange-dish|orange-dish-on-white|orange-bowl-on-white)
    exec bash cap/saved_scripts/shell_scripts/dish_bowl_rack.sh orange-on-white
    ;;
  can-trash|can-in-trash|trash-can)
    exec bash cap/saved_scripts/shell_scripts/dish_bowl_rack.sh can-trash
    ;;
  kitkat|kitkat-handover|kitkat-trash|handover)
    exec bash cap/saved_scripts/shell_scripts/run_kitkat_handover.sh
    ;;
  bottle-rack|bottle)
    exec bash cap/saved_scripts/shell_scripts/bottle_rack.sh
    ;;
  drawer|top-drawer|drawer-pull-push)
    export OPENFORGE_ALLOW_PHYSICAL_MOTION="${OPENFORGE_ALLOW_PHYSICAL_MOTION:-1}"
    exec bash cap/saved_scripts/shell_scripts/drawer_observe_pull_open_x.sh "$@"
    ;;
  drawer-close|drawer-push-close|push-drawer-close)
    export OPENFORGE_ALLOW_PHYSICAL_MOTION="${OPENFORGE_ALLOW_PHYSICAL_MOTION:-1}"
    exec bash cap/saved_scripts/shell_scripts/drawer_push_close_x.sh "$@"
    ;;
  drawer-candy|candy-drawer|drawer-place-candy)
    export OPENFORGE_ALLOW_PHYSICAL_MOTION="${OPENFORGE_ALLOW_PHYSICAL_MOTION:-1}"
    exec bash cap/saved_scripts/shell_scripts/drawer_candy_bar_place.sh "$@"
    ;;
  home)
    exec bash cap/saved_scripts/shell_scripts/home.sh
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
