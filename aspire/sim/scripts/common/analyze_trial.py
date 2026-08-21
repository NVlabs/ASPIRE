#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline trial analysis — diagnose grasp success/failure from trace.json.

Reads trace.json (+ optional keyframe arrays) and produces a structured
diagnostic report covering phase detection, grasp quality, gripper state,
object displacement, and a final verdict.

Works on both old traces (no robot state / gripper_width) and new enhanced
traces produced by the updated TraceLogger.

Usage:
    .venv/bin/python3 scripts/common/analyze_trial.py \
        --trial-dir outputs/.../trial_08_... --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ────────────────────────────────────────────────────────────────
# Phase classification
# ────────────────────────────────────────────────────────────────

PHASE_ORDER = [
    "OBSERVE",
    "SEGMENT",
    "GRASP_PLAN",
    "PRE_GRASP",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "RELEASE",
    "RETREAT",
]

_OBSERVE_FNS = {"get_observation", "point_prompt_molmo"}
_SEGMENT_FNS = {"segment_sam3_text_prompt", "segment_sam3_point_prompt"}
_GRASP_PLAN_FNS = {
    "plan_grasp",
    "select_top_down_grasp",
    "decompose_transform",
    "mask_to_world_points",
    "depth_to_point_cloud",
    "get_oriented_bounding_box_from_3d_points",
}


def classify_phases(entries: list[dict]) -> list[dict]:
    """Assign a phase label to each trace entry.

    The algorithm uses a simple state machine: it walks through the entries
    and transitions forward through the phase order based on function names
    and contextual cues (e.g. close_gripper marks GRASP, open_gripper after
    grasp marks RELEASE).
    """
    phases: list[dict] = []
    gripper_closed = False
    grasp_planned = False
    first_open_gripper_seen = False
    lift_move_count = 0

    for entry in entries:
        fn = entry["function"]

        if not grasp_planned:
            # Pre-grasp-planning: OBSERVE / SEGMENT / GRASP_PLAN
            if fn in _OBSERVE_FNS:
                phase = "OBSERVE"
            elif fn in _SEGMENT_FNS:
                phase = "SEGMENT"
            elif fn in _GRASP_PLAN_FNS:
                phase = "GRASP_PLAN"
                if fn == "select_top_down_grasp":
                    grasp_planned = True
            else:
                phase = "OBSERVE"
        else:
            # Post-grasp-planning
            if fn == "open_gripper" and not first_open_gripper_seen:
                phase = "PRE_GRASP"
                first_open_gripper_seen = True
            elif fn == "close_gripper":
                phase = "GRASP"
                gripper_closed = True
                lift_move_count = 0
            elif fn == "open_gripper" and gripper_closed:
                phase = "RELEASE"
                gripper_closed = False
            elif gripper_closed:
                # After close_gripper, before open_gripper
                if fn in ("solve_ik", "move_to_joints"):
                    lift_move_count += 1
                    # move_to_joints inherits phase from preceding solve_ik
                    if fn == "move_to_joints" and phases:
                        phase = phases[-1]["phase"]
                    elif lift_move_count <= 2:
                        # First solve_ik+move pair = LIFT
                        phase = "LIFT"
                    else:
                        # Check if z is decreasing (PLACE) vs moving laterally (TRANSPORT)
                        pos = _get_target_position(entry)
                        if pos is not None and len(phases) > 0:
                            prev_pos = _find_last_position(phases)
                            if prev_pos is not None and pos[2] < prev_pos[2] - 0.01:
                                phase = "PLACE"
                            else:
                                phase = "TRANSPORT"
                        else:
                            phase = "TRANSPORT"
                else:
                    phase = "TRANSPORT"
            elif not gripper_closed and first_open_gripper_seen:
                if not grasp_planned:
                    phase = "PRE_GRASP"
                elif fn in ("solve_ik", "move_to_joints"):
                    # Before close_gripper or after open_gripper(release)
                    # Check if we already had a RELEASE
                    had_release = any(p["phase"] == "RELEASE" for p in phases)
                    if had_release:
                        phase = "RETREAT"
                    else:
                        phase = "PRE_GRASP"
                else:
                    phase = "PRE_GRASP"
            else:
                phase = "PRE_GRASP"

        phases.append({**entry, "phase": phase})

    return phases


def _get_target_position(entry: dict) -> list[float] | None:
    """Extract target position from a solve_ik entry."""
    args = entry.get("args", {})
    pos = args.get("position")
    if isinstance(pos, list) and len(pos) >= 3:
        return pos
    return None


def _find_last_position(phases: list[dict]) -> list[float] | None:
    """Find the last solve_ik target position in the phase list."""
    for p in reversed(phases):
        pos = _get_target_position(p)
        if pos is not None:
            return pos
    return None


# ────────────────────────────────────────────────────────────────
# Analysis functions
# ────────────────────────────────────────────────────────────────


def analyze_grasp_quality(entries: list[dict]) -> dict[str, Any]:
    """Extract grasp planning quality metrics."""
    info: dict[str, Any] = {}

    for e in entries:
        if e["function"] == "plan_grasp":
            r = e.get("result", {})
            info["num_grasps"] = r.get("num_grasps")
            info["best_score"] = r.get("best_score")
            info["best_grasp_camera_pos"] = r.get("best_grasp_position")

        if e["function"] == "select_top_down_grasp":
            r = e.get("result", {})
            info["found_grasp"] = r.get("found_grasp", False)
            info["grasp_world_pos"] = r.get("grasp_position")
            info["grasp_score"] = r.get("score")

    # Compute offset between grasp position and object center if available
    seg_entries = [e for e in entries if e["function"] in _SEGMENT_FNS]
    if seg_entries and info.get("grasp_world_pos"):
        # Use the first "bowl"-like segmentation bbox center as proxy
        for se in seg_entries:
            prompt = se.get("args", {}).get("text_prompt", "")
            if "bowl" in prompt.lower() or "object" in prompt.lower() or "mug" in prompt.lower():
                bbox = se.get("result", {}).get("mask_0_bbox")
                if bbox:
                    info["target_bbox"] = bbox
                    info["target_bbox_center_px"] = [
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2,
                    ]
                break

    return info


def analyze_gripper_state(entries: list[dict]) -> dict[str, Any]:
    """Track gripper width through the trial.

    For new traces: uses result.gripper_width after close/open_gripper.
    For old traces: uses robot_joint_pos[-1] from get_observation if available.
    """
    info: dict[str, Any] = {"events": [], "has_gripper_width": False}

    for e in entries:
        fn = e["function"]
        r = e.get("result", {})

        if fn in ("close_gripper", "open_gripper"):
            event: dict[str, Any] = {
                "step": e["step"],
                "action": fn,
                "timestamp": e.get("timestamp"),
            }

            # New trace format: gripper_width directly in result
            if "gripper_width" in r:
                event["gripper_width"] = r["gripper_width"]
                info["has_gripper_width"] = True

            info["events"].append(event)

        # Also track robot_joint_pos from get_observation if numeric
        if fn == "get_observation":
            joint_pos = r.get("robot_joint_pos")
            if isinstance(joint_pos, list) and len(joint_pos) > 0:
                info["has_gripper_width"] = True
                info.setdefault("obs_gripper_widths", []).append({
                    "step": e["step"],
                    "gripper_width": joint_pos[-1],
                })

    # Determine grasp outcome from gripper width
    close_events = [ev for ev in info["events"] if ev["action"] == "close_gripper"]
    if close_events:
        last_close = close_events[-1]
        gw = last_close.get("gripper_width")
        if gw is not None:
            info["grasp_gripper_width"] = gw
            # > 0.01 means something is between the fingers
            info["object_in_gripper"] = gw > 0.01

    return info


def analyze_gripper_object_comovement(entries: list[dict], phases: list[dict]) -> dict[str, Any]:
    """Determine if the gripper closed on an object using multiple signals.

    Signals checked:
    1. robot_joint_pos[-1] > 0.03 after close_gripper → object forcing gripper open
    2. During LIFT phase: if EE Z increases, object likely held
    3. Gripper width from close_gripper result (new traces)

    Returns dict with:
      gripper_closed_on_object: "YES" | "NO" | "UNKNOWN"
      confidence: explanation string
      signals: list of individual signal results
    """
    info: dict[str, Any] = {"signals": []}

    # --- Signal 1: gripper_width after close_gripper ---
    for e in entries:
        if e["function"] == "close_gripper":
            r = e.get("result", {})
            gw = r.get("gripper_width")
            if gw is not None:
                has_object = gw > 0.03
                info["signals"].append({
                    "name": "gripper_width_after_close",
                    "value": gw,
                    "interpretation": "OBJECT" if has_object else "AIR",
                    "detail": f"gripper_width={gw:.4f} {'>' if has_object else '<='} 0.03 threshold",
                })

    # --- Signal 2: robot_joint_pos[-1] from get_observation after close_gripper ---
    close_step = None
    release_step = None
    for e in entries:
        if e["function"] == "close_gripper":
            close_step = e["step"]
        elif e["function"] == "open_gripper" and close_step is not None:
            release_step = e["step"]
            break

    if close_step is not None:
        # Look for any get_observation between close and release
        for e in entries:
            if e["function"] != "get_observation":
                continue
            if e["step"] <= close_step:
                continue
            if release_step is not None and e["step"] >= release_step:
                continue
            joint_pos = e.get("result", {}).get("robot_joint_pos")
            if isinstance(joint_pos, list) and len(joint_pos) > 0:
                gw = joint_pos[-1]
                has_object = gw > 0.03
                info["signals"].append({
                    "name": "joint_pos_during_hold",
                    "step": e["step"],
                    "value": gw,
                    "interpretation": "OBJECT" if has_object else "AIR",
                    "detail": f"robot_joint_pos[-1]={gw:.4f} at step {e['step']}",
                })

    # --- Signal 3: EE motion during LIFT phase ---
    lift_positions = []
    for p in phases:
        if p["phase"] == "LIFT" and p["function"] == "solve_ik":
            pos = p.get("args", {}).get("position")
            if isinstance(pos, list) and len(pos) >= 3:
                lift_positions.append(pos)

    if len(lift_positions) >= 1 and close_step is not None:
        # Get the grasp position (last solve_ik before close_gripper)
        grasp_pos = None
        for p in phases:
            if p["function"] == "solve_ik" and p["step"] < close_step:
                pos = p.get("args", {}).get("position")
                if isinstance(pos, list) and len(pos) >= 3:
                    grasp_pos = pos

        if grasp_pos:
            lift_delta_z = lift_positions[-1][2] - grasp_pos[2]
            ee_moved_up = lift_delta_z > 0.03
            info["signals"].append({
                "name": "lift_motion",
                "value": lift_delta_z,
                "interpretation": "LIFTED" if ee_moved_up else "NO_LIFT",
                "detail": f"EE delta_z={lift_delta_z:.4f}m during LIFT phase",
            })

    # --- Aggregate verdict ---
    object_signals = [s for s in info["signals"]
                      if s["interpretation"] in ("OBJECT",)]
    air_signals = [s for s in info["signals"]
                   if s["interpretation"] in ("AIR",)]

    if object_signals:
        info["gripper_closed_on_object"] = "YES"
        info["confidence"] = f"{len(object_signals)} signal(s) indicate object in gripper"
    elif air_signals:
        info["gripper_closed_on_object"] = "NO"
        info["confidence"] = f"{len(air_signals)} signal(s) indicate gripper closed on air"
    else:
        info["gripper_closed_on_object"] = "UNKNOWN"
        info["confidence"] = "No gripper width data available (old trace format)"

    return info


def analyze_object_displacement(entries: list[dict], keyframes_dir: Path | None) -> dict[str, Any]:
    """Detect object movement by comparing segmentation bboxes across calls.

    Also checks depth arrays at mask locations if available (fallback for old traces).
    """
    info: dict[str, Any] = {}

    # Group segmentation calls by text prompt
    seg_by_prompt: dict[str, list[dict]] = {}
    for e in entries:
        if e["function"] in _SEGMENT_FNS:
            prompt = e.get("args", {}).get("text_prompt", "unknown")
            seg_by_prompt.setdefault(prompt, []).append(e)

    # Check bbox displacement for each prompt
    for prompt, seg_list in seg_by_prompt.items():
        if len(seg_list) < 2:
            continue
        bboxes = []
        for s in seg_list:
            bbox = s.get("result", {}).get("mask_0_bbox")
            if bbox:
                bboxes.append({"step": s["step"], "bbox": bbox})
        if len(bboxes) >= 2:
            first = bboxes[0]["bbox"]
            last = bboxes[-1]["bbox"]
            first_center = np.array([(first[0] + first[2]) / 2, (first[1] + first[3]) / 2])
            last_center = np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
            displacement_px = float(np.linalg.norm(last_center - first_center))
            info[f"{prompt}_bbox_displacement_px"] = round(displacement_px, 1)
            info[f"{prompt}_bboxes"] = bboxes

    # Depth-based fallback: compare depth at mask location between observations
    if keyframes_dir and keyframes_dir.exists():
        depth_files = sorted(keyframes_dir.glob("step_*_depth_agentview.npy"))
        mask_files = sorted(keyframes_dir.glob("step_*_mask_0.npy"))

        if len(depth_files) >= 2 and len(mask_files) >= 1:
            try:
                mask = np.load(mask_files[0])
                depths_at_mask = []
                for df in depth_files:
                    depth = np.load(df)
                    if depth.shape[:2] == mask.shape[:2]:
                        masked_depth = depth[mask > 0]
                        if masked_depth.size > 0:
                            depths_at_mask.append({
                                "file": df.name,
                                "mean_depth": float(np.mean(masked_depth)),
                                "std_depth": float(np.std(masked_depth)),
                            })
                if len(depths_at_mask) >= 2:
                    depth_change = abs(
                        depths_at_mask[-1]["mean_depth"] - depths_at_mask[0]["mean_depth"]
                    )
                    info["depth_at_mask"] = depths_at_mask
                    info["depth_change_at_object"] = round(depth_change, 4)
            except Exception as exc:
                info["depth_analysis_error"] = str(exc)

    return info


def analyze_motion(phases: list[dict]) -> dict[str, Any]:
    """Analyze robot motion: grasp approach, lift height, place position."""
    info: dict[str, Any] = {}

    # Find grasp position (from solve_ik in PRE_GRASP/GRASP)
    grasp_positions = []
    lift_positions = []
    place_positions = []

    for p in phases:
        if p["function"] == "solve_ik":
            pos = _get_target_position(p)
            if pos is None:
                continue
            phase = p["phase"]
            if phase in ("PRE_GRASP", "GRASP"):
                grasp_positions.append(pos)
            elif phase == "LIFT":
                lift_positions.append(pos)
            elif phase in ("TRANSPORT", "PLACE"):
                place_positions.append(pos)

    if grasp_positions:
        info["grasp_approach_pos"] = grasp_positions[0]
        info["grasp_final_pos"] = grasp_positions[-1]
        if len(grasp_positions) >= 2:
            approach_z = grasp_positions[0][2]
            grasp_z = grasp_positions[-1][2]
            info["approach_height"] = round(approach_z, 4)
            info["grasp_height"] = round(grasp_z, 4)

    if lift_positions:
        info["lift_pos"] = lift_positions[-1]
        if grasp_positions:
            lift_delta_z = lift_positions[-1][2] - grasp_positions[-1][2]
            info["lift_delta_z"] = round(lift_delta_z, 4)

    if place_positions:
        info["place_pos"] = place_positions[-1]

    return info


# ────────────────────────────────────────────────────────────────
# Verdict
# ────────────────────────────────────────────────────────────────


def compute_verdict(
    gripper_info: dict,
    displacement_info: dict,
    grasp_info: dict,
    motion_info: dict,
    comovement_info: dict | None = None,
) -> dict[str, str]:
    """Produce a structured verdict from analysis results."""
    verdict: dict[str, str] = {}

    # --- Gripper closed on object ---
    if comovement_info:
        status = comovement_info.get("gripper_closed_on_object", "UNKNOWN")
        confidence = comovement_info.get("confidence", "")
        verdict["gripper_closed"] = f"GRIPPER_CLOSED_ON_OBJECT: {status} ({confidence})"

    # --- Grasp verdict ---
    if "object_in_gripper" in gripper_info:
        if gripper_info["object_in_gripper"]:
            gw = gripper_info.get("grasp_gripper_width", "?")
            verdict["grasp"] = f"GRASP LIKELY SUCCEEDED (gripper_width={gw:.4f})"
        else:
            gw = gripper_info.get("grasp_gripper_width", "?")
            verdict["grasp"] = f"GRASP LIKELY FAILED (gripper_width={gw:.4f}, closed on air)"
    else:
        # Fallback: use grasp score + geometric analysis
        score = grasp_info.get("grasp_score")
        found = grasp_info.get("found_grasp")
        if found and score is not None:
            if score > 0.5:
                verdict["grasp"] = f"GRASP PLAUSIBLE (score={score:.3f}, no gripper confirmation)"
            else:
                verdict["grasp"] = f"GRASP UNCERTAIN (score={score:.3f}, no gripper confirmation)"
        else:
            verdict["grasp"] = "GRASP UNKNOWN (no grasp planned or no gripper data)"

    # --- Object movement verdict ---
    displacement_keys = [k for k in displacement_info if k.endswith("_bbox_displacement_px")]
    if displacement_keys:
        max_disp = max(displacement_info[k] for k in displacement_keys)
        if max_disp > 20:
            verdict["object_moved"] = f"OBJECT MOVED (max bbox displacement={max_disp:.1f}px)"
        else:
            verdict["object_moved"] = f"OBJECT DID NOT MOVE (max bbox displacement={max_disp:.1f}px)"
    elif "depth_change_at_object" in displacement_info:
        dc = displacement_info["depth_change_at_object"]
        if dc > 0.02:
            verdict["object_moved"] = f"OBJECT LIKELY MOVED (depth change={dc:.4f}m)"
        else:
            verdict["object_moved"] = f"OBJECT LIKELY DID NOT MOVE (depth change={dc:.4f}m)"
    else:
        verdict["object_moved"] = "OBJECT MOVEMENT UNKNOWN (insufficient data)"

    # --- Lift verdict ---
    lift_dz = motion_info.get("lift_delta_z")
    if lift_dz is not None:
        if lift_dz > 0.05:
            verdict["lift"] = f"LIFT ATTEMPTED (delta_z={lift_dz:.4f}m)"
        else:
            verdict["lift"] = f"MINIMAL LIFT (delta_z={lift_dz:.4f}m)"

    return verdict


# ────────────────────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────────────────────


def format_report(
    trial_dir: Path,
    phases: list[dict],
    grasp_info: dict,
    gripper_info: dict,
    displacement_info: dict,
    motion_info: dict,
    verdict: dict,
    comovement_info: dict | None = None,
    verbose: bool = False,
) -> str:
    lines: list[str] = []
    sep = "─" * 60

    lines.append(sep)
    lines.append(f"TRIAL ANALYSIS: {trial_dir.name}")
    lines.append(sep)

    # Phase summary
    lines.append("\n## Phases")
    current_phase = None
    for p in phases:
        if p["phase"] != current_phase:
            current_phase = p["phase"]
            lines.append(f"\n  [{current_phase}]")
        fn = p["function"]
        step = p["step"]
        dt = p["duration_ms"]
        extra = ""
        if verbose:
            r = p.get("result", {})
            if fn == "close_gripper" and "gripper_width" in r:
                extra = f" → gripper_width={r['gripper_width']:.4f}"
            elif fn == "select_top_down_grasp":
                extra = f" → found={r.get('found_grasp')}, score={r.get('score', '?')}"
            elif fn == "plan_grasp":
                extra = f" → {r.get('num_grasps', '?')} grasps, best={r.get('best_score', '?')}"
            elif fn == "solve_ik":
                pos = p.get("args", {}).get("position")
                if pos:
                    extra = f" → target=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        lines.append(f"    step {step:3d}: {fn} ({dt}ms){extra}")

    # Grasp quality
    lines.append(f"\n## Grasp Quality")
    if grasp_info.get("found_grasp"):
        lines.append(f"  Grasp found: yes")
        lines.append(f"  Score: {grasp_info.get('grasp_score', '?')}")
        pos = grasp_info.get("grasp_world_pos")
        if pos:
            lines.append(f"  World position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        lines.append(f"  Candidates from GraspNet: {grasp_info.get('num_grasps', '?')}")
    else:
        lines.append(f"  No grasp planned or select_top_down_grasp not found")

    # Gripper state
    lines.append(f"\n## Gripper State")
    if gripper_info["has_gripper_width"]:
        for ev in gripper_info["events"]:
            gw_str = f", width={ev['gripper_width']:.4f}" if "gripper_width" in ev else ""
            lines.append(f"  step {ev['step']}: {ev['action']}{gw_str}")
        if "obs_gripper_widths" in gripper_info:
            for ow in gripper_info["obs_gripper_widths"]:
                lines.append(f"  step {ow['step']}: get_observation → gripper_width={ow['gripper_width']:.4f}")
    else:
        lines.append(f"  No gripper width data (old trace format)")
        for ev in gripper_info["events"]:
            lines.append(f"  step {ev['step']}: {ev['action']}")

    # Gripper-object co-movement
    if comovement_info:
        lines.append(f"\n## Gripper-Object Co-movement")
        status = comovement_info.get("gripper_closed_on_object", "UNKNOWN")
        lines.append(f"  GRIPPER_CLOSED_ON_OBJECT: {status}")
        lines.append(f"  Confidence: {comovement_info.get('confidence', 'N/A')}")
        for sig in comovement_info.get("signals", []):
            lines.append(f"    [{sig['interpretation']}] {sig['name']}: {sig['detail']}")

    # Object displacement
    lines.append(f"\n## Object Displacement")
    disp_keys = [k for k in displacement_info if k.endswith("_bbox_displacement_px")]
    if disp_keys:
        for k in disp_keys:
            prompt = k.replace("_bbox_displacement_px", "")
            lines.append(f"  {prompt}: {displacement_info[k]:.1f}px bbox center shift")
    if "depth_change_at_object" in displacement_info:
        lines.append(f"  Depth change at object mask: {displacement_info['depth_change_at_object']:.4f}m")
    if not disp_keys and "depth_change_at_object" not in displacement_info:
        lines.append(f"  Insufficient data for displacement analysis")

    # Motion
    lines.append(f"\n## Motion")
    if motion_info.get("approach_height") is not None:
        lines.append(f"  Approach height: {motion_info['approach_height']}m")
    if motion_info.get("grasp_height") is not None:
        lines.append(f"  Grasp height: {motion_info['grasp_height']}m")
    if motion_info.get("lift_delta_z") is not None:
        lines.append(f"  Lift delta-z: {motion_info['lift_delta_z']}m")
    if motion_info.get("place_pos"):
        pp = motion_info["place_pos"]
        lines.append(f"  Place position: [{pp[0]:.4f}, {pp[1]:.4f}, {pp[2]:.4f}]")

    # Verdict
    lines.append(f"\n{sep}")
    lines.append("## VERDICT")
    lines.append(sep)
    for key, v in verdict.items():
        lines.append(f"  {v}")
    lines.append(sep)

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline trial analysis")
    parser.add_argument("--trial-dir", type=Path, required=True, help="Path to trial directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed step info")
    parser.add_argument("--json", action="store_true", help="Output raw analysis as JSON")
    args = parser.parse_args()

    trial_dir: Path = args.trial_dir
    trace_path = trial_dir / "trace.json"
    if not trace_path.exists():
        print(f"ERROR: {trace_path} not found", file=sys.stderr)
        sys.exit(1)

    entries = json.loads(trace_path.read_text())
    keyframes_dir = trial_dir / "keyframes"
    if not keyframes_dir.exists():
        keyframes_dir = None

    # Run analyses
    phases = classify_phases(entries)
    grasp_info = analyze_grasp_quality(entries)
    gripper_info = analyze_gripper_state(entries)
    comovement_info = analyze_gripper_object_comovement(entries, phases)
    displacement_info = analyze_object_displacement(entries, keyframes_dir)
    motion_info = analyze_motion(phases)
    verdict = compute_verdict(gripper_info, displacement_info, grasp_info, motion_info, comovement_info)

    if args.json:
        output = {
            "trial_dir": str(trial_dir),
            "num_steps": len(entries),
            "grasp": grasp_info,
            "gripper": gripper_info,
            "comovement": comovement_info,
            "displacement": displacement_info,
            "motion": motion_info,
            "verdict": verdict,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        report = format_report(
            trial_dir, phases, grasp_info, gripper_info,
            displacement_info, motion_info, verdict,
            comovement_info=comovement_info,
            verbose=args.verbose,
        )
        print(report)


if __name__ == "__main__":
    main()
