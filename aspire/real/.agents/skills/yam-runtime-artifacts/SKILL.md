---
name: yam-runtime-artifacts
description: "Use when inspecting YAM logs, videos, SAM3 overlays, observations, planner previews, function-call JSON, result files, or failure evidence."
---

# YAM Runtime Artifacts

Run artifact commands from the Aspire real-robot workspace (`aspire/real`).
All `logs/**` paths below are relative to that directory.

Runs write to `logs/<script_name>_<YYYYMMDDTHHMMSS>/`.

Artifact inspection rule: tool return is not success. For physical debugging,
the two highest-value evidence sources are:

1. `debug_events.jsonl`: what the robot was commanded to do, in order, with
   timestamps, arguments, durations, and errors.
2. Video frames from `top.mp4`, `left.mp4`, `right.mp4`, and `bottom.mp4`: what
   physically happened before, during, and after those commands.

Do not infer contact, grasp, drawer motion, collision, or success from JSON
alone. Align command windows from `debug_events.jsonl` with extracted video
frames. If videos are missing or corrupt, treat the run as weak evidence unless
the task is specifically recorder debugging.

Core workflow:

1. Validate recorder output:

   ```bash
   python3 -m json.tool logs/<run>/preview_recording_result.json
   ```

   Each needed camera should have `ok=true`, `ffprobe.ok=true`, nonzero
   `duration_s`, nonzero `nb_frames`, `backend="python"`,
   `codec_name="h264"`, and `pix_fmt="yuv420p"`.

2. Read the robot-command chronology:

   ```bash
   rg -n 'tool_start|tool_end|freespace_move|servo_ee_delta|set_gripper|get_robot_state' logs/<run>/debug_events.jsonl
   ```

   Use this to identify the exact windows for approach, close, contact, push,
   pull, release, retreat, and failures. Prefer actual tool arguments over
   script labels when deciding motion direction.

3. Extract frames around those windows:

   ```bash
   mkdir -p /tmp/yam_frames
   for cam in top left right bottom; do
     for t in 0 10 20 30 40 50 60 70 80; do
       ffmpeg -hide_banner -loglevel error -y -ss "$t" \
         -i "logs/<run>/${cam}.mp4" -frames:v 1 \
         "/tmp/yam_frames/${cam}_${t}.jpg"
     done
   done
   ```

   Then inspect the relevant frames or make contact sheets. Always include
   frames before, during, and after the motion; a single final frame often
   hides whether contact was useful, transient, or accidental.

4. Answer the physical question from the paired evidence:

   - Did the gripper actually reach the target, or only the planned pose?
   - Did the fingers capture the object/handle, or slide along it?
   - Did the object move relative to fixed scene features?
   - Did the gripper open before retreat or while still engaged?
   - Did the final state persist after release?

Supporting artifacts:

- `result.json`: final reward/success packet from `run_script.py`. It may wrap
  script details under `details` or `info`. Treat it as an index, not proof.
- `stage_summary.md`: human-readable summary written by scripts that call
  `write_stage_summary`. Useful for compact config, final state, and
  `why_stopped`, but still verify against `debug_events.jsonl` and videos.
- `task_result.json`: some older scripts write their own result packet here.
  Compare with `result.json` if both exist.
- `exec.log`: process-level stdout/stderr and Python exceptions. Use this for
  import errors, uncaught tracebacks, and recorder startup/shutdown messages.
- `run_<script>_<timestamp>.txt`: dashboard/tool-call transcript. It often
  includes sampled robot state before/after tool calls, in-flight tool status,
  and final concise state even when `result.json` is sparse.
- `profiling.json`: summarized tool timings and per-call results. Useful for
  confirming which motion-capable tools actually ran and whether tool errors
  occurred.
- `episode_config.json`: resolved run configuration such as env, robot mode,
  recording/debug UI settings, and script file.
- `code.py` and `code_snapshot.json`: copy of the executed script/source
  provenance. Use this to match artifacts to the code version that actually
  ran.

Video and recorder artifacts:

- `top.mp4`, `left.mp4`, `right.mp4`, `bottom.mp4`: BundleSDF preview videos.
  Primary physical evidence for contact, scene reset, object motion, grasp
  failure, and collision risk. Current runs should leave these root MP4s
  encoded as H.264 with `yuv420p` pixel format.
- `<camera>.pre_h264.mp4`: original OpenCV/Python preview-recorder output
  preserved before H.264 re-encode. Use it only for recorder debugging or to
  recover evidence if the root H.264 file is missing.
- `<camera>.h264_reencode.log`: ffmpeg stderr from the Python recorder's H.264
  post-encode step. Empty is normal; nonempty output can explain codec or
  finalization failures.
- `preview_recording_preflight.json`: preview availability before the run.
  Use it to prove BundleSDF preview streams were reachable at launch.
- `preview_recording_result.json`: recorder result after the run. A video is
  usable only if the camera entry has `ok=true`, `ffprobe.ok=true`, nonzero
  `duration_s`, nonzero `nb_frames`, a sane size, `codec_name="h264"`, and
  `pix_fmt="yuv420p"`. Prefer `backend="python"` for current real runs.
- `<camera>.preview_probe.log`: per-camera probe diagnostics.
- `<camera>.preview_recorder.log`: per-camera recorder diagnostics. If MP4s are
  48 bytes, missing, or ffprobe reports `moov atom not found`, inspect this log
  and treat the run as lacking visual evidence.
- `observations/`: raw or serialized camera/RGB-D/robot observations captured
  by the script. Use this for exact images, depth, masks, and robot state at
  named stages.
- `vis/observations/`: rendered observation images, overlays, and contact
  sheets. Use these before guessing from raw arrays.
- SAM3 or detector overlays: usually under `vis/`, `observations/`, or
  detector-specific subdirectories. Check selected masks/bboxes against the
  actual target; false positives can make an otherwise valid plan irrelevant.
- BundleSDF/object pose outputs: use these to compare perceived object pose,
  preview camera evidence, and any target pose used by the motion planner.
- `plans/`: candidate poses, waypoint previews, planner packets, failed IK/RRT
  details, and selected trajectory summaries.
- Planner preview images/videos: use these to check approach direction,
  clearance, tool orientation, and whether the gripper is aimed at the selected
  object or a false-positive mask.
- Function-call JSON: some scripts write attempted tool calls, parameters,
  and per-stage results. These are the bridge between perception/plans and
  robot motion.

Common conclusions:

- Command success but bad/missing videos: not enough evidence for physical
  debugging; repair recorder/BundleSDF and repeat or run an observe-only check.
- Valid videos but no object motion: inspect contact geometry, gripper state,
  target pose, and plan direction before changing perception.
- Video shows target reached but no capture: change approach/orientation/close
  sequencing before increasing travel distance.
- JSON says gripper closed but video shows sliding: treat it as a contact
  geometry problem, not a planner success.
- Planner success but visual miss: inspect selected detector/SAM3 mask and
  BundleSDF pose; the plan may have followed a false target.
- Gripper target differs from measured gripper position: likely contact or
  obstruction. Correlate with video before deciding whether it was useful
  contact.
- Robot state in `result.json` absent or sparse: use `run_*.txt`,
  `debug_events.jsonl`, and `profiling.json` for the actual final state and
  tool sequence.
