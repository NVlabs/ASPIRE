# Scene snapshot — a generated CaP program for initial task exploration.
#
# Run with the assigned first development seed (LIBERO: 51; Robosuite: 101):
#
#   cap-harness run --benchmark <benchmark> --suite <suite> --task-id <id> --seed <FIRST_DEV_SEED> \
#     --program .claude/skills/iterative-debugging/scripts/scene_snapshot.py \
#     --output-root "$TASK_DIR/debug/explore" --flat-run-dir --no-videos \
#     --init-mode <seeded for libero-pro | saved for robosuite; omit for behavior>
#
# Generated programs cannot import modules or touch the filesystem; the harness
# records everything. This program:
#   - reads the authoritative task language from get_task_context(),
#   - probes SAM3 prompts (each probe leaves a top-1 overlay in media/overlays/),
#   - opens a controllable gripper or holds a fixed tool's joints for one tick so
#     the recorder saves initial RGB keyframes around the public motion call.
#
# To validate task-specific prompts, copy this file into the task workspace and
# edit PROBE_PROMPTS after inspecting the first keyframes.

PROBE_PROMPTS = []  # e.g. ["white bowl", "silver drawer handle"]
# None = auto: the wide camera each benchmark publishes ("agentview" on LIBERO and
# Robosuite, "head" on BEHAVIOR); set explicitly to probe another camera.
PROBE_CAMERA = None

context = get_task_context()
state = get_robot_state()

report = {
    "task_language": context.language,
    "suite": context.suite,
    "task_id": context.task_id,
    "task_name": context.task_name,
    "task_metadata": dict(context.metadata),
    "base_frame": state.base_frame,
    "cameras": [],
    "prompt_probes": [],
}

observation = get_observation()
for camera_name in observation.cameras:
    report["cameras"].append(camera_name)

probe_camera = PROBE_CAMERA
if probe_camera is None:
    for candidate in ("agentview", "head", "top"):
        if candidate in report["cameras"]:
            probe_camera = candidate
            break
    if probe_camera is None and report["cameras"]:
        probe_camera = report["cameras"][0]
report["probe_camera"] = probe_camera

for prompt in PROBE_PROMPTS:
    probe = {"prompt": prompt, "ok": False, "num_masks": 0, "scores": [], "boxes": []}
    segmented = segment_text(probe_camera, prompt)
    if segmented.ok:
        probe["ok"] = True
        probe["num_masks"] = len(segmented.segmentations)
        for candidate in segmented.segmentations[:5]:
            probe["scores"].append(round(float(candidate.score), 3))
            if candidate.box_xyxy is not None:
                probe["boxes"].append([float(value) for value in candidate.box_xyxy])
    report["prompt_probes"].append(probe)

# Benign motion so the recorder saves before/after keyframes of the initial scene.
if context.family == "robosuite":
    gripper_arms = robosuite.get_controller_metadata()["controllable_gripper_arms"]
    if gripper_arms:
        open_gripper(arm=gripper_arms[0])
    else:
        commands = {}
        for arm in state.joint_positions:
            commands[arm] = ArmCommand("joint_position", state.joint_positions[arm])
        step(RobotAction(commands))
else:
    open_gripper()

result = report
