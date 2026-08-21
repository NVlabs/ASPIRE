# CLAUDE.md

**NEVER git push to ANY remote. Local commits only.**

**Always use Opus 4.6 (1M context) as the model.** Do not downgrade or switch models.

This file is the constitution for Claude Code working in this repository. Read it fully before doing anything.

---

## STARTUP: Read Project Memory First

At the start of every session, **immediately read** `.claude/memory/MEMORY.md` before doing anything else. It contains project vision, key conventions, and the skills index.

---

## What Is ASPIRE

ASPIRE is a framework where LLMs/VLMs write Python code to control robots through a curated tool API. Code is executed in a sandboxed MuJoCo simulator (Robosuite), the simulator verifies success, and the agent improves through failure diagnosis and skill accumulation.

Active benchmarks — **Robosuite** manipulation tasks:
`cube_lifting`, `cube_restack`, `cube_stack`, `nut_assembly`, `spill_wipe`, `two_arm_lift`, `two_arm_handover`

---

## ⛔ FORBIDDEN APIs — DO NOT USE

**These APIs access simulator ground truth and are STRICTLY FORBIDDEN in all fix code, debug scripts, and skill implementations.** Using them invalidates benchmark results — they don't transfer to real robots.

```
❌ env.handle.env.sim                   — no MuJoCo sim object access
❌ sim.data.body_xpos                   — no ground-truth object positions
❌ sim.data.get_site_xpos               — no ground-truth target locations
❌ sim.data.set_joint_qpos              — no joint teleportation
❌ sim.model.body_name2id               — no internal body ID lookup
❌ sim.model.joint_id2name              — no internal joint ID lookup
❌ sim.data.qpos                        — no raw joint state reading
❌ sim.forward()                        — no manual physics stepping
❌ env._step_once()                     — no raw environment stepping
❌ env.handle.env (accessing inner env) — no unwrapping the environment
```

**Why:** Previous fix codes used `body_xpos` for XY localization (bypassing perception) and `set_joint_qpos` for teleporting joints (bypassing manipulation). This produced artificially high success rates that don't transfer to real robots. Full audit: `docs/fix_code_api_audit.md`.

**Rule of thumb:** If a real robot with a camera could do it, it's allowed. If it reads the physics engine's internal state, it's forbidden.

**Also forbidden:** Reading simulator asset files (`.xml`, `.urdf`, MuJoCo model files) to infer object geometry or scene structure. Treat these as inaccessible — diagnose purely from observations and traces.

---

## ✅ ALLOWED APIs

Full source: `cap/integrations/franka/control_reduced_skill_library.py` (extends `control_reduced.py`). Read both for exact signatures, return types, and edge cases.

**Single-arm tasks** (`cube_lifting`, `cube_restack`, `cube_stack`, `nut_assembly`, `spill_wipe`):
```
✅ get_observation()                              — RGB, depth, intrinsics, extrinsics, robot state
✅ segment_sam3_text_prompt(rgb, text)             — SAM3 text-prompted segmentation
✅ segment_sam3_point_prompt(rgb, point_coords)    — SAM3 point-prompted segmentation
✅ plan_grasp(depth, intrinsics, segmentation)     — GraspNet grasp planning (returns camera-frame poses)
✅ select_top_down_grasp(poses, scores, cam_to_world) — filter for top-down grasps, returns world-frame pose
✅ get_oriented_bounding_box_from_3d_points(pts)   — OBB geometry {center, extent, R}
✅ solve_ik(position, quaternion_wxyz)             — IK → joint angles (7,)
✅ move_to_joints(joints)                         — blocking motor control
✅ open_gripper() / close_gripper()               — gripper control
✅ mask_to_world_points(mask, depth, K, T)        — 2D mask → (N,3) world-frame point cloud
✅ depth_to_point_cloud(depth, intrinsics)         — depth image → (H,W,3) camera-frame cloud
✅ pixel_to_world_point(u, v, z, intrinsics, extrinsics) — single pixel → 3D world point
✅ decompose_transform(T)                         — 4×4 → (position (3,), quaternion_wxyz (4,))
✅ rotation_matrix_to_quaternion(R)               — 3×3 rotation matrix → quaternion wxyz (4,)
✅ transform_points(points, T)                    — apply 4×4 transform to (N,3) or (H,W,3) points
✅ interpolate_segment(p1, p2, step=0.03)         — generate waypoints along a line segment
✅ normalize_vector(v)                            — normalize 3D vector
✅ point_prompt_molmo(image, text)                — Molmo pixel grounding → {text: (pixel_x, pixel_y)}
✅ numpy, scipy, standard math                   — computation
```

**Bimanual tasks** (`two_arm_lift`, `two_arm_handover`) replace the single-arm motion/IK functions with arm-suffixed variants:
```
✅ solve_ik_arm0(position, quaternion_wxyz)       — IK for arm 0 → (7,)
✅ solve_ik_arm1(position, quaternion_wxyz)       — IK for arm 1 → (7,), input in robot0 base frame
✅ move_to_joints_arm0(joints)                   — blocking motor control for arm 0
✅ move_to_joints_arm1(joints)                   — blocking motor control for arm 1
✅ move_to_joints_both(joints0, joints1)          — simultaneous blocking control for both arms
✅ open_gripper_arm0() / close_gripper_arm0()    — gripper control for arm 0
✅ open_gripper_arm1() / close_gripper_arm1()    — gripper control for arm 1
```
All geometry, perception, and utility functions are the same across single-arm and bimanual tasks.

**Note — `spill_wipe`:** Uses `tcp_offset=[0, 0, -0.0158]` (shorter than standard `[0, 0, -0.107]`) due to the sponge tool attachment.

**You are encouraged to design new helper functions** (e.g. `localize_object()`, `make_topdown_quat()`) when you find yourself repeating patterns. Add them to the relevant experiment's `skills/` folder so they're captured in that experiment's snapshot. Good helpers = the skill library grows.

---

## Skills & Prompts

This is the **robosuite** task suite. Its experiments live under
`.claude/robosuite/<experiment>/` (see [../README.md](../README.md) for the full index):

- `fix-loop/` — baseline → iterative fix loop → eval (seeds 1–100)
- `training-law/` — cumulative tokens vs. success rate scaling law

**Suite-shared references** (used by every experiment in this suite):

| File | What it covers |
|---|---|
| [`api-reference.md`](api-reference.md) | Full robosuite control API, output structure, TraceLogger format, source files |
| [`run-baseline.md`](run-baseline.md) | Baseline launch command, config paths, output structure for all 7 tasks |

**Per-experiment files** (each `<experiment>/` folder is a self-contained snapshot):

| File | What it covers |
|---|---|
| `<experiment>/SKILL.md` | Master reference: system overview, setup, running experiments, debugging |
| `<experiment>/main-agent-prompt.md` | Fix-loop coordinator prompt |
| `<experiment>/subagent-prompt.md` | Per-task debug subagent prompt |
| `<experiment>/clean-task-slate.md` | Reset checklist before a rerun |
| `<experiment>/skills/grasp.md` | Pick-and-place template: make_topdown_quat(), pre-grasp/lower/close/lift/place skeleton |
| `<experiment>/skills/localize.md` | SAM3/Molmo prompting, multi-prompt fallback, per-object prompt registry |
| `<experiment>/skills/transport.md` | Motion patterns: waypoints, safe transit, interpolated moves |

Read the relevant experiment's `SKILL.md` and `skills/` before starting work.

---

## Key Philosophy: Self-Evolve

The skills library is **self-improving** — that's the whole point of ASPIRE. When you discover a new pattern, fix a recurring bug, or solve a task in a new way:

1. **Update or add the relevant skill** in the experiment's `skills/` folder
2. **Skills must ONLY use allowed APIs** — never add MuJoCo ground-truth patterns
3. **Write reusable analysis as scripts** in `scripts/` (not inline `python3 -c "..."`), then reference them in the skill
4. **Write a log entry** in `./docs/logs/YYYY-MM-DD.md` — what changed, what worked, what failed

Use `grep` on `./docs/logs/` to find relevant context when you hit familiar issues.

---

## Log Everything

Append to `./docs/logs/YYYY-MM-DD.md` after any significant work: experiments run, fixes tried, skills updated, results observed. Be detailed — include trial numbers, success rates, error messages, file paths.
