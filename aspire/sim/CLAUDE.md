# CLAUDE.md

**NEVER git push to ANY remote. Local commits only.**

This file is the constitution for Claude Code working in this repository. Read it fully before doing anything.

---

## STARTUP: Read This First

This file is the project constitution (forbidden APIs and conventions). Before starting suite-specific work, read the suite registry at [.claude/README.md](.claude/README.md), then the relevant suite guide under `.claude/<suite>/`. To reproduce a paper experiment, start at [`.claude/README.md`](.claude/README.md), then open the matching suite and experiment guide.

**Setting up the repo (venvs, submodules, perception servers)?** The canonical install instructions live in the root [`README.md` § Setup](README.md#setup); per-experiment run instructions live under `.claude/<suite>/<experiment>/INSTRUCTIONS.md`. Those are the single source of truth.

---

## What Is ASPIRE

ASPIRE is a framework where LLMs/VLMs write Python code to control robots through a curated tool API. Code is executed in isolated simulator trial processes (LIBERO/MuJoCo), the simulator verifies success, and the agent improves through failure diagnosis and skill accumulation.

## Security Boundary

**Simulator isolation is not a security sandbox.** Generated Python runs with
full import access and can reach anything available to its process. Run trial
workers on an isolated host or container without credentials or sensitive host
mounts, restrict network access, and route model requests through the
loopback-only proxy so provider keys stay outside worker arguments. Do not give
generated code access to physical hardware. Real-robot work must follow
`../real/AGENTS.md` and requires human review plus the documented motion gate.

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
❌ inner.parsed_problem                 — no goal state reading from BDDL
❌ inner._eval_predicate                — no reward function access
❌ inner.obj_body_id                    — no internal object ID mapping
❌ env._step_once()                     — no raw environment stepping
❌ env.handle.env (accessing inner env) — no unwrapping the environment
```

**Why:** Previous fix codes used `body_xpos` for XY localization and `set_joint_qpos` for teleporting joints. This produced artificially high success rates that don't transfer to real robots.

**Rule of thumb:** If a real robot with a camera could do it, it's allowed. If it reads the physics engine's internal state, it's forbidden.

**Also forbidden:** Reading simulator asset files (`.bddl`, `.xml`, `.urdf`, MuJoCo model files) to infer object geometry, success predicates, or scene structure. Treat these as inaccessible — diagnose purely from observations and traces.

---

## ✅ ALLOWED APIs

Full source: `cap/integrations/franka/libero_reduced_skill_library.py` (and `libero_reduced.py` for base class). Read the source for exact signatures, return types, and edge cases.

```
✅ get_observation()                         — RGB, depth, intrinsics, extrinsics, robot state
✅ segment_sam3_text_prompt(rgb, text)        — SAM3 text-prompted segmentation
✅ segment_sam3_point_prompt(rgb, points)     — SAM3 point-prompted segmentation
✅ mask_to_world_points(mask, depth, K, T)   — 3D point cloud from mask + depth
✅ get_oriented_bounding_box_from_3d_points(pts) — OBB geometry
✅ plan_grasp(depth, intrinsics, segmentation) — GraspNet grasp planning
✅ solve_ik(position, quaternion)            — inverse kinematics
✅ move_to_joints(joints)                   — motor control
✅ open_gripper() / close_gripper()         — gripper control
✅ goto_home_joint_position()              — move to robot home configuration
✅ decompose_transform(T)                   — math utility
✅ point_prompt_molmo(image, text)          — Molmo pixel grounding
✅ env.handle.task_language                 — task instruction string
✅ numpy, scipy, standard math             — computation
```

**You are encouraged to design new helper functions** (e.g. `localize_object()`, `make_topdown_quat()`) when you find yourself repeating patterns. Add them to the relevant skill in `.claude/libero/skills/` so future sessions can reuse them. Good helpers = the skill library grows.

---

## Suite Runbooks

Experiment runbooks live in `.claude/<suite>/<experiment>/`. Read the suite `CLAUDE.md`, then the experiment `INSTRUCTIONS.md` and `SKILL.md` before starting work.

> **Reproducing a paper experiment?** Start at [`.claude/README.md`](.claude/README.md), then open the matching suite guide and experiment `INSTRUCTIONS.md`.

| Location | What it covers |
|---|---|
| `.claude/robosuite/CLAUDE.md` | Robosuite suite constitution: allowed/forbidden APIs, baseline collection, fix-loop and training-law conventions. |
| `.claude/robosuite/fix-loop/` | Robosuite fix-loop experiment runbook, coordinator prompt, subagent prompt, and skill snapshots. |
| `.claude/robosuite/training-law/` | Robosuite training-law experiment runbook, coordinator prompt, subagent prompt, and token accounting. |
| `.claude/libero/CLAUDE.md` | LIBERO suite constitution: benchmark roles, API/baseline pointers, env conventions, perception preflight |
| `.claude/libero/fix-loop/` | Experiment 1: LIBERO-Pro fix loop. Coordinator + per-task subagent prompt for debug seeds 51-65 and held-out eval seeds 1-50. |
| `.claude/libero/evosearch/` | Experiment 2: Fix Loop + Evolutionary Search. Coordinator + subagent prompt for candidate search, validation selection, and final eval. |
| `.claude/libero/zeroshot-transfer/` | Experiment 3: LIBERO-90 skill-library build, chunk commits/tags, and zero-shot transfer handoff. |
| `.claude/libero/library-size-scaling/` | Experiment 4: frozen snapshot eval on LIBERO-Long-Pro and scaling tables/plots. |
| `.claude/libero/inference-time-scaling/` | Experiment 5: debug-compute/token-budget scaling on LIBERO-Long-Pro. |
| `.claude/libero/skills/` | Suite-shared LIBERO robot skills: grasp, localize, transport, and manipulation. Analysis procedures live in `.claude/libero/analysis/`. |

---

## Key Philosophy: Self-Evolve

The skills library is **self-improving** — that's the whole point of ASPIRE. When you discover a new pattern, fix a recurring bug, or solve a task in a new way:

1. **Update or add the relevant skill** in `.claude/libero/skills/`
2. **Skills must ONLY use allowed APIs** — never add MuJoCo ground-truth patterns
3. **Write reusable analysis as scripts** in `scripts/` (not inline `python3 -c "..."`), then reference them in the skill
4. **Write a log entry** in `./docs/logs/YYYY-MM-DD.md` — what changed, what worked, what failed

Use `grep` on `./docs/logs/` to find relevant context when you hit familiar issues.

---

## Log Everything

Append to `./docs/logs/YYYY-MM-DD.md` after any significant work: experiments run, fixes tried, skills updated, results observed. Be detailed — include trial numbers, success rates, error messages, file paths.
