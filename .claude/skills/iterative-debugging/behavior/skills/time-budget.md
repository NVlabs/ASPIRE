# Time budget

An episode is bounded by `MAX_STEPS` simulator ticks (30 per simulated second; 6000 = 200 s) and
by wall clock. Rough costs measured or expected on one RTX 4090:

| Call | Steps | Wall clock | Note |
|---|---|---|---|
| Isaac launch + scene load | 0 | 35–50 s with warm caches; about 190 s the first time a scene loads; +5 min shader compile on a fresh machine | outside the step budget |
| `localize_object` | 0 | 1–3 s | one SAM3 round trip |
| `turn_in_place(0.5)` (servo) | 5 (servo, measured 2026-09-12) | seconds | in place |
| `navigate_to_pose(..., "curobo")` | 100–600 | 20–60 s | includes one base plan |
| `move_to_pose` (cuRobo arm plan) | 50–400 | 5–30 s | first plan of a session warms cuRobo up (up to a minute) |
| `close_gripper` / `open_gripper` | ≤40 | 1–2 s | |
| `generate_grasps` | 0 | 2–5 s | one Contact-GraspNet round trip |

Budgets to respect inside a policy: at most 12 search turns; one standoff navigation; four
grasp attempts. A policy that spends its steps searching never reaches the grasp; when a run
ends with `step_limit`, cut an earlier block before touching the grasp.

**Every attempt replays every earlier block**, so a block's cost is paid again on each of the
attempts that follow it. A block that turns twelve times to find the object costs its steps and
seconds for the rest of the seed; once you know which turn found it, that block is the first
thing to trim in Stage 1, and in Stage 2 you cannot trim it at all.

Per attempt, a run took 45–170 s end to end on this bench; budget five minutes. At roughly two
minutes an attempt, ten learning seeds are a few hours and the twenty-five evaluation seeds are
most of a day. Plan around that and run campaigns in the background.
