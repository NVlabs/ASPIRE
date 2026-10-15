# Optional Backends

cuRobo runs in an isolated environment so its CUDA dependencies cannot modify an
embodiment runtime. The harness process uses lightweight HTTP clients only.

## cuRoboV2

The Apache-2.0 source is pinned at `third_party/curobo`:

```bash
scripts/bootstrap_curobo.sh
```

The service's canonical port is `8118`. Runtime GPU placement and bounded concurrency come from the
selected hardware profile; `CAP_HARNESS_CUROBO_GPU` only overrides the bootstrap CUDA smoke check.
That check defaults to GPU `3`, which a single-GPU machine does not have. It passes regardless —
importing cuRobo needs no visible GPU — so this misleads rather than fails; set
`CAP_HARNESS_CUROBO_GPU=0` to keep the smoke check honest.
It supports standalone IK, collision-aware c-space planning to an IK result, integrated pose
planning, and integrated 14-DoF bimanual pose planning.

Collision checking is on for every model. `CAP_HARNESS_CUROBO_DISABLE_COLLISION` (any value
but `0`) disables it everywhere; a plan returned in that mode routes the arm through whatever the
planner was not shown, and nothing downstream re-checks it, so every such plan is logged as blind.

## Start services

```bash
scripts/bootstrap_providers.sh --providers curobo --profile rtx5090
scripts/supervise_services.sh --profile rtx5090 --providers curobo
cap-harness doctor --providers curobo
```

## Select backends in a program

```python
pyroki_curobo = MotionStrategy(
    ik_solver="pyroki",
    trajectory_planner="curobo",
)
curobo_integrated = MotionStrategy(pose_planner="curobo-integrated")

plan = plan_motion(target_pose, strategy=pyroki_curobo)
grasps = generate_grasps("agentview", mask, backend="contact-graspnet")
```

Programs may use several strategies in one episode. Unavailable services, unreachable targets, and
invalid inputs return typed failures. Backend fallback occurs only when the program requests it
explicitly.

## cuRobo inside OmniGibson (BEHAVIOR-1K)

The `behavior` benchmark does not use the cuRobo service on port 8118. Its `curobo` and
`curobo-integrated` backends are OmniGibson's own `CuRoboMotionGenerator`, constructed in the
simulator process with the R1 Pro model shipped in the robot assets and the scene's collision
meshes as the world. `MotionStrategy()` therefore defaults to cuRobo for IK, trajectories and
integrated pose planning on that benchmark; `pyroki` and `interpolation` are not registered for it.
