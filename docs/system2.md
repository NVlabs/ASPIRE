# System 2 operator runbook

Start the arm services, SAM3/cuRobo, and CAP MCP, then attach a fresh coding agent.
All services run on the **same machine** by default. If they are already running,
skip to [Start a fresh agent](#3-start-a-fresh-agent--terminal-5).

This file is for the operator. The launcher supplies the separate
[agent operating guide](system2-agent-guide.md) to the coding agent automatically.
The repository's [AGENTS.md](../AGENTS.md) holds shared robot execution rules;
[CAP tool usage](../.agents/skills/cap-tool-usage/SKILL.md) supplies the observation
and low-level action instructions.

## What the two launchers do

| Command | Starts | Meaning of `--allow-motion` |
| --- | --- | --- |
| `scripts/start_system2.sh` | CAP environment, MCP server, and Viser page | Allows the server to execute physical commands. Without it, the server rejects actuation and disables the Home button. |
| `scripts/start_system2_agent.sh` | A fresh Codex agent connected to MCP | Requests Home at startup and preapproves Home, pose, synchronized motion, joint targets, and single-arm gripper tools for that agent. |

**A motion-enabled agent opens both grippers and homes both arms at startup.**
Starting the MCP server alone does not home. Omitting the flag from the agent
launcher requests observation-only startup, but later tool approvals still follow
your Codex settings. To enforce observation-only operation through MCP, omit the
flag from the **server** command.

## 1. First-time setup

Run all commands from this repository's root. Follow [Host setup](host-setup.md)
for GPU drivers, `uv`, model-weight access, and other machine prerequisites.
Configure SocketCAN and camera access, and install and authenticate the Codex CLI.

Use an existing calibrated station profile; this startup procedure does not
generate camera calibration. See [Real YAM](yam-real.md) for the profile format
and the packaged `yam-example` reference.
Replace these placeholders and set the variables in **each service terminal**:

```bash
export ASPIRE_PROFILE="<hardware-profile>"
export ASPIRE_STATION="<station-id>"
export ASPIRE_CONFIG_ROOT="/absolute/path/to/station-profiles"
```

- `ASPIRE_PROFILE` selects a GPU configuration in `configs/profiles/`, such as
  `rtx5090`. It is independent of the station ID.
- `ASPIRE_STATION` matches the `station` field in your calibrated `station.yaml`.
- `ASPIRE_CONFIG_ROOT` contains the station directories and their calibration
  bundles. Use `127.0.0.1` and distinct ports for the two arm endpoints.

Install the environments once, or after dependency changes:

```bash
scripts/bootstrap_system2.sh
scripts/bootstrap_providers.sh --providers sam3,curobo --profile "$ASPIRE_PROFILE"
```

These create `.venv-system2`, `.venv-sam3`, and `.venv-curobo` inside this checkout.
No manual virtual-environment activation is needed for the commands below.

For **Astra-low**, edit these existing top-level keys in `~/.codex/config.toml`
(add them if absent):

```toml
model = "gpt-6-astra"
model_reasoning_effort = "low"
```

The agent launcher inherits these settings. It accepts `--allow-motion`, optional
`exec`, and task text; it does not forward Codex options such as `-m` or `-c`.

Shell commands run without approval prompts, with writes limited to the workspace
and temporary directories and network access enabled for package installation.
The launcher puts the `uv` package cache in `.uv-cache/`. Install task dependencies
in a repository virtual environment or temporary directory. These settings apply
to newly launched agents.

## 2. Start the services

Keep each command running in its own terminal. Reuse services already running for
this station; do not start a second arm controller or camera owner.

### Terminal 1 — SAM3 and cuRobo

```bash
scripts/supervise_services.sh --providers sam3,curobo --profile "$ASPIRE_PROFILE"
```

Wait for `all provider services ready`. Default ports are SAM3 **8114** and cuRobo
**8118**. Provider logs are in `validation-artifacts/service-logs/`.

### Terminal 2 — left arm

```bash
.venv-system2/bin/yam-servers left --station "$ASPIRE_STATION" \
  --config-root "$ASPIRE_CONFIG_ROOT"
```

### Terminal 3 — right arm

```bash
.venv-system2/bin/yam-servers right --station "$ASPIRE_STATION" \
  --config-root "$ASPIRE_CONFIG_ROOT"
```

Starting an arm service energizes its motors. Normal starts load saved gripper
calibration. Only for initial gripper calibration, add `--calibrate-gripper` with
empty, clear fingers; it moves the fingers and saves their measured travel.

### Terminal 4 — CAP MCP and visualizer

```bash
scripts/start_system2.sh --station "$ASPIRE_STATION" \
  --station-config-root "$ASPIRE_CONFIG_ROOT" --allow-motion \
  --sam3-url http://127.0.0.1:8114 --curobo-url http://127.0.0.1:8118
```

This connects to the arm services and owns the cameras. It serves MCP at
`http://127.0.0.1:8222/mcp/`. Adjust provider URLs if your hardware profile changes
their ports. SAM3 supplies segmentation, local Mink solves IK, and cuRobo plans
collision-aware task trajectories.

Open **http://127.0.0.1:8080** for the Viser page: a live **3D robot model**, a
**Go home** button, and camera views. Drag to orbit the model and scroll to zoom.
Both arms and grippers follow measured joint state from the shared session cache;
the model is hidden when that state is missing or stale. The URDF and meshes ship
with this repository. The model is a nominal station view; use the camera images
to assess objects and the actual surroundings.

The viewer discovers the active station's cameras, including stations with
two or three cameras; camera names and count are not fixed. Missing and stale
views are labeled. The viewer reads the existing session cache and opens no
additional cameras or arm connections.

The button runs the same Home program as MCP and is disabled while another job is
running or when the server has no motion permission. Opening the page does not
move the arms. Use `--viser-port PORT` to change its port or `--no-viser` to disable
the page. Viser starts automatically with MCP; no additional terminal is needed.

Before launching the agent, check the HTTP services from another terminal:

```bash
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8114/healthz
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8118/healthz
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8222/health
```

These establish service availability. The agent checks robot state and camera
freshness through MCP after connecting.

## 3. Start a fresh agent — Terminal 5

With the services running and Astra-low configured above:

```bash
scripts/start_system2_agent.sh --allow-motion
```

This creates a fresh interactive session. The expected startup sequence is:

1. Discover the CAP MCP tools.
2. Call `cap_program_go_home` once. Both grippers open, then both arms home.
3. Wait for the terminal result and inspect fresh camera views.
4. Report `READY`, the measured `home_verified` result, joint residuals, and gripper
   state, then wait for your task. A failed command or unopened gripper blocks the
   task; a home residual outside the reported tolerance is shown as a diagnostic.

Then type your task into that agent, for example:

```text
Pick up the cube using move_to_pose and the single-arm gripper tool.
Verify that it lifts from the table, then hold it.
```

The guide instructs the agent to use
`strategy={"ik_solver":"mink","trajectory_planner":"curobo"}` for pose planning
and verify the physical result from fresh evidence.

Pose targets accept either `rpy_deg: [roll, pitch, yaw]` in degrees or
`quaternion_wxyz: [w, x, y, z]`. Supply exactly one orientation format; see the
[Cartesian pose contract](api-reference.md#cartesian-pose-inputs) for an example.

To run a task noninteractively, use:

```bash
scripts/start_system2_agent.sh --allow-motion exec "Pick up the cube and hold it."
```

That also performs startup Home before the task. To start an agent that only
checks readiness and waits, omit `--allow-motion`. To attach another fresh agent,
rerun the launcher; services can remain running. Each motion-enabled fresh agent
performs Home again.

### Reproduce the manipulation demos

After completing the station setup above, select **one** task recipe for the
physical objects on the table. These text-based skills describe contact strategy,
recovery, and completion evidence; they do not require the original recordings,
Codex rollouts, or saved Cartesian targets.

| Task | Text-based task skill | Scene and completion evidence |
| --- | --- | --- |
| Rubik's Cube | [Task context](../examples/yam_real/rubiks_cube/TASK_CONTEXT.md) + [verified cube guide](../.agents/skills/manipulation-tasks/rubiks-cube.md) | A scrambled cube; verify all six faces uniform, aligned layers, and stable released placement. |
| Unscrew and pour | [Bottle-opening recipe](../examples/yam_real/unscrew_bottle/TASK_CONTEXT.md) | A rigid bottle with a conventional screw cap and a pill inside; verify detached cap, pill on the table, and released upright bottle. |
| Pill insertion and closure | [Pill-and-cap recipe](../examples/yam_real/pill_in_bottle/TASK_CONTEXT.md) | An open bottle, a loose pill that fits its mouth, and a compatible cap; verify containment and threaded seating, and report final tightness separately when unverified. |
| Inverted cup nesting | [Cup-stacking recipe](../examples/yam_real/cup_stacking/TASK_CONTEXT.md) | Compatible tapered cups; count all requested cups and verify one stable, opening-down nested stack after release. |

[AGENTS.md](../AGENTS.md) directs the agent to the
[manipulation skill entrypoint](../.agents/skills/manipulation-tasks/SKILL.md).
The original cube task context and manipulation guide are unchanged. The three
additional recipes are distilled from demo-session experience and checked against
the current tool contracts, but have not yet been revalidated with a fresh robot
session. In particular, retained cap seating does not establish final tightening
torque. Do not treat these instructions as a guarantee of success or skip the
recipe's physical verification.

Prepare the matching objects, leave the arms' startup Home paths clear, and run
one of the commands below from the repository root. Each fresh motion-enabled
agent opens both grippers and homes before acting; do not launch it while holding
an object. The recipes use the native low-level CAP tools, with no WAM checkpoint
or VLM supervisor required. Ground all poses in the current calibrated station.

For an already-ready agent, type the corresponding quoted task prompt into its
terminal instead of launching another session. For a reproducible trial, record the
model/reasoning settings, object setup and count, gripper type, operator
interventions, and verified final predicates; use the runbook's
[recording option](#stop-or-restart) to save the run.

### Unscrew and pour

```bash
scripts/start_system2_agent.sh --allow-motion \
  "Read examples/yam_real/unscrew_bottle/TASK_CONTEXT.md. Unscrew the bottle cap, place it separately, pour the pill onto the table, and leave the bottle upright and released."
```

### Pill insertion and bottle closure

```bash
scripts/start_system2_agent.sh --allow-motion \
  "Read examples/yam_real/pill_in_bottle/TASK_CONTEXT.md. Put the pill into the bottle, fit and tighten its cap, and leave the bottle standing on the table. Verify containment and cap seating; do not claim final tightness unless established."
```

### Inverted cup nesting

```bash
scripts/start_system2_agent.sh --allow-motion \
  "Read examples/yam_real/cup_stacking/TASK_CONTEXT.md. Nest all the cups upside down into one stable stack and release it."
```

### Solve a Rubik's Cube

With the same services running, start a fresh agent with the packaged task:

```bash
scripts/start_system2_agent.sh --allow-motion \
  "Read examples/yam_real/rubiks_cube/TASK_CONTEXT.md. Solve the physical Rubik's Cube and put it down on the table."
```

After startup Home, the agent reads the [task context](../examples/yam_real/rubiks_cube/TASK_CONTEXT.md)
and [cube strategy](../.agents/skills/manipulation-tasks/rubiks-cube.md). These cover
six-face reconstruction, supported layer turns, reorientation, recovery, and
verification of a solved cube released on the table. They use the live camera views
and existing low-level MCP actions. The agent needs an available symbolic cube
solver to compute and validate a solution for the current scramble.

## Home shortcut

Click **Go home** in Viser, or type **`go home`** in the agent terminal. Both use the
same session program. The agent calls the existing MCP tool:

```text
cap_program_go_home({})
```

Home opens each gripper through the single-arm tool, then moves both arms to the
profile's `home_joints` in one interpolated home motion. It reports the measured
joint residuals and `home_verified` using a 0.03 rad diagnostic tolerance.
ASPIRE's controller scales the motion duration with distance, from 1.5 to 8 seconds.
Home runs entirely inside this repository and does not need SAM3 or cuRobo.

Home releases anything held; it does not place an object. For placement first,
say **`Put the cube on the table, then go home`**. The home sweep uses joint
interpolation, so provide a clear path for both arms.

## Stop or restart

Exit the agent to end its conversation. MCP and the arm services remain running.
To shut down the stack, finish any motion, place held objects, and support the arms;
then stop MCP, the two arm services, and the provider supervisor with `Ctrl-C` in
their terminals. Stopping an arm service disables its motors. MCP shutdown waits
for in-flight execution and closes cameras; it is not an emergency stop.

For recording, add `--record-dir NEW_DIRECTORY` to the MCP command; add `--no-video`
to omit video. For an off-robot session, use MCP's `--sim` option instead of starting
the arm services; it still needs a configured station model.

## Included System 2 components

- `agent/server.py`: environment lifecycle, MCP HTTP endpoint, and operator steering.
- `agent/mcp.py`: tool schemas, dispatch, numeric results, and camera images.
- `agent/session.py`: serialized jobs, observations, and queued operator messages.
- `agent/embedded.py`: attaches the same session to a caller-owned CAP API.
- `agent/visualizer.py`: Viser robot model, camera views, and Home on the same session.
- `agent/programs/`: packaged observation and Home shortcuts.
- `yam_real/live_adapter.py` and `yam_real/control/runtime.py`: shared observations
  and one actuation worker.

Use `get_capability_graph` to discover the tools configured in the running session.

### Control path and dependencies

```text
Codex agent ── MCP ──> shared CAP session <── Viser Home button
                           │      └── cached state/images ──> Viser model and cameras
                           └── CAP API / live adapter / control worker
                                 └── RealYamEnv.execute_action_batch
                                       └── arm RPC clients
                                             └── left/right arm services
                                                   └── CAN motor drivers
```

`RealYamEnv` owns the station's RealSense camera readers. Pose tasks use SAM3 over
HTTP for segmentation, calibrated RGB-D for localization, local Mink for IK, and
cuRobo over HTTP for planning. Home uses the station's joint targets directly.

| Layer | Environment | Main dependencies |
| --- | --- | --- |
| Coding agent | Installed Codex CLI | Account authentication and selected model |
| MCP, session, and visualizer | `.venv-system2` | FastAPI, Uvicorn, MCP SDK, Viser, yourdfpy, Pillow |
| Real YAM environment and kinematics | `.venv-system2` | NumPy, SciPy, Portal RPC, RealSense SDK, MuJoCo, Mink |
| Arm services | `.venv-system2` | Packaged YAM controller, DaMiao motor library, SocketCAN |
| SAM3 service | `.venv-sam3` | SAM3, PyTorch, model weights |
| cuRobo service | `.venv-curobo` | cuRobo, CUDA, PyTorch, robot model |

The environment, arm server, session, and viewer implementations all live in this
repository. Station profiles supply arm endpoints, CAN interfaces, joint limits,
home targets, camera serials, and calibration; no other workspace checkout is used.
