# Aspire YAM Station Real Deployment

This directory contains Aspire's real-station deployment surface for YAM. From
the top-level Aspire checkout, enter this workspace before running commands:

```bash
cd aspire/real
```

All paths below are relative to `aspire/real`:

```bash
source .forge_env
```

> A fresh clone is not yet self-contained. Before installing dependencies or
> operating hardware, follow [`SETUP.md`](SETUP.md), including its workstation
> recovery checklist for missing vendor sources, runtime assets, calibration,
> and sanitized local configuration. An agent on the known-good station can
> follow [`WORKSTATION_RECOVERY.md`](WORKSTATION_RECOVERY.md) to locate and
> report the missing inputs safely.

Recovered dependency patches, locked build inputs, and the reviewed installer
are described in [`install/README.md`](install/README.md).

## Canonical Demo Quickstart

This path is for a workstation with the same dual-arm YAM station, four D405
cameras, CAN interface names, calibrated station model, and NVIDIA runtime used
by the checked-in demo. Complete [`SETUP.md`](SETUP.md) once before running it.

Configure the ignored station environment and run the read-only static checks:

```bash
cp .forge_env.example .forge_env
$EDITOR .forge_env
bash tools/yam_demo_preflight.sh
```

With an operator at the station and the emergency stop reachable, start only
the services used by the saved demo:

```bash
bash tmux/launch_yam_demo_services.sh --no-attach
```

The launcher starts both follower arm servers, the read-only camera Portal,
SAM3, and BundleSDF in dependency order. It waits for each service to become
healthy and then runs the service-level preflight. It intentionally does not
start AnyGrasp, cuRobo, PyRoki, an LLM provider, voice services, or evaluation
agents because the canonical saved scripts do not call them.

Inspect service logs with `tmux attach -t yam_demo`. Stop the service group
after the run with `tmux kill-session -t yam_demo`.

With an operator at the station, the emergency stop reachable, and the demo
scene prepared, run:

```bash
bash cap/saved_scripts/yam_demo.sh full
```

Physical motion is enabled by the task wrappers; do not run this command as an
unattended smoke test.

## Manual Service Operation

The demo-specific launcher above is the normal entrypoint. To start one
component manually, use:

```bash
bash tools/run_yam_demo_service.sh left-arm
bash tools/run_yam_demo_service.sh right-arm
bash tools/run_yam_demo_service.sh camera-portal
bash tools/run_yam_demo_service.sh sam3
bash tools/run_yam_demo_service.sh bundlesdf
```

The broader `tmux/launch_realworld_localserver_realsense.sh` launcher remains
available for development workflows that need AnyGrasp, cuRobo, PyRoki, or
provider services. It is not required for `yam_demo.sh full`.

## Canonical Demo

Use `yam_demo.sh` as the operator entrypoint:

```bash
bash cap/saved_scripts/yam_demo.sh full
```

Useful subcommands:

```bash
bash cap/saved_scripts/yam_demo.sh bowls
bash cap/saved_scripts/yam_demo.sh white-dish
bash cap/saved_scripts/yam_demo.sh orange-on-white
bash cap/saved_scripts/yam_demo.sh can-trash
bash cap/saved_scripts/yam_demo.sh kitkat
bash cap/saved_scripts/yam_demo.sh bottle-rack
bash cap/saved_scripts/yam_demo.sh home
```

`full` runs drawer-close, home, white dish, orange-on-white, can-trash, KitKat
handover/trash, then bottle rack.

## Preflight Troubleshooting

`tools/yam_demo_preflight.sh` exits with status 2 if a required item is
missing. Resolve failures before starting the service tmux session:

| Failure | Resolution |
| --- | --- |
| `.forge_env` or calibrated XML missing | Copy `.forge_env.example`, set `YAM_STATION_CALIBRATED_XML`, and verify that the XML and its referenced meshes are readable. |
| `i2rt` or BundleSDF source missing | Materialize the reviewed local source under `third_party/`; see [`install/README.md`](install/README.md). |
| BundleSDF runtime libraries missing | Set `BUNDLESDF_RUNTIME_LIB_DIR` to the reviewed runtime closure. |
| Python module missing | Complete the station installation in the Python 3.11 `.venv`. |
| Camera alias missing | Restore the station's `/dev/video_{top,left,right,bottom}` udev mappings. |
| Follower CAN interface missing | Restore `can_follow_l` and `can_follow_r` and confirm the expected 1 Mbps configuration. |
| Service startup times out | Attach with `tmux attach -t yam_demo` and inspect the failing service window. |

## License

ASPIRE-owned real-robot code is available under the project-level
[Apache License 2.0](../../LICENSE). This workspace also references or contains
separately licensed dependency patches, SDK integrations, robot descriptions,
and assets. Some tracked YAM CAD/calibration provenance remains unresolved and
must be cleared or removed before public redistribution. Review the
repository's [third-party notices](../../THIRD_PARTY_NOTICES.md) for the
attribution summary, and consult the
[third-party license catalog](../../THIRD_PARTY_LICENSES.md) for exact
revisions, component-specific terms, and release blockers.

The static preflight does not contact arm RPCs or send commands. The
`--services` form performs read-only health queries after startup.

## Real Demo Video

Real demo MP4 placeholder:

```text
media/aspire_yam_station_real_demo.mp4
```

## Layout

- `cap/saved_scripts/yam_demo.sh`: only top-level shell entrypoint.
- `cap/saved_scripts/shell_scripts/`: internal shell helpers used by
  `yam_demo.sh`.
- `cap/saved_scripts/*.py`: active task implementations.
- `cap/saved_scripts/yam_runtime/`: shared observation, artifact, gripper,
  planning, and verification helpers.
- `cap/saved_scripts/skill_library/`: `run_script.py` skill-library injection
  support and older helpers; still required.
- `cap/saved_scripts/legacy_codes/`: old probes, sweeps, recovery scripts, and
  alternate attempts.

## Runtime Expectations

Physical robot commands must set `OPENFORGE_ALLOW_PHYSICAL_MOTION=1`; the
saved demo helpers already do this. Keep recording/debug UI enabled for real
runs. Logs are written under `logs/<script>_<timestamp>/` and should include
`result.json`, `task_result.json` when applicable, `stage_summary.md`, `plans/`,
`observations/`, overlays, and camera videos.

Before a real run, verify arm/camera/perception servers are healthy. See the
Codex skill `yam-server-setup` for the current checks and restart commands.

## Agent Skills

Read [`AGENTS.md`](AGENTS.md) before agent-driven real-robot work. Reusable
robotics lessons live in [`.agents/skills/`](.agents/skills/):

- `yam-full-demo`: canonical commands and demo flow.
- `yam-grasp-pickup`: detection, grasp candidates, staged close, and lift
  checks.
- `yam-transport`: held-object transport and rack/bin approach patterns.
- `yam-retreat-recovery`: post-release retreat, home/open, and held-object
  recovery.
- `yam-robot-debugging`: run-observe-act loops and debugging workflow.
- `yam-motion-planner`: cuRobo failures and perturbation robustness sweeps.
- `yam-runtime-artifacts`: log/video/overlay inspection.
- `yam-server-setup`: service launch and readiness checks.
- `yam-simulation-transfer`: adaptation of simulation strategies to YAM.

Simulation code and agent guidance live in the sibling [`../sim/`](../sim/)
workspace. Use `yam-simulation-transfer` to adapt strategy between simulation
and the real station; do not reuse simulator coordinates or APIs directly.
