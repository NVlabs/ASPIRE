# Environment architecture: simulators, providers, profiles

Simulator environments and model-serving provider environments are separate.
Simulators (LIBERO, Robosuite, BEHAVIOR-1K) install only their simulator runtime
plus the cap-harness client, and reach a single, shared, long-lived provider
stack over HTTP. BEHAVIOR-1K is the exception that proves the rule: its Isaac Sim
runtime cannot live in `uv.lock`, so `scripts/bootstrap_behavior.sh` builds
`.venv-behavior` from the pinned upstream installer steps, and motion planning for
it runs in-process on the simulator's own cuRobo instead of the cuRobo service. This keeps simulator installs light (no torch/model-server
packages) and lets one provider instance serve every simulator client.

## Single source of truth

`configs/environments.json` is the declarative topology. It lists every
simulator and provider, their canonical `.venv-<name>` basenames, provider
service modules, canonical ports, the client-only dependency set, and the
acceptance-critical provider ports. The supervisor, doctor, bootstraps, and
structural tests all resolve their configuration through
`cap_harness.environments` instead of hard-coding these values.

`configs/profiles/<name>.json` binds the topology to hardware. A profile sets
GPU placement, the build architecture (resolved against `gpu_runtime.arches` in
`configs/dependency-lock.json`), bounded per-provider request concurrency,
optional port overrides, and an optional node-local venv backing root. Two
profiles ship: `rtx5090` (single Blackwell sm_120 GPU; providers share GPU 0
with serialized access) and `l40` (multi-GPU Ada sm_89 node). Dependency
*versions* are shared and hardware-neutral; only the build *target* is per-arch.

## Virtual environment paths

Canonical, user-facing paths are always `<repo>/.venv-<name>`. On local storage
these are real directories. When `CAP_HARNESS_VENV_ROOT` is set (shared
node-local storage), the real environment lives under that root and
`<repo>/.venv-<name>` is a symlink to it; the symlink target preserves the
`.venv-<name>` basename, so name checks and doctor diagnostics stay correct.

## Bootstrapping

Simulators are bootstrapped independently and never require LIBERO first:

```bash
scripts/bootstrap_libero.sh       # -> .venv-libero   (client-only)
scripts/bootstrap_robosuite.sh    # -> .venv-robosuite (client-only)
scripts/bootstrap_behavior.sh     # -> .venv-behavior  (Python 3.11, Isaac Sim 5.1; datasets with --accept-dataset-tos)
```

Providers are bootstrapped per provider, or selectively via the
orchestration-only aggregator (there is deliberately no `bootstrap_sim.sh`
mega-installer):

```bash
scripts/bootstrap_sam3.sh --profile rtx5090     # -> .venv-sam3
scripts/bootstrap_pyroki.sh --profile rtx5090   # -> .venv-pyroki
scripts/bootstrap_providers.sh \
  --providers sam3,pyroki,curobo --profile rtx5090
```

Each bootstrap writes `<venv>/.cap-fingerprint.json` (a hash of the resolved
deps, pins, architecture, and submodule commits). An unchanged bootstrap is a
no-op instead of a rebuild.

## Running and diagnosing the shared provider stack

```bash
# one long-lived stack for a profile; GPU/ports/concurrency come from the profile
scripts/supervise_services.sh --profile rtx5090 --providers sam3,pyroki,curobo

# five-provider health/diagnostics, ports resolved from the topology
cap-harness doctor --providers sam3,contact_graspnet,pyroki,curobo
```

"Long-lived" means as long as that shell. The supervisor runs in the foreground
and traps `INT`/`TERM`/`EXIT` to `kill -TERM` every child, so a closed terminal,
a dropped SSH session or a process reaper takes **all** the providers down at
once -- and it does not restart them. There is no persistence built in. To
survive a disconnect, detach it yourself:

```bash
setsid nohup scripts/supervise_services.sh --profile rtx5090 \
  --providers sam3,pyroki,curobo >/tmp/providers.log 2>&1 &
disown
```

Worth knowing on a shared bench: a run taking minutes will fail partway if the
stack goes with the terminal, and it surfaces as provider calls timing out
rather than as anything naming the supervisor.

Each provider bounds concurrent inference via `CAP_HARNESS_SERVICE_CONCURRENCY`
(set from the profile: 1 serializes GPU access; CPU providers run higher) and
exposes `/healthz`.
