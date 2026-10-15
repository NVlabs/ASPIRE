# System 2 agent operating guide

These instructions are for the coding agent operating an already configured CAP
MCP session. `scripts/start_system2_agent.sh` includes this file in the agent's
initial instructions. The operator uses [the runbook](system2.md) to install and
start the services; by default they all run on the same machine.

Follow the repository's [AGENTS.md](../AGENTS.md). Before the first robot command,
read [.agents/skills/cap-tool-usage/SKILL.md](../.agents/skills/cap-tool-usage/SKILL.md)
for observation, localization, and the ASPIRE low-level action contract. For a cube
task, read [manipulation-tasks](../.agents/skills/manipulation-tasks/SKILL.md) and its
linked task context and cube strategy. Read each once; reuse it through the session.

Discover the active station and available capabilities through MCP. Camera names
and count come from the runtime. Calibration and service addresses belong to the
operator's configuration.

Follow the launcher's session startup mode. For a motion-enabled fresh session,
discover the tools and call `cap_program_go_home` with `{}` exactly once before
starting the operator task. This opens both grippers and returns both arms to
the station's configured home pose. Wait for completion, inspect fresh views,
and report `READY` with `home_verified`, measured joint residuals, and gripper
state. A residual outside the reported home tolerance is diagnostic; report it
accurately. If the command fails or either gripper is not open, stop and report
the failure. For observation-only startup, observe and report without motion.

When the operator says "go home", call `cap_program_go_home` with `{}`. This is
the Home shortcut and releases both grippers; it does not place a held object.
If the operator asks to put an object down and then home, complete placement first.
Do not home again between tasks unless requested.

Camera frames include acquisition age and stale/missing flags. Cached pixels do
not become fresh merely because they were polled again. CAP motions finish
atomically; the session is not the controller's emergency-stop mechanism.
