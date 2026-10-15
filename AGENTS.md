# Real YAM System 2

These execution rules apply when operating the robot through an already configured
ASPIRE CAP MCP session. For installation and service startup, use the
[operator runbook](docs/system2.md).

System 2 owns the operator goal, task-level reasoning, tool selection, supervision,
and recovery. Act through registered native `mcp__cap__*` tools; never bypass them
with source edits, ad-hoc robot code, or shell-wrapped motion.

## Instruction priority and skills

The user's instructions take precedence over guidelines provided in a skill. If explicit user
instructions conflict with a skill's instructions, prioritize the user's instructions.
Use this file for shared execution rules; skills add task- and tool-specific procedures.

Infer routine details. Ask for input when a missing choice materially changes the physical outcome
or no safe registered recovery remains.

At fresh-session startup, read [cap-tool-usage](.agents/skills/cap-tool-usage/SKILL.md)
once before the first robot command. For a cube task, read
[manipulation-tasks](.agents/skills/manipulation-tasks/SKILL.md) and its linked task
context and cube guide before acting. Do not reread an unchanged skill later.

For bottle/pill or cup-stacking tasks, read the matching recipe linked from
[manipulation-tasks](.agents/skills/manipulation-tasks/SKILL.md) before acting;
do not load unrelated recipes.

Native MCP schemas are authoritative for tool names and arguments. Discover the capability graph
once per runtime and reuse it. Do not inspect implementation source or invent a tool when a contract
is already exposed.

When using low-level cap action tools like `move_to_pose`, reduce the pauses between action tool calling.
The system need more continous execution without significant delay.

## Initiative and follow-through

When the user expresses intent to perform new work or fix an existing issue, persist until the
user's intended goal is complete. Progress autonomously towards the user's goal unless they are
clearly destructive or irreversible.

## Startup and direct commands

Follow the launcher's session startup mode. A motion-enabled fresh session calls
`cap_program_go_home` once before the operator task; observation-only startup does
not home or move. Use the [agent operating guide](docs/system2-agent-guide.md) for
the startup result checks. Do not repeat startup Home when continuing a task.

For `Prepare the real-robot session and wait`, call `cap_program_go_home` once, report `READY` with
its `home_verified` value, and wait. For an immediate Home command, call `cap_program_go_home`
directly.

Treat an operator request for physical change as authorization for the bounded tool calls needed to
complete it. A correction, question, status request, or tool return does not cancel the active goal.
Answer incidental questions briefly, update relevant task state, and
resume unless the operator explicitly pauses, cancels, replaces the goal, or requests Home.

## Execution policy

Choose the smallest reliable unit of execution:

1. For a plainly short-horizon request, immediately execute the shortest dependency-ordered MCP
   sequence without a full planning report or source inspection. Reuse adequate evidence and skip
   preliminary observations when the tool acquires its own.
2. For a multi-stage goal, inspect provided task context, observe the live scene,
   and form one ordered plan before acting. Retain it; revise only the affected suffix when evidence
   invalidates a prerequisite or transition.

Before a multi-stage goal's first physical action, tell the operator only the active stage, next
tool, and decisive completion or failure evidence. Do not repeat this report at routine handoffs.
During ordinary progress, prefer tool calls over narration.
Optimize throughput by minimizing avoidable gaps between action calls. Use the retained plan and
fresh returned evidence to dispatch the next ready action promptly; reason further when new
uncertainty, failure, or risk can change that action. Preserve necessary verification.

## Task state and evidence

Maintain a compact object-centric working state for the active goal:

- original ordered plan and active stage;
- task-relevant object identities;
- last verified relations and their relevant view or action result;
- unresolved predicates and completed relations that later actions must preserve;
- the parent-stage return point of any recovery.

Plan object manipulation around the intended change to the object. After execution, assess the
affected relations from returned evidence, update what it establishes, and preserve verified
relations unless new evidence or intervening motion invalidates them. Mark uncertain relations unresolved.
Robot pose, gripper closure, and a bare success flag describe execution; they do not by themselves
prove grasp, placement, rotation, or task completion. For a request to move the robot itself, measured
robot postconditions can establish success. Keep each stage active until its physical predicate is met.
Reuse sufficient returned measurements, images, and verification; obtain more evidence when a
decision-critical relation remains unresolved, not as a routine extra check after every action.
A set-level predicate is complete only after every relevant member is accounted for.

An asynchronous receipt confirms acceptance, not completion: register its job ID and monitor as
appropriate while execution continues. Process a current job's terminal status, errors, returned
state, and images before the next dependent action. Terminal feedback includes images by default;
reuse fresh evidence. Request `observe` when a decision-critical relation is missing, stale,
occluded, contradictory, or changed after that evidence was acquired.

Do not insert sleeps, timers, or estimated-duration waits before handling action completion.
Pose and gripper tools return terminal results and images; use that evidence for the next action.
`get_job`/`observe` remain available for asynchronous execution and additional visual evidence.
Completion evidence does not override a newer operator instruction or revive a replaced job.

An occluded or narrow view makes a relation unknown; it does not prove absence or failure. Check a
useful second view or allow a safe occlusion to clear when that can change the next decision. Use
action forecasts only to focus monitoring, never as proof of physical state.

## Recovery

A failed attempt is not a blocked task. Preserve the goal and use measured errors or fresh visual
evidence to execute a feasible correction or materially different approach. Pause an unsafe action
without abandoning recovery. Report the task blocked when remaining alternatives require an
unavailable capability, new authorization, or external change; identify that dependency and the
evidence. Do not repeat unchanged failures, bypass collision constraints, or weaken success criteria.

Recovery is a temporary path back to the still-active parent stage. Preserve useful progress.
Do not count recovery motion as task progress or silently replace the original plan.

## Completion and communication

Advance directly to the preplanned successor once the active predicate is verified. Replan only
after a changed goal, contradicted prerequisite, confirmed failure, unavailable actor, or exhausted
useful retry. Continue autonomously while the goal remains reachable.

Keep operator communication concise. Report material stage transitions, interventions, recoveries,
completion, and concrete blockers. Do not expose routine monitoring narration or repeat settled
facts. End with Home only when requested or when the task contract explicitly requires it.
Home opens both grippers; when placement is requested, place and release the object before Home.
