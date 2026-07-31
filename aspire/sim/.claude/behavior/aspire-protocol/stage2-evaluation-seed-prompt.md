# Stage 2 Fresh Evaluation-Seed Agent Prompt

You are a fresh isolated Claude Code agent for exactly one held-out
BEHAVIOR-1K seed. You have no useful memory from any other held-out seed.

## Immutable Inputs

- Task: `<Soda Can|Radio>`
- Held-out seed: `<SEED>`
- Traced config: `<CONFIG_PATH>`
- Frozen manifest: `<CAMPAIGN_ROOT>/manifest.md`
- Frozen skill library: `<CAMPAIGN_ROOT>/skill-library-frozen`
- Your private seed directory: `<CAMPAIGN_ROOT>/stage2/seed_<NN>`
- Isaac Sim GPU: `<GPU_ID>`
- Per-seed limits: `<EVALUATION_SEED_BUDGETS>`

Verify the frozen-library checksum before acting. Your read scope is limited to
the immutable inputs, suite/API/fix-loop documentation at the recorded commit,
and your private seed directory. Do not read Stage 1 policies or traces, other
Stage 2 seed directories, coordinator notes about other evaluation outcomes,
or any mutable skill library.

## Episode

Start with a fresh empty `policy.py`; do not copy a development or evaluation
policy. Construct your policy block by block using the frozen skills:

1. Append a small `# Code block N` section of roughly 5-20 executable lines.
2. Replay only your assigned seed from reset, writing to a new attempt path.
3. Inspect only your own summary, trace, keyframes, video, observations, and
   relevant service logs.
4. Diagnose and append or minimally revise the next block.
5. Repeat until the task succeeds or the fixed budget is exhausted.

Use this command shape only:

```bash
OMNIGIBSON_GPU_ID=<GPU_ID> uv run --no-sync --active \
  scripts/behavior/replay_trial_b1k.py \
  --config-path <CONFIG_PATH> \
  --replay-code <CAMPAIGN_ROOT>/stage2/seed_<NN>/policy.py \
  --trial <SEED> \
  --output-dir <CAMPAIGN_ROOT>/stage2/seed_<NN>/attempts/attempt_<NNN> \
  --record-video
```

This within-seed debugging is the evaluation procedure; it is not a one-shot
frozen-policy replay. However, your episode is the seed's only evaluation
episode. Do not ask for a replacement run if it fails.

At termination, write `seed-summary.md` with the task-completion outcome,
terminal reason, replay count, failure classification, and exact artifact
paths. Do not update any skill library, communicate lessons to later agents,
run another seed, push, delete evidence, invoke the built-in ASPIRE model loop,
or touch real hardware. Return only your seed, terminal status/outcome, replay
count, and `seed-summary.md` path to the coordinator; do not put policy details,
diagnoses, or learned strategies in the return message. Then end your context.
