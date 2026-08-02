# BEHAVIOR-1K ASPIRE Campaign State

Only the coordinator updates this file.

## Manifest

- Campaign/task: `<id> / <Soda Can|Radio>`
- Commit/config/hash: `<values>`
- Model/context: `<value>`
- Stage 1 / per-seed budgets: `<values>`
- Host/GPU: `<value>`
- Approval reference: `<value>`
- Status: `preflight|stage1|freeze|stage2|reporting|complete|stopped`

## Seeds

Append or update one row per seed. Terminal Stage 2 rows are never replaced.

| Stage | Seed | Status | Outcome | Replays | Agent/session | Summary |
|---|---:|---|---|---:|---|---|
| 1 | 26 | pending | | | | |
| 2 | 1 | pending | | | | |

Required Stage 1 seeds: `26-35`. Required Stage 2 seeds: `1-25`.

## Freeze

- Frozen library/hash manifest: `<path / hash>`
- Config/API/prompt hashes: `<values>`
- Stage 1 agent and simulator stopped: `no`
- Frozen manifest verified: `no`

## Deviations

Append timestamp, stage/seed, evidence path, validity impact, and action. Never
erase an event.

## Completion

- Held-out terminal episodes: `0/25`
- Held-out successes: `0/25`
- Final report: `reports/final-report.md`
- Built-in LLM loop used: `no`
- Real hardware used: `no`
