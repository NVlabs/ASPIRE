# BEHAVIOR-1K ASPIRE Campaign State

Only the coordinator updates this file. Record paths relative to the campaign
root when possible. Never remove prior attempts or terminal failures.

## Manifest

- Campaign ID: `<campaign-id>`
- Task: `<Soda Can|Radio>`
- Repository path: `<path>`
- Commit SHA: `<sha>`
- Config: `<path>`
- Config SHA-256: `<sha256>`
- External model/context: `<model and context>`
- Stage 1 budgets: `<limits>`
- Stage 2 per-seed budgets: `<limits>`
- GPU/host: `<host and gpu>`
- User approval reference: `<timestamp or transcript reference>`
- Status: `preflight|stage1|freeze|stage2|reporting|complete|stopped`

## Stage 1 — Development Seeds 26-35

| Seed | Status | Outcome | Replays | Policy | Summary | Notes |
|---:|---|---|---:|---|---|---|
| 26 | pending | | | | | |
| 27 | pending | | | | | |
| 28 | pending | | | | | |
| 29 | pending | | | | | |
| 30 | pending | | | | | |
| 31 | pending | | | | | |
| 32 | pending | | | | | |
| 33 | pending | | | | | |
| 34 | pending | | | | | |
| 35 | pending | | | | | |

## Freeze

- Frozen at: `<timestamp>`
- Working skill-library path: `skill-library-working/`
- Frozen skill-library path: `skill-library-frozen/`
- Frozen manifest: `frozen-manifest.sha256`
- Manifest verified: `no`
- Experimental contract recorded: `no`
- Stage 1 agent/simulator stopped: `no`

## Stage 2 — Held-Out Seeds 1-25

Allowed statuses are `pending`, `running`, `success`, `failed`, `invalid`, and
`stopped`. `success`, `failed`, and `invalid` are terminal and must not be
replaced by reruns.

| Seed | Status | Outcome | Replays | Agent/session | Summary | Notes |
|---:|---|---|---:|---|---|---|
| 1 | pending | | | | | |
| 2 | pending | | | | | |
| 3 | pending | | | | | |
| 4 | pending | | | | | |
| 5 | pending | | | | | |
| 6 | pending | | | | | |
| 7 | pending | | | | | |
| 8 | pending | | | | | |
| 9 | pending | | | | | |
| 10 | pending | | | | | |
| 11 | pending | | | | | |
| 12 | pending | | | | | |
| 13 | pending | | | | | |
| 14 | pending | | | | | |
| 15 | pending | | | | | |
| 16 | pending | | | | | |
| 17 | pending | | | | | |
| 18 | pending | | | | | |
| 19 | pending | | | | | |
| 20 | pending | | | | | |
| 21 | pending | | | | | |
| 22 | pending | | | | | |
| 23 | pending | | | | | |
| 24 | pending | | | | | |
| 25 | pending | | | | | |

## Deviations And Infrastructure Events

Append events chronologically. Include timestamp, seed/stage, evidence path,
whether the event invalidates the result, and the action taken. Never erase a
resolved event.

## Completion

- Frozen manifest reverified: `no`
- Held-out terminal episodes: `0/25`
- Held-out successes: `0/25`
- Final report: `reports/final-report.md`
- Built-in ASPIRE LLM loop used: `no`
- Real hardware used: `no`
