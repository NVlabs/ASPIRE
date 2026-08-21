# Scripts

Scripts are grouped by suite or shared utility scope:

| Directory | Scope |
| --- | --- |
| `scripts/libero/` | LIBERO, LIBERO-90, LIBERO-Long replay, eval, progress, and plotting helpers. |
| `scripts/robosuite/` | Robosuite baseline, fix-loop, training-law, replay, progress, and plotting helpers. |
| `scripts/behavior/` | BEHAVIOR/B1K setup, verification, replay, and debugging helpers. |
| `scripts/common/` | Shared utilities such as perception server startup, token accounting, trace analysis, and legacy skill-library compilation tools. |

Do not commit generated task code under `scripts/`. Use ignored scratch locations such as `outputs/working_codes/` for temporary fixed-code copies.
