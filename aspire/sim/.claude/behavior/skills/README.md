# BEHAVIOR-1K Skills

These files are seed placeholders for reusable BEHAVIOR-1K/R1Pro knowledge.
Keep experiment scaffolding and API references in `system-pipeline.md` and
`r1pro-api.md`; add learned task patterns to the focused files below only after
an experiment validates them.

| Local file | Purpose |
|---|---|
| `system-pipeline.md` | Architecture, launch modes, configs, outputs, critical run rules |
| `r1pro-api.md` | Complete local R1Pro API reference and conventions |
| `interactive-policy.md` | Placeholder for adaptive observe-act-observe policy patterns |
| `perception-and-search.md` | Placeholder for SAM3/search/prompting patterns |
| `navigation.md` | Placeholder for navigation and approach-geometry patterns |
| `grasping.md` | Placeholder for grasp attempt and verification patterns |
| `time-budget.md` | Placeholder for expensive-call timing and gating patterns |
| `search.md` | Placeholder for multi-position exploration patterns |
| `radio-table-tasks.md` | Placeholder for table-object/radio-specific patterns |

Use these placeholders as growth points, not as privileged knowledge. Generated
task code must rely only on public R1Pro API calls and observations.

For a canonical two-stage ASPIRE campaign, do not edit these repository files
in place. Copy this directory to the campaign's `skill-library-working/`, learn
only on seeds 26-35, then freeze that campaign-owned copy for isolated
evaluation on seeds 1-25. See
[`../fix-loop/INSTRUCTIONS.md`](../fix-loop/INSTRUCTIONS.md).
