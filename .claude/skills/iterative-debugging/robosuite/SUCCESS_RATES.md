# Robosuite fix_code success rates

Pulled from `archive_do_not_read/runs/.../validation_result.json` (100-seed pass rate) for each task's
selected fix.

| Task | Fix code file | Pass rate | Passes |
| --- | --- | --- | --- |
| cube_stack | cube_stack_fix_code.py | 0.97 | 97/100 |
| cube_lifting | cube_lifting_fix_code.py | 1.00 | 100/100 |
| cube_restack | cube_restack_fix_code.py | 1.00 | 100/100 |
| nut_assembly | nut_assembly_fix_code.py | 0.99 | 99/100 |
| spill_wipe | spill_wipe_fix_code.py | 0.95 | 95/100 |
| two_arm_handover | two_arm_handover_fix_code.py | 0.90 | 90/100 |
| two_arm_lift | two_arm_lift_fix_code.py | 1.00 | 100/100 |

Source runs:
- `archive_do_not_read/runs/2026-09-15_23-38-07_robosuite-cube-stack/robosuite/cube_stack/task_0`
- `archive_do_not_read/runs/2026-09-16_00-49-44_robosuite-six-tasks-astra-xhigh/robosuite/{cube_lifting,cube_restack,nut_assembly,spill_wipe,two_arm_handover,two_arm_lift}/task_0`
