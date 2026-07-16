# Clean Task Slate: Evolutionary Search

Use this before rerunning a task or suite.

1. Confirm the Fix Loop `fix_code.py` baseline for the task is present.
2. Decide whether previous Evolutionary Search candidates, summaries, and validation outputs are historical records or should be superseded.
3. Check target GPU availability before launching a new candidate batch.
4. Verify perception servers on ports 8114-8116 or let the coordinator preflight start them.
5. Keep Evolutionary Search debug/validation selection separate from final held-out eval.
6. Do not promote candidate-specific tricks into shared skills unless they pass the coordinator review criteria.
