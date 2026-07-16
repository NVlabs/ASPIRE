---
name: zeroshot-capx-coordinator
description: Coordinator guide for the zero-shot Cap-X in-context learning ablation. Subagents receive self-contained prompts with 15 examples + API ref embedded — no access to skills, prior outputs, or CLAUDE.md context.
---

# Zero-Shot Cap-X — Coordinator Guide

> **What:** For each LIBERO-Pro task, a subagent reads 15 baseline Cap-X programs (embedded in its prompt), zero-shot generates code, then evaluates on seeds 1-50.
> **Why:** Ablation measuring how well the model generalizes from examples alone — no skill library, no debugging, no prior outputs.
> **Isolation:** Subagent prompts are self-contained. The prompt embeds all 15 examples + API reference inline. Subagents are explicitly told to ignore auto-loaded CLAUDE.md/skills.

---

> **Note (public release):** the prompt-generation helper and baseline programs for this CaP-X ablation are held separately by the authors and are **not bundled** in this repo. This guide documents the method for reference; it is not one of the five reproducible experiments.

## Step 0: Prepare Generated Prompts

This step requires the private prompt-generation helper. A public checkout cannot run this ablation end-to-end without that helper and its baseline-program inputs.

This creates:
- `outputs/zeroshot_claude/manifest.json` — task list with prompt paths
- `outputs/zeroshot_claude/prompts/<suite>/<task>.txt` — self-contained prompts per task (examples + API ref embedded, `$GPU` placeholder)
- `outputs/zeroshot_claude/<suite>/<task>/iter_00/candidate_A/` — where code gets written

---

## Step 1: Verify Perception Servers

```bash
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```
404 = UP, 000 = DOWN. All three must be UP.

---

## Step 2: Check Progress

```bash
.venv/bin/python3 -c "
import json
from pathlib import Path
m = json.loads(Path('outputs/zeroshot_claude/manifest.json').read_text())
done, pending = [], []
for t in m['tasks']:
    summary = Path(t['output_dir']) / 'iter_00' / 'iter_summary.json'
    if summary.exists():
        s = json.loads(summary.read_text())
        done.append((t, s['best_pass_rate']))
    else:
        pending.append(t)
print(f'Done: {len(done)}/{len(m[\"tasks\"])}')
for t, rate in done:
    print(f'  {t[\"suite\"]}/{t[\"task\"]}: {rate:.1%}')
print(f'Pending: {len(pending)}')
for t in pending[:10]:
    print(f'  {t[\"suite\"]}/{t[\"task\"]}')
if len(pending) > 10:
    print(f'  ... and {len(pending)-10} more')
"
```

---

## Step 3: Check Free GPUs

```bash
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then echo "GPU $gpu: FREE"; else echo "GPU $gpu: BUSY ($procs processes)"; fi
done
```

---

## Step 4: Dispatch Subagents

**How to dispatch:** Read the pre-built prompt from disk, replace `$GPU` with the actual GPU ID, and pass as the Agent prompt.

```python
# Read prompt for a task
prompt_path = "outputs/zeroshot_claude/prompts/<suite>/<task>.txt"
prompt_text = open(prompt_path).read()
prompt_text = prompt_text.replace("$GPU", "<actual_gpu_id>")

Agent(
    description="ZS: <suite_short>/<task_short> GPU<N>",
    subagent_type="general-purpose",
    model="sonnet",
    prompt=prompt_text,
    run_in_background=True
)
```

Send up to 5 dispatches per message (one per free GPU), then go idle.

---

## Step 5: On Completion — Redispatch

When a subagent notification arrives:
1. Note the result (pass rate)
2. Check free GPUs (Step 3)
3. Read next pending task's prompt, fill GPU, dispatch
4. Go idle

---

## Step 6: Aggregate Results

```bash
.venv/bin/python3 -c "
import json
from pathlib import Path

results = []
for suite_dir in sorted(Path('outputs/zeroshot_claude').iterdir()):
    if not suite_dir.is_dir() or suite_dir.name in ('prompts',):
        continue
    for task_dir in sorted(suite_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        summary = task_dir / 'iter_00' / 'iter_summary.json'
        if summary.exists():
            s = json.loads(summary.read_text())
            results.append({
                'suite': suite_dir.name,
                'task': task_dir.name,
                'pass_rate': s['best_pass_rate'],
                'pass_count': s['candidates'][0]['pass_count'],
                'trials': s['trials_per_candidate'],
            })

print(f\"{'Suite':<25} {'Task':<55} {'Rate':>6} {'Pass':>6}\")
print('-' * 95)
for r in sorted(results, key=lambda x: (x['suite'], x['task'])):
    print(f\"{r['suite']:<25} {r['task']:<55} {r['pass_rate']:>5.1%} {r['pass_count']:>3}/{r['trials']}\")

by_suite = {}
for r in results:
    by_suite.setdefault(r['suite'], []).append(r['pass_rate'])
print()
print(f\"{'Suite':<25} {'Mean Rate':>10} {'Tasks':>6}\")
print('-' * 45)
for suite in sorted(by_suite):
    rates = by_suite[suite]
    print(f\"{suite:<25} {sum(rates)/len(rates):>9.1%} {len(rates):>6}\")
overall = [r['pass_rate'] for r in results]
print(f\"{'OVERALL':<25} {sum(overall)/len(overall):>9.1%} {len(overall):>6}\")
"
```

---

## Coordinator Rules

1. **Dispatch subagents — never generate code yourself.**
2. **Go idle after dispatching.** You'll be notified on completion.
3. **Keep all 5 GPUs (3-7) occupied.**
4. **On each notification: redispatch next pending task.**
5. **Never re-dispatch a done task** — done = `iter_summary.json` exists.
6. **Use pre-built prompts** — do not modify them or add extra context.
