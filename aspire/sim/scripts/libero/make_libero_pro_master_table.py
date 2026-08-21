# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile a master per-task table for every LIBERO-Pro task.

Columns:
  Suite | Task | Baseline (gemini_m4, seeds 51..65)
              | Fix (gemini_m4_fixed OR goal_task actor rerun, seeds 1..50)
              | Evolutionary Search Debug (Stage 1 best candidate on seeds 51..65)
              | Evolutionary Search Eval  (Stage 2, seeds 1..50)

Writes:
  - outputs/reports/libero_pro_master_table.md
  - outputs/reports/libero_pro_master_table.csv
"""
import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE        = ROOT / "outputs/libero_baseline_image_diff_gemini"
FIXED           = ROOT / "outputs/libero_fix_loop_eval"
GOAL_TASK_RERUN = ROOT / "outputs/aspire_goal_task_actor_eval"
EVOSEARCH_EVAL     = ROOT / "outputs/aspire_evosearch_eval"
EVOSEARCH_DEBUG    = ROOT / "outputs/claude_evosearch"
EVOSEARCH_RERUN    = ROOT / "outputs/claude_evosearch_rerun"
VAL_DIR         = ROOT / "outputs/validation_comparison_seeds66_80"
# Extra Stage-1 iteration artifacts produced by reruns (same schema).
EVOSEARCH_DEBUG_EXTRAS = [EVOSEARCH_RERUN]

SUITES = [
    "libero_goal_swap",
    "libero_goal_task",
    "libero_object_swap",
    "libero_object_task",
    "libero_spatial_swap",
    "libero_spatial_task",
]

TASK_LABELS = {
    "open_the_middle_drawer_of_the_cabinet":                                                "open_middle_drawer",
    "open_the_top_drawer_and_put_the_bowl_inside":                                          "open_top_drawer+bowl",
    "push_the_plate_to_the_front_of_the_stove":                                             "push_plate→stove",
    "put_the_bowl_on_the_plate":                                                             "bowl→plate",
    "put_the_bowl_on_the_stove":                                                             "bowl→stove",
    "put_the_bowl_on_top_of_the_cabinet":                                                   "bowl→cabinet",
    "put_the_cream_cheese_in_the_bowl":                                                     "cream_cheese→bowl",
    "put_the_wine_bottle_on_the_rack":                                                      "wine→rack",
    "put_the_wine_bottle_on_top_of_the_cabinet":                                            "wine→cabinet",
    "turn_on_the_stove":                                                                     "turn_on_stove",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket":                                 "alphabet_soup",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket":                                     "bbq_sauce",
    "pick_up_the_butter_and_place_it_in_the_basket":                                        "butter",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket":                             "choc_pudding",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket":                                  "cream_cheese",
    "pick_up_the_ketchup_and_place_it_in_the_basket":                                       "ketchup",
    "pick_up_the_milk_and_place_it_in_the_basket":                                          "milk",
    "pick_up_the_orange_juice_and_place_it_in_the_basket":                                  "orange_juice",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket":                                "salad_dressing",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket":                                  "tomato_sauce",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":   "between_plate_ramekin",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":                   "table_center",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": "top_drawer",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":              "next_cookie_box",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":                   "next_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":                 "next_ramekin",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":                   "on_cookie_box",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":                      "on_ramekin",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":                        "on_stove",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":               "on_wooden_cabinet",
}


def find_trials(path: Path):
    if not path.exists():
        return []
    r = subprocess.run(
        ["find", str(path), "-maxdepth", "7", "-type", "d", "-name", "trial_*"],
        capture_output=True, text=True, timeout=180,
    )
    return [l for l in r.stdout.strip().split("\n") if l]


def baseline_rate(suite: str, task: str):
    """Seeds 51..65 — one best-src trial per seed, pass = 'reward_1' in dir name."""
    run_dir = BASELINE / suite / task / "gcp_google_gemini-3.1-pro-preview" / "run"
    trials = find_trials(run_dir)
    if not trials:
        return None, 0
    best = {}
    for t in trials:
        name = Path(t).name
        parts = name.split("_")
        try:
            seed = int(parts[1])
            src  = int(parts[3])
        except (IndexError, ValueError):
            continue
        passed = "reward_1" in name
        if seed not in best or src > best[seed][0]:
            best[seed] = (src, passed)
    total = len(best)
    passes = sum(1 for _, (_, p) in best.items() if p)
    return passes, total


def _rate_from_dir(fix_dir: Path):
    trials = find_trials(fix_dir)
    trial_re = re.compile(r"trial_(\d+)_sandboxrc")
    seeds_to_path = {}
    for t in trials:
        m = trial_re.search(Path(t).name)
        if not m:
            continue
        seed = int(m.group(1))
        if not (1 <= seed <= 50):
            continue
        seeds_to_path[seed] = t
    total = len(seeds_to_path)
    if total == 0:
        return None, 0
    passes = 0
    for seed, t in seeds_to_path.items():
        s = Path(t) / "summary.txt"
        if s.exists() and "Task Completed: True" in s.read_text(errors="ignore"):
            passes += 1
    return passes, total


def fix_rate(suite: str, task: str):
    """Seeds 1..50, pass = 'Task Completed: True' in summary.txt.

    For libero_goal_task we prefer the rerun dir; fall back to the original
    fixed dir if the rerun does not cover the task.
    Returns (passes, n, source_label).
    """
    if suite == "libero_goal_task":
        p, n = _rate_from_dir(GOAL_TASK_RERUN / suite / task)
        if n > 0:
            return p, n, "rerun"
        p, n = _rate_from_dir(FIXED / suite / task)
        return p, n, ("fixed" if n > 0 else "—")
    p, n = _rate_from_dir(FIXED / suite / task)
    return p, n, ("fixed" if n > 0 else "—")


def evosearch_eval_rate(suite: str, task: str):
    """Stage-2 eval: seeds 1..50 from trial dir name `taskcompleted_X` suffix."""
    task_dir = EVOSEARCH_EVAL / suite / task
    if not task_dir.exists():
        return None, 0
    tid_re = re.compile(r"trial_(\d+)")
    tc_re  = re.compile(r"taskcompleted_([01])")
    by_id = {}
    for p in task_dir.rglob("trial_*"):
        if not p.is_dir():
            continue
        mid = tid_re.match(p.name)
        mtc = tc_re.search(p.name)
        if not mid or not mtc:
            continue
        tid = int(mid.group(1))
        if 1 <= tid <= 50:
            by_id[tid] = int(mtc.group(1))
    total = len(by_id)
    if total == 0:
        return None, 0
    return sum(by_id.values()), total


def evosearch_debug_rate(suite: str, task: str):
    """Stage-1 debug: best candidate's pass_count on seeds 51..65.

    Scans every eval_results.json under outputs/claude_evosearch/<suite>/<task>/
    and picks the candidate with the highest pass_count on seeds 51..65.
    This matches the Stage 1 selection rule (the winning candidate is
    promoted to Stage 2 / evosearch_best_code.py).
    """
    roots = [EVOSEARCH_DEBUG / suite / task] + [x / suite / task for x in EVOSEARCH_DEBUG_EXTRAS]
    roots = [r for r in roots if r.exists()]
    if not roots:
        return None, 0
    best_passes = -1
    best_total = 0
    eval_files = []
    for r in roots:
        eval_files.extend(r.rglob("eval_results.json"))
    for jp in eval_files:
        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue
        trial_results = data.get("trial_results") or []
        passes = 0
        seen = set()
        for tr in trial_results:
            seed = tr.get("trial")
            if not isinstance(seed, int) or not (51 <= seed <= 65):
                continue
            if seed in seen:
                continue
            seen.add(seed)
            if int(tr.get("task_completed", 0)) == 1:
                passes += 1
        total = len(seen)
        if total == 0:
            continue
        if passes > best_passes or (passes == best_passes and total > best_total):
            best_passes = passes
            best_total = total
    if best_passes < 0:
        return None, 0
    return best_passes, best_total


def evosearch_rerun_rate(suite: str, task: str):
    """Stage-2 pass rate (seeds 1-50) from claude_evosearch_rerun stage2/iter_summary.json."""
    tdir = EVOSEARCH_RERUN / suite / task
    if not tdir.exists():
        return None, 0
    for run_dir in sorted(tdir.glob("2*"), reverse=True):
        js = run_dir / "stage2" / "iter_summary.json"
        if js.exists():
            d = json.loads(js.read_text())
            rate = d["candidates"][0]["pass_rate"]
            return round(rate * 50), 50
    return None, 0


def val_selected_rate(suite: str, task: str, fix_pct, evosearch_rerun_pct):
    """Pick fix vs evosearch based on validation seeds 66-80. Returns (pct, source)."""
    js = VAL_DIR / suite / task / "iter_summary.json"
    if not js.exists():
        # no validation — fall back to max
        candidates = [(fix_pct, "fix"), (evosearch_rerun_pct, "evosearch*")]
        candidates = [(v, s) for v, s in candidates if v is not None]
        if not candidates:
            return None, "—"
        return max(candidates, key=lambda x: x[0])
    d = json.loads(js.read_text())
    rates = {c["candidate"]: c["pass_rate"] for c in d["candidates"]}
    vf = rates.get("candidate_fix")
    ve = rates.get("candidate_evosearch")
    if vf is not None and ve is not None:
        if ve >= vf:
            return evosearch_rerun_pct, "evosearch"
        else:
            return fix_pct, "fix"
    # partial validation — fall back to max
    candidates = [(fix_pct, "fix"), (evosearch_rerun_pct, "evosearch*")]
    candidates = [(v, s) for v, s in candidates if v is not None]
    if not candidates:
        return None, "—"
    return max(candidates, key=lambda x: x[0])


def list_tasks(suite: str):
    """Canonical task list = union of task folders seen in any source."""
    seen = set()
    candidate_roots = [BASELINE / suite, FIXED / suite, GOAL_TASK_RERUN / suite,
                       EVOSEARCH_EVAL / suite, EVOSEARCH_DEBUG / suite]
    candidate_roots.extend(x / suite for x in EVOSEARCH_DEBUG_EXTRAS)
    for root in candidate_roots:
        if root.exists():
            for p in root.iterdir():
                if p.is_dir():
                    seen.add(p.name)
    return sorted(seen)


def fmt_pct(passes, total):
    if total == 0 or passes is None:
        return "—", None
    pct = 100 * passes / total
    return f"{passes}/{total} ({pct:.1f}%)", pct


def fmt_delta(a, b):
    if a is None or b is None:
        return "—"
    d = b - a
    if d == 0:
        return "0"
    return f"{d:+.1f}"


rows = []
for suite in SUITES:
    for task in list_tasks(suite):
        b_pass, b_n        = baseline_rate(suite, task)
        f_pass, f_n, f_src = fix_rate(suite, task)
        ed_pass, ed_n      = evosearch_debug_rate(suite, task)
        ee_pass, ee_n      = evosearch_eval_rate(suite, task)
        er_pass, er_n      = evosearch_rerun_rate(suite, task)
        b_str,  b_pct  = fmt_pct(b_pass, b_n)
        f_str,  f_pct  = fmt_pct(f_pass, f_n)
        ed_str, ed_pct = fmt_pct(ed_pass, ed_n)
        ee_str, ee_pct = fmt_pct(ee_pass, ee_n)
        er_str, er_pct = fmt_pct(er_pass, er_n)
        vs_pct, vs_src = val_selected_rate(suite, task, f_pct, er_pct)
        vs_str = f"{vs_pct:.1f}%" if vs_pct is not None else "—"
        rows.append({
            "suite": suite,
            "task": task,
            "label": TASK_LABELS.get(task, task),
            "baseline_passes": b_pass, "baseline_n": b_n, "baseline_pct": b_pct,
            "fix_passes": f_pass, "fix_n": f_n, "fix_pct": f_pct, "fix_src": f_src,
            "evosearch_debug_passes": ed_pass, "evosearch_debug_n": ed_n, "evosearch_debug_pct": ed_pct,
            "evosearch_eval_passes": ee_pass,  "evosearch_eval_n": ee_n,  "evosearch_eval_pct": ee_pct,
            "evosearch_rerun_passes": er_pass, "evosearch_rerun_n": er_n, "evosearch_rerun_pct": er_pct,
            "val_selected_pct": vs_pct, "val_selected_src": vs_src,
            "baseline_str": b_str, "fix_str": f_str,
            "evosearch_debug_str": ed_str, "evosearch_eval_str": ee_str,
            "evosearch_rerun_str": er_str, "val_selected_str": vs_str,
            "delta_fix_vs_base":      fmt_delta(b_pct,  f_pct),
            "delta_evosearch_vs_fix":    fmt_delta(f_pct,  ee_pct),
            "delta_val_vs_fix":       fmt_delta(f_pct,  vs_pct),
        })

# ---------- CSV ----------
csv_path = ROOT / "outputs/reports/libero_pro_master_table.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "suite", "task",
        "baseline_passes", "baseline_n", "baseline_pct",
        "fix_passes", "fix_n", "fix_pct", "fix_source",
        "evosearch_debug_passes", "evosearch_debug_n", "evosearch_debug_pct",
        "evosearch_eval_passes",  "evosearch_eval_n",  "evosearch_eval_pct",
        "evosearch_rerun_passes", "evosearch_rerun_n", "evosearch_rerun_pct",
        "aspire_evosearch_pct", "aspire_evosearch_src",
        "delta_fix_vs_base_pp", "delta_evosearch_vs_fix_pp", "delta_val_vs_fix_pp",
    ])
    for r in rows:
        def rnd(v): return None if v is None else round(v, 2)
        writer.writerow([
            r["suite"], r["task"],
            r["baseline_passes"], r["baseline_n"], rnd(r["baseline_pct"]),
            r["fix_passes"], r["fix_n"], rnd(r["fix_pct"]), r["fix_src"],
            r["evosearch_debug_passes"], r["evosearch_debug_n"], rnd(r["evosearch_debug_pct"]),
            r["evosearch_eval_passes"],  r["evosearch_eval_n"],  rnd(r["evosearch_eval_pct"]),
            r["evosearch_rerun_passes"], r["evosearch_rerun_n"], rnd(r["evosearch_rerun_pct"]),
            rnd(r["val_selected_pct"]), r["val_selected_src"],
            r["delta_fix_vs_base"], r["delta_evosearch_vs_fix"], r["delta_val_vs_fix"],
        ])
print(f"Wrote {csv_path}")


# ---------- Markdown (pretty-padded) ----------
def pad(s, n, right=False):
    s = str(s)
    if len(s) >= n:
        return s
    return (" " * (n - len(s)) + s) if right else (s + " " * (n - len(s)))

def render_table(header_cells, align, body_rows):
    """header_cells: list[str]; align: list of 'l'|'r'; body_rows: list[list[str]]."""
    widths = [len(h) for h in header_cells]
    for row in body_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header = "| " + " | ".join(pad(h, widths[i]) for i, h in enumerate(header_cells)) + " |"
    sep_parts = []
    for i, a in enumerate(align):
        w = widths[i]
        if a == "r":
            sep_parts.append("-" * (w + 1) + ":")
        elif a == "c":
            sep_parts.append(":" + "-" * w + ":")
        else:
            sep_parts.append("-" * (w + 2))
    sep = "|" + "|".join(sep_parts) + "|"
    out = [header, sep]
    for row in body_rows:
        cells = [
            pad(cell, widths[i], right=(align[i] == "r"))
            for i, cell in enumerate(row)
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


md = []
md.append("# LIBERO-Pro Master Table\n")
md.append("## Column definitions\n")
md.append(
    "| Column        | Source                                                                                      | Seeds  | Pass rule                                             |"
)
md.append(
    "|---------------|---------------------------------------------------------------------------------------------|:------:|-------------------------------------------------------|"
)
md.append(
    "| Baseline      | `outputs/libero_baseline_image_diff_gemini/...` (Gemini 3.1 pro, best-of-src per seed)                     | 51–65  | `reward_1` in trial folder name                       |"
)
md.append(
    "| Fix           | `outputs/libero_fix_loop_eval/...`; for `libero_goal_task` prefer `outputs/aspire_goal_task_actor_eval/...` (rerun)† | 1–50   | `Task Completed: True` in `summary.txt`              |"
)
md.append(
    "| Evolutionary Search-debug  | `outputs/claude_evosearch/.../iter_*/candidate_*/eval_results.json` — best candidate pass_count | 51–65  | `task_completed == 1` in `trial_results`              |"
)
md.append(
    "| Evolutionary Search-eval   | `outputs/aspire_evosearch_eval/...`                                                              | 1–50   | `taskcompleted_1` in trial folder name                |"
)
md.append("")
md.append("† Rerun rows are flagged with a `*` in the **src** column.\n")
md.append("Δ columns are percentage-point deltas (right-aligned).")

HEADER = ["Task", "Baseline", "Fix", "src", "Evolutionary Search-dbg", "Evolutionary Search-eval",
          "Evolutionary Search-rerun", "ASPIRE+Evolutionary Search", "sel", "Δ Fix−Base", "Δ H+E−Fix"]
ALIGN  = ["l",    "r",        "r",   "c",   "r",          "r",
          "r",            "r",           "c",   "r",          "r"]

for suite in SUITES:
    suite_rows = [r for r in rows if r["suite"] == suite]
    md.append(f"\n## {suite}\n")
    body = []
    for r in suite_rows:
        src_mark = "rerun" if r["fix_src"] == "rerun" else ("fixed" if r["fix_src"] == "fixed" else "—")
        body.append([
            r["label"],
            r["baseline_str"],
            r["fix_str"],
            src_mark,
            r["evosearch_debug_str"],
            r["evosearch_eval_str"],
            r["evosearch_rerun_str"],
            r["val_selected_str"],
            r["val_selected_src"],
            r["delta_fix_vs_base"],
            r["delta_val_vs_fix"],
        ])
    md.append(render_table(HEADER, ALIGN, body))

# ---------- Suite Aggregates ----------
def macro_mean(values):
    vs = [v for v in values if v is not None]
    return None if not vs else sum(vs) / len(vs)

AGG_HEADER = ["Suite", "Baseline", "Fix", "Evolutionary Search-dbg", "Evolutionary Search-eval", "Evolutionary Search-rerun", "ASPIRE+Evolutionary Search"]
AGG_ALIGN  = ["l",     "r",        "r",   "r",          "r",           "r",             "r"]
md.append("\n## Suite aggregates (macro mean across tasks with data; n = task count)\n")
md.append("ASPIRE+Evolutionary Search = val-set (seeds 66–80) selects fix vs evosearch per task; reports winner's seeds 1–50 rate.\n")

agg_body = []
for suite in SUITES:
    sr = [r for r in rows if r["suite"] == suite]
    def col(key):
        vals = [r[key] for r in sr]
        m = macro_mean(vals)
        n = sum(1 for v in vals if v is not None)
        return "—" if m is None else f"{m:.1f}% (n={n})"
    agg_body.append([
        suite,
        col("baseline_pct"),
        col("fix_pct"),
        col("evosearch_debug_pct"),
        col("evosearch_eval_pct"),
        col("evosearch_rerun_pct"),
        col("val_selected_pct"),
    ])
md.append(render_table(AGG_HEADER, AGG_ALIGN, agg_body))

md_path = ROOT / "outputs/reports/libero_pro_master_table.md"
md_path.parent.mkdir(parents=True, exist_ok=True)
md_path.write_text("\n".join(md) + "\n")
print(f"Wrote {md_path}")
