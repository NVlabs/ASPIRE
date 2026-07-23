#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast scan of ASPIRE/outputs using find, then build progress tables."""

import subprocess
import re
from collections import defaultdict

OUTPUTS = "outputs"  # relative to repo root

def find_trials(base_dir):
    """Use find to get all trial dirs quickly."""
    result = subprocess.run(
        ["find", base_dir, "-maxdepth", "12", "-type", "d", "-name", "trial_*"],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []

def parse_trial_path(path, base_dir):
    """Extract suite, task, method info from a trial path."""
    rel = path[len(base_dir):].lstrip("/")
    parts = rel.split("/")
    
    # Find suite and task
    suite = None
    task = None
    for i, p in enumerate(parts):
        if p.startswith("libero_") and i + 1 < len(parts):
            nxt = parts[i + 1]
            if any(nxt.startswith(x) for x in ["put_", "pick_", "open_", "push_", "turn_", "KITCHEN", "LIVING", "STUDY"]):
                suite = p
                task = nxt
                break
    
    is_success = "reward_1" in parts[-1]
    is_before = "/before/" in path
    is_after = "/after/" in path
    
    # Detect m1/m4
    method = None
    if "/m1/" in path:
        method = "m1"
    elif "/m4/" in path:
        method = "m4"
    
    return suite, task, is_success, is_before, is_after, method

def fmt(s, t):
    if t == 0:
        return "—"
    return f"{s}/{t} ({s*100//t}%)"

def shorten(task):
    t = task
    t = t.replace("put_the_", "").replace("pick_up_the_", "")
    t = t.replace("_and_place_it_in_the_basket", " → basket")
    t = t.replace("_on_top_of_the_", " → ")
    t = t.replace("_in_the_", " → ")
    t = t.replace("_on_the_", " → ")
    t = t.replace("_to_the_front_of_the_", " → front of ")
    t = t.replace("open_the_", "open ")
    t = t.replace("turn_on_the_", "turn on ")
    t = t.replace("_and_put_the_bowl_inside", " + bowl inside")
    return t

def main():
    # Scan all trial dirs at once
    all_trials = find_trials(OUTPUTS)
    
    # Categorize by output directory
    # Key structure: (output_dir, suite, task) -> counts
    
    # debug_log: before/after
    debug = defaultdict(lambda: {"before_s": 0, "before_t": 0, "after_s": 0, "after_t": 0})
    # fix_validation: aggregated
    fix = defaultdict(lambda: {"s": 0, "t": 0})
    # m1/m4 baselines: 
    m1 = defaultdict(lambda: {"s": 0, "t": 0})
    m4 = defaultdict(lambda: {"s": 0, "t": 0})
    
    for path in all_trials:
        if not path:
            continue
        rel_to_out = path[len(OUTPUTS):].lstrip("/")
        top_dir = rel_to_out.split("/")[0]
        
        suite, task, is_succ, is_before, is_after, method = parse_trial_path(path, OUTPUTS)
        if not suite or not task:
            continue
        
        key = (suite, task)
        
        if top_dir == "claude_debug_log":
            # For debug_log, use the FIRST suite/task occurrence (top-level)
            # The path is: claude_debug_log/suite/task/model/run/{before,after}/suite/task/model/run/trial_*
            # We want the top-level suite/task (parts[0], parts[1])
            first_suite = None
            first_task = None
            for i, p in enumerate(rel_to_out.split("/")[1:], 1):  # skip "claude_debug_log"
                if p.startswith("libero_"):
                    first_suite = p
                    next_parts = rel_to_out.split("/")
                    if i + 1 < len(next_parts):
                        first_task = next_parts[i + 1]
                    break
            if first_suite and first_task:
                key = (first_suite, first_task)
            
            if is_before:
                debug[key]["before_t"] += 1
                if is_succ:
                    debug[key]["before_s"] += 1
            elif is_after:
                debug[key]["after_t"] += 1
                if is_succ:
                    debug[key]["after_s"] += 1
        elif "fix_validation" in top_dir:
            fix[key]["t"] += 1
            if is_succ:
                fix[key]["s"] += 1
        elif "m1_vs_m4" in top_dir:
            if method == "m1":
                m1[key]["t"] += 1
                if is_succ:
                    m1[key]["s"] += 1
            elif method == "m4":
                m4[key]["t"] += 1
                if is_succ:
                    m4[key]["s"] += 1
    
    # Collect all suites and tasks
    all_keys = set()
    for d in [debug, fix, m1, m4]:
        all_keys.update(d.keys())
    
    suites = sorted(set(k[0] for k in all_keys))
    
    print("# ASPIRE Experiment Progress — Full Data Inventory\n")
    
    for suite in suites:
        tasks = sorted(set(k[1] for k in all_keys if k[0] == suite))
        if not tasks:
            continue
        
        print(f"\n## {suite} ({len(tasks)} tasks)\n")
        print("| Task | Debug Before | Debug After | Fix 50ep | M1 50ep | M4 50ep |")
        print("| --- | --- | --- | --- | --- | --- |")
        
        suite_fix_s = suite_fix_t = 0
        suite_m1_s = suite_m1_t = 0
        suite_m4_s = suite_m4_t = 0
        
        for task in tasks:
            key = (suite, task)
            d = debug.get(key, {"before_s": 0, "before_t": 0, "after_s": 0, "after_t": 0})
            f = fix.get(key, {"s": 0, "t": 0})
            m1d = m1.get(key, {"s": 0, "t": 0})
            m4d = m4.get(key, {"s": 0, "t": 0})
            
            suite_fix_s += f["s"]; suite_fix_t += f["t"]
            suite_m1_s += m1d["s"]; suite_m1_t += m1d["t"]
            suite_m4_s += m4d["s"]; suite_m4_t += m4d["t"]
            
            print(f"| {shorten(task)} | {fmt(d['before_s'], d['before_t'])} | {fmt(d['after_s'], d['after_t'])} | {fmt(f['s'], f['t'])} | {fmt(m1d['s'], m1d['t'])} | {fmt(m4d['s'], m4d['t'])} |")
        
        if len(tasks) > 1:
            print(f"| **Total** | | | **{fmt(suite_fix_s, suite_fix_t)}** | **{fmt(suite_m1_s, suite_m1_t)}** | **{fmt(suite_m4_s, suite_m4_t)}** |")
    
    # Summary
    print("\n## Summary\n")
    print("| Suite | Debugged (after>0) | Fix 50ep | M1 50ep | M4 50ep |")
    print("| --- | --- | --- | --- | --- |")
    for suite in suites:
        tasks = sorted(set(k[1] for k in all_keys if k[0] == suite))
        n_debug = sum(1 for t in tasks if debug.get((suite, t), {}).get("after_t", 0) > 0)
        n_fix = sum(1 for t in tasks if fix.get((suite, t), {}).get("t", 0) > 0)
        n_m1 = sum(1 for t in tasks if m1.get((suite, t), {}).get("t", 0) > 0)
        n_m4 = sum(1 for t in tasks if m4.get((suite, t), {}).get("t", 0) > 0)
        print(f"| {suite} | {n_debug}/{len(tasks)} | {n_fix}/{len(tasks)} | {n_m1}/{len(tasks)} | {n_m4}/{len(tasks)} |")

if __name__ == "__main__":
    main()
