# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import tyro

from aspire.sim.cap.utils.eval_utils import ExperimentParser, analyze_failures, analyze_highlights, TrialData

# Regex to detect trial directories
TRIAL_DIR_PATTERN = re.compile(
    r"trial_(\d+)_sandboxrc_(\d+)_reward_([-\d\.]+)_taskcompleted_(\d+)"
)

MAX_ANALYSIS_TRIALS = 50

# Regex to extract function definitions (captures full function with body)
FUNCTION_DEF_PATTERN = re.compile(
    r'^(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*\n(?:(?:[ \t]+.+\n?)+))',
    re.MULTILINE
)

# Simpler pattern to just get function signatures
FUNCTION_SIG_PATTERN = re.compile(
    r'^def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:',
    re.MULTILINE
)


def extract_functions_from_code(code: str) -> list[dict]:
    """Extract all function definitions from code.
    
    Returns list of dicts with 'name', 'signature', and 'full_def' keys.
    """
    functions = []
    
    # Find all function definitions
    for match in FUNCTION_DEF_PATTERN.finditer(code):
        full_def = match.group(1).strip()
        
        # Extract signature info
        sig_match = FUNCTION_SIG_PATTERN.match(full_def)
        if sig_match:
            name = sig_match.group(1)
            params = sig_match.group(2).strip()
            return_type = sig_match.group(3).strip() if sig_match.group(3) else None
            
            signature = f"def {name}({params})"
            if return_type:
                signature += f" -> {return_type}"
            
            functions.append({
                'name': name,
                'signature': signature,
                'full_def': full_def,
            })
    
    return functions


def compile_successful_functions(sorted_results: Dict[int, TrialData]) -> str:
    """Extract and compile all function definitions from successful trials.
    
    Returns a formatted string listing all unique functions found.
    """
    all_functions = {}  # name -> list of (trial_idx, function_info)
    
    for trial_idx, trial_data in sorted_results.items():
        if not trial_data.task_completed:
            continue
        
        try:
            code = trial_data.summary_txt.read_text()
            functions = extract_functions_from_code(code)
            
            for func in functions:
                name = func['name']
                if name not in all_functions:
                    all_functions[name] = []
                all_functions[name].append((trial_idx, func))
        except Exception:
            continue
    
    if not all_functions:
        return "No function definitions found in successful trials."
    
    # Format output
    lines = [
        "Function Definitions from Successful Trials",
        "=" * 60,
        f"Total unique functions: {len(all_functions)}",
        "",
    ]
    
    for name in sorted(all_functions.keys()):
        occurrences = all_functions[name]
        trial_ids = [str(t[0]) for t in occurrences]
        
        lines.append(f"\n{'─' * 60}")
        lines.append(f"Function: {name}")
        lines.append(f"Found in trials: {', '.join(trial_ids)}")
        lines.append("")
        
        # Show the first occurrence's full definition
        _, func_info = occurrences[0]
        lines.append(func_info['full_def'])
    
    return "\n".join(lines)



def _batch_results_for_analysis(
    results: dict[int, Any], prioritize_completed: bool, limit: int = MAX_ANALYSIS_TRIALS
) -> list[dict[int, Any]]:
    """Sort and batch results for analysis."""
    completed_trials = {k: v for k, v in results.items() if v.task_completed}
    failed_trials = {k: v for k, v in results.items() if not v.task_completed}

    # Primary set
    primary = completed_trials if prioritize_completed else failed_trials
    # Secondary set
    secondary = failed_trials if prioritize_completed else completed_trials

    all_items = list(primary.items()) + list(secondary.items())

    batches = []
    for i in range(0, len(all_items), limit):
        batch_items = all_items[i : i + limit]
        batches.append(dict(sorted(batch_items)))

    return batches


@dataclass
class Config:
    """Parse experiment outputs and analyze failures.
    
    Auto-detects directory level and processes accordingly:
      - outputs/                      → processes all models and experiments
      - outputs/{model}/              → processes all experiments for that model  
      - outputs/{model}/{experiment}/ → processes single experiment
    
    Analysis is saved as analysis.txt inside each experiment directory.
    """

    output_dir: Path = Path("outputs_multimodel")
    """Path to outputs root, model directory, or experiment directory."""
    
    workers: int = 10
    """Number of parallel workers for processing experiments."""


def is_experiment_dir(path: Path) -> bool:
    """Check if a directory is an experiment directory (contains trial folders or initial_prompt.txt)."""
    if not path.is_dir():
        return False
    # Check for initial_prompt.txt or any trial directories
    if (path / "initial_prompt.txt").exists():
        return True
    for item in path.iterdir():
        if item.is_dir() and TRIAL_DIR_PATTERN.match(item.name):
            return True
    return False


def is_model_dir(path: Path) -> bool:
    """Check if a directory is a model directory (contains experiment directories)."""
    if not path.is_dir():
        return False
    for item in path.iterdir():
        if is_experiment_dir(item):
            return True
    return False


def is_outputs_root(path: Path) -> bool:
    """Check if a directory is the outputs root (contains model directories)."""
    if not path.is_dir():
        return False
    for item in path.iterdir():
        if is_model_dir(item):
            return True
    return False


def parse_results(output_dir: str):
    parser = ExperimentParser(output_dir)
    results = parser.parse_trials()
    return results


def process_experiment(experiment_dir: Path, model_name: str | None = None) -> dict:
    """Process a single experiment directory and save analysis.txt and highlights.txt.
    
    Independently checks for each file - only generates if missing.
    Returns a dict with status info for reporting.
    """
    experiment_name = experiment_dir.name
    result = {
        "experiment_dir": experiment_dir,
        "model_name": model_name,
        "status": "unknown",
        "message": "",
    }
    
    try:
        # Check what already exists (basic check only for primary file to determine "skipped" status logic later)
        # But now we process chunks dynamically, so we'll check individually.
        
        # Parse results (needed for generating either file)
        results = parse_results(str(experiment_dir))
        
        if not results:
            result["status"] = "skipped"
            result["message"] = "No trials found"
            return result
        
        # Summary stats
        total = len(results)
        completed = sum(1 for t in results.values() if t.task_completed)
        avg_reward = sum(t.reward for t in results.values()) / total if total > 0 else 0
        
        generated = []
        
        # Generate analysis batches
        # Prioritize failures for failure analysis
        failure_batches = _batch_results_for_analysis(results, prioritize_completed=False)
        for i, batch in enumerate(failure_batches):
            suffix = "" if i == 0 else f"_{i+1}"
            filename = f"analysis{suffix}.txt"
            analysis_file = experiment_dir / filename
            
            if not analysis_file.exists():
                analysis = analyze_failures(batch)
                
                with open(analysis_file, "w") as f:
                    if model_name:
                        f.write(f"Model: {model_name}\n")
                    f.write(f"Experiment: {experiment_name}\n")
                    f.write(f"Batch: {i+1}/{len(failure_batches)}\n")
                    f.write(f"Trials: {total}, Completed: {completed}/{total}, Avg Reward: {avg_reward:.3f}\n")
                    f.write("="*60 + "\n\n")
                    f.write(analysis)
                
                generated.append(filename)
        
        # Generate highlights batches
        # Prioritize successes for highlights
        highlight_batches = _batch_results_for_analysis(results, prioritize_completed=True)
        for i, batch in enumerate(highlight_batches):
            suffix = "" if i == 0 else f"_{i+1}"
            filename = f"highlights{suffix}.txt"
            highlights_file = experiment_dir / filename
            
            if not highlights_file.exists():
                highlights = analyze_highlights(batch)
                
                with open(highlights_file, "w") as f:
                    f.write(highlights)
                
                generated.append(filename)


        functions_exists = (experiment_dir / "functions.txt").exists()
        # Generate functions list if missing (from successful trials only)
        if not functions_exists:
            functions_text = compile_successful_functions(results)
            functions_file = experiment_dir / "functions.txt"
            
            with open(functions_file, "w") as f:
                if model_name:
                    f.write(f"Model: {model_name}\n")
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Trials: {total}, Completed: {completed}/{total}\n")
                f.write("=" * 60 + "\n\n")
                f.write(functions_text)
            
            generated.append("functions")
        
        if not generated:
             result["status"] = "skipped"
             result["message"] = "All analysis/highlights files already exist"
             return result

        result["status"] = "success"
        result["message"] = f"Generated: {', '.join(generated)} | Trials: {total}, Completed: {completed}/{total}, Avg Reward: {avg_reward:.3f}"
        return result
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        return result


def _process_experiment_wrapper(args: tuple[Path, str | None]) -> dict:
    """Wrapper for multiprocessing (needs picklable args)."""
    experiment_dir, model_name = args
    return process_experiment(experiment_dir, model_name)


def collect_experiments_from_model(model_dir: Path) -> list[tuple[Path, str]]:
    """Collect all experiment directories under a model directory."""
    experiments = []
    model_name = model_dir.name
    for experiment_dir in sorted(model_dir.iterdir()):
        if is_experiment_dir(experiment_dir):
            experiments.append((experiment_dir, model_name))
    return experiments


def collect_experiments_from_root(outputs_root: Path) -> list[tuple[Path, str]]:
    """Collect all experiment directories under the outputs root."""
    experiments = []
    for model_dir in sorted(outputs_root.iterdir()):
        if is_model_dir(model_dir):
            experiments.extend(collect_experiments_from_model(model_dir))
    return experiments


def collect_all_experiment_dirs(path: Path) -> list[tuple[Path, str | None]]:
    """Auto-detect directory level and collect all experiment directories."""
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if is_experiment_dir(path):
        print(f"Detected: experiment directory")
        return [(path, None)]
    elif is_model_dir(path):
        print(f"Detected: model directory")
        return collect_experiments_from_model(path)
    elif is_outputs_root(path):
        print(f"Detected: outputs root directory")
        return collect_experiments_from_root(path)
    else:
        raise ValueError(
            f"Could not determine directory type for: {path}\n"
            "Expected one of:\n"
            "  - Experiment dir (contains trial_* folders or initial_prompt.txt)\n"
            "  - Model dir (contains experiment directories)\n"
            "  - Outputs root (contains model directories)"
        )


def check_missing_analyses(experiments: list[tuple[Path, str | None]]) -> list[Path]:
    """Check which experiment directories are missing analysis.txt."""
    missing = []
    for experiment_dir, _ in experiments:
        if not (experiment_dir / "analysis.txt").exists():
            missing.append(experiment_dir)
    return missing


def run_parallel(experiments: list[tuple[Path, str | None]], workers: int) -> None:
    """Process experiments in parallel and report results."""
    if not experiments:
        print("No experiments to process.")
        return
    
    print(f"\nProcessing {len(experiments)} experiment(s) with {workers} workers...\n")
    
    results = []
    if workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_exp = {
                executor.submit(_process_experiment_wrapper, exp): exp 
                for exp in experiments
            }
            
            for future in as_completed(future_to_exp):
                result = future.result()
                results.append(result)
                
                # Print progress
                exp_dir = result["experiment_dir"]
                model = result["model_name"] or "N/A"
                status = result["status"]
                msg = result["message"]
                
                status_icon = {"success": "✓", "skipped": "○", "error": "✗"}.get(status, "?")
                print(f"[{status_icon}] {model}/{exp_dir.name}: {msg}")

    else:
        for exp in experiments:
            result = process_experiment(exp[0], exp[1])
            results.append(result)

            # Print progress
            exp_dir = result["experiment_dir"]
            model = result["model_name"] or "N/A"
            status = result["status"]
            msg = result["message"]
            
            status_icon = {"success": "✓", "skipped": "○", "error": "✗"}.get(status, "?")
            print(f"[{status_icon}] {model}/{exp_dir.name}: {msg}")

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print(f"\n{'='*60}")
    print(f"Summary: {success} created, {skipped} skipped, {errors} errors")
    print('='*60)


def main(cfg: Config) -> None:
    # Collect all experiments
    experiments = collect_all_experiment_dirs(cfg.output_dir)
    
    # Process in parallel
    run_parallel(experiments, cfg.workers)
    
    # Final check for missing analysis.txt
    missing = check_missing_analyses(experiments)
    if missing:
        print(f"\n⚠ Warning: {len(missing)} experiment(s) missing analysis.txt:")
        for path in missing:
            print(f"  - {path}")
    else:
        print(f"\n✓ All {len(experiments)} experiment(s) have analysis.txt")


if __name__ == "__main__":
    tyro.cli(main)