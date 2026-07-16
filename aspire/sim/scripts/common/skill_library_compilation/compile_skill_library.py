from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tyro
from openai import OpenAI

# OpenRouter proxy runs locally — no API key needed for the proxy
API_KEY = os.environ.get("OPENROUTER_API_KEY", "not-needed-for-local-proxy")

# Regex to detect trial directories (for is_experiment_dir check)
TRIAL_DIR_PATTERN = re.compile(
    r"trial_(\d+)_sandboxrc_(\d+)_reward_([-\d\.]+)_taskcompleted_(\d+)"
)

# Functions to exclude (task-specific or boilerplate)
EXCLUDED_FUNCTION_NAMES = {
    "main",
    "run",
    "execute",
    "test",
    "demo",
    "setup",
    "cleanup",
    "init",
    "initialize",
}

SKILL_LIBRARY_PROMPT = """You are an expert robotics software engineer curating a reusable skill library.

Below are function definitions extracted from successful robot manipulation code generations across multiple tasks and models.
These functions were composed by LLM coding agents to solve various manipulation tasks using a reduced/low-level robotics API.

Your task is to:
1. Identify the MOST USEFUL and REUSABLE functions that could form a general-purpose skill library
2. Group similar functions into categories (e.g., perception, motion, grasping, coordinate transforms)
3. For each category, select the BEST implementation(s) - prefer well-documented, general-purpose versions
4. Exclude task-specific or overly narrow functions
5. Note any functions that appear frequently - this indicates high utility

Output format:
- Organize by category
- For each selected function, explain WHY it's useful and reusable
- Include the full function code, along with proper Python docstring and type hints, return types, etc.
- Note how many times similar functions appeared (popularity)

Focus on functions that would be valuable additions to a robotics manipulation toolkit.

---
EXTRACTED FUNCTIONS (with occurrence counts and sources):
{functions}
"""


@dataclass
class Config:
    """Compile a skill library from functions.txt files.
    
    Analyzes functions extracted from reduced_api experiments to identify
    the most commonly used and helpful reusable skills.
    """

    output_dir: Path = Path("outputs")
    """Path to the outputs root directory."""
    
    save_path: Path = Path("outputs/skill_library.txt")
    """Path to save the compiled skill library."""
    
    min_occurrences: int = 2
    """Minimum number of occurrences across experiments to consider a function."""
    
    skip_existing: bool = False
    """Skip if skill library already exists."""


def is_experiment_dir(path: Path) -> bool:
    """Check if a directory is an experiment directory."""
    if not path.is_dir():
        return False
    if (path / "initial_prompt.txt").exists():
        return True
    for item in path.iterdir():
        if item.is_dir() and TRIAL_DIR_PATTERN.match(item.name):
            return True
    return False


def is_model_dir(path: Path) -> bool:
    """Check if a directory is a model directory."""
    if not path.is_dir():
        return False
    for item in path.iterdir():
        if is_experiment_dir(item):
            return True
    return False


def is_reduced_api_experiment(experiment_name: str) -> bool:
    """Check if experiment is a reduced_api variant."""
    return "_reduced_api" in experiment_name


@dataclass
class FunctionInfo:
    """Information about a single function definition."""
    name: str
    signature: str
    full_code: str
    trials: list[int]
    model_name: str
    experiment_name: str
    
    @property
    def trial_count(self) -> int:
        return len(self.trials)


@dataclass 
class AggregatedFunction:
    """A function aggregated across multiple experiments."""
    name: str
    implementations: list[FunctionInfo]
    
    @property
    def total_occurrences(self) -> int:
        return sum(f.trial_count for f in self.implementations)
    
    @property
    def experiment_count(self) -> int:
        return len(self.implementations)
    
    @property
    def models(self) -> set[str]:
        return {f.model_name for f in self.implementations}
    
    def best_implementation(self) -> FunctionInfo:
        """Return the implementation with most occurrences (likely best tested)."""
        return max(self.implementations, key=lambda f: f.trial_count)


def parse_functions_file(file_path: Path, model_name: str, experiment_name: str) -> list[FunctionInfo]:
    """Parse a functions.txt file and extract function information."""
    functions = []
    
    try:
        content = file_path.read_text()
    except Exception:
        return functions
    
    # Split by function separator
    sections = content.split("────────────────────────────────────────────────────────────")
    
    for section in sections[1:]:  # Skip header
        section = section.strip()
        if not section:
            continue
        
        lines = section.split("\n")
        
        # Parse function name
        name_match = re.search(r"Function:\s*(\w+)", lines[0] if lines else "")
        if not name_match:
            continue
        name = name_match.group(1)
        
        # Skip excluded functions
        if name.lower() in EXCLUDED_FUNCTION_NAMES:
            continue
        
        # Parse trials
        trials = []
        trials_match = re.search(r"Found in trials:\s*([\d,\s]+)", section)
        if trials_match:
            trials = [int(t.strip()) for t in trials_match.group(1).split(",") if t.strip()]
        
        # Extract code (everything after the trials line)
        code_start = section.find("def ")
        if code_start == -1:
            continue
        
        full_code = section[code_start:].strip()
        
        # Extract signature (first line of def)
        sig_match = re.match(r"(def\s+\w+\s*\([^)]*\)[^:]*:)", full_code)
        signature = sig_match.group(1) if sig_match else f"def {name}(...)"
        
        functions.append(FunctionInfo(
            name=name,
            signature=signature,
            full_code=full_code,
            trials=trials,
            model_name=model_name,
            experiment_name=experiment_name,
        ))
    
    return functions


def collect_all_functions(outputs_root: Path) -> dict[str, AggregatedFunction]:
    """Collect all functions from reduced_api experiments."""
    all_functions: dict[str, list[FunctionInfo]] = defaultdict(list)
    
    for model_dir in sorted(outputs_root.iterdir()):
        if not is_model_dir(model_dir):
            continue
        
        model_name = model_dir.name
        
        for experiment_dir in sorted(model_dir.iterdir()):
            if not is_experiment_dir(experiment_dir):
                continue
            
            experiment_name = experiment_dir.name
            
            # Only process reduced_api experiments
            if not is_reduced_api_experiment(experiment_name):
                continue
            
            functions_file = experiment_dir / "functions.txt"
            if not functions_file.exists():
                continue
            
            functions = parse_functions_file(functions_file, model_name, experiment_name)
            
            for func in functions:
                all_functions[func.name].append(func)
    
    # Convert to AggregatedFunction objects
    aggregated = {}
    for name, implementations in all_functions.items():
        aggregated[name] = AggregatedFunction(name=name, implementations=implementations)
    
    return aggregated


def filter_functions(
    functions: dict[str, AggregatedFunction],
    min_occurrences: int = 2,
) -> dict[str, AggregatedFunction]:
    """Filter functions based on criteria."""
    filtered = {}
    
    for name, func in functions.items():
        # Skip if too few occurrences
        if func.total_occurrences < min_occurrences:
            continue
        
        # Skip very short functions (likely trivial)
        best = func.best_implementation()
        if len(best.full_code.split("\n")) < 3:
            continue
        
        # Skip if name suggests task-specificity
        task_specific_patterns = [
            r"cube", r"stack", r"lift", r"wipe", r"spill", r"place_.*_on",
            r"pick_.*_up", r"grab_the", r"move_to_goal",
        ]
        if any(re.search(p, name.lower()) for p in task_specific_patterns):
            continue
        
        filtered[name] = func
    
    return filtered


def format_functions_for_prompt(functions: dict[str, AggregatedFunction]) -> str:
    """Format aggregated functions for the LLM prompt."""
    lines = []
    
    # Sort by total occurrences (most popular first)
    sorted_funcs = sorted(
        functions.values(),
        key=lambda f: f.total_occurrences,
        reverse=True
    )
    
    for func in sorted_funcs:
        best = func.best_implementation()
        models = ", ".join(sorted(func.models))
        
        lines.append(f"\n{'='*70}")
        lines.append(f"FUNCTION: {func.name}")
        lines.append(f"Total occurrences: {func.total_occurrences} (across {func.experiment_count} experiments)")
        lines.append(f"Models: {models}")
        lines.append(f"Best implementation from: {best.model_name}/{best.experiment_name} ({best.trial_count} trials)")
        lines.append("")
        lines.append(best.full_code)
    
    return "\n".join(lines)


def call_llm(prompt: str) -> str:
    """Call the LLM via the local OpenRouter proxy."""
    client = OpenAI(
        base_url="http://127.0.0.1:8110/",
        api_key=API_KEY,
    )
    completion = client.chat.completions.create(
        model="google/gemini-3.1-pro-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def main(cfg: Config) -> None:
    if not cfg.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.output_dir}")
    
    if cfg.skip_existing and cfg.save_path.exists():
        print(f"Skill library already exists at {cfg.save_path}, skipping.")
        return
    
    # Step 1: Collect all functions from reduced_api experiments
    print("="*60)
    print("Step 1: Collecting functions from reduced_api experiments")
    print("="*60)
    
    all_functions = collect_all_functions(cfg.output_dir)
    print(f"Found {len(all_functions)} unique function names")
    
    if not all_functions:
        print("No functions found. Make sure functions.txt files exist in reduced_api experiments.")
        return
    
    # Step 2: Filter functions
    print("\n" + "="*60)
    print("Step 2: Filtering functions")
    print("="*60)
    
    filtered = filter_functions(all_functions, min_occurrences=cfg.min_occurrences)
    print(f"After filtering: {len(filtered)} functions")
    
    if not filtered:
        print(f"No functions meet the minimum occurrence threshold ({cfg.min_occurrences}).")
        print("Try lowering --min-occurrences")
        return
    
    # Show stats
    print("\nTop functions by occurrence:")
    sorted_funcs = sorted(filtered.values(), key=lambda f: f.total_occurrences, reverse=True)
    for func in sorted_funcs[:15]:
        print(f"  {func.name}: {func.total_occurrences} occurrences across {func.experiment_count} experiments")
    
    # Step 3: Generate skill library with LLM
    print("\n" + "="*60)
    print("Step 3: Generating skill library with LLM analysis")
    print("="*60)
    
    functions_text = format_functions_for_prompt(filtered)
    prompt = SKILL_LIBRARY_PROMPT.format(functions=functions_text)
    
    print("Calling LLM to curate skill library...")
    
    try:
        skill_library = call_llm(prompt)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        print("\nSaving raw function list instead...")
        skill_library = f"[LLM analysis failed: {e}]\n\nRaw function list:\n{functions_text}"
    
    # Step 4: Save
    print("\n" + "="*60)
    print("Step 4: Saving skill library")
    print("="*60)
    
    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cfg.save_path, "w") as f:
        f.write("SKILL LIBRARY - Reusable Functions from LLM Robot Code Generation\n")
        f.write("="*70 + "\n")
        f.write(f"Source: reduced_api and reduced_api_exampleless experiments\n")
        f.write(f"Total unique functions analyzed: {len(all_functions)}\n")
        f.write(f"Functions after filtering: {len(filtered)}\n")
        f.write(f"Minimum occurrence threshold: {cfg.min_occurrences}\n")
        f.write("="*70 + "\n\n")
        f.write(skill_library)
    
    print(f"✓ Skill library saved to: {cfg.save_path}")
    
    # Also save raw functions for reference
    raw_path = cfg.save_path.with_name("skill_library_raw.txt")
    with open(raw_path, "w") as f:
        f.write("RAW FUNCTION LIST (before LLM curation)\n")
        f.write("="*70 + "\n\n")
        f.write(functions_text)
    
    print(f"✓ Raw function list saved to: {raw_path}")


if __name__ == "__main__":
    tyro.cli(main)
