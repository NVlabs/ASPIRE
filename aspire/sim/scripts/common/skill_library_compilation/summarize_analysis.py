from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from openai import OpenAI

# OpenRouter proxy runs locally — no API key needed for the proxy
API_KEY = os.environ.get("OPENROUTER_API_KEY", "not-needed-for-local-proxy")

# Regex to detect trial directories (for is_experiment_dir check)
TRIAL_DIR_PATTERN = re.compile(
    r"trial_(\d+)_sandboxrc_(\d+)_reward_([-\d\.]+)_taskcompleted_(\d+)"
)

MODEL_ANALYSIS_PROMPT = """You are an expert analyzing LLM code generation performance across multiple robotics tasks.

Below are failure analyses from multiple experiments run with the same model

Your task is to identify MODEL-SPECIFIC patterns and quirks:
1. What systematic mistakes does this model make across different tasks?
2. Are there consistent failure modes (e.g., syntax errors, API misuse, reasoning errors)?
3. What are the model's strengths and weaknesses for robotics code generation?

Provide a concise but comprehensive summary.

---
EXPERIMENT ANALYSES:
{analyses}
"""

TASK_ANALYSIS_PROMPT = """You are an expert analyzing LLM code generation performance for robotics tasks.

Below are failure analyses from multiple models attempting the same task family: {task_name}

suffix descriptions:
- _multiturn: multiturn conversation with stderr and stdout feedback (such that the model can see the output of the previous turn). Interesting behaviors would include any retry behaviors.
- _multiturn_vdm: multiturn conversation with a visual-differencing module (A VLM model is used to compare the visual differences between the current and previous images of the environment and provide a textual description of the differences). Interesting behaviors would include any retry behaviors.
- _multiturn_vf: multiturn conversation with visual-feedback (raw images are provided directly to the model). Interesting behaviors would include any retry behaviors.
- _reduced_api_exampleless: reduced API without any in-context usage examples.
- _reduced_api: reduced API (lower level of abstraction of the API).
- _privileged: privileged API
- (no suffix): basic, possibly task-specific API

Your task is to identify TASK-SPECIFIC patterns:
1. What makes this task challenging? What are the common pitfalls?
2. Which failure modes are task-inherent vs model-dependent?

Provide a concise but comprehensive summary.

---
EXPERIMENT ANALYSES:
{analyses}
"""


@dataclass
class Config:
    """Summarize analysis.txt files across models and tasks.
    
    First checks that all experiments have analysis.txt, then generates:
      - Per-model summaries (model quirks across tasks)
      - Per-task summaries (task challenges across models)
    """

    output_dir: Path = Path("outputs")
    """Path to the outputs root directory."""
    
    save_dir: Path = Path("outputs/summaries")
    """Directory to save summary files."""
    
    skip_existing: bool = True
    """Skip generating summaries that already exist."""


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


@dataclass
class ExperimentInfo:
    """Info about a single experiment."""
    path: Path
    model_name: str
    experiment_name: str
    task_name: str  # base task name (without suffixes like _reduced_api)
    analysis_path: Path
    
    @property
    def has_analysis(self) -> bool:
        return self.analysis_path.exists()
    
    def read_analysis(self) -> str:
        if self.has_analysis:
            return self.analysis_path.read_text()
        return ""


def extract_task_name(experiment_name: str) -> str:
    """Extract task name from experiment name.
    
    Each experiment variant (e.g., _multiturn, _multiturn_vdm, _reduced_api) 
    is treated as a separate task for analysis purposes.
    """
    return experiment_name


def collect_all_experiments(outputs_root: Path) -> list[ExperimentInfo]:
    """Collect all experiment info from the outputs directory."""
    experiments = []
    
    for model_dir in sorted(outputs_root.iterdir()):
        if not is_model_dir(model_dir):
            continue
        
        model_name = model_dir.name
        
        for experiment_dir in sorted(model_dir.iterdir()):
            if not is_experiment_dir(experiment_dir):
                continue
            
            experiment_name = experiment_dir.name
            task_name = extract_task_name(experiment_name)
            
            experiments.append(ExperimentInfo(
                path=experiment_dir,
                model_name=model_name,
                experiment_name=experiment_name,
                task_name=task_name,
                analysis_path=experiment_dir / "analysis.txt",
            ))
    
    return experiments


def check_all_analyses_exist(experiments: list[ExperimentInfo]) -> tuple[bool, list[ExperimentInfo]]:
    """Check if all experiments have analysis.txt. Returns (all_exist, missing_list)."""
    missing = [exp for exp in experiments if not exp.has_analysis]
    return len(missing) == 0, missing


def group_by_model(experiments: list[ExperimentInfo]) -> dict[str, list[ExperimentInfo]]:
    """Group experiments by model name."""
    groups = defaultdict(list)
    for exp in experiments:
        groups[exp.model_name].append(exp)
    return dict(groups)


def group_by_task(experiments: list[ExperimentInfo]) -> dict[str, list[ExperimentInfo]]:
    """Group experiments by base task name."""
    groups = defaultdict(list)
    for exp in experiments:
        groups[exp.task_name].append(exp)
    return dict(groups)


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


def generate_model_summary(model_name: str, experiments: list[ExperimentInfo]) -> str:
    """Generate a summary for a single model across all its experiments."""
    # Compile all analyses for this model
    analyses_text = []
    for exp in experiments:
        analysis = exp.read_analysis()
        if analysis:
            analyses_text.append(
                f"### Experiment: {exp.experiment_name}\n"
                f"Task: {exp.task_name}\n\n"
                f"{analysis}\n"
                f"{'='*60}\n"
            )
    
    if not analyses_text:
        return f"No analyses available for model: {model_name}"
    
    prompt = MODEL_ANALYSIS_PROMPT.format(
        model_name=model_name,
        analyses="\n".join(analyses_text)
    )
    
    return call_llm(prompt)


def generate_task_summary(task_name: str, experiments: list[ExperimentInfo]) -> str:
    """Generate a summary for a single task across all models."""
    # Compile all analyses for this task
    analyses_text = []
    for exp in experiments:
        analysis = exp.read_analysis()
        if analysis:
            analyses_text.append(
                f"### Model: {exp.model_name}\n"
                f"Experiment variant: {exp.experiment_name}\n\n"
                f"{analysis}\n"
                f"{'='*60}\n"
            )
    
    if not analyses_text:
        return f"No analyses available for task: {task_name}"
    
    prompt = TASK_ANALYSIS_PROMPT.format(
        task_name=task_name,
        analyses="\n".join(analyses_text)
    )
    
    return call_llm(prompt)


def main(cfg: Config) -> None:
    if not cfg.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {cfg.output_dir}")
    
    # Collect all experiments
    print(f"Scanning {cfg.output_dir}...")
    experiments = collect_all_experiments(cfg.output_dir)
    print(f"Found {len(experiments)} experiment(s)")
    
    if not experiments:
        print("No experiments found.")
        return
    
    # Step 1: Check all analyses exist
    print("\n" + "="*60)
    print("Step 1: Checking for missing analysis.txt files")
    print("="*60)
    
    all_exist, missing = check_all_analyses_exist(experiments)
    
    if not all_exist:
        print(f"\n✗ {len(missing)} experiment(s) missing analysis.txt:")
        for exp in missing:
            print(f"  - {exp.model_name}/{exp.experiment_name}")
        print("\nPlease run parse_outputs.py first to generate missing analyses.")
        return
    
    print(f"✓ All {len(experiments)} experiment(s) have analysis.txt")
    
    # Create save directory
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Per-model analysis
    print("\n" + "="*60)
    print("Step 2: Generating per-model summaries")
    print("="*60)
    
    model_groups = group_by_model(experiments)
    print(f"Found {len(model_groups)} model(s)")
    
    model_summaries_dir = cfg.save_dir / "per_model"
    model_summaries_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_experiments in sorted(model_groups.items()):
        output_file = model_summaries_dir / f"{model_name}.txt"
        
        if cfg.skip_existing and output_file.exists():
            print(f"  [○] {model_name}: already exists, skipping")
            continue
        
        print(f"  [→] {model_name}: analyzing {len(model_experiments)} experiment(s)...")
        
        try:
            summary = generate_model_summary(model_name, model_experiments)
            
            with open(output_file, "w") as f:
                f.write(f"Model Summary: {model_name}\n")
                f.write(f"Experiments analyzed: {len(model_experiments)}\n")
                f.write("="*60 + "\n\n")
                f.write(summary)
            
            print(f"  [✓] {model_name}: saved to {output_file}")
        except Exception as e:
            print(f"  [✗] {model_name}: error - {e}")
    
    # Step 3: Per-task analysis
    print("\n" + "="*60)
    print("Step 3: Generating per-task summaries")
    print("="*60)
    
    task_groups = group_by_task(experiments)
    print(f"Found {len(task_groups)} task(s)")
    
    task_summaries_dir = cfg.save_dir / "per_task"
    task_summaries_dir.mkdir(parents=True, exist_ok=True)
    
    for task_name, task_experiments in sorted(task_groups.items()):
        output_file = task_summaries_dir / f"{task_name}.txt"
        
        if cfg.skip_existing and output_file.exists():
            print(f"  [○] {task_name}: already exists, skipping")
            continue
        
        print(f"  [→] {task_name}: analyzing {len(task_experiments)} experiment(s) across {len(set(e.model_name for e in task_experiments))} model(s)...")
        
        try:
            summary = generate_task_summary(task_name, task_experiments)
            
            with open(output_file, "w") as f:
                f.write(f"Task Summary: {task_name}\n")
                f.write(f"Experiments analyzed: {len(task_experiments)}\n")
                f.write(f"Models: {', '.join(sorted(set(e.model_name for e in task_experiments)))}\n")
                f.write("="*60 + "\n\n")
                f.write(summary)
            
            print(f"  [✓] {task_name}: saved to {output_file}")
        except Exception as e:
            print(f"  [✗] {task_name}: error - {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Per-model summaries: {model_summaries_dir}")
    print(f"Per-task summaries:  {task_summaries_dir}")


if __name__ == "__main__":
    tyro.cli(main)

