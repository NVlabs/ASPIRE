# Skill Library Compilation

Analyze evaluation outputs and compile reusable skill libraries from successful trials.

These are legacy/offline analysis utilities. The public LIBERO experiment runbooks use the suite-shared skills in `.claude/libero/skills/` and the per-experiment skill folders under `.claude/libero/<experiment>/skills/`.

## Prerequisites

- Evaluation outputs in `outputs/` from completed runs.
- For summarization/compilation scripts that call a model, point the script at an OpenAI-compatible endpoint or adapt the default `base_url` in the script.

## Usage

```bash
# 1. Parse outputs → generates analysis.txt, highlights.txt, functions.txt per experiment
uv run --no-sync --active scripts/common/skill_library_compilation/parse_outputs.py \
    --cfg.output-dir outputs/

# 2. Summarize across models and tasks
uv run --no-sync --active scripts/common/skill_library_compilation/summarize_analysis.py \
    --cfg.output-dir outputs/

# 3. Compile skill library from successful reduced-API experiments
uv run --no-sync --active scripts/common/skill_library_compilation/compile_skill_library.py \
    --cfg.output-dir outputs/
```

Output: `outputs/skill_library.txt` — curated reusable functions for future evaluations.

## Utilities

- **`trial_folder_rename.py`** — Standardize trial folder numbering (`--cfg.dry-run` to preview)
- **`eval_dir_to_code.py`** — Consolidate all `code.py` files into one file for review

Run any script with `--help` for full options.
