#!/usr/bin/env python3
"""Replay a specific Robosuite trial with enhanced tracing.

Supports three modes:
  - Default (fresh LLM query): generates new code via the LLM
  - --replay-code <path>: re-executes saved code.py blocks without calling LLM
  - --interactive: drops into a Python REPL after env.reset() with all API functions

Usage:
    # Fresh LLM query
    .venv/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
      --args.config env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml \
      --args.trial 51 \
      --args.model aws/anthropic/bedrock-claude-sonnet-4-6 \
      --args.output-dir ./outputs/robosuite_debug

    # Replay saved code
    .venv/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
      --args.config env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml \
      --args.trial 51 \
      --args.replay-code /tmp/fix_attempt.py \
      --args.output-dir ./outputs/robosuite_debug

    # Interactive REPL (no LLM needed)
    .venv/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
      --args.config env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml \
      --args.trial 51 \
      --args.interactive
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Force weights_only=False for PyTorch loading of legacy files
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

# Ensure repo root is on PYTHONPATH
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import tyro
import yaml

from aspire.sim.cap.envs.configs.instantiate import instantiate


@dataclass
class ReplayTrialArgs:
    """Command-line arguments for replay_trial_robosuite.py."""

    config: str
    """Path to the traced YAML config (e.g. env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml)."""

    trial: int
    """Trial number (1-indexed). Determines env.reset(seed=trial)."""

    output_dir: str = "./outputs/robosuite_debug"
    """Base output directory. Nested under {config_stem}/{model}/run unless flat_output."""

    flat_output: bool = False
    """If True, write trials directly under output_dir (no config_stem/model/run nesting)."""

    model: str = "aws/anthropic/bedrock-claude-sonnet-4-6"
    """LLM model name."""

    server_url: str = "https://inference-api.nvidia.com/v1/chat/completions"
    """LLM API endpoint."""

    api_key: str | None = None
    """API key for the LLM. Reads from NVIDIA_API_KEY env var if not provided."""

    temperature: float = 1.0
    """Sampling temperature for code generation."""

    max_tokens: int = 2048 * 10
    """Maximum tokens for LLM response."""

    reasoning_effort: str = "medium"
    """Reasoning effort for models that support it."""

    replay_code: str | None = None
    """Path to a fix.py or code.py file. If provided, re-executes code without calling LLM."""

    interactive: bool = False
    """Drop into a Python REPL after env.reset() with all API functions in scope."""

    record_video: bool = True
    """Whether to record and save rollout video."""

    debug: bool = False
    """Enable debug logging."""


def _parse_code_blocks(code_path: str) -> list[str]:
    """Parse a saved code.py file into individual code blocks.

    Splits on '# Code block N' comment headers.
    """
    code_text = Path(code_path).read_text()
    parts = re.split(r'^# Code block \d+\s*\n', code_text, flags=re.MULTILINE)
    blocks = [p.strip() for p in parts if p.strip()]
    if not blocks:
        blocks = [code_text.strip()]
    return blocks


def _query_llm(args: ReplayTrialArgs, prompt: list[dict]) -> dict[str, Any]:
    """Query the LLM for code generation with API key rotation."""
    import requests

    raw_key = args.api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not raw_key:
        raise RuntimeError("No API key. Set NVIDIA_API_KEY or pass --api-key.")

    api_keys = [k.strip() for k in raw_key.split(",") if k.strip()]
    key_idx = 0

    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": prompt,
    }

    from aspire.sim.cap.utils.launch_utils import CLAUDE_MODELS
    if args.model in CLAUDE_MODELS:
        payload["thinking"] = {"type": "enabled", "budget_tokens": 4096}

    print(f"Querying LLM ({args.model})...")
    t0 = time.time()

    max_retries = len(api_keys) * 5
    for attempt in range(max_retries):
        current_key = api_keys[key_idx % len(api_keys)]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}",
        }
        resp = requests.post(args.server_url, headers=headers, data=json.dumps(payload), timeout=200)
        if resp.status_code == 429:
            key_idx += 1
            wait = 15
            print(f"  429 rate limit on key[{key_idx-1}], rotating to key[{key_idx % len(api_keys)}], waiting {wait}s...")
            time.sleep(wait)
            continue
        elif resp.status_code in [500, 502, 503, 504]:
            wait = 30 + (attempt * 10)
            print(f"  Retry {attempt+1}: status {resp.status_code}, waiting {wait}s...")
            time.sleep(wait)
            continue
        else:
            break

    resp.raise_for_status()
    body = resp.json()
    dt = time.time() - t0
    print(f"  LLM responded in {dt:.1f}s")

    if args.debug:
        print(json.dumps(body, indent=2))

    content = body["choices"][0]["message"]["content"]
    reasoning = body["choices"][0]["message"].get("reasoning")
    return {"content": content, "reasoning": reasoning}


def _extract_code(content: str) -> list[str]:
    """Extract Python code from markdown fenced blocks."""
    fence_start = "```python\n"
    fence_end = "```"
    if fence_start in content:
        start_idx = content.find(fence_start) + len(fence_start)
        content = content[start_idx:]
    if fence_end in content:
        end_idx = content.rfind(fence_end)
        content = content[:end_idx]
    return [content.strip()]


def _parse_multi_turn_decision(content: str) -> tuple[str, str | None]:
    """Parse multi-turn decision from LLM response."""
    content_stripped = content.strip()
    if content_stripped.upper().startswith("REGENERATE"):
        code_part = content_stripped[len("REGENERATE"):].strip()
        blocks = _extract_code(code_part)
        return "regenerate", blocks[0] if blocks else code_part
    elif content_stripped.upper().startswith("FINISH"):
        return "finish", None
    else:
        blocks = _extract_code(content_stripped)
        if blocks and blocks[0]:
            return "regenerate", blocks[0]
        return "finish", None


def _find_apis(obj, depth=0):
    """Walk the env wrapper chain to find the _apis dict."""
    if depth > 5:
        return {}
    if hasattr(obj, "_apis"):
        return obj._apis
    for attr in ("env", "low_level_env", "_executor", "executor"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            result = _find_apis(child, depth + 1)
            if result:
                return result
    return {}


def _run_interactive_repl(env, obs, args: ReplayTrialArgs, config_stem: str) -> None:
    """Drop into an interactive Python REPL with all API functions in scope."""
    import code as code_module

    apis = _find_apis(env)

    repl_ns: dict[str, Any] = {
        "env": env,
        "obs": obs,
        "np": np,
        "args": args,
    }

    fn_names = []
    trace_logger = None
    for api in apis.values():
        for fn_name, fn in api.functions().items():
            repl_ns[fn_name] = fn
            fn_names.append(fn_name)
        if hasattr(api, "get_trace_logger"):
            trace_logger = api.get_trace_logger()
            repl_ns["trace_logger"] = trace_logger

    def step(code_str: str):
        """Execute a code string through env.step() and return (obs, reward, terminated, truncated, info)."""
        return env.step(code_str)
    repl_ns["step"] = step

    print("\n" + "=" * 70)
    print("  INTERACTIVE REPL — ASPIRE Trial Debugger (Robosuite)")
    print("=" * 70)
    print(f"  Config: {config_stem}")
    print(f"  Trial:  {args.trial} (seed={args.trial})")
    print(f"  Env reset complete. Scene is ready.")
    print()
    print("  Available API functions:")
    for fn in sorted(fn_names):
        repl_ns_fn = repl_ns[fn]
        doc = getattr(repl_ns_fn, "__doc__", "") or ""
        first_line = doc.strip().split("\n")[0][:60] if doc.strip() else ""
        print(f"    {fn}()" + (f"  — {first_line}" if first_line else ""))
    print()
    print("  Other objects in scope:")
    print("    env            — the gym environment")
    print("    obs            — last observation dict")
    print("    np             — numpy")
    if trace_logger:
        print("    trace_logger   — TraceLogger instance")
    print("    step(code_str) — execute a code string via env.step()")
    print()
    print("  Example:")
    print("    >>> obs = get_observation()")
    print("    >>> cam = next(v for v in obs.values() if isinstance(v, dict) and 'images' in v)")
    print("    >>> rgb = cam['images']['rgb']")
    print("    >>> masks = segment_sam3_text_prompt(rgb, 'red cube')")
    print("    >>> print(masks[0]['score'])")
    print()
    print("  Type exit() or Ctrl-D to quit.")
    print("=" * 70 + "\n")

    code_module.interact(
        banner="",
        local=repl_ns,
        exitmsg="REPL exited.",
    )


def main(args: ReplayTrialArgs) -> None:
    """Main replay entry point."""
    config_stem = Path(args.config).stem
    mode = "Interactive REPL" if args.interactive else ("Replay code" if args.replay_code else "Fresh LLM query")
    print(f"=" * 80)
    print(f"Replay Trial {args.trial}")
    print(f"Config: {config_stem}")
    print(f"Mode: {mode}")
    print(f"=" * 80)

    # 1. Load YAML config
    config_path = os.path.expanduser(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["trials"] = 1

    if args.flat_output:
        run_dir = Path(args.output_dir)
        config_save_dir = run_dir
    else:
        model_dir_name = args.model.replace("/", "_")
        run_dir = Path(args.output_dir) / config_stem / model_dir_name / "run"
        config_save_dir = Path(args.output_dir) / config_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(run_dir)

    # Save config for reference
    config_save_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = config_save_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)

    # 2. Instantiate environment
    print("Instantiating environment...")
    env = instantiate(config["env"])

    # 3. Reset environment with trial seed
    print(f"Resetting env with seed={args.trial}...")
    obs, info = env.reset(seed=args.trial)
    if "full_prompt" in obs:
        obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])

    if args.record_video and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True)

    # 3b. Interactive REPL mode
    if args.interactive:
        _run_interactive_repl(env, obs, args, config_stem)
        return

    # 4. Execute code blocks
    code_blocks: list[str] = []
    code_block_metadata: list[dict] = []
    all_responses: list[dict] = []
    num_regenerations = 0
    num_finishes = 0
    reward = 0.0
    terminated = False
    truncated = False
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": "", "task_completed": False}

    multi_turn_prompt_template = config["env"].get("cfg", {}).get("multi_turn_prompt")

    if args.replay_code:
        # Mode 2: Replay saved code
        print(f"Loading code from: {args.replay_code}")
        code_blocks = _parse_code_blocks(args.replay_code)
        code_block_metadata = [{"generation": 0, "regenerated": False}] * len(code_blocks)
        print(f"  Found {len(code_blocks)} code blocks")

        for i, code in enumerate(code_blocks):
            print(f"\n--- Executing code block {i} ---")
            obs_next, reward, terminated, truncated, info_step = env.step(code)
            obs = obs_next
            print(f"  reward={reward}, terminated={terminated}, task_completed={info_step.get('task_completed')}")
            if info_step.get("stderr"):
                print(f"  stderr: {info_step['stderr'][:200]}")
            if terminated or truncated:
                break

    else:
        # Mode 1: Fresh LLM query
        print("Querying LLM for initial code...")
        llm_response = _query_llm(args, obs["full_prompt"])
        raw_code = llm_response["content"]

        initial_blocks = _extract_code(raw_code)
        code_blocks.extend(initial_blocks)
        code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
        all_responses.append({
            "block_idx": [0],
            "code_blocks": initial_blocks,
            "decision": "initial",
            "reasoning": llm_response.get("reasoning", ""),
        })

        code_block_idx = 0
        max_turns = 30

        while code_block_idx < len(code_blocks) and code_block_idx <= max_turns:
            code = code_blocks[code_block_idx]
            code_block_idx += 1

            print(f"\n--- Executing code block {code_block_idx - 1} ---")
            obs_next, reward, terminated, truncated, info_step = env.step(code)
            obs = obs_next

            print(f"  reward={reward}, terminated={terminated}, task_completed={info_step.get('task_completed')}")
            if info_step.get("stderr"):
                print(f"  stderr: {info_step['stderr'][:200]}")

            if multi_turn_prompt_template:
                if "terminated episode" in info_step.get("stderr", ""):
                    truncated = True
                    break

                executed_code = "\n".join(code_blocks[:code_block_idx])
                complete_mt_prompt = multi_turn_prompt_template.format(
                    executed_code=executed_code,
                    console_stdout=info_step["stdout"],
                    console_stderr=info_step["stderr"],
                )

                mt_prompt = copy.deepcopy(obs["full_prompt"])
                mt_prompt.append({
                    "role": "user",
                    "content": [{"type": "text", "text": complete_mt_prompt}],
                })

                mt_response = _query_llm(args, mt_prompt)
                decision, new_code = _parse_multi_turn_decision(mt_response["content"])

                if decision == "regenerate" and new_code:
                    print("  LLM chose: REGENERATE")
                    new_blocks = _extract_code(new_code) if "```" in new_code else [new_code]
                    all_responses.append({
                        "block_idx": [code_block_idx],
                        "code_blocks": new_blocks,
                        "decision": "regenerate",
                        "reasoning": mt_response.get("reasoning", ""),
                    })
                    del code_blocks[code_block_idx:]
                    del code_block_metadata[code_block_idx:]
                    code_blocks.extend(new_blocks)
                    code_block_metadata.extend([{
                        "generation": num_regenerations + 1,
                        "regenerated": True,
                    }] * len(new_blocks))
                    num_regenerations += 1
                elif decision == "finish":
                    print("  LLM chose: FINISH")
                    num_finishes += 1
                    break

    # 5. Build final annotated code
    annotated_blocks = []
    for i, (block, meta) in enumerate(zip(code_blocks, code_block_metadata)):
        annotated_blocks.append(f"# Code block {i}\n{block}")
    final_code = "\n\n".join(annotated_blocks)

    # 6. Build summary
    sandbox_rc = info_step.get("sandbox_rc", 0)
    task_completed = info_step.get("task_completed", False)

    log_lines = [
        "-" * 100,
        f"Trial {args.trial} — {config_stem}",
        f"Mode: {'replay' if args.replay_code else 'fresh LLM query'}",
        f"Model: {args.model}",
        "Generated program:",
        final_code,
        "",
        "Environment response:",
        f"  Sandbox failed: {sandbox_rc}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {info_step.get('stderr', '')}",
        f"  Reward: {reward}",
        f"  Task Completed: {task_completed}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {len(code_blocks)}",
        "-" * 100,
    ]

    # 7. Save artifacts
    trial_dir = run_dir / f"trial_{args.trial:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}_taskcompleted_{int(task_completed or False)}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    (trial_dir / "code.py").write_text(final_code)
    (trial_dir / "summary.txt").write_text("\n".join(log_lines))
    if all_responses:
        (trial_dir / "all_responses.json").write_text(json.dumps(all_responses, indent=2))

    # Save trace from the traced API
    apis = _find_apis(env)
    if apis:
        for api in apis.values():
            if hasattr(api, "get_trace_logger"):
                trace_logger = api.get_trace_logger()
                trace_logger.save(str(trial_dir))
                trace_logger.reset()
                print(f"  Saved trace to {trial_dir}/keyframes/")
    else:
        print("  Warning: could not find _apis for trace saving")

    # Save video + evenly-spaced keyframes
    if args.record_video and hasattr(env, "get_video_frames"):
        frames = env.get_video_frames(clear=True)
        if frames:
            import imageio.v2 as imageio
            video_path = trial_dir / f"video_{reward:.3f}.mp4"
            with imageio.get_writer(str(video_path), fps=30, format="FFMPEG", codec="libx264") as writer:
                for frame in frames:
                    writer.append_data(np.ascontiguousarray(frame))
            print(f"Saved video: {video_path} ({len(frames)} frames)")

            from PIL import Image as _Image
            n_keyframes = 10
            keyframes_dir = trial_dir / "keyframes"
            keyframes_dir.mkdir(parents=True, exist_ok=True)
            n_kf = min(n_keyframes, len(frames))
            indices = [int(i * (len(frames) - 1) / max(n_kf - 1, 1)) for i in range(n_kf)]
            for rank, idx in enumerate(indices):
                img = _Image.fromarray(frames[idx])
                img.save(keyframes_dir / f"video_frame_{rank:02d}_of_{n_keyframes}_step_{idx:04d}.jpg", quality=85)
            print(f"Saved {n_keyframes} video keyframes to {keyframes_dir}")

    print(f"\n{'=' * 80}")
    print(f"Trial {args.trial} complete!")
    print(f"  Reward: {reward}")
    print(f"  Task Completed: {task_completed}")
    print(f"  Output: {trial_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
