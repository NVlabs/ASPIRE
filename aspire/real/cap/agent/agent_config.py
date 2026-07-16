"""Hydra structured configs for the agent pipeline.

All experiment configuration lives here as dataclasses. Hydra validates
YAML configs against these schemas at load time — one source of truth.

Usage::

    # Run an experiment (everything in the YAML)
    uv run python run_agent.py experiment=pick_place_sink_to_counter

    # Override a field from CLI
    uv run python run_agent.py experiment=pick_place_sink_to_counter env.seed=99

    # Override multiple fields
    uv run python run_agent.py experiment=pick_place_sink_to_counter \\
        max_iterations=10 runtime.dry_run=true
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """Environment configuration (absorbs ROBOCASA_* env vars)."""

    name: Optional[str] = None
    viewer: bool = False
    layout_id: int = -3  # -3 = random in RoboCasa
    style_id: int = -3
    seed: Optional[int] = None
    # RoboCasaEnv is pinned to OSC_POSE — no controller_type knob (the old
    # field has been removed along with ensure_controller_type). GR00T N1.5
    # was trained at 256×256; skill tools work at any resolution.
    camera_height: int = 480
    camera_width: int = 640


@dataclass
class PolicyConfig:
    """External closed-loop policy infrastructure (e.g. GR00T, Pi0.5).

    Only the fields that are NOT agent-script decisions live here — the
    server endpoint and an optional default backend/model pair for when a
    caller omits them. Rollout budgets, chunking, and model selection are
    written directly in the agent-generated script. Task prompt and seed
    come from the env, not Hydra. Video recording is owned by
    ``ScriptRecorder`` at the session level, not this config.
    """

    backend: str = "grootpool"
    model: str = "n15"
    endpoint: str = ""  # e.g. "tcp://127.0.0.1:7070" — empty = backend default


@dataclass
class LLMConfig:
    """LLM backend and model selection."""

    backend: str = "bridge:claude_code"
    model: Optional[str] = None
    cap_agent_url: str = "http://localhost:8200"
    mcp_config_path: str = "cap/bridge/mcp_config.json"


@dataclass
class RuntimeConfig:
    """Runtime / infrastructure settings (ports, servers, modes)."""

    agent_name: str = ""
    dry_run: bool = False
    log_dir: str = "logs"
    no_cameras: bool = False
    quiet: bool = True  # suppress profiler console output (still logs to file)
    saved_scripts_dir: str = "cap/saved_scripts"

    # --- Service ports & hosts ---
    curobo_host: str = "127.0.0.1"
    curobo_port: int = 9400
    pyroki_host: str = "127.0.0.1"
    pyroki_port: int = 9600
    mppi_host: str = "127.0.0.1"
    mppi_port: int = 0
    anygrasp_host: str = "localhost"
    anygrasp_port: int = 8122
    sam3_host: str = "localhost"
    sam3_port: int = 9500
    bundlesdf_host: str = "localhost"
    bundlesdf_port: int = 8119
    viser_port: int = 8080
    bridge_host: str = "localhost"
    bridge_port: int = 8201
    voice_host: str = "localhost"
    voice_port: int = 8202


@dataclass
class RetryPromptConfig:
    """Controls how the code generator builds its prompt on iteration > 0."""

    failure_first: bool = True
    include_previous_code: bool = True
    include_previous_thoughts: bool = True
    include_all_history: bool = True
    max_history_chars: int = 8000
    failure_prefix: str = ""
    thoughts_instruction: str = ""
    code_instruction: str = ""


@dataclass
class CodeGeneratorConfig:
    """Code generator: prompt files, context flags, retry behaviour."""

    # Prompt files to concatenate as system prompt
    # Each entry is a path relative to cap/prompt/, without .md extension.
    prompt: List[str] = field(default_factory=list)
    context_robot_state: bool = True
    context_task_info: bool = True
    context_tool_history: bool = True
    tool_history_max_chars: int = 3000
    require_thoughts: bool = True
    initial_thoughts_instruction: str = ""
    initial_code_instruction: str = ""
    retry: RetryPromptConfig = field(default_factory=RetryPromptConfig)
    # Generate multiple independent programs in one iteration, execute all of
    # them, and keep the best-scoring candidate.  This is intended for offline
    # study/harness tasks; real robot candidate search is blocked unless
    # ``allow_candidate_search_on_robot`` is explicitly enabled.
    num_candidates: int = 1
    candidate_parallelism: int = 1
    allow_candidate_search_on_robot: bool = False


@dataclass
class SkillAuthorConfig:
    """Stage-1 of the two-stage code generator (see docs/plan/two_stage_code_generator.md)."""

    # Prompt files concatenated as the stage-1 system prompt.
    prompt: List[str] = field(default_factory=list)
    # Max retries when the parser / validator rejects the LLM's authoring plan.
    max_retries: int = 2
    # If true, an empty plan ("no new skills needed") is a successful outcome.
    # If false, exhausted retries raise — useful in unit tests.
    allow_no_new_skills: bool = True
    # Per-attempt feedback cap injected into PRIOR ATTEMPTS. Default is
    # generous because cross-seed reflections (Phase B) now include every
    # seed's oracle predicates + stdout tail + VLM note — a 30-seed report
    # easily runs to 6-10 KB. Truncation lands mid-seed when too small.
    max_feedback_chars: int = 12000
    # How many recent attempts to include in PRIOR ATTEMPTS.
    max_prior_attempts: int = 3
    # Number of consecutive discard iterations before injecting the escalation
    # prompt that forces new:/replace: instead of refine:.
    escalation_threshold: int = 3


@dataclass
class AssemblyGeneratorConfig:
    """Stage-2 of the two-stage code generator (see docs/plan/two_stage_code_generator.md)."""

    prompt: List[str] = field(default_factory=list)
    max_retries: int = 2
    # Reject any top-level def/class/lambda/asyncdef in the generated code.
    forbid_toplevel_def: bool = True
    # Require at least one import from skill_library.*  — off by default so
    # iter 0 with an empty library can still produce pure-namespace code.
    require_skill_library_imports: bool = False
    # See SkillAuthorConfig.max_feedback_chars.
    max_feedback_chars: int = 12000
    max_prior_code_chars: int = 4000
    max_prior_stdout_chars: int = 2000
    max_prior_attempts: int = 3


@dataclass
class VisualEvidenceConfig:
    """Controls what frames are captured during execution and how they are presented to the VLM.

    Designed for ablation: change ``mode`` in a YAML config and every VLM
    call (Phase A per-seed + Phase B synthesis) automatically uses the new
    evidence.

    Modes
    -----
    single_end
        One frame per camera at the very end of execution (cheapest).
    before_after
        Iteration-level begin/end frame per camera. Default. Matches the
        prior behavior of CompositeReflectionStrategy.
    uniform_sample
        N frames sampled at a fixed interval during execution via a
        background thread in run_script.py. Gives the VLM a timeline.
    skill_boundaries
        One frame captured before and after *each* tool call via profiler
        hooks. Lets the VLM attribute scene changes to specific skills.
    keyframes
        Same capture as ``uniform_sample`` but filtered at read-time:
        only frames with MSE > ``delta_threshold`` relative to the
        previous kept frame are included.
    """

    mode: str = "before_after"
    # None → inherit reflection.cameras
    cameras: Optional[List[str]] = None
    # Hard cap on images per VLM call (cost / rate-limit control)
    max_images: int = 12
    # multi_image: separate images with labels (best quality)
    # grid: stitched NxM grid (one image, saves tokens)
    # strip: single-row horizontal strip
    presentation: str = "multi_image"

    # uniform_sample / keyframes: background sampler interval (seconds)
    sample_interval_s: float = 1.0
    # uniform_sample: how many frames to target (subsampled from captured)
    n_frames: int = 8

    # keyframes: MSE fraction threshold to qualify as a visual change
    delta_threshold: float = 0.05

    # skill_boundaries: restrict to these tool names; empty = all tools
    skills: List[str] = field(default_factory=list)

    # grid presentation column count
    grid_cols: int = 4


@dataclass
class ReflectionConfig:
    """Self-reflection step configuration.

    ``vlm_backend`` / ``vlm_model`` also drive the per-seed visual analysis
    inside ``SubprocessExecutorStep`` — pick the same provider once and both
    uses stay aligned. Set via Hydra group (``reflection=nvidia_gemini``)
    or per-field override (``reflection.vlm_backend=nvidia``).
    """

    prompt: List[str] = field(default_factory=list)
    strategy: str = "text"  # text | vision | composite
    cameras: List[str] = field(default_factory=lambda: ["top"])
    vlm_backend: str = "gemini"
    vlm_model: Optional[str] = None  # None → backend default
    scene_query: str = ""
    analysis_prompt: str = ""
    visual_evidence: VisualEvidenceConfig = field(default_factory=VisualEvidenceConfig)


@dataclass
class ExecutionConfig:
    """Controls how code is executed (subprocess via run_script.py)."""

    n_seeds: int = 1
    seed_start: int = 0
    timeout_s: int = 300
    record: bool = False
    mode: str = "sequential"  # "sequential", "tmux", or "parallel"
    n_gpus: int = 0          # render GPUs for parallel mode; 0 = auto-detect via nvidia-smi
    seeds_per_gpu: int = 1   # concurrent seeds per GPU (increase to run more in parallel)


@dataclass
class RewardConfig:
    """Pluggable task-success evaluator (``cap.reward``).

    Runs inside ``SubprocessExecutorStep`` between execution and the
    per-seed VLM reflection (Phase A). Output is injected into both the
    per-seed VLM prompt and the cross-seed synthesis prompt (Phase B).
    """

    # ``oracle`` — hardcoded per-task predicate recipes over
    # ``result.json.details`` (no sim access).  ``noop`` / ``none`` — skip
    # evaluator entirely and use only the binary ``success`` flag.
    evaluator: str = "oracle"

    # Override task name. Defaults to ``cfg.env.name`` (e.g.
    # ``robocasa:PickPlaceSinkToCounter``). Only needed when the env_name
    # can't be matched to a registered recipe directly.
    task: Optional[str] = None

    # Inject oracle diagnostics into the per-seed VLM reflection prompt
    # (Phase A). Turn off to keep Phase A purely visual.
    inject_into_per_seed_vlm: bool = True

    # Hardware / no-oracle tasks can expose get_task_info() as a VLM reward.
    # These fields are read by real-YAM's namespace implementation and are
    # forwarded into run_script.py subprocesses by SubprocessExecutorStep.
    vlm_backend: str = "nvidia"
    vlm_model: Optional[str] = "gcp/google/gemini-3.1-pro-preview"
    vlm_camera: str = "top"
    vlm_reasoning_effort: str = "high"


@dataclass
class RecordingConfig:
    enabled: bool = False
    per_iteration: bool = True
    # Camera streams to write when a robot backend supports video recording.
    # RoboCasa still auto-discovers simulator cameras; real YAM uses these names.
    cameras: List[str] = field(default_factory=lambda: ["top", "left", "right", "bottom"])


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "cap-robocasa"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SkillLibraryConfig:
    """Episodic skill library configuration."""

    enabled: bool = False
    seed_dir: str = ""  # path to human-curated seed skills (copied in at run start)


@dataclass
class SkillReflectionConfig:
    """Per-skill async VLM reflection after each @skill call.

    When enabled, captures top+wrist frames before and after every
    @skill-decorated function, then fires an async Gemini/VLM query
    describing what changed. Results are written to
    exec_dir/skill_steps/NNN_skillname.md and merged into per-seed
    feedback in Phase 5.
    """

    enabled: bool = False
    cameras: List[str] = field(default_factory=lambda: ["top", "wrist"])
    vlm_backend: str = "nvidia"
    vlm_model: Optional[str] = None  # None → backend default
    max_workers: int = 4  # async thread pool size


@dataclass
class DebugUIConfig:
    """Live log-watching debug UI for run_script.py.

    The UI is a separate process. run_script only launches it and writes
    structured JSONL events into the run log directory.
    """

    enabled: bool = False
    launch: bool = True
    host: str = "0.0.0.0"
    port: int = 8788
    auto_open: bool = False
    auto_exit_on_run_end: bool = True
    replace_existing_on_port: bool = True
    event_log_name: str = "debug_events.jsonl"
    poll_interval_s: float = 0.25


@dataclass
class PromptsConfig:
    """Selects which prompt files to load from ``cap/prompt/`` subfolders.

    - ``null`` (default / key absent) -> load category defaults.
    - ``[]`` (explicit empty list) -> load nothing for that category.
    - ``["a", "b"]`` -> load exactly those files.
    """

    system: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    embodiment: Optional[List[str]] = None
    task: Optional[List[str]] = None
    heuristics: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Full experiment configuration — Hydra structured config.

    The dataclass defines the schema. YAML configs are validated against it.
    CLI overrides use dot notation: ``env.seed=42``, ``runtime.direct=true``.
    """

    # Experiment metadata
    name: str = "unnamed"
    task: str = MISSING  # required — set by experiment config or CLI
    max_iterations: int = 5

    # Component configs
    env: EnvConfig = field(default_factory=EnvConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    # Pipeline steps
    steps: List[str] = field(
        default_factory=lambda: [
            "observer",
            "code_generator",
            "executor",
            "reward_evaluator",
        ]
    )

    # Sub-configs
    code_generator: CodeGeneratorConfig = field(default_factory=CodeGeneratorConfig)
    skill_author: SkillAuthorConfig = field(default_factory=SkillAuthorConfig)
    assembly_generator: AssemblyGeneratorConfig = field(
        default_factory=AssemblyGeneratorConfig
    )
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation_method: str = "reward"  # reward | vlm | none
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    skill_library: SkillLibraryConfig = field(default_factory=SkillLibraryConfig)
    skill_reflection: SkillReflectionConfig = field(default_factory=SkillReflectionConfig)
    debug_ui: DebugUIConfig = field(default_factory=DebugUIConfig)
    oracle_api: bool = False  # expose get_oracle_targets in the tool namespace
    prompts: PromptsConfig = field(default_factory=PromptsConfig)

    # Extra: review step
    review: bool = False
    review_model: Optional[str] = None

    # Extra: reflect step
    reflect: bool = False
    reflect_model: Optional[str] = None

    # Robot adapter/backend — set via robot=robocasa or robot=real_yam config
    # group.  Typed Any to accept arbitrary Hydra _target_ modules + kwargs
    # from YAML without strict schema.  When absent (None), runtime code
    # auto-derives the adapter/backend from env.name.
    robot: Any = None

    # Script-only fields (used by run_script.py, ignored by run_agent.py)
    script_file: Optional[str] = None
    script_output_dir: Optional[str] = None
    script_no_log: bool = False
    skill_library_path: Optional[str] = None

    # Oracle mode — run_agent executes a fixed script instead of calling the
    # LLM. code_generator is replaced with an OracleCodeStep that loads this
    # file; code_reviewer + self_reflection are skipped.
    oracle: Optional[str] = None


# ---------------------------------------------------------------------------
# ConfigStore registration
# ---------------------------------------------------------------------------


def register_configs() -> None:
    """Register structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=AgentConfig)


# ---------------------------------------------------------------------------
# Saving / serialization helpers
# ---------------------------------------------------------------------------


def save_config_to_log(cfg: DictConfig, log_dir: Path) -> Path:
    """Save the fully-resolved config to a log directory."""
    dest = log_dir / "config.yaml"
    dest.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return dest


def config_to_dict(cfg: DictConfig) -> dict:
    """Convert config to a plain dict for metadata.json."""
    return OmegaConf.to_container(cfg, resolve=True)
