"""Shared configuration for the CAP system."""

import os
import urllib.parse
from pathlib import Path

import numpy as np

CAP_AGENT_NAME = os.environ.get("CAP_AGENT_NAME", "").strip()
# No longer raise at import time — agent_name is configured via Hydra
# (runtime.agent_name in experiment YAML) and validated at run time.

# ---------------------------------------------------------------------------
# Network ports
# ---------------------------------------------------------------------------

# Portal RPC ports for follower (YAM) arm servers (matches constants.py)
LEFT_FOLLOWER_PORT = 11333
RIGHT_FOLLOWER_PORT = 11334

# Portal RPC ports for leader arm servers (matches constants.py)
LEFT_LEADER_PORT = 11335
RIGHT_LEADER_PORT = 11336

# CAP server (CONTROL_FREQ_HZ control loop) — Portal RPC
CAP_SERVER_PORT = 8300

# CAP agent (orchestrator) — REST + WebSocket for UI
CAP_AGENT_PORT = 8200

# Detection server (cap/skills/serve_pose.py) — HTTP
DETECTION_SERVER_PORT = 8118

# BundleSDF tracking server (tools/vision/serve_bundlesdf.py) — HTTP
BUNDLESDF_SERVER_HOST = os.environ.get("BUNDLESDF_SERVER_HOST", "localhost")
BUNDLESDF_SERVER_PORT = int(os.environ.get("BUNDLESDF_SERVER_PORT", "8119"))

# SAM3 segmentation server (tools/vision/serve_sam3.py) — HTTP
# Runs on the same machine as cap_server/cap_agent by default.
SAM3_SERVER_HOST = os.environ.get("SAM3_SERVER_HOST", "localhost")
SAM3_SERVER_PORT = int(os.environ.get("SAM3_SERVER_PORT", "9500"))

# Robot grasp frame → fingertip/TCP offset used to convert planner-facing
# grasp poses into the world-frame pose that should be passed to motion planning.
GRIPPER_TCP_OFFSET_Z_M = 0.0

# Table surface height in world frame. Used by 2D top-down grasp planning.
TABLE_SURFACE_Z_M = float(os.environ.get("TABLE_SURFACE_Z_M", "0.75"))

# AnyGrasp planner-facing safety floor. Raw/native AnyGrasp poses remain
# unchanged, but planner-facing AnyGrasp poses are clipped to stay at or above
# this world-frame Z.
ANYGRASP_MIN_PLANNER_Z_M = float(os.environ.get("ANYGRASP_MIN_PLANNER_Z_M", "0.80"))


def make_bundlesdf_name(text: str, name: str | None = None) -> str:
    """URL-safe BundleSDF session key derived from *text* or an explicit *name*.

    Used by serve_bundlesdf, bundlesdf_track tools, and the detection backend
    to agree on the same session key for a given object description.
    """
    raw = (name or text).strip().lower().replace("/", "-")
    return urllib.parse.quote(raw, safe="")[:80]


# Policy server (flow matching) — Portal RPC / HTTP
POLICY_SERVER_PORT = 8964

# Named external policy models that CAP can call on demand.
PI05_FUNCTIONAL_GRASP_MODEL = "Pi05-stateless-functional-grasp"
PI05_POLICY_SERVER = os.environ.get(
    "PI05_POLICY_SERVER",
    f"localhost:{POLICY_SERVER_PORT}",
)
PI05_POLICY_LAUNCH_SCRIPT = os.environ.get(
    "PI05_POLICY_LAUNCH_SCRIPT",
    "",
)
POLICY_MODEL_CONFIGS: dict[str, dict[str, str | int]] = {
    PI05_FUNCTIONAL_GRASP_MODEL: {
        "server": PI05_POLICY_SERVER,
        "default_task_description": "functional grasp",
        "embodiment_tag": "xdof",
        "resolution": 480,
        "launch_script": PI05_POLICY_LAUNCH_SCRIPT,
        "reset_behavior": "noop",
    },
}

# Skill-name → policy-server routing for learn_skill.
# "hold" is a built-in keyword: holds current joint positions, no server needed.
# All other values are "host:port" addresses passed directly to the policy client.
SKILL_POLICY_SERVER: dict[str, str] = {
    "hold_still": "hold",
    "pick up stick": f"localhost:{POLICY_SERVER_PORT}",
}

# Reward server — Portal RPC
REWARD_SERVER_PORT = 8500

# Viser 3D visualization — WebSocket
VISER_PORT = 8080

# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

CONTROL_FREQ_HZ = 60.0
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ_HZ

POLICY_FREQ_HZ = 30.0
POLICY_PERIOD_S = 1.0 / POLICY_FREQ_HZ

# During human-in-the-loop takeover the control loop still runs at
# POLICY_FREQ_HZ, but obs building, reward evaluation, and the RL server RPC
# are skipped for N-1 out of every N steps.
# Set to 1 to disable (send every step, same as RL mode).
HIL_POLICY_SLOWDOWN: int = 3  # RL step every 3rd control step → ~10 Hz

# ---------------------------------------------------------------------------
# Human-in-the-loop leader arm takeover
# ---------------------------------------------------------------------------

# Leader-arm takeover hardware is not included in the pruned runtime.
USE_FELLO = False

# When True, leader-arm takeover overrides commanded joint positions in the
# CONTROL_FREQ_HZ control loop, not just during learn_skill.
ALWAYS_TAKEOVERABLE = False

# ---------------------------------------------------------------------------
# learn_skill data path
# ---------------------------------------------------------------------------

# Base directory for episode recordings from learn_skill.
# Episodes are saved as: <LEARN_SKILL_DATA_PATH>/<skill_name>/YYYYMMDDTHHMMSS######/
_DEFAULT_DATA_HOME = Path(
    os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
).expanduser()
ASPIRE_DATA_DIR = Path(
    os.environ.get("ASPIRE_DATA_DIR", str(_DEFAULT_DATA_HOME / "aspire"))
).expanduser()
LEARN_SKILL_DATA_PATH = os.environ.get(
    "LEARN_SKILL_DATA_PATH", str(ASPIRE_DATA_DIR / "learn_skill")
)

# USB drive label for insert_usb reward detection
USB_DRIVE_NAME = "Ubuntu 24.04.4 LTS amd64"

# ---------------------------------------------------------------------------
# RL policy server
# ---------------------------------------------------------------------------

RL_POLICY_HOST = (
    "localhost"  # RL policy server host (override with cap_server --rl-host)
)
RL_POLICY_PORT = 8965  # Portal RPC port for rl_policy_server
RL_EPISODE_MAX_STEPS = 1000  # Default max steps per episode
RL_DATA_PATH = os.environ.get(
    "RL_DATA_PATH", str(ASPIRE_DATA_DIR / "learn_skill_rl")
)

# ---------------------------------------------------------------------------
# Joint limits (from station.xml actuator ctrlrange)
# ---------------------------------------------------------------------------

# fmt: off
JOINT_LIMITS_LOW = np.array([
    -1.3962634016, 0.0, 0.0, -1.5708, -1.5708, -2.0944,  # left arm
    -1.3962634016, 0.0, 0.0, -1.5708, -1.5708, -2.0944,  # right arm
], dtype=np.float64)

JOINT_LIMITS_HIGH = np.array([
    1.3962634016, 3.66519, 3.66519, 1.5708, 1.5708, 2.0944,  # left arm
    1.3962634016, 3.66519, 3.66519, 1.5708, 1.5708, 2.0944,  # right arm
], dtype=np.float64)
# fmt: on

# Gripper range: 0.0 (closed) to 1.0 (open)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
GRIPPER_DEFAULT_WIDTH_M: float = 0.08

# ---------------------------------------------------------------------------
# Default motion parameters
# ---------------------------------------------------------------------------

# go_home: max joint speed (rad/s); duration is inferred from displacement / velocity
GO_HOME_MAX_JOINT_VEL: float = 1.0

# set_gripper: settle timeout, poll interval, and position-error threshold
GRIPPER_SETTLE_TIMEOUT_S: float = 1.5
GRIPPER_POLL_S: float = 0.05
GRIPPER_SETTLE_THRESH: float = 0.1
# Torque-limit stall detection: exit early when gripper position stops changing
GRIPPER_TORQUE_LIMIT_HOLD_S: float = 0.2  # seconds of stall before early exit
GRIPPER_STALL_THRESH: float = 0.02  # min pos change per poll to not be stalled

# _ik_servo defaults
MOVE_EEF_MAX_DURATION_S: float = 5.0
MOVE_EEF_MAX_VEL: float = 0.3  # m/s

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Motor gains (used during interpolation / settling)
# ---------------------------------------------------------------------------

INTERP_KP = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 10.0])
INTERP_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5])

# ---------------------------------------------------------------------------
# Camera names — derived from active station profile
# ---------------------------------------------------------------------------


def _resolve_camera_names() -> tuple[str, ...]:
    try:
        from robot.station_profiles import active_station_cameras

        return active_station_cameras().names
    except Exception:
        return ("top", "left", "right")


CAMERA_NAMES: tuple[str, ...] = _resolve_camera_names()

# ---------------------------------------------------------------------------
# Home / zero joint configuration
# ---------------------------------------------------------------------------

HOME_JOINT_STATE: dict[str, np.ndarray] = {
    "left_joint_pos": np.zeros(6),
    "left_gripper_pos": np.zeros(1),
    "right_joint_pos": np.zeros(6),
    "right_gripper_pos": np.zeros(1),
}

# ---------------------------------------------------------------------------
# VLM backends (used by vlm_query tool)
# ---------------------------------------------------------------------------

# Optional local or remote VLM server. Configure non-local endpoints explicitly.
SMOL_VLM_URL = os.environ.get("SMOL_VLM_URL", "http://127.0.0.1:8401/v1")
SMOL_VLM_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Gemini — Google cloud API (requires GEMINI_API_KEY env var).
# gemini-2.5-flash
GEMINI_VL_MODEL = os.environ.get("GEMINI_VL_MODEL", "gemini-2.5-flash")

# Default backend for vlm_query(...) when callers do not pass backend=...
DEFAULT_VLM_BACKEND = os.environ.get("DEFAULT_VLM_BACKEND", "nvidia")

# Qwen3-VL via SSH tunnel (ssh -N -L 8402:localhost:8000 <gpu-node>).
QWEN_VL_URL = os.environ.get("QWEN_VL_URL", "http://localhost:8402/v1")
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen3-VL-8B-Instruct")

# Gemini 3.1 Pro — Google cloud API with thinking (requires GEMINI_API_KEY env var).
GEMINI_PRO_VL_MODEL = os.environ.get("GEMINI_PRO_VL_MODEL", "gemini-3.1-pro-preview")

# GPT-5.4 — OpenAI API (requires OPENAI_API_KEY env var).
GPT_VL_MODEL = os.environ.get("GPT_VL_MODEL", "gpt-5.4")

# NVIDIA inference gateway — OpenAI-compatible, routes to Gemini / Bedrock Claude
# / Azure GPT / etc. based on the `<cloud>/<provider>/<model>` model string.
# Auth is the top-level NVIDIA_API_KEY env var OR any of NVIDIA_API_KEY_1..N
# (round-robin-able for parallel inference fanout).
NVIDIA_VL_BASE_URL = os.environ.get(
    "NVIDIA_VL_BASE_URL", "https://inference-api.nvidia.com/v1/"
)
NVIDIA_VL_MODEL = os.environ.get("NVIDIA_VL_MODEL", "gcp/google/gemini-3-flash-preview")

# ---------------------------------------------------------------------------
# Portal RPC
# ---------------------------------------------------------------------------

PORTAL_EMPTY_SENTINEL = "__none__"
"""Sentinel for Portal RPC empty-string arguments (Portal cannot transmit zero-length strings)."""

# ---------------------------------------------------------------------------
# Agent bridge
# ---------------------------------------------------------------------------

BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "localhost")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8201"))

# ---------------------------------------------------------------------------
# Voice API
# ---------------------------------------------------------------------------

CAP_VOICE_HOST = os.environ.get("CAP_VOICE_HOST", "localhost")
CAP_VOICE_PORT = int(os.environ.get("CAP_VOICE_PORT", "8202"))

# ---------------------------------------------------------------------------
# Grasp planning services
# ---------------------------------------------------------------------------

ANYGRASP_SERVER_HOST = os.environ.get("ANYGRASP_SERVER_HOST", "localhost")
ANYGRASP_SERVER_PORT = int(os.environ.get("ANYGRASP_SERVER_PORT", "8122"))
ANYGRASP_SERVER_URL = f"http://{ANYGRASP_SERVER_HOST}:{ANYGRASP_SERVER_PORT}"

# ---------------------------------------------------------------------------
# LLM and policy defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
DEFAULT_POLICY_SERVER = os.environ.get(
    "DEFAULT_POLICY_SERVER", f"localhost:{POLICY_SERVER_PORT}"
)

# ---------------------------------------------------------------------------
# HIL logging
# ---------------------------------------------------------------------------

HIL_LOG_DIR = Path(os.environ.get("HIL_LOG_DIR", "logs/hil"))
