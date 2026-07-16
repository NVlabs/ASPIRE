"""VLM transport for the openforge NVIDIA-only runtime."""

from __future__ import annotations

from cap.agent.tools.vlm.transport import VLMBackend, query, register  # noqa: F401

# Importing the backend module registers it with the transport registry.
from cap.agent.tools.vlm.backends import nvidia  # noqa: F401

from cap.agent.tools.vlm.backends.nvidia import (  # noqa: F401
    list_nvidia_keys,
    pick_nvidia_key,
)
