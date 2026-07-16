"""LIBERO traced API wrappers.

Kept separate from robosuite traced wrappers so `.venv-libero` can register
LIBERO tracing without importing robosuite-specific API classes.
"""

from __future__ import annotations

from aspire.sim.cap.envs.base import BaseEnv
from aspire.sim.cap.integrations.franka.libero_reduced_skill_library import (
    FrankaLiberoApiReducedSkillLibrary,
)
from aspire.sim.cap.integrations.trace_logger import TracedApiMixin


class TracedFrankaLiberoApiReducedSkillLibrary(
    TracedApiMixin, FrankaLiberoApiReducedSkillLibrary
):
    """FrankaLiberoApiReducedSkillLibrary with execution trace logging."""

    def __init__(self, env: BaseEnv) -> None:
        FrankaLiberoApiReducedSkillLibrary.__init__(self, env)
        self.__init_trace__()
        self._trace_logger.set_env(env)
