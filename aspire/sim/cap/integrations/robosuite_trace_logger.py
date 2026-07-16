"""Robosuite traced API wrappers."""

from __future__ import annotations

from aspire.sim.cap.envs.base import BaseEnv
from aspire.sim.cap.integrations.franka.control_reduced_skill_library import (
    FrankaControlApiReducedSkillLibrary,
)
from aspire.sim.cap.integrations.trace_logger import TracedApiMixin


class TracedFrankaControlApiReducedSkillLibrary(
    TracedApiMixin, FrankaControlApiReducedSkillLibrary
):
    """FrankaControlApiReducedSkillLibrary with execution trace logging."""

    def __init__(self, env: BaseEnv, **kwargs) -> None:
        FrankaControlApiReducedSkillLibrary.__init__(self, env, **kwargs)
        self.__init_trace__()
        self._trace_logger.set_env(env)
