# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episodic agent skill library — registry, profiling, and decorator.

SkillProfile tracks execution logs for a single versioned skill function.
SkillRegistry is a process-level singleton that owns all profiles and exposes
a ``@skill`` decorator.  Decorated functions must return ``(val, execution_log)``
where *execution_log* is a dict; the registry intercepts the pair, appends the
log, then returns both unchanged so caller code is unaffected.

Typical usage in a skill file::

    from cap.agent.skill_registry import skill

    @skill
    def pick_object_v2(ctx, target):
        \"\"\"Pick a target object.\"\"\"
        ...
        return result, {"success": True, "target": target}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class SkillProfile:
    name: str  # e.g. "pick_object_v1"
    base_name: str  # e.g. "pick_object"
    version: int  # e.g. 1
    docstring: str
    source_file: str | None = None
    source_line: int | None = None
    calls: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.calls:
            return 0.0
        return sum(1 for c in self.calls if c.get("success")) / len(self.calls)

    def record(self, log: dict) -> None:
        self.calls.append(log)

    def logs(self) -> list[dict]:
        """All execution logs for this skill. Fed raw to reflection LLM."""
        return self.calls


class SkillRegistry:
    """Singleton owning all SkillProfiles for the current process."""

    _instance: SkillRegistry | None = None

    def __init__(self) -> None:
        self._profiles: dict[str, SkillProfile] = {}
        # Optional hooks: each called with (skill_name, call_id, phase)
        # where phase is "before" or "after".
        self._capture_hooks: list[Callable] = []

    def set_capture_hook(self, hook: Callable | None) -> None:
        """Replace hooks called before and after every @skill function.

        Signature: hook(skill_name: str, call_id: int, phase: str) -> None
        phase is "before" (pre-call) or "after" (post-call, result available).
        """
        self._capture_hooks = [] if hook is None else [hook]

    def add_capture_hook(self, hook: Callable | None) -> None:
        """Append a before/after hook without replacing existing hooks."""
        if hook is not None:
            self._capture_hooks.append(hook)

    @classmethod
    def get(cls) -> SkillRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def register(self, fn: Callable) -> Callable:
        """@skill decorator: registers fn, wraps it to capture execution_log automatically.

        The wrapped function intercepts the (val, execution_log) return value,
        injects ``time_s`` (wall-clock duration of the call) into *execution_log*
        if not already present, appends it to the profile's calls list, then
        returns (val, execution_log) unchanged so caller code is unaffected.
        """
        import time as _time

        name = fn.__name__
        base, version = _parse_name(name)
        profile = SkillProfile(
            name=name,
            base_name=base,
            version=version,
            docstring=fn.__doc__ or "",
            source_file=str(Path(fn.__code__.co_filename).resolve()),
            source_line=fn.__code__.co_firstlineno,
        )
        self._profiles[name] = profile

        _call_seq = [0]

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _call_seq[0] += 1
            call_id = _call_seq[0]
            hooks = list(self._capture_hooks)
            for hook in hooks:
                try:
                    hook(name, call_id, "before")
                except Exception:
                    pass
            t0 = _time.time()
            try:
                val, log = fn(*args, **kwargs)
            except Exception:
                for hook in hooks:
                    try:
                        hook(name, call_id, "error")
                    except Exception:
                        pass
                raise
            if isinstance(log, dict) and "time_s" not in log:
                log["time_s"] = round(_time.time() - t0, 3)
            profile.record(log)
            for hook in hooks:
                try:
                    hook(name, call_id, "after")
                except Exception:
                    pass
            return val, log

        wrapper.__name__ = name
        wrapper.__doc__ = fn.__doc__
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    def profiles(self) -> dict[str, SkillProfile]:
        return dict(self._profiles)

    def to_json(self) -> list[dict[str, Any]]:
        """Serialize all profiles for writing to skill_log.json in subprocess."""
        return [
            {
                "name": p.name,
                "base_name": p.base_name,
                "version": p.version,
                "docstring": p.docstring,
                "source_file": p.source_file,
                "source_line": p.source_line,
                "calls": len(p.calls),
                "success_rate": round(p.success_rate, 3),
                "logs": p.logs(),
            }
            for p in self._profiles.values()
        ]


def _parse_name(name: str) -> tuple[str, int]:
    """'pick_object_v2' -> ('pick_object', 2); 'pick_object' -> ('pick_object', 1)"""
    m = re.match(r"^(.+?)_v(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 1)


# Convenient decorator alias used in skill files
skill = SkillRegistry.get().register
