"""Common result object returned by every planner (failures are recorded, not raised)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mapf.grid import Cell

SOLVED = "solved"
FAILED = "failed"
TIMEOUT = "timeout"
ERROR = "error"


@dataclass
class PlanResult:
    planner: str
    status: str
    paths: list[list[Cell]] | None = None
    runtime_s: float = 0.0
    nodes_expanded: int = 0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        """True when the planner returned paths for all agents. This does NOT
        guarantee they are collision-free (e.g. the independent baseline). Use
        ``mapf.validate`` to check that."""
        return self.status == SOLVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner": self.planner,
            "status": self.status,
            "paths": None if self.paths is None else [[list(c) for c in p] for p in self.paths],
            "runtime_s": self.runtime_s,
            "nodes_expanded": self.nodes_expanded,
            "message": self.message,
            "extra": dict(self.extra),
        }
