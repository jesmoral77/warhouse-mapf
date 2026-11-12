"""Seeded random start/goal generation on free cells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mapf.grid import Cell, Grid


@dataclass
class Scenario:
    starts: list[Cell]
    goals: list[Cell]
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_agents(self) -> int:
        return len(self.starts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "starts": [list(c) for c in self.starts],
            "goals": [list(c) for c in self.goals],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        return cls(
            starts=[tuple(c) for c in d["starts"]],
            goals=[tuple(c) for c in d["goals"]],
            seed=d.get("seed"),
            meta=dict(d.get("meta", {})),
        )


def generate_scenario(grid: Grid, n_agents: int, seed: int) -> Scenario:
    """Sample ``n_agents`` start/goal pairs from the grid's largest connected
    component. All 2n cells are distinct, so starts are distinct, goals are
    distinct, and no start coincides with any goal. Every goal is reachable."""
    if n_agents < 0:
        raise ValueError("n_agents must be non-negative")
    cells = grid.largest_component()
    if 2 * n_agents > len(cells):
        raise ValueError(
            f"need {2 * n_agents} distinct cells but largest component has {len(cells)}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cells), size=2 * n_agents, replace=False)
    picked = [cells[i] for i in idx]
    return Scenario(starts=picked[:n_agents], goals=picked[n_agents:], seed=seed)
