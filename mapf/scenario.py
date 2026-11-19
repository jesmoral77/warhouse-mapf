"""Seeded random start/goal generation on free cells."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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


def check_scenario(grid: Grid, starts: list[Cell], goals: list[Cell]) -> None:
    problems = []
    if len(starts) == 0:
        problems.append("scenario has no agents")
    if len(starts) != len(goals):
        problems.append(f"{len(starts)} starts but {len(goals)} goals")
    for kind, cells in (("start", starts), ("goal", goals)):
        for i, cell in enumerate(cells):
            if not grid.in_bounds(cell):
                problems.append(f"agent {i} {kind} {cell} is outside the {grid.height}x{grid.width} map")
            elif not grid.is_free(cell):
                problems.append(f"agent {i} {kind} {cell} is on a blocked cell")
        if len(set(cells)) != len(cells):
            problems.append(f"two agents share the same {kind} cell")
    if not problems:
        for i, (s, g) in enumerate(zip(starts, goals)):
            if g not in grid.component(s):
                problems.append(f"agent {i} goal {g} is not reachable from start {s}")
    if problems:
        raise ValueError("; ".join(problems))


def load_scenario(path: str | Path, grid: Grid) -> Scenario:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scenario = Scenario.from_dict(data)
    except OSError as e:
        raise ValueError(f"cannot read scenario file {path}: {e}") from e
    except KeyError as e:
        raise ValueError(f"{path}: not a valid scenario file (missing key {e})") from e
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: not a valid scenario file ({e!r})") from e
    for cell in scenario.starts + scenario.goals:
        if len(cell) != 2 or not all(isinstance(x, int) for x in cell):
            raise ValueError(f"{path}: cell {list(cell)} is not a [row, col] pair of integers")
    check_scenario(grid, scenario.starts, scenario.goals)
    return scenario
