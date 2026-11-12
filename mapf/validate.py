"""Independent solution checker. Deliberately imports nothing from the planners
or A* so it can be trusted to audit their output.

A path is a list of cells; ``path[t]`` is the robot's position at time ``t``.
After its path ends the robot stays at ``path[-1]`` forever.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

Cell = tuple[int, int]


@dataclass(frozen=True)
class Conflict:
    kind: str
    agents: tuple[int, int]
    t: int
    cells: tuple[Cell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "agents": list(self.agents),
            "t": self.t,
            "cells": [list(c) for c in self.cells],
        }


@dataclass
class ValidationResult:
    valid: bool
    conflicts: list[Conflict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_conflicts(self) -> int:
        return len(self.conflicts)


def position(path: Sequence[Cell], t: int) -> Cell:
    """Position at time t, holding the final cell forever after the path ends."""
    return tuple(path[min(t, len(path) - 1)])


def find_conflicts(paths: Sequence[Sequence[Cell]]) -> list[Conflict]:
    """All vertex and edge (swap) conflicts. Empty paths are skipped."""
    agents = [i for i, p in enumerate(paths) if len(p) > 0]
    if not agents:
        return []
    horizon = max(len(paths[i]) for i in agents)
    conflicts: list[Conflict] = []

    for t in range(horizon):
        occupants: dict[Cell, list[int]] = defaultdict(list)
        for i in agents:
            occupants[position(paths[i], t)].append(i)
        for cell, who in occupants.items():
            for a in range(len(who)):
                for b in range(a + 1, len(who)):
                    conflicts.append(Conflict("vertex", (who[a], who[b]), t, (cell,)))

        if t + 1 < horizon:
            moves: dict[tuple[Cell, Cell], int] = {}
            for i in agents:
                u, v = position(paths[i], t), position(paths[i], t + 1)
                if u == v:
                    continue
                j = moves.get((v, u))
                if j is not None:
                    a, b = (j, i) if j < i else (i, j)
                    conflicts.append(
                        Conflict(
                            "edge",
                            (a, b),
                            t,
                            (position(paths[a], t), position(paths[b], t)),
                        )
                    )
                moves[(u, v)] = i

    conflicts.sort(key=lambda c: (c.t, c.kind != "vertex", c.agents))
    return conflicts


def validate(
    paths: Sequence[Sequence[Cell]],
    grid: Any = None,
    starts: Sequence[Cell] | None = None,
    goals: Sequence[Cell] | None = None,
) -> ValidationResult:
    """Check a full solution. ``grid`` only needs ``in_bounds`` and ``is_free``.
    Never raises on malformed input; problems are reported in ``errors``."""
    errors: list[str] = []

    for name, ref in (("starts", starts), ("goals", goals)):
        if ref is not None and len(ref) != len(paths):
            errors.append(f"{len(paths)} paths but {len(ref)} {name}")

    for i, path in enumerate(paths):
        if len(path) == 0:
            errors.append(f"agent {i}: empty path")
            continue
        for t, cell in enumerate(path):
            cell = tuple(cell)
            if grid is not None:
                if not grid.in_bounds(cell):
                    errors.append(f"agent {i}: t={t} cell {cell} out of bounds")
                elif not grid.is_free(cell):
                    errors.append(f"agent {i}: t={t} cell {cell} is blocked")
            if t > 0:
                prev = tuple(path[t - 1])
                if abs(cell[0] - prev[0]) + abs(cell[1] - prev[1]) > 1:
                    errors.append(f"agent {i}: illegal move {prev}->{cell} at t={t - 1}")
        if starts is not None and i < len(starts) and tuple(path[0]) != tuple(starts[i]):
            errors.append(f"agent {i}: starts at {tuple(path[0])}, expected {tuple(starts[i])}")
        if goals is not None and i < len(goals) and tuple(path[-1]) != tuple(goals[i]):
            errors.append(f"agent {i}: ends at {tuple(path[-1])}, expected {tuple(goals[i])}")

    conflicts = find_conflicts(paths)
    return ValidationResult(valid=not conflicts and not errors, conflicts=conflicts, errors=errors)
