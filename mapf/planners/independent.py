"""Baseline: each robot plans its own shortest path and ignores the others.
The resulting paths will usually contain conflicts; that is the point of the baseline."""

from __future__ import annotations

import time
from typing import Sequence

from mapf.astar import astar
from mapf.grid import Cell, Grid
from mapf.planners.base import ERROR, FAILED, SOLVED, TIMEOUT, PlanResult

NAME = "independent"


def plan(
    grid: Grid,
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    timeout: float | None = None,
) -> PlanResult:
    t0 = time.perf_counter()
    expanded = 0
    paths: list[list[Cell]] = []
    try:
        for i, (s, g) in enumerate(zip(starts, goals)):
            if timeout is not None and time.perf_counter() - t0 > timeout:
                return PlanResult(NAME, TIMEOUT, None, time.perf_counter() - t0, expanded,
                                  f"timed out after planning {i} agents")
            res = astar(grid, s, g)
            expanded += res.nodes_expanded
            if res.path is None:
                return PlanResult(NAME, FAILED, None, time.perf_counter() - t0, expanded,
                                  f"agent {i}: no path from {tuple(s)} to {tuple(g)}")
            paths.append(res.path)
    except Exception as e:
        return PlanResult(NAME, ERROR, None, time.perf_counter() - t0, expanded, repr(e))
    return PlanResult(NAME, SOLVED, paths, time.perf_counter() - t0, expanded)
