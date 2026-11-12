from __future__ import annotations

import time
from typing import Sequence

from mapf.astar import Constraints, space_time_astar
from mapf.grid import Cell, Grid
from mapf.planners.base import ERROR, FAILED, SOLVED, TIMEOUT, PlanResult

NAME = "prioritized"


def reserve(reservations: Constraints, path: list[Cell]) -> None:
    last = len(path) - 1
    for t in range(last):
        reservations.add_vertex(path[t], t)
        reservations.add_edge(path[t + 1], path[t], t)
    reservations.add_permanent(path[last], last)


def plan(
    grid: Grid,
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    timeout: float | None = None,
    order: Sequence[int] | None = None,
) -> PlanResult:
    t0 = time.perf_counter()
    deadline = None if timeout is None else t0 + timeout
    order = list(range(len(starts))) if order is None else list(order)
    expanded = 0
    paths: list[list[Cell] | None] = [None] * len(starts)
    reservations = Constraints()

    def result(status: str, message: str = "") -> PlanResult:
        out = paths if status == SOLVED else None
        return PlanResult(NAME, status, out, time.perf_counter() - t0, expanded, message,
                          {"order": order})

    try:
        for rank, i in enumerate(order):
            if deadline is not None and time.perf_counter() > deadline:
                return result(TIMEOUT, f"timed out after planning {rank} agents")
            res = space_time_astar(grid, starts[i], goals[i], reservations, deadline=deadline)
            expanded += res.nodes_expanded
            if res.timed_out:
                return result(TIMEOUT, f"timed out while planning agent {i}")
            if res.path is None:
                return result(FAILED, f"agent {i} (priority {rank}): no path avoiding earlier agents")
            paths[i] = res.path
            reserve(reservations, res.path)
    except Exception as e:
        return result(ERROR, repr(e))
    return result(SOLVED)
