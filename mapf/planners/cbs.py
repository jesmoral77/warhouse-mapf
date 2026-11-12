from __future__ import annotations

import heapq
import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from mapf.astar import Constraints, space_time_astar
from mapf.grid import Cell, Grid
from mapf.planners.base import ERROR, FAILED, SOLVED, TIMEOUT, PlanResult

NAME = "cbs"
log = logging.getLogger("mapf.cbs")


def _at(path: list[Cell], t: int) -> Cell:
    return path[min(t, len(path) - 1)]


def find_conflicts(paths: list[list[Cell]]) -> list[tuple]:
    horizon = max((len(p) for p in paths), default=0)
    out: list[tuple] = []
    for t in range(horizon):
        occupants: dict[Cell, list[int]] = defaultdict(list)
        for i, p in enumerate(paths):
            occupants[_at(p, t)].append(i)
        for cell, who in occupants.items():
            for a in range(len(who)):
                for b in range(a + 1, len(who)):
                    out.append(("vertex", who[a], who[b], cell, t))
        if t + 1 < horizon:
            moves: dict[tuple[Cell, Cell], int] = {}
            for i, p in enumerate(paths):
                u, v = _at(p, t), _at(p, t + 1)
                if u == v:
                    continue
                j = moves.get((v, u))
                if j is not None:
                    out.append(("edge", j, i, v, u, t))
                moves[(u, v)] = i
    return out


def cost(paths: list[list[Cell]]) -> int:
    return sum(len(p) - 1 for p in paths)


@dataclass
class Node:
    constraints: list[Constraints]
    paths: list[list[Cell]]
    cost: int
    conflicts: list[tuple]


def plan(
    grid: Grid,
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    timeout: float | None = None,
) -> PlanResult:
    t0 = time.perf_counter()
    deadline = None if timeout is None else t0 + timeout
    stats = {"high_level_expanded": 0, "high_level_generated": 0}
    expanded = 0

    def result(status: str, paths=None, message: str = "") -> PlanResult:
        return PlanResult(NAME, status, paths, time.perf_counter() - t0, expanded, message,
                          dict(stats))

    try:
        n = len(starts)
        root_cons = [Constraints() for _ in range(n)]
        root_paths = []
        for i in range(n):
            res = space_time_astar(grid, starts[i], goals[i], root_cons[i], deadline=deadline)
            expanded += res.nodes_expanded
            if res.timed_out:
                return result(TIMEOUT, message="timed out planning root")
            if res.path is None:
                return result(FAILED, message=f"agent {i}: no path even without other agents")
            root_paths.append(res.path)

        tie = itertools.count()
        root = Node(root_cons, root_paths, cost(root_paths), find_conflicts(root_paths))
        stats["high_level_generated"] += 1
        log.info("root: cost %d, %d conflicts", root.cost, len(root.conflicts))
        open_heap = [(root.cost, len(root.conflicts), next(tie), root)]

        while open_heap:
            if deadline is not None and time.perf_counter() > deadline:
                return result(TIMEOUT, message=f"timed out after {stats['high_level_expanded']} nodes")
            _, _, _, node = heapq.heappop(open_heap)
            stats["high_level_expanded"] += 1
            conflict = node.conflicts[0] if node.conflicts else None
            log.info("expand #%d: cost %d, %d conflicts, open %d, branching on %s",
                     stats["high_level_expanded"], node.cost, len(node.conflicts),
                     len(open_heap), conflict)
            if conflict is None:
                log.info("solution found: cost %d, %d high-level nodes expanded, %d generated",
                         node.cost, stats["high_level_expanded"], stats["high_level_generated"])
                return result(SOLVED, node.paths)

            if conflict[0] == "vertex":
                _, i, j, cell, t = conflict
                branches = [(i, ("vertex", cell, t)), (j, ("vertex", cell, t))]
            else:
                _, i, j, u, v, t = conflict
                branches = [(i, ("edge", u, v, t)), (j, ("edge", v, u, t))]

            for agent, con in branches:
                cons = node.constraints[agent].copy()
                if con[0] == "vertex":
                    cons.add_vertex(con[1], con[2])
                else:
                    cons.add_edge(con[1], con[2], con[3])
                res = space_time_astar(grid, starts[agent], goals[agent], cons, deadline=deadline)
                expanded += res.nodes_expanded
                if res.timed_out:
                    return result(TIMEOUT, message="timed out in low-level search")
                if res.path is None:
                    continue
                constraints = list(node.constraints)
                constraints[agent] = cons
                paths = list(node.paths)
                paths[agent] = res.path
                child = Node(constraints, paths, cost(paths), find_conflicts(paths))
                stats["high_level_generated"] += 1
                heapq.heappush(open_heap, (child.cost, len(child.conflicts), next(tie), child))

        return result(FAILED, message="constraint tree exhausted")
    except Exception as e:
        return result(ERROR, message=repr(e))
