"""Spatial A* and space-time A* on a 4-connected Grid.

Constraint semantics (matching ``mapf.validate`` conflicts):
- vertex ``(cell, t)``: the agent may not be at ``cell`` at time ``t``.
- edge ``(u, v, t)``: the agent may not move ``u -> v`` between ``t`` and ``t+1``.
- permanent ``cell -> t0``: ``cell`` is forbidden at every ``t >= t0`` (used by
  prioritized planning to keep out of earlier robots' parked goal cells).
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

from mapf.grid import Cell, Grid

INF = float("inf")


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class SearchResult:
    path: list[Cell] | None
    nodes_expanded: int = 0
    timed_out: bool = False

    @property
    def found(self) -> bool:
        return self.path is not None


@dataclass
class Constraints:
    vertex: set[tuple[Cell, int]] = field(default_factory=set)
    edge: set[tuple[Cell, Cell, int]] = field(default_factory=set)
    permanent: dict[Cell, int] = field(default_factory=dict)

    def add_vertex(self, cell: Cell, t: int) -> None:
        self.vertex.add((tuple(cell), t))

    def add_edge(self, u: Cell, v: Cell, t: int) -> None:
        self.edge.add((tuple(u), tuple(v), t))

    def add_permanent(self, cell: Cell, t0: int) -> None:
        cell = tuple(cell)
        self.permanent[cell] = min(t0, self.permanent.get(cell, t0))

    def copy(self) -> "Constraints":
        return Constraints(set(self.vertex), set(self.edge), dict(self.permanent))

    def max_time(self) -> int:
        """Latest time step at which any constraint has an effect (-1 if none)."""
        ts = [t for _, t in self.vertex]
        ts += [t + 1 for _, _, t in self.edge]
        ts += list(self.permanent.values())
        return max(ts, default=-1)

    def vertex_blocked(self, cell: Cell, t: int) -> bool:
        if (cell, t) in self.vertex:
            return True
        t0 = self.permanent.get(cell)
        return t0 is not None and t >= t0

    def edge_blocked(self, u: Cell, v: Cell, t: int) -> bool:
        return (u, v, t) in self.edge

    def last_vertex_time(self, cell: Cell) -> float:
        """Last t at which ``cell`` is forbidden; INF if permanently; -1 if never."""
        if cell in self.permanent:
            return INF
        return max((t for c, t in self.vertex if c == cell), default=-1)


def _reconstruct(parent: dict, node) -> list[Cell]:
    path = []
    while node is not None:
        path.append(node[0])
        node = parent[node]
    path.reverse()
    return path


# Spatial A*


def astar(grid: Grid, start: Cell, goal: Cell) -> SearchResult:
    """Shortest 4-connected path ignoring time and other robots."""
    start, goal = tuple(start), tuple(goal)
    if not (grid.is_free(start) and grid.is_free(goal)):
        return SearchResult(None, 0)

    tie = itertools.count()
    g = {start: 0}
    parent: dict = {(start,): None}
    open_heap = [(manhattan(start, goal), 0, next(tie), start)]
    closed: set[Cell] = set()
    expanded = 0

    while open_heap:
        _, gc, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal:
            return SearchResult(_reconstruct(parent, (cur,)), expanded)
        for n in grid.neighbors(cur):
            ng = gc + 1
            if ng < g.get(n, INF):
                g[n] = ng
                parent[(n,)] = (cur,)
                heapq.heappush(open_heap, (ng + manhattan(n, goal), ng, next(tie), n))
    return SearchResult(None, expanded)


# Space-time A*


def space_time_astar(
    grid: Grid,
    start: Cell,
    goal: Cell,
    constraints: Constraints | None = None,
    max_time: int | None = None,
    deadline: float | None = None,
) -> SearchResult:
    """Earliest-arrival path in (cell, t) space honoring ``constraints``.

    Goal-stay rule: the search may only finish at ``goal`` at time ``t`` if no
    constraint forbids ``goal`` at any time >= t, because the robot stays there
    forever. ``max_time`` defaults to ``grid.num_free + constraints.max_time()``.
    ``deadline`` is a ``time.perf_counter()`` value; if passed, the search stops
    with ``timed_out=True``.
    """
    start, goal = tuple(start), tuple(goal)
    cons = constraints or Constraints()
    if not (grid.is_free(start) and grid.is_free(goal)):
        return SearchResult(None, 0)
    if cons.vertex_blocked(start, 0):
        return SearchResult(None, 0)

    goal_last = cons.last_vertex_time(goal)
    if goal_last == INF:
        return SearchResult(None, 0)

    t_static = cons.max_time() + 1
    horizon = max_time if max_time is not None else grid.num_free + t_static

    tie = itertools.count()
    start_node = (start, 0)
    parent: dict = {start_node: None}
    best_g: dict = {(start, 0): 0}
    open_heap = [(manhattan(start, goal), manhattan(start, goal), next(tie), start_node)]
    closed: set = set()
    expanded = 0

    while open_heap:
        _, _, _, node = heapq.heappop(open_heap)
        cell, t = node
        key = (cell, min(t, t_static))
        if key in closed:
            continue
        closed.add(key)
        expanded += 1

        if deadline is not None and (expanded & 1023) == 0 and time.perf_counter() > deadline:
            return SearchResult(None, expanded, timed_out=True)

        if cell == goal and t > goal_last:
            return SearchResult(_reconstruct(parent, node), expanded)
        if t >= horizon:
            continue

        nt = t + 1
        for n in [cell, *grid.neighbors(cell)]:
            if cons.vertex_blocked(n, nt) or cons.edge_blocked(cell, n, t):
                continue
            nkey = (n, min(nt, t_static))
            if nkey in closed or best_g.get(nkey, INF) <= nt:
                continue
            best_g[nkey] = nt
            child = (n, nt)
            parent[child] = node
            h = manhattan(n, goal)
            heapq.heappush(open_heap, (nt + h, h, next(tie), child))

    return SearchResult(None, expanded)
