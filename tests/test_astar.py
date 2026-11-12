from collections import deque
from pathlib import Path

import numpy as np
import pytest

from mapf.astar import Constraints, astar, space_time_astar
from mapf.grid import Grid, load_map
from mapf.planners import independent
from mapf.scenario import generate_scenario
from mapf.validate import validate

MAP_SMALL = Path(__file__).resolve().parent.parent / "maps" / "warehouse_small.map"


def open_grid(h, w):
    return Grid(np.zeros((h, w), dtype=bool))


def corridor(n=5):
    return open_grid(1, n)


def bfs_dist(grid, s, g):
    """Reference distance computed independently of A*."""
    dist = {s: 0}
    q = deque([s])
    while q:
        c = q.popleft()
        if c == g:
            return dist[c]
        for n in grid.neighbors(c):
            if n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return None


def assert_legal(grid, path):
    for t, c in enumerate(path):
        assert grid.is_free(c)
        if t:
            p = path[t - 1]
            assert abs(c[0] - p[0]) + abs(c[1] - p[1]) <= 1



def test_single_robot_shortest_path_open_grid():
    g = open_grid(5, 5)
    res = astar(g, (0, 0), (4, 4))
    assert res.path[0] == (0, 0) and res.path[-1] == (4, 4)
    assert len(res.path) - 1 == 8
    assert_legal(g, res.path)


def test_single_robot_shortest_path_matches_bfs_on_warehouse():
    g = load_map(MAP_SMALL)
    sc = generate_scenario(g, 15, seed=7)
    for s, goal in zip(sc.starts, sc.goals):
        res = astar(g, s, goal)
        assert len(res.path) - 1 == bfs_dist(g, s, goal)
        assert_legal(g, res.path)


def test_path_around_shelf():
    blocked = np.zeros((3, 5), dtype=bool)
    blocked[1, 0:4] = True
    g = Grid(blocked)
    res = astar(g, (0, 0), (2, 0))
    assert len(res.path) - 1 == bfs_dist(g, (0, 0), (2, 0)) == 10


def test_unreachable_returns_none():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[:, 1] = True
    assert astar(Grid(blocked), (0, 0), (0, 2)).path is None


def test_start_equals_goal():
    assert astar(open_grid(2, 2), (1, 1), (1, 1)).path == [(1, 1)]



def test_space_time_without_constraints_matches_spatial():
    g = load_map(MAP_SMALL)
    sc = generate_scenario(g, 10, seed=2)
    for s, goal in zip(sc.starts, sc.goals):
        assert len(space_time_astar(g, s, goal).path) == len(astar(g, s, goal).path)


def test_waits_when_vertex_constraint_blocks_corridor():
    g = corridor(5)
    cons = Constraints()
    cons.add_vertex((0, 1), 1)
    res = space_time_astar(g, (0, 0), (0, 4), cons)
    assert res.path == [(0, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert len(res.path) - 1 == 5


def test_waits_for_longer_vertex_block():
    g = corridor(5)
    cons = Constraints()
    for t in (1, 2, 3):
        cons.add_vertex((0, 1), t)
    res = space_time_astar(g, (0, 0), (0, 4), cons)
    assert res.path[:4] == [(0, 0)] * 4
    assert res.path[-1] == (0, 4) and len(res.path) - 1 == 7
    assert all(not cons.vertex_blocked(c, t) for t, c in enumerate(res.path))


def test_edge_constraint_forces_wait():
    g = corridor(3)
    cons = Constraints()
    cons.add_edge((0, 0), (0, 1), 0)
    res = space_time_astar(g, (0, 0), (0, 2), cons)
    assert res.path == [(0, 0), (0, 0), (0, 1), (0, 2)]


def test_goal_stay_rule_future_constraint_on_goal():
    g = corridor(5)
    cons = Constraints()
    cons.add_vertex((0, 4), 6)
    res = space_time_astar(g, (0, 0), (0, 4), cons)
    assert res.path[-1] == (0, 4)
    assert len(res.path) - 1 == 7
    assert res.path[6] != (0, 4)
    assert all(not cons.vertex_blocked(c, t) for t, c in enumerate(res.path))


def test_goal_constraint_before_arrival_is_harmless():
    g = corridor(5)
    cons = Constraints()
    cons.add_vertex((0, 4), 2)
    res = space_time_astar(g, (0, 0), (0, 4), cons)
    assert len(res.path) - 1 == 4


def test_permanent_constraint_on_goal_has_no_solution():
    g = corridor(5)
    cons = Constraints()
    cons.add_permanent((0, 4), 10)
    assert space_time_astar(g, (0, 0), (0, 4), cons).path is None


def test_permanent_constraint_on_corridor_cell_blocks_passage():
    g = corridor(5)
    cons = Constraints()
    cons.add_permanent((0, 2), 3)
    res = space_time_astar(g, (0, 0), (0, 4), cons)
    assert len(res.path) - 1 == 4
    cons.add_vertex((0, 1), 1)
    assert space_time_astar(g, (0, 0), (0, 4), cons).path is None


def test_unreachable_terminates():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[:, 1] = True
    cons = Constraints()
    cons.add_vertex((0, 0), 50)
    res = space_time_astar(Grid(blocked), (0, 0), (0, 2), cons)
    assert res.path is None and not res.timed_out


def test_start_blocked_at_t0():
    cons = Constraints()
    cons.add_vertex((0, 0), 0)
    assert space_time_astar(corridor(3), (0, 0), (0, 2), cons).path is None


# Independent planner


def test_independent_returns_shortest_paths_even_if_conflicting():
    g = corridor(4)
    res = independent.plan(g, [(0, 0), (0, 3)], [(0, 3), (0, 0)])
    assert res.solved and res.status == "solved"
    assert [len(p) - 1 for p in res.paths] == [3, 3]
    assert not validate(res.paths, g).valid


def test_independent_failure_is_recorded():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[:, 1] = True
    res = independent.plan(Grid(blocked), [(0, 0)], [(0, 2)])
    assert res.status == "failed" and res.paths is None and "agent 0" in res.message


def test_independent_on_warehouse():
    g = load_map(MAP_SMALL)
    sc = generate_scenario(g, 6, seed=1)
    res = independent.plan(g, sc.starts, sc.goals, timeout=10)
    assert res.solved and res.nodes_expanded > 0
    v = validate(res.paths, g, sc.starts, sc.goals)
    assert v.errors == []
