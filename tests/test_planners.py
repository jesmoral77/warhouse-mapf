from pathlib import Path

import numpy as np
import pytest

from mapf.grid import Grid, load_map
from mapf.planners import cbs, independent, prioritized
from mapf.scenario import generate_scenario
from mapf.validate import validate

MAP_SMALL = Path(__file__).resolve().parent.parent / "maps" / "warehouse_small.map"
COORDINATED = [prioritized, cbs]


def soc(paths):
    return sum(len(p) - 1 for p in paths)


def cross_instance():
    grid = Grid(np.zeros((5, 5), dtype=bool))
    starts = [(2, 0), (2, 4), (0, 2), (4, 2)]
    goals = [(2, 4), (2, 0), (4, 2), (0, 2)]
    return grid, starts, goals


def corridor_with_bay():
    blocked = np.ones((2, 7), dtype=bool)
    blocked[0, :] = False
    blocked[1, 4] = False
    return Grid(blocked), [(0, 0), (0, 6)], [(0, 6), (0, 0)]


@pytest.mark.parametrize("planner", COORDINATED, ids=lambda p: p.NAME)
def test_coordinated_planners_valid_on_cross(planner):
    grid, starts, goals = cross_instance()
    res = planner.plan(grid, starts, goals, timeout=10)
    assert res.solved, res.message
    v = validate(res.paths, grid, starts, goals)
    assert v.valid, (v.conflicts, v.errors)


def test_independent_reports_conflicts_on_cross():
    grid, starts, goals = cross_instance()
    res = independent.plan(grid, starts, goals, timeout=10)
    assert res.solved
    v = validate(res.paths, grid, starts, goals)
    assert v.errors == []
    assert v.n_conflicts > 0 and not v.valid


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_all_planners_on_warehouse_4_robots(seed):
    grid = load_map(MAP_SMALL)
    sc = generate_scenario(grid, 4, seed)
    for planner in COORDINATED:
        res = planner.plan(grid, sc.starts, sc.goals, timeout=10)
        assert res.solved, (planner.NAME, res.message)
        v = validate(res.paths, grid, sc.starts, sc.goals)
        assert v.valid, (planner.NAME, v.conflicts, v.errors)
    res = independent.plan(grid, sc.starts, sc.goals, timeout=10)
    assert res.solved
    assert validate(res.paths, grid, sc.starts, sc.goals).errors == []


def test_corridor_swap_cbs_no_worse_than_prioritized():
    grid, starts, goals = corridor_with_bay()
    pr = prioritized.plan(grid, starts, goals, timeout=10)
    cb = cbs.plan(grid, starts, goals, timeout=10)
    assert pr.solved and cb.solved
    assert validate(pr.paths, grid, starts, goals).valid
    assert validate(cb.paths, grid, starts, goals).valid
    assert soc(cb.paths) <= soc(pr.paths)
    assert soc(cb.paths) == 15
    assert (1, 4) in cb.paths[1]


def test_cbs_beats_prioritized_with_bad_order():
    grid, starts, goals = corridor_with_bay()
    pr = prioritized.plan(grid, starts, goals, order=[1, 0])
    assert pr.status == "failed" and pr.paths is None and "agent 0" in pr.message
    assert cbs.plan(grid, starts, goals).solved


def test_cbs_timeout_is_recorded():
    grid, starts, goals = cross_instance()
    res = cbs.plan(grid, starts, goals, timeout=0)
    assert res.status == "timeout" and res.paths is None


def test_cbs_unsolvable_is_recorded():
    grid = Grid(np.zeros((1, 3), dtype=bool))
    res = cbs.plan(grid, [(0, 0), (0, 2)], [(0, 2), (0, 0)], timeout=0.5)
    assert res.status == "timeout" and res.paths is None
