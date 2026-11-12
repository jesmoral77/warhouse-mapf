"""Validator tests on hand-built path sets (no planners involved)."""

import numpy as np
import pytest

from mapf.grid import Grid
from mapf.validate import find_conflicts, validate


def kinds(conflicts):
    return [(c.kind, c.agents, c.t) for c in conflicts]



def test_disjoint_paths_valid():
    paths = [
        [(0, 0), (0, 1), (0, 2)],
        [(2, 0), (2, 1), (2, 2)],
    ]
    res = validate(paths)
    assert res.valid
    assert res.conflicts == [] and res.errors == []


def test_following_is_not_a_conflict():
    paths = [
        [(0, 1), (0, 2), (0, 3)],
        [(0, 0), (0, 1), (0, 2)],
    ]
    assert validate(paths).valid


def test_different_lengths_never_meet():
    paths = [
        [(0, 0)],
        [(2, 0), (2, 1), (2, 2), (2, 3), (1, 3)],
    ]
    assert validate(paths).valid


def test_single_and_empty():
    assert validate([[(0, 0), (0, 1)]]).valid
    assert validate([]).valid


def test_waiting_in_place_is_fine():
    paths = [
        [(0, 0), (0, 0), (0, 1)],
        [(1, 1), (1, 1), (1, 1)],
    ]
    assert validate(paths).valid



def test_vertex_conflict():
    paths = [
        [(0, 0), (0, 1), (0, 2)],
        [(2, 2), (1, 2), (0, 2)],
    ]
    res = validate(paths)
    assert not res.valid
    assert kinds(res.conflicts) == [("vertex", (0, 1), 2)]
    assert res.conflicts[0].cells == ((0, 2),)


def test_swap_conflict():
    paths = [
        [(0, 0), (0, 1)],
        [(0, 1), (0, 0)],
    ]
    res = validate(paths)
    assert not res.valid
    assert kinds(res.conflicts) == [("edge", (0, 1), 0)]
    assert res.conflicts[0].cells == ((0, 0), (0, 1))


def test_swap_detected_regardless_of_agent_order():
    paths = [
        [(5, 5), (5, 5), (5, 5)],
        [(1, 1), (1, 2)],
        [(1, 2), (1, 1)],
    ]
    assert kinds(find_conflicts(paths)) == [("edge", (1, 2), 0)]


def test_robot_at_goal_conflict():
    paths = [
        [(0, 0), (0, 1)],
        [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0)],
    ]
    res = validate(paths)
    assert not res.valid
    assert ("vertex", (0, 1), 3) in kinds(res.conflicts)
    assert all(c.cells == ((0, 1),) for c in res.conflicts if c.kind == "vertex")


def test_robot_at_goal_conflict_reverse_order():
    paths = [
        [(1, 3), (1, 2), (1, 1), (1, 0)],
        [(1, 1)],
    ]
    res = validate(paths)
    assert not res.valid
    assert kinds(res.conflicts) == [("vertex", (0, 1), 2)]


def test_arriving_onto_parked_robot():
    paths = [
        [(0, 0), (0, 1)],
        [(0, 3), (0, 2), (0, 1)],
    ]
    res = validate(paths)
    assert kinds(res.conflicts) == [("vertex", (0, 1), 2)]


def test_multiple_conflicts_three_robots():
    paths = [
        [(1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1)],
        [(1, 3), (1, 2), (1, 1)],
    ]
    res = validate(paths)
    assert not res.valid
    assert kinds(res.conflicts) == [
        ("vertex", (0, 1), 1),
        ("edge", (0, 2), 1),
    ]


def test_three_way_vertex_reports_all_pairs():
    paths = [
        [(0, 1), (1, 1)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 1)],
    ]
    res = validate(paths)
    assert kinds(res.conflicts) == [
        ("vertex", (0, 1), 1),
        ("vertex", (0, 2), 1),
        ("vertex", (1, 2), 1),
    ]
    assert res.n_conflicts == 3



@pytest.fixture
def small_grid():
    blocked = np.zeros((3, 4), dtype=bool)
    blocked[1, 1] = True
    return Grid(blocked)


def test_jump_is_an_error(small_grid):
    res = validate([[(0, 0), (0, 2)]], grid=small_grid)
    assert not res.valid and res.conflicts == []
    assert any("illegal move" in e for e in res.errors)


def test_diagonal_is_an_error():
    res = validate([[(0, 0), (1, 1)]])
    assert not res.valid
    assert any("illegal move" in e for e in res.errors)


def test_blocked_cell_is_an_error(small_grid):
    res = validate([[(0, 1), (1, 1), (2, 1)]], grid=small_grid)
    assert not res.valid and res.conflicts == []
    assert any("blocked" in e for e in res.errors)


def test_out_of_bounds_is_an_error(small_grid):
    res = validate([[(0, 0), (-1, 0)]], grid=small_grid)
    assert not res.valid
    assert any("out of bounds" in e for e in res.errors)


def test_start_and_goal_mismatch(small_grid):
    paths = [[(0, 0), (0, 1)]]
    res = validate(paths, grid=small_grid, starts=[(2, 0)], goals=[(0, 3)])
    assert not res.valid and res.conflicts == []
    assert any("starts at" in e for e in res.errors)
    assert any("ends at" in e for e in res.errors)


def test_matching_start_and_goal_ok(small_grid):
    paths = [[(0, 0), (0, 1), (0, 2)]]
    assert validate(paths, grid=small_grid, starts=[(0, 0)], goals=[(0, 2)]).valid


def test_empty_path_is_an_error():
    res = validate([[], [(0, 0)]])
    assert not res.valid and res.conflicts == []
    assert any("empty path" in e for e in res.errors)


def test_agent_count_mismatch():
    res = validate([[(0, 0)]], starts=[(0, 0), (1, 1)])
    assert not res.valid
    assert any("1 paths but 2 starts" in e for e in res.errors)
