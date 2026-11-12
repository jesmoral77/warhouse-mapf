import numpy as np
import pytest

from mapf.grid import Grid, generate_warehouse, load_map, save_map
from mapf.scenario import Scenario, generate_scenario


def test_map_roundtrip(tmp_path):
    g = generate_warehouse(20, 15)
    p = tmp_path / "w.map"
    save_map(g, p)
    assert load_map(p) == g


def test_load_movingai_chars(tmp_path):
    p = tmp_path / "m.map"
    p.write_text("type octile\nheight 2\nwidth 3\nmap\n.@T\nG.W\n")
    g = load_map(p)
    assert g.blocked.tolist() == [[False, True, True], [False, False, True]]


def test_load_rejects_bad_dims(tmp_path):
    p = tmp_path / "bad.map"
    p.write_text("type octile\nheight 2\nwidth 3\nmap\n...\n..\n")
    with pytest.raises(ValueError):
        load_map(p)


def test_neighbors():
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    g = Grid(blocked)
    assert set(g.neighbors((1, 1))) == {(2, 1), (1, 2), (1, 0)}
    assert set(g.neighbors((0, 0))) == {(1, 0)}


@pytest.mark.parametrize("w,h", [(20, 15), (40, 30), (13, 10)])
def test_warehouse_structure(w, h):
    g = generate_warehouse(w, h)
    assert (g.height, g.width) == (h, w)
    assert not g.blocked[0].any() and not g.blocked[-1].any()
    assert not g.blocked[:, 0].any() and not g.blocked[:, -1].any()
    assert len(g.largest_component()) == g.num_free
    for c in range(w):
        col = g.blocked[:, c]
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, col.astype(int), 0])))[::2]
        assert all(r == 2 for r in runs)
    for r in range(h):
        row = g.blocked[r]
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, row.astype(int), 0])))[::2]
        assert all(x == 5 for x in runs)
    assert g.blocked.any()


def test_scenario_deterministic_and_distinct():
    g = generate_warehouse(20, 15)
    a = generate_scenario(g, 8, seed=3)
    b = generate_scenario(g, 8, seed=3)
    assert a.starts == b.starts and a.goals == b.goals
    assert generate_scenario(g, 8, seed=4).starts != a.starts
    cells = a.starts + a.goals
    assert len(set(cells)) == 16
    assert all(g.is_free(c) for c in cells)


def test_scenario_roundtrip_dict():
    g = generate_warehouse(20, 15)
    s = generate_scenario(g, 4, seed=1)
    s2 = Scenario.from_dict(s.to_dict())
    assert s2.starts == s.starts and s2.goals == s.goals and s2.seed == 1


def test_scenario_too_many_agents():
    g = generate_warehouse(20, 15)
    with pytest.raises(ValueError):
        generate_scenario(g, g.num_free, seed=0)
