import json
from pathlib import Path

import numpy as np
import pytest

from mapf import run
from mapf.grid import Grid, load_map
from mapf.scenario import check_scenario, load_scenario
from mapf.trace import ACTIONS, action_name, build_trace, step_line

MAP_SMALL = str(Path(__file__).resolve().parent.parent / "maps" / "warehouse_small.map")


def small_grid():
    blocked = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 0],
    ], dtype=bool)
    return Grid(blocked)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_scenario_valid(tmp_path):
    path = write_json(tmp_path / "ok.json", {"starts": [[0, 0], [2, 3]], "goals": [[2, 0], [0, 3]]})
    scenario = load_scenario(path, small_grid())
    assert scenario.starts == [(0, 0), (2, 3)]
    assert scenario.goals == [(2, 0), (0, 3)]


@pytest.mark.parametrize("starts, goals, expected", [
    ([(1, 1)], [(0, 0)], "blocked"),
    ([(9, 9)], [(0, 0)], "outside"),
    ([(0, 0), (0, 1)], [(2, 2), (2, 2)], "same goal"),
    ([(0, 0), (0, 1)], [(2, 2)], "2 starts but 1 goals"),
    ([(0, 0)], [(4, 0)], "not reachable"),
    ([], [], "no agents"),
])
def test_check_scenario_rejects(starts, goals, expected):
    with pytest.raises(ValueError, match=expected):
        check_scenario(small_grid(), starts, goals)


def test_load_scenario_rejects_bad_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid scenario"):
        load_scenario(path, small_grid())
    with pytest.raises(ValueError, match="missing key"):
        load_scenario(write_json(tmp_path / "nokey.json", {"starts": [[0, 0]]}), small_grid())


def blocked_cell_of_small_map():
    grid = load_map(MAP_SMALL)
    rows, cols = np.nonzero(grid.blocked)
    return [int(rows[0]), int(cols[0])]


def test_run_invalid_scenario_is_recorded(tmp_path):
    bad = write_json(tmp_path / "bad.json", {"starts": [blocked_cell_of_small_map()], "goals": [[0, 0]]})
    out = tmp_path / "run"
    metrics = run.main(["--map", MAP_SMALL, "--scenario", str(bad), "--planner", "cbs",
                        "--out", str(out)])
    assert metrics["status"] == "invalid_input"
    assert not metrics["success"]
    assert "blocked" in metrics["message"]
    assert json.loads((out / "metrics.json").read_text()) == metrics
    assert "invalid_input" in (out / "run.log").read_text()
    assert not (out / "plan.json").exists()


def test_run_too_many_agents_is_recorded(tmp_path):
    out = tmp_path / "run"
    metrics = run.main(["--map", MAP_SMALL, "--agents", "500", "--seed", "0", "--planner", "cbs",
                        "--out", str(out)])
    assert metrics["status"] == "invalid_input"
    assert "distinct cells" in metrics["message"]


def test_run_missing_map_is_recorded(tmp_path):
    out = tmp_path / "run"
    metrics = run.main(["--map", str(tmp_path / "nope.map"), "--agents", "2", "--seed", "0",
                        "--planner", "cbs", "--out", str(out)])
    assert metrics["status"] == "invalid_input"
    assert "cannot read map" in metrics["message"]


def test_run_accepts_scenario_file(tmp_path):
    grid = load_map(MAP_SMALL)
    free = grid.largest_component()
    task = write_json(tmp_path / "task.json", {"starts": [free[0], free[1]], "goals": [free[-1], free[-2]]})
    out = tmp_path / "run"
    metrics = run.main(["--map", MAP_SMALL, "--scenario", str(task), "--planner", "cbs",
                        "--out", str(out), "--substeps", "1"])
    assert metrics["success"]
    plan = json.loads((out / "plan.json").read_text())
    assert plan["scenario_file"] == str(task)
    assert plan["n_agents"] == 2


def test_run_requires_one_task_source(tmp_path):
    with pytest.raises(SystemExit):
        run.main(["--map", MAP_SMALL, "--planner", "cbs", "--out", str(tmp_path / "run")])


def test_action_name():
    assert action_name((2, 2), (1, 2)) == "N"
    assert action_name((2, 2), (3, 2)) == "S"
    assert action_name((2, 2), (2, 3)) == "E"
    assert action_name((2, 2), (2, 1)) == "W"
    assert action_name((2, 2), (2, 2)) == "Wait"
    with pytest.raises(ValueError):
        action_name((2, 2), (4, 2))


def test_build_trace_holds_robot_at_goal():
    paths = [[(0, 0), (0, 1), (0, 2)], [(2, 0), (2, 0), (1, 0), (0, 0)]]
    goals = [(0, 2), (0, 0)]
    steps = build_trace(paths, goals)
    assert [s["t"] for s in steps] == [0, 1, 2, 3]
    robot0 = [s["robots"][0] for s in steps]
    assert [r["action"] for r in robot0] == ["E", "E", "Wait", "Wait"]
    assert [r["at_goal"] for r in robot0] == [False, False, True, True]
    assert [r["cell"] for r in robot0][-1] == [0, 2]
    robot1 = [s["robots"][1] for s in steps]
    assert [r["action"] for r in robot1] == ["Wait", "N", "N", "Wait"]
    assert step_line(steps[0]) == "t=0   a0 (0,0) E | a1 (2,0) Wait"


def test_run_writes_trace(tmp_path):
    out = tmp_path / "run"
    metrics = run.main(["--map", MAP_SMALL, "--agents", "4", "--seed", "1", "--planner", "cbs",
                        "--timeout", "5", "--out", str(out), "--substeps", "1"])
    trace = json.loads((out / "trace.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    paths = plan["result"]["paths"]
    assert trace["available_actions"] == ACTIONS
    assert trace["percept"]["starts"] == plan["scenario"]["starts"]
    assert trace["status"] == metrics["status"] == "solved"
    assert len(trace["steps"]) == max(len(p) for p in paths)
    assert len(trace["steps"]) >= metrics["makespan"] + 1
    for step in trace["steps"]:
        for r in step["robots"]:
            assert r["action"] in ACTIONS
            assert r["cell"] == paths[r["agent"]][min(step["t"], len(paths[r["agent"]]) - 1)]
    assert all(r["at_goal"] for r in trace["steps"][-1]["robots"])
    assert "execution trace" in (out / "run.log").read_text()


def test_run_trace_is_empty_on_timeout(tmp_path):
    out = tmp_path / "run"
    metrics = run.main(["--map", MAP_SMALL, "--agents", "10", "--seed", "2", "--planner", "cbs",
                        "--timeout", "0.2", "--out", str(out), "--substeps", "1"])
    assert metrics["status"] == "timeout"
    trace = json.loads((out / "trace.json").read_text())
    assert trace["steps"] == []
    assert trace["planner_stats"]["high_level_expanded"] > 0
    assert "child: agent" in (out / "run.log").read_text()
