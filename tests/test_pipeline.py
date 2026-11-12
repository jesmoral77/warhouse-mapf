import csv
import json
from pathlib import Path

from mapf import experiment, run
from mapf.metrics import agent_cost

MAP_SMALL = str(Path(__file__).resolve().parent.parent / "maps" / "warehouse_small.map")


def test_agent_cost_ignores_trailing_waits_and_counts_final_arrival():
    assert agent_cost([(0, 0), (0, 1), (0, 1), (0, 1)], (0, 1)) == 1
    assert agent_cost([(0, 1), (0, 0), (0, 1)], (0, 1)) == 2
    assert agent_cost([(0, 1)], (0, 1)) == 0
    assert agent_cost([(0, 0)], (0, 1)) is None


def test_run_writes_all_artifacts(tmp_path):
    out = tmp_path / "run"
    metrics = run.main(["--map", MAP_SMALL, "--agents", "3", "--seed", "0", "--planner", "cbs",
                        "--timeout", "5", "--out", str(out), "--substeps", "1"])
    for name in ["command.txt", "plan.json", "metrics.json", "run.log", "paths.png", "anim.gif"]:
        assert (out / name).stat().st_size > 0, name
    assert (out / "command.txt").read_text().startswith("python -m mapf.run --map")
    assert json.loads((out / "metrics.json").read_text()) == metrics
    assert metrics["success"] and "expand #1" in (out / "run.log").read_text()


def test_experiment_writes_csv_and_plot(tmp_path):
    out = tmp_path / "sweep.csv"
    experiment.main(["--map", MAP_SMALL, "--agents", "2", "4", "--seeds", "2", "--timeout", "5",
                     "--out", str(out)])
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2 * 2 * 3
    assert {r["planner"] for r in rows} == {"independent", "prioritized", "cbs"}
    assert (tmp_path / "summary.png").stat().st_size > 0
    assert (tmp_path / "sweep_summary.csv").exists()
    assert (tmp_path / "sweep_command.txt").exists()
