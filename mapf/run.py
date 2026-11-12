from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from mapf.grid import Grid, load_map
from mapf.metrics import compute_metrics
from mapf.planners import cbs, independent, prioritized
from mapf.planners.base import PlanResult
from mapf.scenario import Scenario, generate_scenario
from mapf.validate import validate
from mapf.visualize import animate, plot_paths

PLANNERS: dict[str, Callable[..., PlanResult]] = {
    "independent": independent.plan,
    "prioritized": prioritized.plan,
    "cbs": cbs.plan,
}

log = logging.getLogger("mapf.run")


def run_planner(
    name: str, grid: Grid, scenario: Scenario, timeout: float | None
) -> tuple[PlanResult, dict[str, Any], Any]:
    result = PLANNERS[name](grid, scenario.starts, scenario.goals, timeout=timeout)
    validation = None
    if result.paths is not None:
        validation = validate(result.paths, grid, scenario.starts, scenario.goals)
    metrics = compute_metrics(result, grid, scenario.starts, scenario.goals, validation)
    return result, metrics, validation


def _attach_log(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)-16s %(message)s"))
    root = logging.getLogger("mapf")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return handler


def main(argv: list[str] | None = None) -> dict[str, Any]:
    argv = sys.argv[1:] if argv is None else list(argv)
    p = argparse.ArgumentParser(description="Plan one MAPF instance and write all artifacts")
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--agents", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--planner", choices=sorted(PLANNERS), required=True)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--substeps", type=int, default=4)
    args = p.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    command = "python -m mapf.run " + subprocess.list2cmdline(argv)
    (out / "command.txt").write_text(command + "\n", encoding="utf-8")

    handler = _attach_log(out / "run.log")
    try:
        log.info("command: %s", command)
        grid = load_map(args.map)
        scenario = generate_scenario(grid, args.agents, args.seed)
        log.info("map %s (%dx%d, %d free), %d agents, seed %d, planner %s, timeout %.1fs",
                 args.map, grid.width, grid.height, grid.num_free, args.agents, args.seed,
                 args.planner, args.timeout)
        for i, (s, g) in enumerate(zip(scenario.starts, scenario.goals)):
            log.info("agent %d: %s -> %s", i, s, g)

        result, metrics, validation = run_planner(args.planner, grid, scenario, args.timeout)

        log.info("status %s %s", result.status, result.message)
        conflicts = validation.conflicts if validation else []
        if validation is not None:
            log.info("validation: valid=%s, %d conflicts, %d errors", validation.valid,
                     validation.n_conflicts, len(validation.errors))
            for c in conflicts:
                log.info("  conflict %s agents %s t=%d cells %s", c.kind, c.agents, c.t, c.cells)
            for e in validation.errors:
                log.info("  error %s", e)
        log.info("metrics: %s", json.dumps(metrics))

        plan = {
            "command": command,
            "map": str(args.map),
            "planner": args.planner,
            "n_agents": args.agents,
            "seed": args.seed,
            "timeout": args.timeout,
            "scenario": scenario.to_dict(),
            "result": result.to_dict(),
        }
        (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        soc = metrics["sum_of_costs"]
        title = (f"{args.planner} | {args.agents} agents, seed {args.seed} | {result.status}"
                 + (f" | SoC {soc}, makespan {metrics['makespan']}" if soc is not None else ""))
        plot_paths(grid, result.paths, scenario.starts, scenario.goals, out / "paths.png",
                   conflicts, title)
        animate(grid, result.paths, scenario.starts, scenario.goals, out / "anim.gif", conflicts,
                args.planner, args.substeps, args.fps)
        log.info("wrote artifacts to %s", out)
    except Exception:
        log.exception("run failed")
        raise
    finally:
        logging.getLogger("mapf").removeHandler(handler)
        handler.close()

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
