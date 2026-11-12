from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from mapf.grid import load_map
from mapf.run import PLANNERS, run_planner
from mapf.scenario import generate_scenario
from mapf.visualize import plot_summary

FIELDS = [
    "map", "planner", "n_agents", "seed", "status", "success", "valid", "sum_of_costs",
    "makespan", "runtime_s", "nodes_expanded", "n_conflicts", "high_level_expanded",
    "high_level_generated", "message",
]
SUMMARY_FIELDS = [
    "planner", "n_agents", "runs", "successes", "success_rate", "mean_sum_of_costs",
    "mean_makespan", "mean_runtime_s", "timeouts",
]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["planner"], r["n_agents"]), []).append(r)
    out = []
    for (planner, n), rs in groups.items():
        ok = [r for r in rs if r["success"]]
        out.append({
            "planner": planner,
            "n_agents": n,
            "runs": len(rs),
            "successes": len(ok),
            "success_rate": len(ok) / len(rs),
            "mean_sum_of_costs": mean(r["sum_of_costs"] for r in ok) if ok else None,
            "mean_makespan": mean(r["makespan"] for r in ok) if ok else None,
            "mean_runtime_s": mean(r["runtime_s"] for r in rs),
            "timeouts": sum(r["status"] == "timeout" for r in rs),
        })
    order = {p: i for i, p in enumerate(PLANNERS)}
    out.sort(key=lambda s: (order.get(s["planner"], 99), s["n_agents"]))
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    argv = sys.argv[1:] if argv is None else list(argv)
    p = argparse.ArgumentParser(description="Sweep agents x seeds x planners")
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--agents", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, required=True, help="number of seeds, 0..N-1")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--planners", nargs="+", choices=list(PLANNERS), default=list(PLANNERS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plot", type=Path, default=None)
    args = p.parse_args(argv)

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_path = args.plot or out.parent / "summary.png"
    summary_path = out.with_name(out.stem + "_summary.csv")
    command = "python -m mapf.experiment " + subprocess.list2cmdline(argv)
    out.with_name(out.stem + "_command.txt").write_text(command + "\n", encoding="utf-8")

    grid = load_map(args.map)
    rows: list[dict[str, Any]] = []
    total = len(args.agents) * args.seeds * len(args.planners)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for n in args.agents:
            for seed in range(args.seeds):
                scenario = generate_scenario(grid, n, seed)
                for planner in args.planners:
                    result, metrics, _ = run_planner(planner, grid, scenario, args.timeout)
                    row = {"map": str(args.map), "seed": seed, "message": result.message, **metrics}
                    writer.writerow(row)
                    fh.flush()
                    rows.append(row)
                    print(f"[{len(rows)}/{total}] n={n:<3} seed={seed:<3} {planner:<12} "
                          f"{result.status:<8} success={metrics['success']!s:<5} "
                          f"soc={_fmt(metrics['sum_of_costs']):<5} t={metrics['runtime_s']:.3f}s",
                          flush=True)

    summary = summarize(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    plot_summary(summary, plot_path, args.timeout,
                 f"{args.map.stem}: {args.seeds} seeds per point, {args.timeout:g}s timeout")

    print()
    print("  ".join(f"{h:>17}" for h in SUMMARY_FIELDS))
    for s in summary:
        print("  ".join(f"{_fmt(s[h]):>17}" for h in SUMMARY_FIELDS))
    print(f"\nwrote {out}, {summary_path}, {plot_path}")
    return rows


if __name__ == "__main__":
    main()
