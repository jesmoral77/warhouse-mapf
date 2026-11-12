from __future__ import annotations

from typing import Any, Sequence

from mapf.grid import Cell, Grid
from mapf.planners.base import PlanResult
from mapf.validate import ValidationResult, validate


def agent_cost(path: Sequence[Cell], goal: Cell) -> int | None:
    if not path or tuple(path[-1]) != tuple(goal):
        return None
    k = len(path) - 1
    while k > 0 and tuple(path[k - 1]) == tuple(goal):
        k -= 1
    return k


def compute_metrics(
    result: PlanResult,
    grid: Grid,
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    validation: ValidationResult | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "planner": result.planner,
        "status": result.status,
        "n_agents": len(starts),
        "success": False,
        "valid": None,
        "sum_of_costs": None,
        "makespan": None,
        "runtime_s": round(result.runtime_s, 6),
        "nodes_expanded": result.nodes_expanded,
        "n_conflicts": None,
        "n_validation_errors": None,
        "per_agent_costs": None,
    }
    out.update(result.extra)
    if result.paths is None:
        return out

    if validation is None:
        validation = validate(result.paths, grid, starts, goals)
    costs = [agent_cost(p, g) for p, g in zip(result.paths, goals)]
    complete = all(c is not None for c in costs)
    out.update(
        valid=validation.valid,
        success=result.solved and validation.valid,
        n_conflicts=validation.n_conflicts,
        n_validation_errors=len(validation.errors),
        per_agent_costs=costs,
        sum_of_costs=sum(costs) if complete else None,
        makespan=max(costs, default=0) if complete else None,
    )
    return out
