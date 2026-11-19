from __future__ import annotations

from typing import Any, Sequence

from mapf.grid import Cell
from mapf.validate import position

ACTIONS = ["N", "S", "E", "W", "Wait"]

MOVE_NAMES = {(-1, 0): "N", (1, 0): "S", (0, 1): "E", (0, -1): "W", (0, 0): "Wait"}


def action_name(u: Cell, v: Cell) -> str:
    step = (v[0] - u[0], v[1] - u[1])
    if step not in MOVE_NAMES:
        raise ValueError(f"{u} -> {v} is not a single N/S/E/W/Wait action")
    return MOVE_NAMES[step]


def build_trace(paths: Sequence[Sequence[Cell]], goals: Sequence[Cell]) -> list[dict[str, Any]]:
    horizon = max((len(p) for p in paths), default=0)
    steps = []
    for t in range(horizon):
        robots = []
        for i, path in enumerate(paths):
            here = position(path, t)
            after = position(path, t + 1)
            robots.append({
                "agent": i,
                "cell": list(here),
                "action": action_name(here, after),
                "at_goal": here == tuple(goals[i]),
            })
        steps.append({"t": t, "robots": robots})
    return steps


def step_line(step: dict[str, Any]) -> str:
    parts = []
    for r in step["robots"]:
        cell = f"({r['cell'][0]},{r['cell'][1]})"
        tag = " goal" if r["at_goal"] else ""
        parts.append(f"a{r['agent']} {cell} {r['action']}{tag}")
    return f"t={step['t']:<3} " + " | ".join(parts)
