from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from PIL import Image

from mapf.grid import Cell, Grid
from mapf.validate import Conflict

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
SHELF = "#3d3d3a"
GRIDLINE = "#e4e3dc"
INK = "#1f1f1e"
INK_MUTED = "#6b6a63"


PLANNER_STYLE = {
    "independent": (PALETTE[0], "o"),
    "prioritized": (PALETTE[1], "s"),
    "cbs": (PALETTE[2], "^"),
}


def plot_summary(summary: Sequence[dict], out: str | Path, timeout: float | None = None,
                 title: str = "") -> None:
    planners = [p for p in PLANNER_STYLE if any(r["planner"] == p for r in summary)]
    planners += sorted({r["planner"] for r in summary} - set(planners))
    agents = sorted({r["n_agents"] for r in summary})
    panels = [
        ("success_rate", "Success rate", lambda v: v * 100, "% of seeds solved and valid"),
        ("mean_sum_of_costs", "Mean sum of costs", lambda v: v, "successful runs only"),
        ("mean_runtime_s", "Mean runtime (s)", lambda v: v, "all runs, log scale"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label, f, note) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, planner in enumerate(planners):
            col, marker = PLANNER_STYLE.get(planner, (color(i + 3), "D"))
            pts = [(r["n_agents"], f(r[key])) for r in summary
                   if r["planner"] == planner and r[key] is not None]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, color=col, linewidth=2, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=planner, zorder=3)
        ax.set_title(label, color=INK, fontsize=11, loc="left", pad=20)
        ax.text(0, 1.025, note, transform=ax.transAxes, color=INK_MUTED, fontsize=8.5,
                va="bottom")
        ax.set_xlabel("agents", color=INK_MUTED, fontsize=9)
        ax.set_xticks(agents)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(GRIDLINE)
    axes[0].set_ylim(-4, 104)
    axes[2].set_yscale("log")
    if timeout:
        axes[2].axhline(timeout, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        axes[2].text(agents[-1], timeout * 0.8, f"timeout {timeout:g}s", color=INK_MUTED,
                     fontsize=8, va="top", ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.005, 0.95))
    if title:
        fig.suptitle(title, x=0.01, y=0.99, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _position(path: Sequence[Cell], t: float) -> tuple[float, float]:
    lo = min(int(np.floor(t)), len(path) - 1)
    hi = min(lo + 1, len(path) - 1)
    a = t - np.floor(t) if lo < len(path) - 1 else 0.0
    r = path[lo][0] + (path[hi][0] - path[lo][0]) * a
    c = path[lo][1] + (path[hi][1] - path[lo][1]) * a
    return c, r


def _figure(grid: Grid, extra_width: float = 0.0):
    scale = 0.42
    fig, ax = plt.subplots(figsize=(grid.width * scale + extra_width, grid.height * scale + 0.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(grid.blocked.astype(int), cmap=ListedColormap([SURFACE, SHELF]), vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which="minor", color=GRIDLINE, linewidth=0.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_color(GRIDLINE)
    return fig, ax


def plot_paths(
    grid: Grid,
    paths: Sequence[Sequence[Cell]] | None,
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    out: str | Path,
    conflicts: Sequence[Conflict] = (),
    title: str = "",
) -> None:
    n = len(starts)
    fig, ax = _figure(grid, extra_width=2.4)
    handles = []
    for i in range(n):
        off = ((i % 5) - 2) * 0.08
        col = color(i)
        if paths is not None:
            p = np.array(paths[i], dtype=float)
            ax.plot(p[:, 1] + off, p[:, 0] + off, color=col, linewidth=2, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3)
        sr, sc = starts[i]
        gr, gc = goals[i]
        ax.scatter([sc], [sr], s=150, color=col, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax.text(sc, sr, str(i), color="white", fontsize=7, fontweight="bold", ha="center",
                va="center", zorder=6)
        ax.scatter([gc], [gr], s=110, marker="s", facecolors=SURFACE, edgecolors=col,
                   linewidths=2, zorder=4)
        ax.text(gc, gr, str(i), color=INK, fontsize=6.5, ha="center", va="center", zorder=6)
        handles.append(Line2D([], [], color=col, linewidth=2, marker="o", markersize=7,
                              label=f"agent {i}"))
    if conflicts:
        xs = [c.cells[0][1] for c in conflicts if c.kind == "vertex"]
        ys = [c.cells[0][0] for c in conflicts if c.kind == "vertex"]
        for c in conflicts:
            if c.kind == "edge":
                xs.append((c.cells[0][1] + c.cells[1][1]) / 2)
                ys.append((c.cells[0][0] + c.cells[1][0]) / 2)
        ax.scatter(xs, ys, marker="X", s=120, color=INK, edgecolors=SURFACE, linewidths=1.5, zorder=7)
        handles.append(Line2D([], [], color=INK, marker="X", linestyle="none", markersize=9,
                              label=f"conflict ({len(conflicts)})"))
    handles += [
        Line2D([], [], color=INK_MUTED, marker="o", linestyle="none", markersize=7, label="start"),
        Line2D([], [], color=INK_MUTED, marker="s", markerfacecolor=SURFACE, linestyle="none",
               markersize=7, label="goal"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
              fontsize=8, labelcolor=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def animate(
    grid: Grid,
    paths: Sequence[Sequence[Cell]] | None,
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    out: str | Path,
    conflicts: Sequence[Conflict] = (),
    title: str = "",
    substeps: int = 4,
    fps: int = 12,
) -> None:
    n = len(starts)
    if paths is None:
        paths = [[tuple(s)] for s in starts]
    horizon = max((len(p) for p in paths), default=1) - 1
    colliding = {(a, c.t) for c in conflicts if c.kind == "vertex" for a in c.agents}
    colliding |= {(a, c.t + 1) for c in conflicts if c.kind == "edge" for a in c.agents}

    fig, ax = _figure(grid)
    cols = [color(i) for i in range(n)]
    for i, (gr, gc) in enumerate(goals):
        ax.scatter([gc], [gr], s=110, marker="s", facecolors=SURFACE, edgecolors=cols[i],
                   linewidths=2, zorder=3)
    robots = ax.scatter([0] * n, [0] * n, s=170, c=cols, edgecolors=SURFACE, linewidths=2, zorder=5)
    labels = [ax.text(0, 0, str(i), color="white", fontsize=7, fontweight="bold", ha="center",
                      va="center", zorder=6) for i in range(n)]
    clock = ax.set_title("", color=INK, fontsize=10, loc="left")
    fig.tight_layout()

    frames, durations = [], []
    frame_ms = int(1000 / fps)
    total = horizon * substeps
    for k in range(total + 1):
        t = k / substeps
        xy = np.array([_position(p, t) for p in paths]) if n else np.zeros((0, 2))
        robots.set_offsets(xy)
        if k % substeps == 0:
            ti = k // substeps
            robots.set_edgecolors([INK if (i, ti) in colliding else SURFACE for i in range(n)])
            robots.set_linewidths([3.5 if (i, ti) in colliding else 2 for i in range(n)])
        for i, lab in enumerate(labels):
            lab.set_position(xy[i])
        clock.set_text(f"{title}   t = {int(t)}" if title else f"t = {int(t)}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
        durations.append(frame_ms)
    durations[0] = 600
    durations[-1] = 1500
    plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
