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
