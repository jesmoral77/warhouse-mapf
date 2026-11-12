"""Grid model: MovingAI .map I/O, 4-connected neighbors, warehouse map generator.

Cells are ``(row, col)`` tuples of ints.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

Cell = tuple[int, int]

MOVES: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, 1), (0, -1))

FREE_CHARS = frozenset(".G")


class Grid:
    """A 4-connected occupancy grid. ``blocked[r, c]`` is True for obstacles."""

    def __init__(self, blocked: np.ndarray):
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.ndim != 2:
            raise ValueError("blocked must be a 2-D array")
        self.blocked = blocked
        self.height, self.width = blocked.shape

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.blocked[cell[0], cell[1]]

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Free 4-connected neighbors (no Wait; planners add that themselves)."""
        r, c = cell
        out = []
        for dr, dc in MOVES:
            n = (r + dr, c + dc)
            if self.is_free(n):
                out.append(n)
        return out

    def free_cells(self) -> list[Cell]:
        rs, cs = np.nonzero(~self.blocked)
        return [(int(r), int(c)) for r, c in zip(rs, cs)]

    @property
    def num_free(self) -> int:
        return int((~self.blocked).sum())

    def component(self, start: Cell) -> set[Cell]:
        """All free cells reachable from ``start`` (BFS)."""
        if not self.is_free(start):
            return set()
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self.neighbors(cur):
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        return seen

    def largest_component(self) -> list[Cell]:
        """Free cells of the largest connected component, in row-major order."""
        remaining = set(self.free_cells())
        best: set[Cell] = set()
        while remaining:
            comp = self.component(next(iter(remaining)))
            remaining -= comp
            if len(comp) > len(best):
                best = comp
        return sorted(best)

    def to_lines(self) -> list[str]:
        return ["".join("@" if b else "." for b in row) for row in self.blocked]

    def __str__(self) -> str:
        return "\n".join(self.to_lines())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Grid) and np.array_equal(self.blocked, other.blocked)


# I/O


def load_map(path: str | Path) -> Grid:
    """Load a MovingAI-format .map file. '.' and 'G' are free; all else blocked."""
    lines = Path(path).read_text().splitlines()
    header: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if line == "map":
            break
        if line:
            key, _, value = line.partition(" ")
            header[key.lower()] = value.strip()
    else:
        raise ValueError(f"{path}: missing 'map' line")

    try:
        height = int(header["height"])
        width = int(header["width"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{path}: bad or missing height/width header") from e

    rows = [ln.rstrip("\r\n") for ln in lines[i : i + height]]
    if len(rows) != height or any(len(r) != width for r in rows):
        raise ValueError(f"{path}: map body does not match {height}x{width} header")

    blocked = np.array([[ch not in FREE_CHARS for ch in row] for row in rows], dtype=bool)
    return Grid(blocked)


def save_map(grid: Grid, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(grid.to_lines())
    path.write_text(
        f"type octile\nheight {grid.height}\nwidth {grid.width}\nmap\n{body}\n"
    )


# Warehouse generator


def _fit(extent: int, block: int, gap: int) -> tuple[int, int]:
    """Number of blocks that fit in ``extent`` with ``gap`` between them, and the
    offset (inside ``extent``) at which the centred lattice starts."""
    n = (extent + gap) // (block + gap)
    if n < 1:
        return 0, 0
    used = n * (block + gap) - gap
    return n, (extent - used) // 2


def generate_warehouse(
    width: int,
    height: int,
    shelf_length: int = 5,
    shelf_depth: int = 2,
    aisle: int = 1,
) -> Grid:
    """Warehouse layout: rows of ``shelf_depth``-deep, ``shelf_length``-long shelf
    blocks separated by ``aisle``-wide horizontal aisles and vertical cross-aisles,
    surrounded by a free border corridor (>= 1 cell; leftover slack is split
    between opposite sides so the shelf lattice is centred). Deterministic.
    """
    if min(width, height) < 3 or min(shelf_length, shelf_depth, aisle) < 1:
        raise ValueError("map too small or non-positive shelf/aisle size")

    inner_w, inner_h = width - 2, height - 2
    nx, off_x = _fit(inner_w, shelf_length, aisle)
    ny, off_y = _fit(inner_h, shelf_depth, aisle)

    blocked = np.zeros((height, width), dtype=bool)
    for by in range(ny):
        r0 = 1 + off_y + by * (shelf_depth + aisle)
        for bx in range(nx):
            c0 = 1 + off_x + bx * (shelf_length + aisle)
            blocked[r0 : r0 + shelf_depth, c0 : c0 + shelf_length] = True

    grid = Grid(blocked)
    assert len(grid.largest_component()) == grid.num_free, "warehouse not connected"
    return grid



def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description="Generate a warehouse .map file")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--shelf-length", type=int, default=5)
    p.add_argument("--shelf-depth", type=int, default=2)
    p.add_argument("--aisle", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    grid = generate_warehouse(
        args.width, args.height, args.shelf_length, args.shelf_depth, args.aisle
    )
    save_map(grid, args.out)
    cmd = "python -m mapf.grid " + " ".join(argv)
    args.out.with_suffix(".cmd.txt").write_text(cmd + "\n")
    print(grid)
    print(f"wrote {args.out} ({grid.width}x{grid.height}, {grid.num_free} free cells)")


if __name__ == "__main__":
    main()
