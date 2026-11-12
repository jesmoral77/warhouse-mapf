# Warehouse MAPF

Collision-free path planning for N robots in a grid warehouse, comparing three planners:
independent A\* (baseline), prioritized planning, and Conflict-Based Search (CBS).

## Proof of Concept: Run Instructions

Requires Python 3.11+ and Git on Windows PowerShell. All commands run from the repository root.

### 1. Clone and install

```powershell
git clone https://github.com/jesmoral77/warhouse-mapf.git
cd warhouse-mapf
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands call `.\.venv\Scripts\python.exe` directly, so the virtual environment never has to be activated and the PowerShell execution policy does not matter.

### 2. Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### 3. Plan one instance with each planner

```powershell
foreach ($p in 'independent', 'prioritized', 'cbs') {
    .\.venv\Scripts\python.exe -m mapf.run --map maps/warehouse_small.map --agents 6 --seed 1 --planner $p --timeout 30 --out results/poc/ws6_s1_$p
}
```

Each `results/poc/ws6_s1_<planner>/` folder contains:

| File | Contents |
|---|---|
| `command.txt` | the exact command that produced the folder |
| `plan.json` | scenario (starts, goals, seed) and every robot's path |
| `metrics.json` | success, sum of costs, makespan, runtime, nodes expanded, conflicts |
| `run.log` | planner trace (CBS high-level expansions, prioritized plan order) and validation result |
| `paths.png` | paths drawn over the map; conflicts marked with an X |
| `anim.gif` | step-by-step animation; colliding robots are outlined |

### 4. Run the sweep

```powershell
.\.venv\Scripts\python.exe -m mapf.experiment --map maps/warehouse_small.map --agents 2 4 6 8 10 12 --seeds 10 --timeout 10 --out results/poc/sweep.csv
```

This runs every planner on the same 60 scenarios (6 agent counts × seeds 0–9) and writes:

- `results/poc/sweep.csv`: one row per run
- `results/poc/sweep_summary.csv`: per planner and agent count, success rate, mean sum of costs, mean runtime and timeouts
- `results/poc/summary.png`: success rate, mean sum of costs (successful runs only) and mean runtime (log scale) against the number of agents
- `results/poc/sweep_command.txt`: the exact command

A run counts as a success only if the planner returned paths and `mapf.validate` finds no conflicts. Independent A\* therefore fails whenever its paths collide. The mean sum of costs averages only each planner's own successful runs, so at high agent counts the planners are averaged over different subsets of seeds.

### Optional: regenerate the maps

The maps are committed. These commands rebuild them identically:

```powershell
.\.venv\Scripts\python.exe -m mapf.grid --width 20 --height 15 --out maps/warehouse_small.map
.\.venv\Scripts\python.exe -m mapf.grid --width 40 --height 30 --out maps/warehouse_medium.map
```

Everything under `results/` is git-ignored.
