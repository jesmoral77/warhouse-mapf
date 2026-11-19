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

Run outputs under `results/` are git-ignored, except the retained evidence in `results/poc/` and `results/vertical_slice/`.

## Vertical Slice

The vertical slice runs one task end to end. The task is either a seeded random scenario or an explicit scenario file. Each planner is a search-based agent that returns a joint plan. The independent validator checks the plan, and the run records what the agent perceived, which action each robot took at every time step, and how it reached the result.

### Give the agent an explicit task

A scenario file lists one start and one goal per robot as `[row, col]` cells:

```json
{"starts": [[4, 1], [4, 5], [1, 6]], "goals": [[4, 5], [4, 1], [7, 6]]}
```

```powershell
.\.venv\Scripts\python.exe -m mapf.run --map maps/warehouse_small.map --scenario scenarios/aisle_swap.json --planner cbs --timeout 10 --out results/vertical_slice/C_aisle_swap_cbs
```

Use either `--scenario` or both `--agents` and `--seed`.

### Terminal statuses

| Status | Meaning |
|---|---|
| `solved` | The planner returned a path for every robot. The run counts as a `success` only if the validator also finds 0 conflicts |
| `timeout` | The wall-clock limit (`--timeout`) expired |
| `failed` | The search proved there is no plan, e.g. prioritized planning finds a later robot boxed in |
| `error` | An unexpected exception inside a planner, recorded instead of raised |
| `invalid_input` | The map or scenario was rejected before planning, e.g. a start on a shelf, a goal off the map, duplicate goals, an unreachable goal, more robots than free cells, or a missing file. No plan is attempted and the process exits with code 2 |

### Extra output per run

Along with the files listed above, each run folder now contains:

| File | Contents |
|---|---|
| `trace.json` | `percept` (map size, starts, goals), `available_actions`, status, planner statistics, and `steps`: for every time step, each robot's cell, the action it takes next (N/S/E/W/Wait) and whether it is at its goal |
| `run.log` | the same per-step trace as readable lines. For CBS, every high-level expansion with the conflict it branched on, plus one line per child naming the constraint added and the child's cost |

### Reproduce the baseline

```powershell
.\scripts\vertical_slice.ps1           # pytest + the retained cases
.\scripts\vertical_slice.ps1 -Sweep    # also the 25-seed sweep (several minutes)
```

| Case | Kind | Command arguments |
|---|---|---|
| `A_cbs_12_s0` | representative, target robot count | `--agents 12 --seed 0 --planner cbs` |
| `B_prio_12_s0` | same instance, prioritized planning | `--agents 12 --seed 0 --planner prioritized` |
| `C_aisle_swap_cbs`, `C_aisle_swap_prio` | explicit task that needs coordination | `--scenario scenarios/aisle_swap.json` |
| `D_cbs_10_s2_timeout` | unsuccessful (CBS timeout) | `--agents 10 --seed 2 --planner cbs` |
| `E_bad_scenario` | unsupported input | `--scenario scenarios/bad_start_on_shelf.json` |
| `F_too_many_agents` | boundary input | `--agents 200 --seed 0` |

All cases use `--map maps/warehouse_small.map --timeout 10`. The sweep summary (`sweep_summary.csv`) adds median, 90th-percentile and maximum runtime per planner and robot count.
