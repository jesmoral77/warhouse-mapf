param(
    [switch]$Sweep
)

$py = ".\.venv\Scripts\python.exe"
$map = "maps/warehouse_small.map"
$out = "results/vertical_slice"

& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

$cases = @(
    @{ name = "A_cbs_12_s0";         args = @("--agents", "12", "--seed", "0", "--planner", "cbs", "--timeout", "10") },
    @{ name = "B_prio_12_s0";        args = @("--agents", "12", "--seed", "0", "--planner", "prioritized", "--timeout", "10") },
    @{ name = "C_aisle_swap_cbs";    args = @("--scenario", "scenarios/aisle_swap.json", "--planner", "cbs", "--timeout", "10") },
    @{ name = "C_aisle_swap_prio";   args = @("--scenario", "scenarios/aisle_swap.json", "--planner", "prioritized", "--timeout", "10") },
    @{ name = "D_cbs_10_s2_timeout"; args = @("--agents", "10", "--seed", "2", "--planner", "cbs", "--timeout", "10") },
    @{ name = "E_bad_scenario";      args = @("--scenario", "scenarios/bad_start_on_shelf.json", "--planner", "cbs", "--timeout", "10") },
    @{ name = "F_too_many_agents";   args = @("--agents", "200", "--seed", "0", "--planner", "cbs", "--timeout", "10") }
)

foreach ($case in $cases) {
    Write-Host "=== $($case.name) ==="
    & $py -m mapf.run --map $map @($case.args) --out "$out/$($case.name)" | Out-Null
    Write-Host "exit code $LASTEXITCODE"
    Get-Content "$out/$($case.name)/metrics.json" | Select-String '"(status|success|sum_of_costs|makespan|runtime_s|high_level_expanded|message)"'
}

if ($Sweep) {
    & $py -m mapf.experiment --map $map --agents 2 4 6 8 10 12 14 16 --seeds 25 --timeout 10 --out "$out/sweep.csv"
}
