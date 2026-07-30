# sweep.sh — Guide

Automated parameter sweep for Athena++.
Runs a series of simulations with modified input-file parameters (source is never touched), detects hangs, and generates a Markdown report.

---

## Quick start

```bash
# 1. Build the binary (if not already built)
bash build.sh

# 2. Define tests (TESTS array near the top of the script)
#    → edit scripts/practice/run/sweep.sh

# 3. Check without running
bash scripts/practice/run/sweep.sh --dry-run

# 4. Run overnight in the background (results land in results/sweeps/sweep-*)
#    The script already logs its own output to results/sweeps/sweep-*/sweep.log,
#    so stdout here can just be discarded.
nohup bash scripts/practice/run/sweep.sh > /dev/null 2>&1 & echo $!
```

---

## Configuration (edit near the top of the script)

| Variable | Default | Purpose |
|---|---|---|
| `PROBLEM` | `acc_disk_visc` | Problem name (matches the `.cpp` filename) |
| `INPUT_TEMPLATE` | `inputs/hydro/athinput.acc_disk_visc` | Input-file template |
| `HANG_TIMEOUT` | `90` | Seconds without sim_time progress → kill |
| `ETA_POLL_INTERVAL` | `5` | How often to check progress in the log (sec) |
| `USE_MPI` | `0` | `1` — enable MPI (needs a build with `-mpi`) |
| `NUM_MPI_PROCS` | `4` | Number of MPI processes |

### `TESTS` array

Each line is a separate test — a space-separated list of `key=value`:

```bash
TESTS=(
  "nu_iso=0.1   alpha=0.0"
  "nu_iso=0.5   alpha=0.0"
  "dfloor=1.0e-6   pfloor=1.0e-8   cfl_number=0.3"
  "mesh_nx1=64   mesh_nx2=64   block_nx1=32  block_nx2=32"
)
```

**Special key prefixes:**

| Key in `TESTS` | Where it's replaced |
|---|---|
| `mesh_nx1`, `mesh_nx2` | `<mesh>` block |
| `block_nx1`, `block_nx2` | `<meshblock>` block |
| Everything else | First occurrence in any block |

### Abbreviation table (for directory names)

Add new abbreviations to `ABBREV` as needed:

```bash
declare -A ABBREV=(
  [nu_iso]="nu"   [alpha]="a"
  [dfloor]="df"   [pfloor]="pf"
  [mesh_nx1]="nx1" [mesh_nx2]="nx2"
  ...
)
```

---

## Results structure

```
results/sweeps/sweep-YYYYMMDD-HHMMSS/
  sweep.log                  ← full console transcript
  REPORT.md                  ← final Markdown report
  acc_disk_visc.cpp          ← source snapshot at run time
  athinput.acc_disk_visc     ← input-template snapshot
  t01_nu0.0_a0.0/
    athinput.in              ← modified input for this test
    params.txt               ← which parameters were overridden
    bin.txt                  ← which binary version was used
    run.log                  ← Athena++ output (progress, errors)
    status                   ← COMPLETED | KILLED_HANG | FAILED | UNKNOWN
    start_time / end_time    ← Unix timestamps
    vis1d.py / vis2d.py / vishst.py  ← visualization scripts (copied in)
    data/                    ← *.tab, *.hst and other Athena++ output files
    figs_2d/                 ← PNG plots (if vis2d.py ran)
    figs_hst/                ← plots from vishst.py (if it ran)
  t02_nu1.0_a0.001/
    ...
```

---

## Monitoring during a run

```bash
# Live log (shows current progress):
tail -f results/sweeps/sweep-*/sweep.log

# Find the PID if you didn't note it down:
pgrep -af "sweep.sh"

# Check statuses of tests already finished:
grep -h "" results/sweeps/sweep-<timestamp>/t*/status

# Is a test currently running:
ps aux | grep athena
```

---

## Stopping

| Command | Behavior |
|---|---|
| `kill -SIGTERM <PID>` | Graceful stop: waits for the current test to finish, generates the report |
| `kill -SIGINT <PID>` | Immediate stop: kills Athena++, generates a partial report |
| `Ctrl+C` (if not backgrounded) | Same as SIGINT |

> Both paths always generate `REPORT.md` with results for the tests that already ran.

---

## The next morning: reviewing results

```bash
# List all runs:
ls -dt results/sweeps/sweep-*/ | head -5

# Open the report from the most recent run:
cat results/sweeps/sweep-*/REPORT.md | head -100

# Log for a specific test (e.g. t02):
cat results/sweeps/sweep-<timestamp>/t02_*/run.log

# Quick overview of all statuses:
ls -d results/sweeps/sweep-<timestamp>/t*/ | while read d; do echo "$(basename $d): $(cat $d/status)"; done
```

---

## Hang detection

The script monitors simulation progress two ways:

1. **`sim_time` progress (PRIMARY)**: if `sim_time` in the log hasn't changed for `HANG_TIMEOUT` (90s by default) → kill.
   This catches real hangs (timestep collapse, `dt → 1e-88`).

2. **Output check (FALLBACK)**: if `sim_time` hasn't appeared in the log yet (early stage), checks whether new lines are appearing in the log.
   If no new lines for `HANG_TIMEOUT` → kill.

> **Note**: hang detection is based on `sim_time` progress, not ETA.
> This avoids false kills when the ETA stalls but the timestep hasn't actually collapsed.

---

## Statuses in the report

| Status | Meaning |
|---|---|
| `✅ COMPLETED` | Simulation reached `tlim` or `nlim` |
| `⏱️ KILLED (hang)` | `sim_time` didn't progress for `HANG_TIMEOUT` seconds → force-stopped |
| `❌ FAILED` | Athena++ printed `FATAL ERROR`, Segfault, `Aborted`, or `Error:` |
| `❓ UNKNOWN` | Process exited but no success/failure marker was found |
| `🔍 DRY RUN` | Test wasn't actually run (`--dry-run` mode) |

---

## Post-processing (automatic visualization)

At the end of the sweep, the script runs `vis2d.py` and `vishst.py` for each test:

- **`vis2d.py`** reads the `.tab` files and draws contour plots (PNG) → `test_dir/figs_2d/`
- **`vishst.py`** parses the `.hst` (history) file → tables, plots → `test_dir/figs_hst/`

If a simulation has no output data (KILLED_HANG, FAILED), visualization is skipped with a WARN.

> Post-processing **doesn't block** the report: even if the vis scripts fail, `REPORT.md` is still generated.
