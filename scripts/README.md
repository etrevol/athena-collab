# `materials/scripts/`

Split by two axes: **practical** (drives or post-processes actual Athena++ runs) vs
**theoretical** (derives the dimensionless parameters before any run), and **current**
vs **legacy**. Legacy is kept for provenance only — nothing there is maintained.

```
practice/          run the code and post-process its output
├── run/           launch and housekeeping
├── vis/           plotting
└── analysis/      quantitative diagnostics
practice-legacy/   superseded versions of the above
theory/            parameter derivation notebooks
theory-legacy/     superseded notebooks and reports
```

## practice/run

| | |
|---|---|
| `sweep.sh` | parameter sweep driver: runs a series of simulations with overridden input parameters, detects hangs, writes `REPORT.md`. Output goes to `results/sweeps/sweep-YYYYMMDD-HHMMSS/`. Finds the repository root by walking up to `configure.py`, so it can be moved. |
| `sweep.md` | usage notes for `sweep.sh` |
| `cleanup_data_folders.py` | removes the bulky `data/` folders, **preserving the `.hst` history files**. Takes a path, is recursive, supports `--dry-run` and `--exclude`. |

```bash
./sweep.sh                                              # full sweep
python3 cleanup_data_folders.py results -n              # what would be freed
python3 cleanup_data_folders.py results -x sweep-20260728-103026 -y
```

## practice/vis

| | |
|---|---|
| `vis1d.py` | radial profiles from `.tab` |
| `vis2d.py` | 2D maps |
| `vishst.py` | history time series (`disk_mass`, `mdot_in`, …) |
| `visforces.py` | radial force balance from the `uov` output (`f_grav`, `f_centr`, `f_press`, `f_sum`) |

Note on `visforces.py`: `f_grav` is the *continuum* `-beta/r^2`, while the source term uses
the discretely consistent form, so `f_sum` is not exactly zero even in perfect discrete
equilibrium — the residual is ~7e-5 relative.

## practice/analysis

| | |
|---|---|
| `analyze.py` | mass / energy / angular-momentum conservation across runs |
| `profile.py` | azimuthally averaged radial profiles and a mass budget by zone |
| `scan_hst.py` | scans every `.hst` in a tree for `dm/m0` and `m_max/m0` — this is what found the runs that blew up to 1e102 |
| `floor_analysis.py` | density/pressure floor activity |
| `analyze_disk_disappearance.py` | earlier mass-loss investigation |

The health checks that matter, in order of reliability, are described in
`materials/reports/ZVIT.md` §13.

## theory

Notebooks that turn the five physical inputs (`M_bh`, `rho_0`, `T_0`, `mu`, `chi`) into the
dimensionless parameters used by the input file. `parameters_summary.csv` and
`disk_profiles.png` are their outputs.

## theory-legacy

`REPORT.md` here was generated for a **different** model — `M_BH = 1.7e7`, `chi = 1000`,
`C' = 0.3`, `gamma = 1.667`, inviscid — whereas the current setup is `4.5e7 / 500 / 0.2 /
1.3`. Do not read numbers off it.
