# Working conventions

How simulations are set up, run and recorded on this branch. Method only — results live
in each run's own `README.md`, so this document does not go stale.

## Layout

| path | what |
|---|---|
| `src/pgen/sg_torus_m1.cpp` | 3D Cartesian self-gravitating torus |
| `src/pgen/acc_disk_visc.cpp` | 2D cylindrical viscous torus |
| `scripts/torus/theory/` | the model: parameters, derived geometry, pre-run checks |
| `scripts/torus/analysis/` | mode analysis and figures |
| `scripts/torus/run/` | campaign driver and run descriptions |
| `results/<campaign>/<run>/` | one directory per run, never shared between runs |

Each run directory holds its own `athinput`, `run.log`, history, snapshots, `figs/` and
an auto-generated `README.md`. A run is therefore self-contained: nothing about it has to
be remembered elsewhere.

## Running

```bash
python3 scripts/torus/theory/torus_sg_model.py --q_rot 0.3 -o results/mycamp/myrun/athinput
CAMPAIGN=mycamp ./scripts/torus/run/sg_campaign.sh myrun
```

The campaign script runs the queue sequentially and, after each run, performs the
analysis, the figures and the animation, then regenerates `README.md`. Nothing needs to
be invoked by hand afterwards. Post-processing failures are reported but never abort the
queue.

Sequential on purpose: Multigrid dominates the cost and the machine is memory bound, so
two concurrent runs finish later than two consecutive ones.

## Rules that came from being wrong

**Measure before scaling.** Every performance choice here was made from a measurement,
not from expectation. Multigrid was 90% of the runtime, and the fastest-looking solver
setting was also the least accurate — the settings and their measured errors are recorded
in the generator.

**Screening runs are 50 orbits, production runs are 100.** Measured: averaging the late
mode amplitude over orbits 30..T, the ratio between two runs reads 1.18 at T=40, 1.57 at
50, 1.88 at 70 and 2.04 at 100. Fifty orbits buys the sign and direction of an effect at
a third of the cost; a hundred buys the number. Both curves oscillate on a ten-orbit
period, so a single late snapshot says nothing — comparisons must be window averages.

**2D first, where it is honest.** A 2D run has ~50x fewer cells than the 3D equivalent:
minutes against hours. Parameter surveys are done in 2D and only the interesting points
are repeated in 3D. This is only valid once 2D and 3D have been checked against each
other on the same physics — do that check before trusting any 2D survey.

**Every run is visualised when it finishes**, not when someone gets round to it. A figure
made months later competes with the memory of what the run was for.

**The pre-run checks are the specification.** `torus_sg_model.py` refuses or warns on
configurations that have previously produced worthless output. Add a check whenever a run
turns out to have been unusable; do not rely on remembering.

## Traps this project has actually hit

- **Stock `outflow` boundaries inject mass in a gravitating problem.** Ghost zones copy
  the interior state along with its inward velocity and the boundary becomes an infinite
  reservoir. All faces must be diodes.
- **Flooring density without flooring momentum** sets `v = m/rho_floor`. Floor density,
  momentum and energy together, and count the activations in the history file.
- **The equilibrium must use the same softened potential the solver applies**, or the
  torus starts out of balance by the difference.
- **A viscous torus spreads by `sqrt(nu t)`** and will leave a box sized for a static
  one. Check the box against the run length.
- **An instability amplifies whatever asymmetry is already present**, including the
  grid's own. Seed the low harmonics coherently and check the seed sits above the grid
  imprint, or the measurement is of the mesh.
- **A collapsed timestep freezes model time while cycles keep running.** Watch model
  time, not cycle count, and not the last logged `dt`.

## Diagnostics

Mode coefficients are volume sums written to the history file every cycle, so mode
evolution is sampled hundreds of times per orbit rather than twice. Amplitudes are
normalised by the torus mass, which makes ratios comparable with published work that
counts particles instead of mass.

The floors are a safety net, not a background: in a healthy run the density-floor counter
stays at zero, and any nonzero value needs to be explained before the run is used.
