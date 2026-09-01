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

Sequential on purpose: the machine is memory bound, so two concurrent runs finish later
than two consecutive ones.

Measured cost of self-gravity, 96^3, 100 orbits: 3.9 h with Multigrid against 2.5 h
without, i.e. 2.44e5 zone-cycles/s against 3.84e5. Multigrid is ~36% of the runtime, not
the 90% recorded here earlier — that figure was from a different configuration and was
being used to plan run sets, so an SG on/off pair costs 1.6x a single run, not 1.1x.

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

**Measure a growth-rate window in e-foldings, not in orbits.** An exponential fit given
less than `e^2` in amplitude is fitting noise, and the failure is invisible in the
residual because a short window is also a straight one. Every window in the q-scan spans
under 2 e-foldings (0.96 at q=0.35), so those rates are worth about 10%, not the 0.004
residual of the line. The cause is the seed: 2e-3 against a saturation near 0.3 leaves
only ~5 e-foldings before the transient and the nonlinear turnover are excluded. The fix
is a smaller seed and a longer run - never a wider window, which readmits the nonlinear
data the window exists to exclude. `scan_figure.py` warns below 2.

**Frequencies converge, amplitudes do not.** The constant-`l` run repeated at 128^3
against 96^3 moves the pattern speed by 1.8% and the harmonic ratio `k` by 1.3%, but the
growth rate by 8% and the peak amplitude by 18%. Quote amplitudes to two significant
figures and no further, and run every scan at one resolution throughout so that its points
differ by the parameter rather than by the mesh. The two quantities compared against
published work happen to fall in the converged group; that is luck, not design, and it
needs checking again whenever a new quantity is compared.

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
- **The cubic mesh stamps only multiples of 4.** Measured on an unseeded 3D run:
  `A_4 = 2.3e-4` while `A_1 = 1e-16`, i.e. round-off. Mode coupling is convolution in
  `m`, so +/-4 combinations close on 4Z at every order and odd harmonics are unreachable
  from the imprint. The m=1 seed floor is therefore round-off, not the 2e-5 imprint - the
  seed can drop by orders, buying e-foldings of fit span for run time alone. Holds only
  while the torus axis is a cube axis and the torus is centred; any tilt or offset
  destroys the 4-fold selection. If `A_1`, `A_2`, `A_3` ever sit at the *same* value,
  that is summation order breaking symmetry, not the mesh.
- **An instability amplifies whatever asymmetry is already present**, including the
  grid's own. Seed the low harmonics coherently and check the seed sits above the grid
  imprint, or the measurement is of the mesh. **In 2D cylindrical there is no imprint at
  all**: an axisymmetric start makes every cell in a phi-ring bitwise identical to its
  neighbours, so the scheme preserves axisymmetry exactly and no mode can ever start.
  Measured, an unseeded inviscid run holds `A_1 = 6.5e-14` flat for 23 orbits while the
  instability would have multiplied it by 2e9. A quiet unperturbed run is therefore a
  statement about the initial conditions, never about stability. The seed does not set
  the rate — verified
  in `results/seedtest/`, where a factor of 1100 in initial amplitude and a change of
  seed type move the growth rate by 11% and the onset of the exponential phase by 4.5
  orbits. Fit the rate strictly before the first saturation: a saturated mode oscillates
  back down through the fitting band and flattens the slope.
- **A collapsed timestep freezes model time while cycles keep running.** Watch model
  time, not cycle count, and not the last logged `dt`.

## Measuring an amplitude on a disk that is disappearing

Two mistakes here were each found only after they had produced a plausible number.

**Take the peak while the disk is still there.** A viscously draining disk grows lopsided
as it accretes, so the mode amplitude climbs monotonically to the last snapshot without
ever saturating. A maximum over the whole run then reports the drainage: in the
dissipation scan it inflated one point by a factor of 37 and lifted it off an otherwise
monotonic curve. Measure only while the disk holds most of its mass, and say which
window was used.

**A short window has two causes and they are opposite.** Below a suppression threshold
the window closes because the mode destroyed the disk, and the peak inside it is the real
saturated value. Above it, the window closes because something else drained the disk
first, and the value is only an upper limit. Distinguish them - whether the amplitude
ever rose far above its seed does it - and mark the difference on the figure.

## Diagnostics

Mode coefficients are volume sums written to the history file every cycle, so mode
evolution is sampled hundreds of times per orbit rather than twice. Amplitudes are
normalised by the torus mass, which makes ratios comparable with published work that
counts particles instead of mass.

The floors are a safety net, not a background: in a healthy run the density-floor counter
stays at zero, and any nonzero value needs to be explained before the run is used.
