# PHANTOM: what is there, and what it would take

Reconnaissance on the night of 2026-09-02/03. Nothing scientific was run; the point was
to find out whether the tool works and what stands between it and the question.

## Why PHANTOM at all

Two independent reasons, and the second is the stronger one.

Bannikova recommended it. And Lithwick et al. (2025, arXiv:2510.12871) find that a
self-gravitating disc supports a coherent eccentric mode **only** where the surface
density is sharply truncated - where `Sigma` actually reaches zero at the edge. A grid
code cannot provide that: `rho_atm` is a numerical regularisation, not a vacuum, and
this project's own V&V records it as such. In SPH a vacuum is the absence of particles,
so the condition is met by construction rather than approximated. If the pattern speed
is being set by the softened edge that the grid is forced to keep, SPH is where that
would show.

## State

Source at `/home/etrevol/phantom/phantom`. Builds and runs:

```bash
cd /home/etrevol/phantom/phantom
make        SETUP=torus SYSTEM=gfortran GRAVITY=yes    # bin/phantom
make setup  SETUP=torus SYSTEM=gfortran GRAVITY=yes    # bin/phantomsetup
```

`SYSTEM=gfortran` is required - without it the build aborts asking for one. The build is
**not parallel-safe**: `make -j2` fails on a missing `physcon.mod`. Build serially.

Verified end to end on a small heavy torus: `phantomsetup torus` writes `torus.setup`,
is re-run to read it back, writes `torus.in` and the initial dump, and `phantom torus.in`
integrates and writes dumps. 37 s of wall time for `t = 0.1` on one thread.

## What the stock setup is, in this project's variables

`src/setup/setup_torus.f90` builds a Papaloizou-Pringle torus. Its density function is

    rho ~ [ M/R0 ( R0/sqrt(r^2+z^2) - (R0/R)^2/2 - 1/(2 dfac) ) ]^n

which is exactly the shape function of `sg_torus_m1.cpp` at `q = 0` and `eps_soft = 0`,
with

    dfac = 1 / (2 C')      so  C' = 0.18  is  dfac = 2.778

(their default `dfac = 1.01` is `C' = 0.495`, a very thin torus). `gamma = 5/3`,
`Mstar` and `Mtorus` are free, and `GRAVITY=yes` compiles in the tree gravity.

## The two gaps

1. **The rotation law is constant angular momentum only.** `q` does not exist in the
   setup. The near-Keplerian runs that this project's whole argument rests on would need
   `rhofunc` and `setup_velocities_and_Bfields` generalised to `l = l_0 R^q` - the same
   algebra already written in `ShapeFunction()`, so this is a transcription rather than
   a derivation.

2. **The equilibrium omits the gas self-gravity**, exactly the defect just fixed on the
   Athena side. Measured symptom in the trial run: RMS Mach number 2.16 and the shock
   viscosity saturated at `alpha = 1` within the first tenth of a time unit, at
   `Mtorus = 0.3`. In SPH the usual remedy is different from ours - relax with velocity
   damping until the virial settles, then restart - and it is cheaper to write than an
   SCF iteration. It must be done, or the SPH run will measure its own readjustment for
   the same reason the grid runs did.

## What a first real run would cost

The stock setup allocates for 5.2M particles at `nrings = 200, nlayers = 40`. The N-body
comparison uses 128k particles, so the SPH run does not need to be large to be
comparable in Poisson noise - but it does need enough to resolve the vertical structure.
Not estimated here; estimate it before queueing anything.
