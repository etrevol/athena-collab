# PHANTOM: the SPH side of the same question

Worktree `/home/etrevol/phantom-sgtorus`, branch `sg-torus`, off `master` of the fork at
`github.com/etrevol/phantom`. The source proper is `/home/etrevol/phantom/phantom`.

## Why there has to be an SPH side

Two reasons, and the second is the stronger one.

Bannikova recommended it. And Lithwick et al. (2025, arXiv:2510.12871) find that a
self-gravitating disc supports a coherent eccentric mode **only** where the surface
density truncates sharply - where `Sigma` actually reaches zero at the edge. A grid code
cannot provide that: `rho_atm` is a numerical regularisation, not a vacuum, and this
project's own V&V records it as such. In SPH a vacuum is the absence of particles, so the
condition is met by construction. If the pattern speed is being set by the softened edge
the grid is forced to keep, SPH is where that shows.

## Building

```bash
cd /home/etrevol/phantom-sgtorus
make       SETUP=torus SYSTEM=gfortran GRAVITY=yes    # bin/phantom
make setup SETUP=torus SYSTEM=gfortran GRAVITY=yes    # bin/phantomsetup
```

`SYSTEM=gfortran` is required - without it the build aborts asking for one. The build is
**not parallel-safe**: `make -j2` fails on a missing `physcon.mod`. Build serially.

## What the stock setup was, and what was wrong with it

`src/setup/setup_torus.f90` builds a Papaloizou-Pringle torus whose density function is
exactly this project's shape function at `q = 0` and `eps_soft = 0`, with

    dfac = 1 / (2 C')      so  C' = 0.18  is  dfac = 2.7778

(the stock default `dfac = 1.01` is `C' = 0.495`, a very thin torus). The translation is
therefore exact rather than approximate. Four things had to be fixed before it could run
this problem at all; all four are on the branch, each defaulting to the previous
behaviour where that behaviour was defensible.

1. **No rotation-law exponent.** Added as `qrot`, `l = l0 (R/Rtorus)^q` with
   `l0^2 = G Mstar Rtorus`, so the pressure maximum stays at `Rtorus` for every `q`.
   Default 0, which is what the setup always built.

2. **No central mass at all.** `iexternalforce` defaults to 0 and nothing in the setup
   set it, so the torus was built in the potential of `Mstar` and then integrated with
   nothing at the centre. It is now a **sink particle**, not `iext_star`: a sink is free
   to move, and the displacement of the central mass in response to the lopsided torus is
   precisely what the paper measures. A fixed potential cannot show it. The grid code
   reaches the same physics through its `indirect` term.

3. **`Mtorus` was ignored.** It was the third argument of `calculate_polyk_and_rhofac`,
   declared `intent(out)`, so the value in the `.setup` file was overwritten by the mass
   integral and the real control was `densmax`. Since the integrated mass is exactly
   linear in `densmax` - `polyk ~ densmax^-(g-1)` and `rhofac ~ densmax` - one evaluation
   solves for the `densmax` that delivers the requested mass. `Mtorus` is now an input.

4. **The equilibrium omitted the gas self-gravity**, the same defect fixed on the grid
   side. `scf_iter > 0` now iterates it, by the same algorithm: at `t = 0` the torus is
   axisymmetric, the azimuthal integral of the Green's function over a ring is the
   complete elliptic integral `K`, and the 3D Poisson problem collapses to a 2D
   quadrature. Same closure as the grid code - the density maximum pinned at `Rtorus` and
   the half-thickness held fixed, the midplane edges moving and reported. Default 0.

## Verification

**The vanishing-mass limit.** At `Mtorus = 1e-4` the iteration reproduces the analytic
model to 0.3-0.4% in `Psi_max`, both edges and the half-thickness, and the SCF's own
quadrature agrees with PHANTOM's independent `(r,z)` mass integral to 3e-6.

**Against the grid code.** At `Mtorus/Mstar = 0.10`, `q = 0.30`, `C' = 0.18` the two
independently written implementations agree on the physics:

| | PHANTOM (`eps = 0`) | Athena++ (`eps_soft = 0.17`) |
|---|---|---|
| `Phi_gas / Phi_c` at `R_tor` | 7.7% | 7.5% |
| enthalpy rise, analytic -> SCF | +29.8% | +25.8% |
| `r_in`, analytic -> SCF | 0.5636 -> 0.5333 | 0.5723 -> 0.5463 |
| `r_out`, analytic -> SCF | 2.9971 -> 2.8106 | 3.1421 -> 2.9756 |
| half-thickness held to | 0.36% | 0.01% |

The residual difference is the softening, which the grid needs because a cell can sit on
the singularity and SPH does not.

**The equilibrium itself.** Virial residual `V = (2E_kin + 3(g-1)E_therm + E_pot)/|E_pot|`
of the initial condition, measured over a tenth of an orbit:

| | V | `E_therm` swing | energy drift |
|---|---|---|---|
| analytic IC (`scf_iter = 0`) | -0.0448 | +2.7% | +1.1e-5 |
| self-consistent (`scf_iter = 30`) | **+0.0013** | +2.1% | +3e-6 |

A factor of 34, and better than the grid code manages (0.0589 -> 0.0165) because SPH has
neither a discretised potential nor an ambient medium to relax against.

## Things worth knowing before the next run

- **Neither built-in damping suits a torus.** `idamp = 1` relaxes velocities towards zero,
  which would kill the rotation; `idamp = 3` relaxes towards a *Keplerian* profile in
  boundary zones, which is not the rotation law here. This is why the SCF was ported
  rather than the usual SPH remedy of damped relaxation. A custom damper towards
  `l = l0 (R/Rtorus)^q` would still be worth having for the particle distribution, which
  the SCF does not touch.
- **Particle number is capped in the setup.** `maxp = 100000` and `np = 0.5*maxp` are
  hardcoded in `setpart`, giving ~54k particles for these parameters. The N-body study
  uses 128k. Raising it is an edit, not a parameter.
- **The Poisson noise is the seed, and that is the point.** With 54k particles the
  intrinsic `m = 1` noise is `~1/sqrt(2N) = 3e-3`, which is the level the N-body study's
  mode grows from and roughly 100x the coherent seed used in the grid runs. Nothing has to
  be imposed.
- **`.ev` gives the paper's diagnostics directly.** Columns 16-18 are the centre of mass,
  i.e. the barycentre offset `r_tb`; the separate `*Sink*.ev` gives the central mass
  displacement `r_c`. No dump post-processing is needed for the analogue of their Fig. 4.
- **The accretion radius is not softening.** The grid softens the central potential
  (`eps_soft = 0.17`, about 0.3 of the inner edge); the sink instead removes particles
  inside `accrad`. Outside it the potential is the bare point mass. Keep `accrad` well
  inside `r_in` and watch the accreted count.
