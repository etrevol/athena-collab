# `acc_disk_visc` — thermally scaled Papaloizou–Pringle disk with alpha viscosity

2D (r, φ) hydrodynamic model of an accretion torus around a point mass, in cylindrical
coordinates. Files:

| | |
|---|---|
| problem generator | `src/pgen/acc_disk_visc.cpp` |
| input | `inputs/hydro/athinput.acc_disk_visc` |
| build + run | `./build.sh` |

Configure with `--prob=acc_disk_visc --coord=cylindrical`.

## Model

Equilibrium torus with constant specific angular momentum `l = sqrt(beta*r_center)`:

```
f(r)   = r_c/r - 0.5*(r_c/r)^2 - C'          f_c = 0.5 - C'      n = 1/(gamma-1)
rho_d  = (f/f_c)^n
p_d    = rho_d * beta*f/(r_c*(n+1))
```

embedded in a static, centrifugally balanced ambient medium (`rho_atm`, `p_atm`,
`v_phi = sqrt(beta/r)`). The two are superposed and the rotation profile comes from one
analytic radial force balance,

```
v_phi^2(r) = beta/r + (r/rho) * dp/dr ,
```

which reproduces the `l = const` torus in the interior and Keplerian rotation in the
ambient, with no discontinuity at the torus surface.

The gravity source term is discretised with the same geometric factor Athena++ uses for
the cylindrical centrifugal term (`coord_src1_i_ = 2/(r_m + r_p)`), so a centrifugally
supported ambient medium is an exact discrete equilibrium.

## Scaling

Everything dimensionless is derived from five physical inputs (`M_bh`, `rho_0`, `T_0`,
`mu`, `chi`); the run itself is scale free in density. `rho_0` enters only the
`mass_scale` / `mdot_scale` conversions used for history output, so changing it rescales
`disk_mass` and `mdot_in` proportionally and leaves the dynamics untouched (there is no
self-gravity).

Accretion time in orbits depends only on `alpha`, `gamma` and `C'`:

```
N_orbits = 1 / (2*pi * alpha * (gamma-1) * (0.5 - C'))
```

## Two things that will bite you

**The radial domain must strictly enclose the torus** (`x1min < r_inner`,
`x1max > r_outer`). The torus surface is a free surface where `rho`, `p` and `c_s` all go
to zero; a zero-gradient outflow condition there is ill posed, and the profile is
unresolvable (`rho` and `p` change by one to two orders of magnitude per cell). Putting
the boundary on the surface drains the disk at ~1% per orbit even with `alpha = 0`. The
problem generator prints a warning if the domain does not enclose the torus.

**Alpha viscosity needs `nu_iso > 0` as well.** Athena++ gates `ViscousFluxIso` on
`<problem>/nu_iso`, so `alpha > 0` with `nu_iso = 0` silently runs as an inviscid disk.
The generator now raises a fatal error instead. `nu_iso` is only an enable switch — its
value is overwritten by `DiskViscosity`.

## Floors

`dfloor` / `pfloor` are a hard numerical safety net only. The physical background is the
ambient medium, which sits far above them; in a healthy run **no cell ever reaches a
floor**. That is the single most useful health check:

```
python3 -c "
import numpy as np,glob
for f in sorted(glob.glob('<rundir>/*.out1.*.tab')):
    d=np.loadtxt(f,comments='#'); print(f, (d[:,4]<=1.0000001e-12).sum(), d[:,4].min())"
```

A floor violation is repaired conservatively: a pressure violation corrects the **energy
only**, leaving density and momentum untouched (as `EquationOfState::ConservedToPrimitive`
does). Repair lives inside the source term so that it runs before `SEND_HYD` on both VL2
stages, and therefore no neighbouring MeshBlock can ever receive an unphysical ghost
state.

## Status

Validated, `alpha = 0`, 100 orbits, 176 x 128: `dm/m0 = -1.4e-6`, `dE/E0 = 1.4e-5`, no
cell at a floor, and a 16-MeshBlock decomposition is bit-identical to a single block.

**Known open issue — viscous runs.** With viscosity on, a single radial ring at the torus
surface (r ~ 0.54) can run away (`p ~ 1e10`, `v_r ~ 2e5`) and the timestep collapses to
~1e-88, which looks like a hang. The trigger is that the torus surface is a *sub-grid*
density discontinuity: its width is set by `rho_atm` (because `rho ~ (r - r_inner)^n`),
not by `dr`, so refining the radial grid does **not** resolve it. Measured behaviour at
`alpha = 0.01`:

| `rho_atm` | collapses at |
|---|---|
| 1e-6 | t = 0.004 (0.4 orbits) |
| 1e-5 | t = 0.013 (1.3 orbits) |
| 1e-4 | no collapse observed through t = 0.11 |

`visc_rho_cut`, `cfl_number` and radial resolution were each tested and make no
difference. `xorder = 1` also avoids it. Use `rho_atm = 1e-4` for viscous runs until this
is properly fixed.
