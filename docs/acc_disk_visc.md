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

**Viscous runs need `rho_atm = 1e-4`.** With a lighter ambient the run appears to hang:
the timestep collapses to ~1e-88, so time stops advancing while cycles keep running and the
ETA grows without bound.

The mechanism is in the funnel between the inner boundary and the torus, not at the
boundary itself. Viscosity makes the inner edge of the torus spread inwards, and the
near-empty cell just ahead of the advancing front is accelerated without bound, because the
viscous flux across it is built from the face-averaged density (dominated by the dense side)
but deposited into a cell some 1e4 times lighter. Measured at `alpha = 0.01`, in the cell at
r = 0.5126:

| cycle | rho(0.4831) | rho(0.5126) | rho(0.5422) | v_r(0.5126) |
|---|---|---|---|---|
| 650 | 4.70e-7 | 4.30e-7 | 3.08e-3 | 74.6 |
| 675 | 4.72e-7 | 3.94e-7 | 3.87e-3 | 513.8 |
| 700 | 6.66e-7 | 8.46e-8 | 4.14e-3 | 22790.5 |

`rho_atm = 1e-4` gives that cell enough inertia and costs almost nothing:

| `rho_atm` | `alpha = 0.01` | `alpha = 0`, 100 orbits |
|---|---|---|
| 1e-6 | collapses at t = 0.004 | `dm/m0 = -1.4e-6` |
| 1e-5 | collapses at t = 0.013 | — |
| **1e-4** | **stable past t = 0.14 (34x further)** | **`dm/m0 = -2.6e-5`** |

Tried and found *not* to help — each only postpones the collapse, so do not spend time on
them again: `cfl_number` (0.4 / 0.2 / 0.1), radial resolution (`nx1` x 2, the surface width
is set by `rho_atm` not by `dr`), `visc_rho_cut` (1e-5 / 1e-3 / 1e-2), a cap on the
viscosity scale height or sound speed, a radial switch-off of the viscosity near the inner
edge, a radially declining ambient, moving `x1min` closer to the torus, and `xorder = 1`
(delays it 19x).

**A trap that will cost you a day if you hit it.** `nu_iso > 0` with `alpha = 0` is *not* an
inviscid run: `DiskViscosity` is only enrolled when `alpha > 0`, so Athena++ quietly applies
a *constant* `nu = nu_iso` everywhere instead. An `alpha = 0` control run needs `nu_iso = 0`
as well. The problem generator now warns about this. (An earlier revision of this document
claimed `rho_atm = 1e-4` destroyed mass conservation; that measurement was made with
`nu_iso = 1.0` still set and was simply physical viscous accretion.)
