# V&V of `acc_disk_visc_vv` with vvkit — findings

An independent check of the 2D Papaloizou–Pringle torus, measured against **exact
analytic solutions** with volume-weighted norms and a cylindrical Jacobian. This
complements `scripts/vv/`, which is built almost entirely on self-convergence (grid
against grid) and on fitting parameters.

Everything below was produced by `vvkit` on branch `athena-vv-improvements`, driving the
binary built in this worktree. Configurations live next to this file; every run is
reproducible with `./vv run --config-path <study>/vvcase_*.yaml`.

---

## 1. Summary of measured orders

All studies: 2D cylindrical `(r, φ)`, square mesh in index space, `reference:
cell_average`, `quadrature_order: 5`, ladder `nx1 ∈ {64, 128, 256, 512}` unless noted.

| study | quantity | reference | L1 | L2 | L∞ | verdict |
|---|---|---|---|---|---|---|
| **M1** | `f_sum` | exactly 0 | **1.832** (R²=0.982) | 1.549 | 1.148 | fails at L∞ |
| **M2** | `rho` after 1 orbit | analytic torus profile | **1.699** (R²=0.998) | 1.493 | 1.022 | fails at L2, L∞ |
| **M2b** | `rho`, `press` | analytic profiles, one run | 1.620 / **1.740** | 1.297 / 1.608 | — | fails |
| **M3** | `nu_applied` | α-law | **1.812** (R²=0.999) | 1.097 | *does not converge* | fails |
| **M4** | `T_rphi` | analytic shear stress | **2.018** (R²=1.0000) | 2.017 | 1.899 | **PASSES** |

M2b used `nx1 ∈ {64, 128, 256}`.

---

## 2. The central result: the viscous stress is second order

**M4 is the important one.** `T_rphi` is the quantity the whole transport story rests on,
and it is the only study here that exercises the **azimuthal derivative**. Its errors
fall by almost exactly a factor of four per refinement:

```
1.3216e-01  →  3.3490e-02  →  8.1806e-03  →  1.9947e-03
```

p = 2.018 in L1 with R² = 1.0000, and — unlike every other study — **2nd order survives
into L∞** (1.899). The discrete shear stress is second-order accurate everywhere,
including at the worst cell.

Before this study was run, the analytic expression for `T_rphi` was validated directly
against the generator's own output column at 512²: agreement is 1e-5 to 1e-3 relative
across the entire domain. So this is also an independent confirmation that
`UserWorkBeforeOutput`'s stress implementation is correct, not merely self-consistent.

Deriving the reference required differentiating branch by branch across the torus
surface. Differentiating `Max(f, 0)` directly produces `Heaviside` and then `DiracDelta`,
which cannot be evaluated numerically — a trap worth recording for anyone repeating this.

## 3. Everything else loses order at the torus surface, and by how much

M1, M2 and M3 all show the same signature: **L1 > L2 > L∞**, monotonically. That is the
fingerprint of a solution whose worst behaviour is concentrated in a small region — here
the torus surface, where `ρ ~ f³ → 0` and smoothness is limited by the power, not by the
scheme.

This is consistent with, and quantifies, what the formulary already records from
`scripts/vv` (`p = 1.97` in the core, `1.41` at the surface):

- **L1** is volume-weighted and reports the body: 1.70–1.83, close to the core figure.
- **L∞** takes the single worst cell, which sits on the surface: 1.0–1.15.

The new quantity here is the **global, volume-weighted order** — 1.70 for density over a
full orbit — which is what a user of the simulation actually experiences, and which
self-convergence on azimuthally averaged profiles does not measure.

**Pressure converges better than density** (M2b: 1.740 vs 1.620 in L1). This is
physically sensible rather than incidental: `p ~ f⁴` approaches zero at the surface more
gently than `ρ ~ f³`, so the pressure profile is the smoother of the two.

## 4. `nu_applied` does not converge in L∞

M3's L∞ error series does not decrease at all:

```
6.5696e-04  →  1.1530e-03  →  8.4168e-04  →  8.2015e-04
```

It rises from 64 to 128 and then flattens at ~8.2e-4. Against `ν(r_c) = 0.313` that is a
worst-cell relative error of roughly 0.26% that **refinement does not remove**.

L1 is healthy (1.812, R² = 0.999), so the disk body is fine; the non-convergence is
localised. The likely site is the viscosity taper `w = ρ²/(ρ² + ρ_cut²)` combined with
`p/ρ` where `ρ → 0` — that is, the surface again, but here with a second sharp feature
(the taper) laid on top of it.

This is worth taking seriously, because it is exactly where the formulary's known
limitation lives: the α = 1e-2 run whose timestep collapses when the viscous front
reaches the funnel. A stress coefficient with an irreducible worst-cell error at the
surface is a plausible contributor.

### A prediction that did not survive contact

Before running M3 I predicted **order ≈ 1**, on the grounds that `hdif.nu` lags the
output primitives by one VL2 stage, giving an O(dt) ∝ O(h) error. The measurement is
1.812.

The mechanism is not refuted; the test was not capable of seeing it. These runs are two
cycles from a stationary initial condition, so `dν/dt ≈ 0` and the lag has nothing to lag
behind. What was actually measured is the spatial representation error. Testing the lag
claim needs either a run in which ν genuinely evolves, or refinement of `dt` at fixed `h`.

## 5. Cross-check against the existing suite

| quantity | vvkit (this work) | `scripts/vv` | formulary | agree? |
|---|---|---|---|---|
| residual order, body | L1 = 1.832 | band `[1.5, 2.5]` | `p = 1.97` core | yes |
| residual order, worst cell | L∞ = 1.148 | — | `p = 1.41` surface | yes, same direction |
| density order | L1 = 1.699 | self-conv. band `[1.0, 2.5]` | `p = 2.21` at N=512 | consistent; vvkit measures a different, stricter thing |
| `nu_applied` vs α-law | L1 = 1.812, L∞ flat | `< 1e-4` in the smooth core | Conclusion 10 | agrees in the core; **L∞ behaviour is new** |
| `T_rphi` | p = 2.018, PASSES | recovered α in `[0.9, 1.1]·α` | — | **new** |

The one apparent disagreement — 1.699 here versus 2.21 in the formulary — is not a
contradiction. The formulary's figure is self-convergence of the azimuthally averaged
`⟨ρ⟩_φ`; averaging over φ removes variance and the comparison is grid-to-grid. This work
compares the full 2D field, cell by cell, against the exact profile. The stricter
measurement gives the lower number, which is the expected ordering.

## 6. GCI and the asymptotic range

With the GCI computed from a proper integral functional (see `FEEDBACK.md` — it was
previously sampled from a single array cell and was meaningless), the studies report:

| study | GCI on 512² | asymptotic ratio R | in asymptotic range? |
|---|---|---|---|
| M1 | 4.17e-4 | 2.07 → 1.32 | not yet |
| M4 | 3.77e-4 | 0.551 → 0.533 | no |

M1's R moves towards 1 as the grids refine, which is the signature of a study
*approaching* the asymptotic range without having reached it at 512². The honest reading
is that grid-convergence error bands at this resolution are indicative, not certified.

M4 is the interesting case: the norms are textbook second order, yet R is stable at
~0.53 rather than 1. R is built from the **integral functional**, not from the norms, so
the two can converge at different rates — a volume-weighted mean of a signed quantity
admits cancellation that the norms do not. Worth following up; it does not undermine the
order measurement.

## 7. What was not done

Stated plainly rather than left to inference:

- **M5 (2D MMS on the production operators)** — not run. The vvkit defect that blocked it
  is fixed (`vv mms` now emits compiling multi-variable code, verified on three cases),
  and per-field norms are in place, but the new `src/pgen/acc_disk_mms.cpp` with exact
  boundary conditions from `u_m(x, y, t)` was not written.
- **M6 (mass budget from `.hst`)** — not run. Requires a `.hst` reader in the adapter and
  the conservation path reconnected to the report; neither was done.
- **A1 (Lynden-Bell & Pringle ring)** — not run. The binary is built
  (`bin/athena_visc_ring`) and the anisotropic-mesh limitation that would have forced a
  1D workaround is now fixed, so this is ready to run as-is.
- **Temporal convergence** — not attempted.
