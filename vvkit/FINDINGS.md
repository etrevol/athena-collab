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
| **A1** | ring `rho` at τ=0.1 | Lynden-Bell & Pringle Σ(x,τ) | 0.490 | 0.445 | 0.501 | fails — see §5 |

M2b and A1 used `nx1 ∈ {64, 128, 256}`.

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

## 5. The spreading ring has an error floor that refinement does not remove

A1 compares the density profile against the Lynden-Bell & Pringle solution at τ = 0.1,
i.e. 10.9 orbits of viscous spreading. It runs — this needed the special-function fix,
since a numpy-only `lambdify` cannot evaluate `besseli` — and the normalisation is
right: the analytic peak is 0.4317 against 0.4325 measured at 64², reproduced to 0.2%.

But the L1 error stalls:

```
4.530e-04  →  2.673e-04  →  2.297e-04        (64 → 128 → 256)
```

An apparent order of 0.49 is not a scheme property; it is a floor at ~2.3e-4 that the
grid cannot reach past.

**A hypothesis that did not survive testing.** The obvious suspect was disc thickness:
LBP solves the razor-thin diffusion equation with no pressure, while the code solves the
full Euler system with `c_s = aspect · v_K`. With `aspect = 0.02`, `aspect² = 4e-4` sits
suspiciously close to the observed floor. So the study was repeated with `aspect = 0.01`,
where the floor should have dropped fourfold. It did not:

| | 64 | 128 | 256 |
|---|---|---|---|
| `aspect = 0.02` | 4.53e-4 | 2.67e-4 | 2.30e-4 |
| `aspect = 0.01` | 2.11e-4 | 2.28e-4 | 2.49e-4 |

The floor is unchanged, and in the thin case the error *grows* with refinement. Finite
disc thickness is not the explanation.

The most likely remaining cause is **domain truncation**. LBP assumes an infinite disc;
this run has diode boundaries at `r ∈ [0.2, 2.0]`, and by τ = 0.1 the ring's inner tail
has reached the inner boundary, where the analytic Σ is of order 3e-5 — the same size as
the discrepancy. That is a property of the setup, not of the grid, so refinement cannot
remove it. Confirming this would need a wider domain or a shorter run; it was not done.

This also explains a design decision in the existing suite. `scripts/vv/vvlib/ring.py`
does not compare profiles: it **fits τ** and infers `ν_eff` from it. A fit is insensitive
to a boundary-induced offset, which a pointwise comparison is not. Their reported
`ν_eff` accurate to −0.4% and this study's error floor are consistent — they measure
different things, and for this configuration the fit is the sounder instrument.

The general lesson, which applies to vvkit as much as to this model: **the exact solution
of a simplified model is not the exact solution of what the code solves.** Comparing
against it measures the sum of the discretisation error and the modelling difference, and
only the first of those converges.

## 6. Cross-check against the existing suite

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

## 7. GCI and the asymptotic range

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

## 8. Regression: the existing suite is unaffected

`python3 scripts/vv/run_vv.py` was run in this worktree after all commits: **36 passed,
2 failed, 0 warnings** in 15 minutes. The two failures are the two already documented in
the formulary as known limitations -- the transient pressure-floor ring on the coarsest
grid, and `visc_transport` reaching 42% of its tlim as the viscous front enters the
funnel. No new failure was introduced.

The PP84 growth rates reproduce the recorded values exactly: m = 1 at 1.0648 per orbit
against 1.0725 from linear theory (0.7% apart), m = 2 at 0.9830 against 0.9710, m = 3 at
0.7466 against 0.7656. Splitting the instrumented generator out as `acc_disk_visc_vv.cpp`
changed nothing physical.

## 9. Manufactured solution: the operators are second order

M5 is the first manufactured solution in this project. The solution is chosen to be
awkward -- varying in both r and phi, every conserved variable non-trivial -- and the
source term derived so that it is exact anyway. Because it is steady, `u_m` is an exact
steady state of the modified equations, so whatever separates the run from it is
discretisation error and nothing else.

| variable | L1 | L2 | R2 |
|---|---|---|---|
| `rho` | 2.466 | 2.409 | 0.9999 |
| `press` | 2.215 | 2.266 | 1.0000 |
| `vel1` | 2.197 | 2.211 | 0.9995 |
| `vel2` | 2.373 | 2.339 | 0.9999 |

Four variables at second order at once, on a 32-256 ladder. The orders sit above 2 rather
than at it, which is the pre-asymptotic slope of a second-order scheme at these
resolutions.

**Why this is evidence and not just a number.** If the operators declared in the study
differed from what Athena++ advances, `u_m` would stop being a solution of the modified
system, the run would converge to something else, and the order would collapse toward
zero. Getting 2.2-2.5 in four variables simultaneously means the whole chain lines up:
the declared operators, the generated source, the geometric source Athena++ applies
internally, the exact boundary values, and the reconstruction. This verifies the
production hydro path in cylindrical geometry, which nothing here did before.

An earlier note in this file argued that a wrong operator would fail *silently*. That was
wrong, and the correction matters because it changes whether MMS is worth attempting when
you are unsure of the operator: it is, because a mismatch is loud.

### The viscous operator too

M5b repeats the study with the full Newtonian stress, `nu_iso = 0.05`. The strain terms
were transcribed from what Athena++ computes rather than from memory --
`FaceXdx = 2 dv_r/dr`, `FaceXdy = r d(v_phi/r)/dr + (1/r) dv_r/dphi`, and the flux is
MINUS the stress -- and the geometric term in the r-momentum equation carries `-tau_phiphi/r`
to match.

| variable | L1 | L2 | R2 |
|---|---|---|---|
| `rho` | 2.510 | 2.434 | 0.9999 |
| `press` | 2.170 | 2.234 | 0.9996 |
| `vel1` | 2.223 | 2.245 | 0.9991 |
| `vel2` | 2.466 | 2.444 | 0.9997 |

**PASSED.** So the viscous operator this whole project rests on is second order, and the
Newtonian stress written above is the one Athena++ actually applies -- had it not been,
`u_m` would have stopped being a solution of the modified system and the order would have
collapsed toward zero.

Cost note: 1658 s for the 256 case against 111 s at 128, because the viscous timestep
falls as `dx^2/nu` while the hyperbolic one falls as `dx`. A viscous MMS ladder is
roughly fifteen times the cost of the inviscid one per level.

## 10. The mass budget closes, and the diagnostic nearly lied about it

M6 checks `d(disk_mass)/dt` against `mdot_in - mdot_out` -- the check a snapshot cannot
do, since falling disk mass alone cannot be told from accretion, an outer-boundary leak,
or the floors.

| grid | imbalance, relative to the initial mass |
|---|---|
| 64 | +3.84e-5 |
| 128 | +3.56e-5 |
| 256 | **+7.32e-6** |

It closes everywhere and improves with refinement.

Getting there produced a finding about the *diagnostic*. At the ten history writes per
orbit the input file used, the budget FAILED at 256 with an imbalance of -1.85e-4 while
passing at 64 and 128 -- worse on the finer grid, which is not physical. Sampling a
hundred times per orbit closes it everywhere. The failure was aliasing: the boundary flux
varies faster as the grid refines while the history cadence stayed fixed.

**The history sampling rate has to be refined along with the mesh**, or the budget
measures the output cadence rather than the solver. vvkit now says so when halving the
sampling rate moves the flux integral by more than the tolerance.

A second measurement artefact, worth recording because it invalidated numbers before it
was found: Athena++ opens its history file in append mode, so re-running a study into an
existing directory produced a `.hst` whose time column restarted partway down. The same
case reported -1.9e-5 on its first run and -7.5e-5 on a rerun of exactly the same
configuration.

## 11. What was not done

Stated plainly rather than left to inference:

- **Temporal convergence** — not attempted.

A1 *was* run (see §5) but does not yield an order for this configuration; a wider domain
or a shorter integration would be needed to separate the scheme's error from the
boundary's.
