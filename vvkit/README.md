# vvkit studies of `acc_disk_visc_vv`

Verification of the 2D Papaloizou–Pringle torus against **exact analytic solutions**,
driven by [vvkit](https://github.com/etrevol/vvkit). Complements `scripts/vv/`, which is
built on self-convergence; this measures error against known-exact references with
volume-weighted norms and a cylindrical Jacobian.

Results and conclusions: **[FINDINGS.md](FINDINGS.md)**.
Assessment of the tool itself: **[FEEDBACK.md](FEEDBACK.md)** (Ukrainian).

## Running

vvkit needs Python >= 3.12 and this WSL distro has 3.10, so the CLI runs from the Windows
venv over interop. That is also the mode the Athena++ plugin documents: `use_wsl: true`
makes the adapter shell back into WSL to run the solver. The `vv` wrapper hides this.

```bash
cd m4_stress && ../vv run --config-path vvcase_trphi.yaml --workdir-base workdir
```

Reports land in `<study>/reports/` as HTML, JSON and a PNG. `vv run` exits non-zero when
a study fails, so it can gate CI.

Binaries are built into **this worktree** (`bin/`), never into the main checkout — the
adapter's on-the-fly build mode runs `make clean` and would desynchronise `.build_state`
there:

```bash
./vvkit/tools/build_binaries.sh
```

## Studies

| dir | measures | reference | runtime |
|---|---|---|---|
| `m1_residual/` | `f_sum`, the discrete force residual | exactly 0 | 15 s |
| `m2_equilibrium/` | `rho` (and `press`) after one orbit | analytic torus profile | 10 min to 512² |
| `m3_nu/` | `nu_applied` | the α-law | 12 s |
| `m4_stress/` | `T_rphi` | analytic shear stress | 17 s |
| `a1_ring/` | *(not run)* Lynden-Bell & Pringle ring | closed-form Σ(x, τ) | — |
| `m5_mms/` | *(not run)* 2D MMS on the production operators | manufactured | — |
| `m6_budget/` | *(not run)* mass budget | `mdot_in − mdot_out` | — |

`m1`, `m3` and `m4` run the solver for two cycles only, so they cost almost nothing in CPU
and are limited by `.tab` parsing (59 MB per dump at 512²).

## Notes on the configurations

- **The mesh must be square in index space** for a 2D study. vvkit takes each
  coordinate's cell count from the data now, but the athinput templates use one refinement
  value for both directions, which is also the production configuration.
- **Constants come from `scripts/theory/disk_model.py`**, printed at full precision, not
  transcribed from the input file's comment block.
- `m3` and `m4` share an identical run (α = 0.01, two cycles, `uov` output); they differ
  only in which column they measure.
- The `T_rphi` reference in `m4` is differentiated branch by branch across the torus
  surface. Differentiating `Max(f, 0)` directly yields `Heaviside`, then `DiracDelta`,
  which will not lambdify.

## Layout

```
vv                     CLI wrapper
tools/                 build_binaries.sh
common/                shared athinput fragments
notes/                 raw first-touch observations, kept as evidence
<study>/               vvcase_*.yaml + athinput.*.template + reports/ + workdir/
```

`workdir/` and `reports/` are gitignored.
