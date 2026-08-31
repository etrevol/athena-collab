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
cd m4_stress && ../vv run vvcase_trphi.yaml --jobs 4
```

Reports land in `<study>/reports/` as HTML, JSON and a PNG. `vv run` exits non-zero when
a study fails, so it can gate CI.

`--jobs N` runs N refinement levels at once; the ladder is the natural place for it,
since the cases are independent and the finest one decides the wall clock. `--profile`
picks who the HTML is written for (`working`, the default, `thesis`, `oss`, `ci`), and
`vv report <study>.json --profile thesis` re-renders an existing result under a
different one without touching the solver.

Binaries are built into **this worktree** (`bin/`), never into the main checkout — the
adapter's on-the-fly build mode runs `make clean` and would desynchronise `.build_state`
there:

```bash
./vvkit/tools/build_binaries.sh
```

## Studies

| dir | measures | reference | ladder |
|---|---|---|---|
| `m1_residual/` | `f_sum`, the discrete force residual | exactly 0 | 64–512 |
| `m2_equilibrium/` | `rho`, `press` after one orbit | analytic torus profile | 64–512 |
| `m3_nu/` | `nu_applied` | the α-law | 64–512 |
| `m4_stress/` | `T_rphi` | analytic shear stress | 64–512 |
| `m5_mms/` | 2D MMS, Euler and viscous operators | manufactured | 32–256 |
| `m6_budget/` | mass budget, sparse and dense sampling | `mdot_in − mdot_out` | 64–256 |
| `a1_ring/` | Lynden-Bell & Pringle spreading ring | closed-form Σ(x, τ) | 64–256 |

All of them have been run; the results are in [FINDINGS.md](FINDINGS.md).
`a1_ring/vvcase_ring_short.yaml` is the one exception — it is the discriminator for A1's
error floor and has not been run yet.

`m1`, `m3` and `m4` run the solver for two cycles only, so they cost almost nothing in CPU
and are limited by `.tab` parsing (59 MB per dump at 512²). Do not trust remembered
runtimes: every report now carries a per-case cost table built from the solver's own
cycle counts and throughput, which is the only figure worth quoting. Measured for `m4`,
one core: 0.24 s at 64², 2.8 s at 512², about 2.8e5 zone-cycles/s.

## Reproducibility

Every report records what produced it: the binary's SHA-256, the checkout and revision
it was built from, the `.build_state` beside it, the input deck as the solver actually
received it, and the per-case cost. This matters here more than in most projects,
because the studies point at `${ATHENA_ROOT}/bin/athena_*` and those binaries are built
by hand from a tree that carries three problem generators — nothing else in the
directory says which one a given number came from.

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
vv                     CLI wrapper; derives and exports ATHENA_ROOT from its own location
tools/                 build_binaries.sh -- builds all three pgens into this worktree's bin/
toolchain/             the vvcase template `vv init` starts from
notes/                 raw first-touch observations, kept as evidence
wiki_example/          the tool's own shipped demo, re-measured here as a cross-check
<study>/               vvcase_*.yaml + athinput.*.template + reports/ + workdir/
```

`workdir/` and `reports/` are gitignored.
