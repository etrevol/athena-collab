# Phase 1 — raw first-touch notes (before any vvkit modification)

Recorded while driving the CLI as a new user. Raw observations; the polished write-up
goes to `FEEDBACK.md` at the end.

Environment: vvkit run from the Windows venv over WSL interop
(`/mnt/d/MyProjects/vvkit/.venv/Scripts/python.exe -m vvkit.cli.main`), Python 3.12.1,
editable install. Athena++ built in the `artem-vvkit` worktree.

---

## Baseline before touching anything

`pytest -q` → **29 passed, 1 failed**, on a clean checkout.

`tests/test_norms_convergence.py::test_synthetic_order_recovery` fails at line 74:
`compute_least_squares_order` is expected to reject fewer than 3 grids and does not.
`convergence/order.py:109` guards `len(...) < 2` while its own message says "At least 3
grid data points are required".

Consequence is not cosmetic: with exactly 2 grids `linregress` fits a line through 2
points, so it returns `r_squared = 1.0` and `std_err = 0.0` **by construction**. The HTML
report renders that as a perfect fit with zero uncertainty, which is exactly the kind of
unsupported claim `PROJECT_SPEC.md:437-448` says the tool must never make.

`docs/STATE.md:28` claims "All 30 unit and integration tests passing" and >=95% coverage.
Neither matches the tree.

## Line endings

Every source file in the worktree is CRLF while every blob in HEAD is LF, and the repo
had no `core.autocrlf` setting for a non-Windows git. From WSL, `git status` reported 79
modified files and `git diff --stat` 8835 insertions / 8820 deletions, of which the real
content changes were **two files**. Committing from WSL without noticing would have
rewritten the line endings of the entire repository in one commit.

Set `core.autocrlf=true` locally, which matches what Windows git was evidently already
doing. Worth a `.gitattributes` with `* text=auto` so this cannot depend on which git
happens to run.

Also: **22 `__pycache__/*.pyc` files are tracked**. Three of them show up as modified
after simply importing the package, so any commit picks up unrelated bytecode churn.

## `vv --help`

- Usage line reads `python -m vvkit.cli.main`, not `vv`. Typer's `prog_name` is not set,
  so the help does not name the console script the user actually installed.
- Non-ASCII characters are mangled in the Windows console: `vvkit ? Verification Harness`,
  and every markdown bullet in `plugin info` renders as `?`. The em dash and `•` do not
  survive the default codepage. This is the documented Windows+WSL mode, so it is the
  default experience, not an edge case.

## `vv plugin info athena++`

Documents `athena_source`, `configure_args`, `executable`, `output_stream`, `use_wsl`,
`wsl_distro` — but **not `coords` and `fields`**, which the wiki documents and which are
the only way to select which column of a `.tab` file the study measures. A user who reads
`vv plugin info` (the CLI's own self-description, and what the agent-operating prompt in
`Command-Line-Interface-(CLI)-Reference.md:34-62` tells an agent to read) cannot discover
them. For a multi-field Athena++ output that is not optional information.

Claim "No need for a `reader` block in your `vvcase.yaml`!" is true and genuinely nice —
the header parsing does work.

## `vv init --plugin athena++`

The scaffolded template contains a silent correctness bug:

```yaml
mms:
  domain: {x1: [-0.5, 0.5]}
```

The adapter's default `coords` map renames `x1v -> x`, so `coord_names == ["x"]` and
`cli/main.py:181` does `config.mms.domain.get("x", [0.0, 1.0])`. The key `x1` is never
read; the domain silently falls back to `[0, 1]`. That corrupts `h_val`, the cell bounds
used for the cell average, and the computed cell volumes — i.e. the error norms and the
observed order — with no warning anywhere. A first-time user following the scaffold gets
numbers that look plausible and are wrong.

The template also references `athinput.sod.template`, which it does not create and does
not describe. `vv init` scaffolds half of the required input.

Related: pydantic models use the default `extra="ignore"`, so a mistyped or misplaced key
is silently discarded rather than reported. `PROJECT_SPEC.md:277-279` asks for the
opposite ("point at the offending key, suggest the closest valid key on typos").
