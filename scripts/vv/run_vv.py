#!/usr/bin/env python3
"""Run the verification & validation suite.

    python3 scripts/vv/run_vv.py                 # quick profile, ~3-5 min on 16 cores
    python3 scripts/vv/run_vv.py --full          # adds a 512^2 grid level
    python3 scripts/vv/run_vv.py --only ppi      # one group, reusing nothing
    python3 scripts/vv/run_vv.py --reuse <dir>   # re-analyse (or extend) an existing run

Writes results/tests/test-<timestamp>/ with data/, materials/, figs/, REPORT.md and
results.json, matching the layout build.sh and sweep.sh use. Exit status is 1 if any
check failed, so this can gate a commit.

The suite is deliberately sized so that the whole thing runs between edits: the longest
single case is a 256^2 run of two orbits, and the rest fit in its shadow. Depth is bought
by adding grid levels (--full), never by making the default slower.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from vvlib import cases as case_mod                                   # noqa: E402
from vvlib import checks, report                                      # noqa: E402
from vvlib.core import FAIL, WARN, OK, INFO                           # noqa: E402
from vvlib.runner import Suite                                        # noqa: E402

_MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL ", INFO: " info "}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="add the 512^2 grid level (costs more than everything else)")
    ap.add_argument("--only", default=None,
                    help="comma-separated tags: static,resid,convergence,temporal,"
                         "invariance,sensitivity,viscosity,ppi")
    ap.add_argument("--jobs", type=int, default=None, help="parallel runs")
    ap.add_argument("--reuse", default=None, help="an existing results/tests/... dir")
    ap.add_argument("--no-run", action="store_true", help="analyse only, run nothing")
    ap.add_argument("--tex", default=None,
                    help="also write the LaTeX result tables here, e.g. "
                         "docs/astroformular/sections/vv_tables.tex")
    a = ap.parse_args(argv)

    t0 = time.time()
    profile = "full" if a.full else "quick"
    suite = Suite(reuse=a.reuse)
    model = case_mod.MODEL
    ladder = case_mod.LADDER_FULL if a.full else case_mod.LADDER_QUICK

    all_cases = case_mod.build(profile)
    if a.only:
        want = {t.strip() for t in a.only.split(",")}
        all_cases = [c for c in all_cases if want & set(c.tags)]

    print(f"V&V suite ({profile}) -> {suite.dir}")
    prov = suite.record_provenance()
    print(f"  commit {prov['git_sha'][:12]}, {prov['git_dirty_files']} modified file(s)")

    records = {}
    if not a.no_run:
        records = suite.run(all_cases, jobs=a.jobs)
        bad = [r for r in records.values() if r["status"] == "failed"]
        if bad:
            print(f"\n  {len(bad)} case(s) failed to run: "
                  f"{', '.join(r['cid'] for r in bad)}")

    print("\nanalysing...")
    ran = {c.cid for c in all_cases}
    results, data = [], {}

    def collect(fn, *args, **kw):
        try:
            cs, d = fn(*args, **kw)
        except Exception as exc:                       # a broken check is a failure,
            from vvlib.core import Check               # not a crashed suite
            cs, d = [Check(fn.__name__.replace("_checks", ""), "internal", FAIL,
                           f"{type(exc).__name__}: {exc}")], {}
        results.extend(cs)
        data.update(d)

    if records:
        collect(checks.health_checks, suite, model, records)
    if "header" in ran or a.no_run:
        collect(checks.static_checks, model)
        collect(checks.mapping_checks, suite, model)
    if any(c.startswith("resid_n") for c in ran):
        collect(checks.residual_checks, suite, model, ladder)
    if any(c.startswith("conv_n") for c in ran):
        collect(checks.convergence_checks, suite, model, ladder)
    if any(c.startswith("cfl_") for c in ran):
        collect(checks.temporal_checks, suite, model)
    if "inv_1block" in ran:
        collect(checks.invariance_checks, suite, model)
    if any(c.startswith("sens_") for c in ran):
        gci = data.get("conv_gci", {}).get("gci", float("nan"))
        collect(checks.sensitivity_checks, suite, model, gci)
    if any(c.startswith("visc_") for c in ran):
        collect(checks.viscosity_checks, suite, model)
    if any(c.startswith("ring_") for c in ran):
        collect(checks.ring_checks, suite, model, case_mod)
    if any(c.startswith("ppi_") for c in ran):
        collect(checks.ppi_checks, suite, model)

    figs = report.make_figures(suite, model, data)
    elapsed = time.time() - t0
    counts = report.write_report(suite, model, results, records, figs, elapsed, prov)
    report.write_tex(suite, model, results, data, path=a.tex)

    width = max((len(c.name) for c in results), default=10)
    group = None
    print()
    for c in results:
        if c.group != group:
            group, _ = c.group, print(f"\n{c.group}")
        print(f"  [{_MARK[c.status]}] {c.name:<{width}}  {c.detail}")

    print(f"\n{counts[FAIL]} failure(s), {counts[WARN]} warning(s), "
          f"{counts[OK]} passed, {counts[INFO]} recorded   [{elapsed:.0f} s]")
    print(f"report: {suite.dir / 'REPORT.md'}")
    for f in figs:
        print(f"figure: {f}")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
