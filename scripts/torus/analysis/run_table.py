#!/usr/bin/env python3
"""One markdown table across several runs: configuration, equilibrium, outcome.

Every run already carries its own README.md, which is the record. This is for the
comparisons those individual files cannot make - a campaign is a set of differences, and
a difference is only visible when the runs sit on the same rows.

Usage:
    python3 run_table.py <label>=<run_dir> [...]
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "run"))
from describe_run import read_athinput, geometry, outcome, scf_report, _f  # noqa: E402

ROWS = [
    ("M_tor",        lambda p, g, s, r: f"{_f(p,'M_tor',0):.2f}"),
    ("self-gravity", lambda p, g, s, r: "on" if _f(p, "self_grav", 1) else "off"),
    ("initial condition", lambda p, g, s, r: "self-consistent" if s else "analytic"),
    ("r_in / r_out", lambda p, g, s, r: f"{g['r_in']:.3f} / {g['r_out']:.3f}"),
    ("z_max",        lambda p, g, s, r: f"{g['z_max']:.3f}"),
    ("Psi_c",        lambda p, g, s, r: f"{g['Psi_c']:.4f}"),
    ("c_s at R_tor", lambda p, g, s, r:
        f"{(_f(p,'gamma',5/3.)*g['Psi_c']/(1.0/(_f(p,'gamma',5/3.)-1)+1))**0.5:.3f}"),
    ("orbits",       lambda p, g, s, r: f"{r['orbits']:.0f}"),
    ("mass drift",   lambda p, g, s, r: f"{r['mass_drift']*100:+.1f}%"),
    ("peak A_1",     lambda p, g, s, r: f"{r['peak_A1']:.3f} @ {r['peak_orbit']:.0f}"),
    ("late A_1",     lambda p, g, s, r: f"{r['late_A1']:.3f} +- {r['late_sd']:.3f}"),
    ("k = A_2/A_1",  lambda p, g, s, r: f"{r['k']:.3f}"),
    ("r_tb",         lambda p, g, s, r: f"{r['r_tb']:.3f}"),
    ("Omega_p",      lambda p, g, s, r: f"{r['omega_p']:.3f}"),
    ("corotation R", lambda p, g, s, r: f"{r['R_co']:.2f}"),
    ("growth rate",  lambda p, g, s, r: f"{r['sigma']:.3f}"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    args = ap.parse_args()

    cols = []
    for spec in args.runs:
        label, _, path = spec.rpartition("=")
        label = label or os.path.basename(os.path.normpath(path))
        p = read_athinput(os.path.join(path, "athinput"))
        s = scf_report(path)
        cols.append((label, p, geometry(p, s), s, outcome(path, "sg_torus_m1")))

    print("| | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, fn in ROWS:
        cells = []
        for _, p, g, s, r in cols:
            try:
                cells.append(fn(p, g, s, r))
            except Exception:
                cells.append("--")
        print(f"| `{name}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
