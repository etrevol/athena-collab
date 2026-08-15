"""Summarise a parameter scan: one measured quantity per point, with a fitted trend.

    python3 scan_figure.py <run_dir>=<x> [...] --xlabel "q" -o figs/qscan.png

Each run contributes one point, taken from its history file the same way the per-run
report takes it, so the scan and the individual runs cannot disagree. Two panels: the
growth rate and the pattern speed, which are the two things a scan over the rotation law
or over viscosity is actually asking about.

A straight line is fitted to each and its zero crossing is reported, because that is
usually the question - at what value of the parameter does the instability switch off,
and does the pattern speed reach zero at the same place or somewhere else. The fit is
drawn dashed beyond the measured range so that extrapolation is visibly extrapolation.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_modes import load_history, pattern_speed                  # noqa: E402
from torus_figures import linear_window                              # noqa: E402


def measure(run_dir, problem_id):
    """Growth rate in units of the orbital frequency, and the saturated pattern speed."""
    d = load_history(os.path.join(run_dir, f"{problem_id}.hst"))
    o, A = d["orbits"], d["A"]
    w = linear_window(o, A[1])
    if w.sum() < 4:
        return None
    c = np.polyfit(o[w], np.log(A[1][w]), 1)[0]
    om = pattern_speed(d)
    wp = (o > o[w][0]) & (o < o[w][-1])
    late = o > 0.7 * o[-1]
    return dict(sigma=float(c) / (2.0 * np.pi), omega_p=float(np.mean(om[wp])),
                peak=float(A[1].max()), late=float(np.mean(A[1][late])))


def figure(points, out_path, xlabel="q", problem_id="sg_torus_m1"):
    xs, res = [], []
    for run, x in points:
        m = measure(run, problem_id)
        if m:
            xs.append(x)
            res.append(m)
    if len(xs) < 2:
        return None
    xs = np.array(xs)
    order = np.argsort(xs)
    xs = xs[order]
    res = [res[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    span = np.linspace(xs.min(), max(xs.max() * 1.6, xs.max() + 0.2), 100)
    for ax, key, lab, col in ((axes[0], "sigma", "$\\sigma\\,/\\,\\Omega$", "#1f77b4"),
                              (axes[1], "omega_p", "$\\Omega_p\\,/\\,\\Omega$", "#d62728")):
        y = np.array([r[key] for r in res])
        ax.plot(xs, y, "o", color=col, ms=7)
        p = np.polyfit(xs, y, 1)
        inside = span <= xs.max()
        ax.plot(span[inside], np.polyval(p, span[inside]), "-", color=col, lw=1.4)
        ax.plot(span[~inside], np.polyval(p, span[~inside]), "--", color=col, lw=1.4,
                alpha=0.8)
        ax.axhline(0.0, color="0.6", lw=0.8)
        zero = -p[1] / p[0] if p[0] != 0 else np.nan
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lab)
        rms = float(np.std(y - np.polyval(p, xs)))
        ax.set_title(f"{lab}: slope {p[0]:+.3f}, zero at {xlabel} = {zero:.3f}"
                     f"  (rms {rms:.4f})", fontsize=10)
    axes[1].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[1].text(xs.min(), 1.02, "corotating with the torus centre", fontsize=8,
                 color="0.4")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("points", nargs="+", help="run_dir=x pairs")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--xlabel", default="q")
    ap.add_argument("--id", default="sg_torus_m1")
    args = ap.parse_args()

    pts = []
    for spec in args.points:
        run, x = spec.rsplit("=", 1)
        pts.append((run, float(x)))
    out = figure(pts, args.out, args.xlabel, args.id)
    print(out or "not enough usable points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
