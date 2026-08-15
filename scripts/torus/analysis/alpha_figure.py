"""The dissipation scan as one figure: does the mode survive, and does the disk.

    python3 alpha_figure.py <campaign_dir> -o figs/alpha_threshold.png

Reads every alpha_* run in a campaign directory, takes the peak m = 1 amplitude and the
surviving mass out of each history file, and plots both against alpha on a log axis. The
upper axis gives the same thing as an interval between cloud collisions, using the
particulate-disc correspondence alpha ~ 1/(Omega t_coll) in the rarely-colliding branch.

Two curves rather than one, because the interesting result is that they switch over
together: it is the mode that destroys the disk, so where the mode is suppressed the
disk survives. A single curve would show the threshold; the pair shows why it is there.

The amplitude is measured only while the disk still holds 90% of its mass. Taking the
maximum over the whole run does not work: a viscously draining disk becomes lopsided as
it accretes, so A_1 climbs monotonically to the last snapshot without ever saturating,
and max() then reports the drainage rather than the mode. At alpha = 0.01 that inflated
the point by a factor of 37 and lifted it visibly off an otherwise monotonic curve.

A short intact-disk window has two possible causes and they must not be conflated. Below
the threshold the mode itself destroys the disk, so the window closes early but the peak
inside it is the real saturated amplitude. Above the threshold viscosity drains the disk
before the mode can grow, and the peak is only an upper limit. The two are told apart by
whether the amplitude ever rose far above the seed: a point is drawn open when its window
is short AND its amplitude stayed within a hundred times the seed.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_modes import load_history, read_hst        # noqa: E402


def collect(campaign, problem_id="acc_disk_visc", min_window=20.0, mass_floor=0.90):
    """One point per run: alpha, peak amplitude, seed, surviving mass, trustworthiness."""
    out = []
    for d in sorted(glob.glob(os.path.join(campaign, "alpha_*"))):
        m = re.search(r"alpha_([0-9.]+)$", d)
        hst = os.path.join(d, "data", f"{problem_id}.hst")
        if not m or not os.path.isfile(hst):
            continue
        a = float(m.group(1))
        x = load_history(hst)
        h = read_hst(hst)
        mass = np.asarray(h["mass"])
        frac = mass / mass[0]
        intact = frac >= mass_floor
        if not intact.any():
            continue
        window = float(x["orbits"][intact][-1])
        gam = 5 / 3.
        t_acc = np.inf if a == 0 else 1.0 / (2 * np.pi * a * (gam - 1) * 0.3)
        peak = float(x["A"][1][intact].max())
        seed = float(x["A"][1][0])
        grew = peak > 100.0 * seed          # the mode, not the drainage, closed the window
        out.append(dict(alpha=a, peak=peak, seed=seed,
                        mass=100.0 * frac[-1], t_acc=t_acc, window=window,
                        drains=(window < min_window) and not grew))
    return out


def figure(points, out_path, title=None):
    if not points:
        return None
    a = np.array([p["alpha"] for p in points])
    pk = np.array([p["peak"] for p in points])
    ms = np.array([p["mass"] for p in points])
    dr = np.array([p["drains"] for p in points])
    seed = points[0]["seed"]
    # the inviscid point has no place on a log axis; park it left of the smallest alpha
    left = a[a > 0].min() / 2.5
    x = np.where(a > 0, a, left)

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ok = (~dr) & (a > 0)
    ax.loglog(x[ok], pk[ok], "o-", color="#1f77b4", ms=7, label="peak $A_1$")
    if dr.any():
        ax.loglog(x[dr], pk[dr], "o", mfc="none", color="#1f77b4", ms=7,
                  label="upper limit: disk drains before the mode can grow")
    if (a == 0).any():
        ax.loglog(x[a == 0], pk[a == 0], "*", color="#1f77b4", ms=16,
                  label="inviscid ($\\alpha = 0$)")
    ax.axhline(seed, color="0.55", lw=0.9, ls=":")
    # mid-axis, where neither the legend nor either curve reaches
    # opaque box: the label sits on the dotted line and must not blend into it
    ax.annotate("seed", xy=(x.max() / 8.0, seed), xytext=(0, 0),
                textcoords="offset points", ha="center", va="center",
                fontsize=8.5, color="0.35",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    # room below the seed line so the legend does not sit on the data
    ax.set_ylim(seed / 6.0, max(pk) * 3.0)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("peak $A_1 / M_{\\rm tor}$")

    ax2 = ax.twinx()
    ax2.semilogx(x, ms, "s--", color="#d62728", ms=5, alpha=0.75)
    ax2.set_ylabel("mass remaining [%]", color="#d62728")
    ax2.set_ylim(0, 105)

    top = ax.secondary_xaxis("top", functions=(
        lambda v: 1.0 / (2 * np.pi * np.maximum(v, 1e-12)),
        lambda v: 1.0 / (2 * np.pi * np.maximum(v, 1e-12))))
    top.set_xlabel("orbits between cloud collisions")
    leg = ax.legend(loc="lower left", fontsize=8.5, borderaxespad=0.6,
                    frameon=True, framealpha=1.0, edgecolor="0.8", borderpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    # no axes title: the document caption names the figure, and a title here only
    # crowds the secondary axis label sitting immediately above it
    if title:
        ax.set_title(title, fontsize=11, pad=26)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--id", default="acc_disk_visc")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    pts = collect(args.campaign, args.id)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(figure(pts, args.out, args.title) or "no usable runs")
    for p in pts:
        flag = "  (upper limit)" if p["drains"] else ""
        print(f"  alpha={p['alpha']:<8} peak={p['peak']:.2e}  mass={p['mass']:5.0f}%"
              f"  intact window {p['window']:5.1f} orbits{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
