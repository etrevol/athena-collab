#!/usr/bin/env python3
"""How much of the early evolution is the initial condition settling down.

The analytic equilibrium balances the gas against the central point mass alone, which
is wrong by O(M_tor/M_c) - a third of the gravity on the gravity branch. The torus then
spends its first tens of orbits readjusting, and any quantity measured in that window
is a measurement of the readjustment. This figure is how that is judged, and how the
self-consistent initial condition (scf_iter > 0) is verified to have removed it.

Two diagnostics, both dimensionless so that runs of different mass can share an axis:

    E_int(t) / E_int(0)     the internal energy, whose swing is the readjustment doing
                            pdV work on itself
    V = (2 E_kin + 3(gamma-1) E_int + E_pot) / |E_pot|
                            the scalar virial residual; zero for a body in equilibrium,
                            and its rms over a window says whether that window is one

Usage:
    python3 ic_relaxation.py <label>=<run_dir> [...] -o figs/ic_relaxation.png
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from torus_modes import load_history, run_root      # noqa: E402

GAMMA = 5.0 / 3.0
COLORS = ["#1f4e79", "#c0392b", "#27ae60", "#8e44ad", "#d68910", "#16a085"]


def load(run_dir):
    hst = next(pathlib.Path(run_dir).glob("data/*.hst"))
    d = load_history(str(hst))
    E_pot = d["E_pot"]
    d["V"] = (2.0 * d["E_kin"] + 3.0 * (GAMMA - 1.0) * d["E_int"] + E_pot) \
        / np.abs(E_pot)
    d["E_int_rel"] = d["E_int"] / d["E_int"][0]
    return d


def window(d, key, lo, hi, how="rms"):
    o = d["orbits"]
    sel = (o >= lo) & (o <= hi)
    if not sel.any():
        return float("nan")
    v = d[key][sel]
    if how == "rms":
        return float(np.sqrt(np.nanmean(v ** 2)))
    if how == "swing":
        return float(np.nanmax(np.abs(v - 1.0)))
    return float(np.nanmean(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("-o", "--out", default="ic_relaxation.png")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        label, _, path = spec.rpartition("=")
        if not path:
            label, path = pathlib.Path(spec).name, spec
        runs.append((label, load(path)))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for i, (label, d) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        axes[0].plot(d["orbits"], d["E_int_rel"], color=c, lw=1.1, label=label)
        axes[1].plot(d["orbits"], d["V"], color=c, lw=1.0, label=label)
        axes[2].semilogy(d["orbits"], d["A"][1], color=c, lw=1.0, label=label)

    axes[0].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[0].set_ylabel(r"$E_{\rm int}(t)\,/\,E_{\rm int}(0)$")
    axes[0].set_title("internal energy: the readjustment")
    axes[1].axhline(0.0, color="0.6", lw=0.8, ls=":")
    axes[1].set_ylabel(r"$(2E_{\rm kin}+3(\gamma-1)E_{\rm int}+E_{\rm pot})/|E_{\rm pot}|$")
    axes[1].set_title("virial residual: zero means equilibrium")
    axes[2].set_ylabel(r"$A_1 / M_{\rm tor}$")
    axes[2].set_title(r"the $m=1$ mode, for reference")
    for ax in axes:
        ax.set_xlabel(r"$t\,/\,T_{\rm orb}$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)

    hdr = (f"{'run':<22}{'E_int swing 0-10':>18}{'|V| rms 0-5':>14}"
           f"{'10-20':>10}{'35-50':>10}")
    print(hdr)
    print("-" * len(hdr))
    for label, d in runs:
        print(f"{label:<22}{window(d,'E_int_rel',0,10,'swing'):>18.3f}"
              f"{window(d,'V',0,5):>14.4f}{window(d,'V',10,20):>10.4f}"
              f"{window(d,'V',35,50):>10.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
