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
from torus_modes import (load_history, pattern_speed, run_root,   # noqa: E402
                          run_parameters, corotation_radius)

GAMMA = 5.0 / 3.0
COLORS = ["#1f4e79", "#c0392b", "#27ae60", "#8e44ad", "#d68910", "#16a085"]


def load(run_dir):
    hst = next(pathlib.Path(run_dir).glob("data/*.hst"))
    d = load_history(str(hst))
    E_pot = d["E_pot"]
    d["V"] = (2.0 * d["E_kin"] + 3.0 * (GAMMA - 1.0) * d["E_int"] + E_pot) \
        / np.abs(E_pot)
    d["E_int_rel"] = d["E_int"] / d["E_int"][0]
    d["om_p"] = pattern_speed(d)
    d["r_tb_mag"] = np.linalg.norm(d["r_tb"], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["k"] = d["A"][2] / d["A"][1]
    d["params"] = run_parameters(run_dir)
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

    fig, axg = plt.subplots(2, 3, figsize=(14.0, 7.6))
    axes = axg.ravel()
    for i, (label, d) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        axes[0].plot(d["orbits"], d["E_int_rel"], color=c, lw=1.1, label=label)
        axes[1].plot(d["orbits"], d["V"], color=c, lw=1.0, label=label)
        axes[2].semilogy(d["orbits"], d["A"][1], color=c, lw=1.0, label=label)
        axes[3].plot(d["orbits"], d["om_p"], color=c, lw=1.0, label=label)
        axes[4].plot(d["orbits"], d["r_tb_mag"], color=c, lw=1.0, label=label)
        axes[5].plot(d["orbits"], d["k"], color=c, lw=0.9, label=label)

    axes[0].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[0].set_ylabel(r"$E_{\rm int}(t)\,/\,E_{\rm int}(0)$")
    axes[0].set_title("internal energy: the readjustment")
    axes[1].axhline(0.0, color="0.6", lw=0.8, ls=":")
    axes[1].set_ylabel(r"$(2E_{\rm kin}+3(\gamma-1)E_{\rm int}+E_{\rm pot})/|E_{\rm pot}|$")
    axes[1].set_title("virial residual: zero means equilibrium")
    axes[2].set_ylabel(r"$A_1 / M_{\rm tor}$")
    axes[2].set_title(r"the $m=1$ mode")
    axes[3].set_ylim(0.0, 1.2)
    axes[3].axhline(1.0, color="0.6", lw=0.8, ls=":")
    axes[3].set_ylabel(r"$\Omega_p\,/\,\Omega(R_{\rm tor})$")
    axes[3].set_title(r"pattern speed; a slow mode would sit near $\mu_{\rm tor}$")
    axes[4].set_ylabel(r"$|r_{\rm tb}|\,/\,R_{\rm tor}$")
    axes[4].set_title("barycentre offset (theirs: 0.24)")
    axes[5].set_ylim(0.0, 0.6)
    axes[5].axhspan(0.20, 0.25, color="0.85", zorder=0)
    axes[5].set_ylabel(r"$k = A_2/A_1$")
    axes[5].set_title("harmonic ratio; grey band is theirs")
    for ax in axes:
        ax.set_xlabel(r"$t\,/\,T_{\rm orb}$")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)

    hdr = (f"{'run':<26}{'E_int swing 0-10':>18}{'|V| rms 0-5':>13}"
           f"{'10-20':>9}{'35-50':>9}{'Om_p 35-50':>12}{'R_co':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label, d in runs:
        om = window(d, "om_p", 35, 50, "mean")
        rco = corotation_radius(om, **d["params"]) if om == om and om > 0 else float("nan")
        print(f"{label:<26}{window(d,'E_int_rel',0,10,'swing'):>18.3f}"
              f"{window(d,'V',0,5):>13.4f}{window(d,'V',10,20):>9.4f}"
              f"{window(d,'V',35,50):>9.4f}{om:>12.3f}{rco:>8.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
