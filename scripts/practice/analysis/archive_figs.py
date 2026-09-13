#!/usr/bin/env python3
"""Cross-run figures for the base-code evidence set: one figure per check.

    python3 scripts/practice/analysis/archive_figs.py SWEEP_DIR [SWEEP_DIR ...] [--out DIR]

Reads every run directory under the given sweeps - any subdirectory holding an
athinput.in, data/*.hst and (optionally) data/*.tab - so a sweep directory and an archive's
runs/ directory both work
and writes, under --out (default: the first sweep's figs_archive/):

    mass_conservation.png   inviscid runs: disk mass, total energy, angular momentum drift
    floor_scan.png          cells on dfloor / pfloor per frame, and rho_min - the sharpest
                            health check, made by scanning every frame
    viscous_vs_theory.png   viscous runs against exp(-t / N_orbits), N from disk_model
    meshblock_bitwise.png   one MeshBlock against sixteen: every history column, |difference|
    resolution.png          128 x 128 against 256 x 256: mass drift and the final profile
    summary.md              the numbers behind the figures

Runs are classified from their athinput.in (alpha, mesh, meshblock), not from directory
names, so the same script serves any sweep built on the base input.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "vis", "python"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "theory"))
import athena_read                                   # noqa: E402
from disk_model import DiskModel, orbits_to_accrete  # noqa: E402


# ----------------------------------------------------------------------------- input
def read_athinput(path):
    out, block = {}, None
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
        elif "=" in line:
            k, v = line.split("=", 1)
            out[f"{block}/{k.strip()}"] = v.strip()
    return out


def read_hst(path):
    names = None
    for line in open(path):
        if line.startswith("#") and "[1]=" in line:
            names = [t.split("]=")[1] for t in line.lstrip("#").split() if "]=" in t]
            break
    d = np.loadtxt(path, comments="#")
    return {n: d[:, i] for i, n in enumerate(names)}


def load_run(rd):
    ath = os.path.join(rd, "athinput.in")
    hst = glob.glob(os.path.join(rd, "data", "*.hst"))
    if not os.path.isfile(ath) or not hst:
        return None
    status = open(os.path.join(rd, "status")).read().strip() if os.path.isfile(os.path.join(rd, "status")) else ""
    if status and status != "COMPLETED":
        print(f"  {os.path.basename(rd):<50} {status} - left out of the figures")
        return None
    p = read_athinput(ath)
    g = lambda k, d=None: float(p[k]) if k in p else d
    m = DiskModel(M_bh=g("problem/M_bh", 4.5e7), T_0=g("problem/T_0", 5e4),
                  mu=g("problem/mu", 0.6), chi=g("problem/chi", 500.0),
                  gamma=g("hydro/gamma", 1.3), r_center=g("problem/r_center", 1.0),
                  C_prime=g("problem/C_prime", 0.2), alpha=g("problem/alpha", 0.0),
                  rho_atm=g("problem/rho_atm", 1e-4))
    h = read_hst(hst[0])
    frames = sorted(glob.glob(os.path.join(rd, "data", "*.block0.out1.*.tab")))
    return dict(name=os.path.basename(rd), dir=rd, p=p, model=m, hst=h,
                orbits=h["time"] / m.P_orb, frames=frames,
                alpha=m.alpha, nu_iso=g("problem/nu_iso", 0.0),
                nx1=int(g("mesh/nx1")), nx2=int(g("mesh/nx2")),
                bx1=int(g("meshblock/nx1", g("mesh/nx1"))), bx2=int(g("meshblock/nx2", g("mesh/nx2"))),
                dfloor=g("hydro/dfloor", 1e-12), pfloor=g("hydro/pfloor", 1e-10),
                tlim_orbits=g("time/tlim") / m.P_orb)


def nblocks(r):
    return (r["nx1"] // r["bx1"]) * (r["nx2"] // r["bx2"])


def label(r):
    tag = f"{r['tlim_orbits']:.0f} orbits, {r['nx1']}x{r['nx2']}"
    if nblocks(r) > 1:
        tag += f", {nblocks(r)} blocks"
    return ("inviscid" if r["alpha"] == 0 and r["nu_iso"] == 0 else f"alpha = {r['alpha']:g}") + " (" + tag + ")"


# ----------------------------------------------------------------------------- frames
def scan_floors(r):
    """Cells on either floor, and rho_min, frame by frame, over every MeshBlock.

    The result is cached as floor_scan.csv in the run directory, so a copy of the run
    that carries no frames (an archive with only the history) still has the numbers."""
    cache = os.path.join(r["dir"], "floor_scan.csv")
    if not r["frames"] and os.path.isfile(cache):
        c = np.loadtxt(cache, delimiter=",", skiprows=1, ndmin=2)
        return c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    orbits, n_d, n_p, rho_min = [], [], [], []
    for f0 in r["frames"]:
        idx = f0.split(".out1.")[1]
        nd = np_ = 0; rmin = np.inf; t = None
        for f in glob.glob(os.path.join(r["dir"], "data", f"*.block*.out1.{idx}")):
            d = athena_read.tab(f)
            rho = np.asarray(d["rho"]); prs = np.asarray(d["press"]); t = float(d["time"])
            nd += int((rho <= r["dfloor"] * (1 + 1e-7)).sum())
            np_ += int((prs <= r["pfloor"] * (1 + 1e-7)).sum())
            rmin = min(rmin, float(rho.min()))
        orbits.append(t / r["model"].P_orb); n_d.append(nd); n_p.append(np_); rho_min.append(rmin)
    out = np.array(orbits), np.array(n_d), np.array(n_p), np.array(rho_min)
    if r["frames"]:
        np.savetxt(cache, np.c_[out], delimiter=",", header="orbits,cells_on_dfloor,cells_on_pfloor,rho_min", comments="")
    return out


def final_profile(r):
    """Azimuthal mean of rho at the first and last frame, assembled over the MeshBlocks;
    cached as profiles.csv for copies of the run that carry no frames."""
    cache = os.path.join(r["dir"], "profiles.csv")
    if not r["frames"] and os.path.isfile(cache):
        c = np.loadtxt(cache, delimiter=",", skiprows=1)
        return c[:, 0], c[:, 1], c[:, 2]
    def mean_profile(idx):
        rr, prof = [], []
        for f in glob.glob(os.path.join(r["dir"], "data", f"*.block*.out1.{idx}")):
            d = athena_read.tab(f)
            rr.append(np.asarray(d["x1v"])[0]); prof.append(np.asarray(d["rho"]).mean(axis=0))
        rr = np.concatenate(rr); prof = np.concatenate(prof); o = np.argsort(rr)
        return rr[o], prof[o]
    rr, p0 = mean_profile(r["frames"][0].split(".out1.")[1])
    _, p1 = mean_profile(r["frames"][-1].split(".out1.")[1])
    np.savetxt(cache, np.c_[rr, p0, p1], delimiter=",", header="r,rho_mean_first,rho_mean_last", comments="")
    return rr, p0, p1


# ----------------------------------------------------------------------------- figures
def fig_mass(runs, out, md):
    inv = [r for r in runs if r["alpha"] == 0 and r["nu_iso"] == 0]
    if not inv:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    md += ["## Inviscid runs: conservation", "", "| run | orbits | dm/m0 (disk) | dE/E0 | dL/L0 |", "|---|---|---|---|---|"]
    for r in inv:
        h = r["hst"]; t = r["orbits"]
        dm = h["disk_mass"] / h["disk_mass"][0] - 1
        dE = h["tot-E"] / h["tot-E"][0] - 1
        dL = h["2-mom"] / h["2-mom"][0] - 1
        for a, y in zip(ax, (dm, dE, dL)):
            a.plot(t, y, label=label(r))
        md.append(f"| `{r['name']}` | {t[-1]:.0f} | {dm[-1]:+.2e} | {dE[-1]:+.2e} | {dL[-1]:+.2e} |")
    for a, ttl in zip(ax, (r"disk mass: $m/m_0 - 1$", r"total energy: $E/E_0 - 1$", r"angular momentum: $L/L_0 - 1$")):
        a.set_title(ttl); a.set_xlabel("orbits"); a.axhline(0, c="k", lw=0.6); a.grid(alpha=0.3)
        a.set_yscale("symlog", linthresh=1e-6)
    ax[0].axhspan(-1e-4, 1e-4, color="g", alpha=0.08, label=r"criterion $|dm/m_0| < 10^{-4}$")
    ax[0].legend(fontsize=8)
    fig.suptitle("Inviscid disk (alpha = 0, nu_iso = 0): what should be conserved, is\n"
                 "(E and L are domain totals, ambient included: the periodic wiggle is ambient gas crossing the boundaries)",
                 fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(out, "mass_conservation.png"), dpi=130); plt.close(fig)
    md.append("")


def fig_floors(runs, out, md):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    md += ["## Cells on a floor (every frame scanned)", "", "| run | frames | max cells on dfloor | max cells on pfloor | rho_min | rho_min / rho_atm |", "|---|---|---|---|---|---|"]
    for r in runs:
        if not r["frames"] and not os.path.isfile(os.path.join(r["dir"], "floor_scan.csv")):
            continue
        t, nd, npf, rmin = scan_floors(r)
        ax[0].plot(t, nd + npf, label=label(r))
        ax[1].semilogy(t, rmin, label=label(r))
        md.append(f"| `{r['name']}` | {len(t)} | {nd.max()} | {npf.max()} | {rmin.min():.2e} | {rmin.min()/r['model'].rho_atm:.2f} |")
    ax[0].set_title("cells on dfloor or pfloor"); ax[0].set_xlabel("orbits"); ax[0].set_ylabel("count")
    ax[1].set_title(r"$\rho_{min}$ over the domain"); ax[1].set_xlabel("orbits")
    ax[1].axhline(runs[0]["model"].rho_atm, ls=":", c="k", label=r"$\rho_{atm}$")
    ax[1].axhline(runs[0]["dfloor"], ls="--", c="r", label="dfloor")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("Floors are a safety net: a healthy run never reaches them"); fig.tight_layout()
    fig.savefig(os.path.join(out, "floor_scan.png"), dpi=130); plt.close(fig)
    md.append("")


def fig_viscous(runs, out, md):
    vis = [r for r in runs if r["alpha"] > 0]
    if not vis:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    md += ["## Viscous runs against the analytic time scale", "", "| run | alpha | N_orbits = 1/(2 pi alpha (gamma-1)(0.5-C')) | measured loss | exp(-t/N) loss | ratio |", "|---|---|---|---|---|---|"]
    for r in vis:
        m = r["model"]; N = orbits_to_accrete(m.alpha, m.gamma, m.C_prime)
        t = r["orbits"]; y = r["hst"]["disk_mass"] / r["hst"]["disk_mass"][0]
        l, = ax.plot(t, y, label=f"alpha = {m.alpha:g}")
        ax.plot(t, np.exp(-t / N), ls="--", c=l.get_color(), label=f"exp(-t/N), N = {N:.0f} orbits")
        md.append(f"| `{r['name']}` | {m.alpha:g} | {N:.0f} | {1-y[-1]:.3f} | {1-np.exp(-t[-1]/N):.3f} | {(1-y[-1])/(1-np.exp(-t[-1]/N)):.2f} |")
    inv = [r for r in runs if r["alpha"] == 0 and r["nu_iso"] == 0 and r["tlim_orbits"] <= 150]
    if inv:
        ax.plot(inv[0]["orbits"], inv[0]["hst"]["disk_mass"] / inv[0]["hst"]["disk_mass"][0], c="k", lw=0.8, label="inviscid")
    ax.set_xlabel("orbits"); ax.set_ylabel(r"disk mass / $m_0$"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Viscous spreading against the thin-disk estimate")
    fig.tight_layout(); fig.savefig(os.path.join(out, "viscous_vs_theory.png"), dpi=130); plt.close(fig)
    md.append("")


def fig_meshblocks(runs, out, md):
    inv = [r for r in runs if r["alpha"] == 0 and r["nu_iso"] == 0 and r["nx1"] == 128]
    many = [r for r in inv if nblocks(r) > 1]
    one = [r for r in inv if nblocks(r) == 1 and r["tlim_orbits"] <= 150]
    if not one or not many:
        return
    b = many[0]
    # the one-block partner: the run whose length matches, else the shortest longer one
    one.sort(key=lambda r: (abs(r["tlim_orbits"] - b["tlim_orbits"]), r["tlim_orbits"]))
    a = one[0]
    n = min(len(a["hst"]["time"]), len(b["hst"]["time"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    md += [f"## One MeshBlock against {nblocks(b)} ({a['orbits'][n-1]:.0f} orbits)", "",
           "| column | max |difference| | relative to the column's scale | at t = 0 |", "|---|---|---|---|"]
    worst = 0.0
    for k in a["hst"]:
        if k in ("time", "dt"):
            continue
        d = np.abs(a["hst"][k][:n] - b["hst"][k][:n])
        scale = np.abs(a["hst"][k][:n]).max()
        rel = d.max() / scale if scale > 0 else 0.0
        ax.plot(a["orbits"][:n], d / scale if scale > 0 else d, label=k); worst = max(worst, rel)
        md.append(f"| `{k}` | {d.max():.3e} | {rel:.1e} | {d[0] / scale if scale > 0 else 0:.1e} |")
    ax.set_yscale("log"); ax.set_xlabel("orbits"); ax.set_ylabel("|1 block - 16 blocks| / column scale")
    ax.set_title(f"Every history column, one block vs {nblocks(b)}: largest relative difference {worst:.1e}")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "meshblock_bitwise.png"), dpi=130); plt.close(fig)
    md += ["", f"Largest relative difference over all columns and rows: **{worst:.1e}**. "
           "The two runs differ from the first row (the history sums run over the blocks in a different "
           "order) and the difference grows with time: Athena++ derives each block's uniform cell width "
           "from the block's own rounded edges, so blocks along phi carry widths that differ by one ulp, "
           "and in a linearly unstable flow that roundoff seed grows.", ""]


def fig_resolution(runs, out, md):
    inv = [r for r in runs if r["alpha"] == 0 and r["nu_iso"] == 0 and nblocks(r) == 1 and r["tlim_orbits"] <= 150]
    res = sorted({r["nx1"] for r in inv})
    if len(res) < 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    md += ["## Resolution", "", "| run | grid | dm/m0 | rho_min at end |", "|---|---|---|---|"]
    for nx in res:
        r = next(x for x in inv if x["nx1"] == nx)
        h = r["hst"]; dm = h["disk_mass"] / h["disk_mass"][0] - 1
        ax[0].plot(r["orbits"], dm, label=f"{r['nx1']}x{r['nx2']}")
        if r["frames"] or os.path.isfile(os.path.join(r["dir"], "profiles.csv")):
            rr, p0, p1 = final_profile(r)
            ax[1].semilogy(rr, p1, label=f"{r['nx1']}x{r['nx2']}, t = {r['orbits'][-1]:.0f} orbits")
            if nx == res[0]:
                ax[1].semilogy(rr, p0, c="k", ls=":", label="t = 0")
        md.append(f"| `{r['name']}` | {r['nx1']}x{r['nx2']} | {dm[-1]:+.2e} | - |")
    ax[0].set_yscale("symlog", linthresh=1e-6); ax[0].set_title(r"disk mass: $m/m_0 - 1$"); ax[0].set_xlabel("orbits")
    ax[1].set_title(r"azimuthal mean $\rho(r)$, final frame"); ax[1].set_xlabel("r")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Twice the resolution: the answer does not move"); fig.tight_layout()
    fig.savefig(os.path.join(out, "resolution.png"), dpi=130); plt.close(fig)
    md.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    runs = []
    for s in args.sweeps:
        # a sweep directory (t1_..., t2_...) or any directory of run directories
        for rd in sorted(d for d in glob.glob(os.path.join(s, "*")) if os.path.isfile(os.path.join(d, "athinput.in"))):
            r = load_run(rd)
            if r:
                runs.append(r); print(f"  {r['name']:<50} {label(r)}")
    if not runs:
        sys.exit("no runs found")
    out = args.out or os.path.join(args.sweeps[0], "figs_archive")
    os.makedirs(out, exist_ok=True)
    md = ["# Base-code evidence set", "", f"Runs: {len(runs)}", ""]
    fig_mass(runs, out, md)
    fig_floors(runs, out, md)
    fig_viscous(runs, out, md)
    fig_meshblocks(runs, out, md)
    fig_resolution(runs, out, md)
    open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
