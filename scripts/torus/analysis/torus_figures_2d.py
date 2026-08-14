"""Figures for the 2D cylindrical torus runs, matching what the 3D tool produces.

    python3 torus_figures_2d.py <run_dir> [-o figdir] [--id acc_disk_visc]

torus_figures.py stays Cartesian and three-dimensional: it reads .athdf and assumes a
cube. The 2D model writes .tab on an (r, phi) grid, which is the right choice there - one
small readable file per snapshot instead of one per MeshBlock - so the two need separate
readers. Everything downstream is shared: the residual definition, the fit window and
the growth figure are imported from the 3D tool rather than reimplemented, and the output
files carry the same names so a run directory looks the same whichever model produced it.

Two differences are unavoidable and deliberate:

  * there is no edge-on view, so the animation has three panels rather than four;
  * the azimuthal residual is exact here rather than binned. In 3D the mean at each
    radius has to be estimated from Cartesian cells falling in a radial bin; on a polar
    grid it is simply the mean over the phi index.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.animation import FFMpegWriter      # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from torus_figures import growth                    # noqa: E402
from torus_modes import load_history, orbital_period   # noqa: E402


def _athena_read():
    p = _HERE
    for _ in range(10):
        cand = os.path.join(p, "vis", "python")
        if os.path.isfile(os.path.join(cand, "athena_read.py")):
            sys.path.insert(0, cand)
            import athena_read
            return athena_read
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    raise FileNotFoundError("vis/python/athena_read.py not found above " + _HERE)


MODE_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}


def snapshots(run_dir, problem_id="acc_disk_visc"):
    """Snapshot files in time order. The 2D runs write one block, hence one file."""
    out = []
    for fn in sorted(glob.glob(os.path.join(run_dir, f"{problem_id}*out1*.tab"))):
        m = re.search(r"\.(\d+)\.tab$", fn)
        if m:
            out.append((int(m.group(1)), fn))
    return [fn for _, fn in sorted(out)]


def read_slice(fn, ar):
    """(time, rho, r, phi) from one .tab snapshot, with rho indexed [phi, r]."""
    d = ar.tab(fn)
    rho = np.asarray(d["rho"])
    r = np.asarray(d["x1v"])[0, :]
    phi = np.asarray(d["x2v"])[:, 0]
    return float(d["time"]), rho, r, phi


def residual(rho):
    """(Sigma - <Sigma>_phi)/<Sigma>_phi. Exact on a polar grid: a mean over phi."""
    bg = rho.mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bg > 0, rho / bg - 1.0, 0.0)


def _polar_mesh(r, phi):
    """Cell-corner Cartesian mesh for pcolormesh, so the annulus is drawn undistorted."""
    dr = np.diff(r).mean()
    dp = np.diff(phi).mean()
    re = np.concatenate([r - 0.5 * dr, [r[-1] + 0.5 * dr]])
    pe = np.concatenate([phi - 0.5 * dp, [phi[-1] + 0.5 * dp]])
    P, R = np.meshgrid(pe, re, indexing="ij")
    return R * np.cos(P), R * np.sin(P)


def _draw_density(ax, X, Y, rho, vmin, vmax):
    im = ax.pcolormesh(X, Y, np.log10(np.maximum(rho, vmin)), cmap="inferno",
                       vmin=np.log10(vmin), vmax=np.log10(vmax), shading="flat")
    ax.plot(0, 0, "o", ms=2.5, color="cyan")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return im


def _draw_residual(ax, X, Y, res):
    im = ax.pcolormesh(X, Y, res, cmap="RdBu_r", vmin=-0.6, vmax=0.6, shading="flat")
    ax.plot(0, 0, "o", ms=2.5, color="k")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return im


def montage(run_dir, orbits, out_path, problem_id="acc_disk_visc"):
    ar = _athena_read()
    snaps = snapshots(run_dir, problem_id)
    if not snaps:
        return None
    T = orbital_period(run_dir)
    t_end = read_slice(snaps[-1], ar)[0] / T
    per_orbit = len(snaps) / (t_end or 1)
    picks = [snaps[min(int(round(o * per_orbit)), len(snaps) - 1)] for o in orbits]

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 5.8), constrained_layout=True)
    vmax = vmin = None
    for c, fn in enumerate(picks):
        t, rho, r, phi = read_slice(fn, ar)
        X, Y = _polar_mesh(r, phi)
        if vmax is None:
            vmax = rho.max()
            vmin = vmax * 3.0e-3
        im = _draw_density(axes[0, c], X, Y, rho, vmin, vmax)
        axes[0, c].set_title(f"{t / T:.0f} $T_{{\\rm orb}}$", fontsize=10)
        imr = _draw_residual(axes[1, c], X, Y, residual(rho))
        if c:
            axes[0, c].set_ylabel("")
            axes[1, c].set_ylabel("")
        axes[0, c].set_xlabel("")
    fig.colorbar(im, ax=axes[0, :].tolist(), label="$\\log_{10}\\Sigma$", fraction=0.02)
    fig.colorbar(imr, ax=axes[1, :].tolist(), label="relative", fraction=0.02)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def multiview(run_dir, out_path, fps=12, problem_id="acc_disk_visc"):
    """Three panels: density, residual, and A_m(t) with a cursor at the current frame."""
    ar = _athena_read()
    snaps = snapshots(run_dir, problem_id)
    if not snaps:
        return None
    T = orbital_period(run_dir)
    hst = os.path.join(run_dir, f"{problem_id}.hst")
    d = load_history(hst) if os.path.isfile(hst) else None

    vmax = None
    for fn in snaps[::max(1, len(snaps) // 12)]:
        v = read_slice(fn, ar)[1].max()
        vmax = v if vmax is None else max(vmax, v)
    vmin = vmax * 3.0e-3

    t0, rho0, r, phi = read_slice(snaps[0], ar)
    X, Y = _polar_mesh(r, phi)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    imf = _draw_density(axes[0], X, Y, rho0, vmin, vmax)
    axes[0].set_title("face-on", fontsize=10)
    fig.colorbar(imf, ax=axes[0], label="$\\log_{10}\\Sigma$", fraction=0.046)
    imr = _draw_residual(axes[1], X, Y, residual(rho0))
    axes[1].set_title("$\\Sigma/\\langle\\Sigma\\rangle_\\varphi - 1$", fontsize=10)
    fig.colorbar(imr, ax=axes[1], label="relative", fraction=0.046)

    cursor = None
    if d is not None:
        for m in (1, 2, 3):
            axes[2].semilogy(d["orbits"], np.maximum(d["A"][m], 1e-6), lw=1.0,
                             color=MODE_COLORS[m], label=f"$A_{m}$")
        axes[2].set_xlabel("$t\\,/\\,T_{\\rm orb}$")
        axes[2].set_ylabel("$A_m\\,/\\,M_{\\rm tor}$")
        axes[2].legend(frameon=False, fontsize=9, loc="lower right")
        axes[2].set_title("mode amplitudes", fontsize=10)
        cursor = axes[2].axvline(0.0, color="0.3", lw=1.2)
    else:
        axes[2].axis("off")

    sup = fig.suptitle("")
    name = os.path.basename(os.path.normpath(run_dir))
    writer = FFMpegWriter(fps=fps, bitrate=3600)
    with writer.saving(fig, out_path, dpi=110):
        for fn in snaps:
            t, rho, _, _ = read_slice(fn, ar)
            imf.set_array(np.log10(np.maximum(rho, vmin)).ravel())
            imr.set_array(residual(rho).ravel())
            if cursor is not None:
                cursor.set_xdata([t / T, t / T])
            sup.set_text(f"{name}     t = {t / T:.1f} $T_{{\\rm orb}}$")
            writer.grab_frame()
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--id", default="acc_disk_visc")
    ap.add_argument("--orbits", default="0,5,10,15,20,30")
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run_dir, "figs")
    os.makedirs(out, exist_ok=True)
    made = [montage(args.run_dir, [float(v) for v in args.orbits.split(",")],
                    os.path.join(out, "montage.png"), args.id)]

    hst = os.path.join(args.run_dir, f"{args.id}.hst")
    if os.path.isfile(hst):
        made.append(growth([(os.path.basename(os.path.normpath(args.run_dir)),
                             args.run_dir)], os.path.join(out, "growth.png"),
                           problem_id=args.id))
    if not args.no_movie:
        try:
            made.append(multiview(args.run_dir, os.path.join(out, "multiview.mp4"),
                                  problem_id=args.id))
        except Exception as exc:
            print(f"multiview skipped: {exc}", file=sys.stderr)

    for f in made:
        if f:
            print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
