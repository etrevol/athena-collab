"""Figures for the self-gravitating / Papaloizou-Pringle torus runs.

    python3 torus_figures.py <run_dir> [<run_dir2>] [-o figdir]

Complements torus_modes.py, which reports the numbers. This module draws the pictures:

    montage      face-on column density at a sequence of times - the classic view of
                 an m = 1 mode growing on a torus and then destroying it
    compare      two runs side by side at the same times, which is how the effect of
                 self-gravity on the instability is actually seen
    growth       ln A_1 against time with the fitted exponential, plus the grid's own
                 m = 4 imprint drawn as a floor, so the window where the measurement
                 means anything is visible rather than asserted
    movie        the growth phase as an mp4

One convention throughout: the colour scale is fixed across all panels of a figure, and
taken from the FIRST panel. Letting each panel autoscale makes a torus that is losing
mass look like a torus that is merely changing shape.
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
from matplotlib.animation import FFMpegWriter   # noqa: E402


def _athena_read():
    here = os.path.dirname(os.path.abspath(__file__))
    p = here
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
    raise FileNotFoundError("vis/python/athena_read.py not found above " + here)


T_ORB = 2.0 * np.pi


def snapshots(run_dir, problem_id="sg_torus_m1"):
    """Every snapshot in a run, as (orbit, path), sorted in time."""
    out = []
    for fn in sorted(glob.glob(os.path.join(run_dir, f"{problem_id}.out1.*.athdf"))):
        idx = int(re.search(r"\.(\d+)\.athdf$", fn).group(1))
        out.append((idx, fn))
    return out


def _columns(fn, ar):
    """Face-on and edge-on column density of one snapshot, plus the time."""
    d = ar.athdf(fn, quantities=["rho"])
    rho = d["rho"]                                    # (z, y, x)
    x, y, z = d["x1v"], d["x2v"], d["x3v"]
    face = rho.sum(axis=0) * (z[1] - z[0])            # (y, x)
    edge = rho.sum(axis=1) * (y[1] - y[0])            # (z, x)
    return d["Time"] / T_ORB, face, edge, x, y, z


def azimuthal_residual(face, x, y, nbins=60):
    """(Sigma - <Sigma>_phi) / <Sigma>_phi, the map on which an m = 1 mode is visible.

    A 13% lopsidedness is invisible on a log density map spanning four decades, because
    the eye is spending its dynamic range on the radial profile. Dividing out the
    azimuthal mean at each radius removes exactly that profile and leaves the mode.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.hypot(X, Y)
    edges = np.linspace(0.0, R.max(), nbins + 1)
    idx = np.clip(np.digitize(R, edges) - 1, 0, nbins - 1)
    mean = np.zeros(nbins)
    for b in range(nbins):
        sel = idx == b
        if sel.any():
            mean[b] = face[sel].mean()
    bg = mean[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(bg > 0, face / bg - 1.0, 0.0)
    return res


def _draw(ax, img, extent, vmin, vmax, labels):
    im = ax.imshow(np.log10(np.maximum(img, vmin)), origin="lower", extent=extent,
                   cmap="inferno", aspect="equal",
                   vmin=np.log10(vmin), vmax=np.log10(vmax))
    ax.plot(0, 0, "o", ms=2.5, color="cyan")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return im


def montage(run_dir, orbits, out_path, problem_id="sg_torus_m1", title=None):
    """Face-on column density at a sequence of orbits, on one shared colour scale."""
    ar = _athena_read()
    snaps = snapshots(run_dir, problem_id)
    if not snaps:
        return None
    # snapshot cadence is fixed, so the index follows from the requested orbit
    per_orbit = len(snaps) / (_columns(snaps[-1][1], ar)[0] or 1)
    picks = []
    for orb in orbits:
        i = int(round(orb * per_orbit))
        i = min(max(i, 0), len(snaps) - 1)
        picks.append(snaps[i][1])

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 5.8), constrained_layout=True)
    vmax = vmin = None
    for c, fn in enumerate(picks):
        t, face, _, x, y, _ = _columns(fn, ar)
        if vmax is None:
            vmax = face.max()
            vmin = vmax * 3.0e-3     # one and a half decades: the torus, not the vacuum
        im = _draw(axes[0, c], face.T, [x[0], x[-1], y[0], y[-1]], vmin, vmax,
                   ("", "$y$"))
        axes[0, c].set_title(f"{t:.0f} $T_{{\\rm orb}}$", fontsize=10)

        res = azimuthal_residual(face.T, x, y)
        imr = axes[1, c].imshow(res.T, origin="lower",
                                extent=[x[0], x[-1], y[0], y[-1]],
                                cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="equal")
        axes[1, c].plot(0, 0, "o", ms=2.5, color="k")
        axes[1, c].set_xlabel("$x$")
        if c == 0:
            axes[0, c].set_ylabel("$\\Sigma$\n$y$")
            axes[1, c].set_ylabel("$\\Sigma/\\langle\\Sigma\\rangle_\\varphi - 1$\n$y$")
        else:
            axes[0, c].set_ylabel("")
            axes[1, c].set_ylabel("")
    fig.colorbar(im, ax=axes[0, :].tolist(), label="$\\log_{10}\\Sigma$", fraction=0.02)
    fig.colorbar(imr, ax=axes[1, :].tolist(), label="relative", fraction=0.02)
    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compare(run_a, run_b, orbits, out_path, labels=("canonical", "no self-gravity"),
            problem_id="sg_torus_m1"):
    """Two runs, same times, same scale, as azimuthal residuals.

    The residual rather than Sigma itself, for the same reason as in the montage: the
    two runs differ by well under a factor of two in amplitude, which is invisible on a
    log density map but obvious once the axisymmetric profile is divided out.
    """
    ar = _athena_read()
    rows = []
    for run in (run_a, run_b):
        snaps = snapshots(run, problem_id)
        if not snaps:
            return None
        per_orbit = len(snaps) / (_columns(snaps[-1][1], ar)[0] or 1)
        rows.append([snaps[min(int(round(o * per_orbit)), len(snaps) - 1)][1]
                     for o in orbits])

    n = len(orbits)
    fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 6.0), constrained_layout=True)
    vmax = vmin = None
    for r, (row, lab) in enumerate(zip(rows, labels)):
        for c, fn in enumerate(row):
            t, face, _, x, y, _ = _columns(fn, ar)
            ax = axes[r, c]
            res = azimuthal_residual(face.T, x, y)
            im = ax.imshow(res.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                           cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="equal")
            ax.plot(0, 0, "o", ms=2.5, color="k")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            if r == 0:
                ax.set_title(f"{t:.0f} $T_{{\\rm orb}}$", fontsize=10)
            else:
                ax.set_title("")
            if c == 0:
                ax.set_ylabel(f"{lab}\n$y$")
            else:
                ax.set_ylabel("")
            if r == 0:
                ax.set_xlabel("")
    fig.colorbar(im, ax=axes.ravel().tolist(),
                 label="$\\Sigma/\\langle\\Sigma\\rangle_\\varphi - 1$",
                 fraction=0.02)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def linear_window(o, A1, lo_factor=20.0, hi_fraction=0.3):
    """The interval over which A_1 is genuinely exponential, chosen by the data.

    Hand-picking the fit window is not harmless here: extending it from orbit 12 to 18
    pulls the measured rate from 0.72 to 0.47 per orbit, because the later points are
    already saturating. So the window is defined by the physics instead - from where
    the mode has climbed well clear of its starting value, to where it has reached a
    fixed fraction of its peak and linear theory stops applying. The fit residual is
    reported alongside, and it is the number that says whether the choice was sound.
    """
    peak = float(np.nanmax(A1))
    start = A1[0] * lo_factor
    w = np.zeros_like(o, dtype=bool)
    above = np.flatnonzero(A1 > start)
    if above.size == 0:
        return w
    i0 = int(above[0])
    # the FIRST crossing of the saturation threshold, not the last: A_1 falls back
    # through it again once the mode decays, and taking the last one would fit the
    # rise and the collapse together and return a growth rate of about zero.
    sat = np.flatnonzero(A1[i0:] >= hi_fraction * peak)
    i1 = i0 + int(sat[0]) if sat.size else len(A1) - 1
    if i1 - i0 < 5:
        return w
    w[i0:i1 + 1] = True
    return w & (A1 > 0)


def growth(runs, out_path, fit_window=None):
    """ln A_1 against time with the fitted exponential, and the m = 4 grid floor.

    The floor is what makes this figure worth drawing: a Cartesian mesh stamps an m = 4
    pattern on a circular torus before anything evolves, and a measurement of A_1 only
    means something while it sits well above that.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from torus_modes import load_history

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    text = []
    for (label, run), col in zip(runs, colors):
        hst = os.path.join(run, "sg_torus_m1.hst")
        if not os.path.isfile(hst):
            continue
        d = load_history(hst)
        o, A = d["orbits"], d["A"]
        ax.semilogy(o, A[1], color=col, lw=1.2, label=f"{label}: $A_1$")
        ax.semilogy(o, A[4], color=col, lw=0.8, ls=":", alpha=0.7,
                    label=f"{label}: $A_4$ (grid)")
        if fit_window is not None:
            w = (o > fit_window[0]) & (o < fit_window[1]) & (A[1] > 0)
        else:
            w = linear_window(o, A[1])
        if w.sum() > 5:
            c, res, *_ = np.polyfit(o[w], np.log(A[1][w]), 1, full=True)
            rms = float(np.sqrt(res[0] / w.sum())) if len(res) else float("nan")
            ax.semilogy(o[w], np.exp(np.polyval(c, o[w])), color="k", lw=2.0,
                        alpha=0.55)
            ax.axvspan(o[w][0], o[w][-1], color=col, alpha=0.10, zorder=0)
            text.append(f"{label}: $\\sigma$ = {c[0]:.2f}/orbit "
                        f"= {c[0] / (2 * np.pi):.3f}$\\,\\Omega$  "
                        f"(orbits {o[w][0]:.0f}-{o[w][-1]:.0f}, rms {rms:.2f})")
    ax.set_xlim(0, 60)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel("$t\\,/\\,T_{\\rm orb}$")
    ax.set_ylabel("$A_m\\,/\\,M_{\\rm tor}$")
    ax.set_title("m = 1 growth, against the grid's own m = 4 imprint")
    if text:
        ax.text(0.98, 0.04, "\n".join(text), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def movie(run_dir, out_path, max_orbit=40.0, fps=12, problem_id="sg_torus_m1"):
    """The growth phase as an mp4, face-on, on a fixed colour scale."""
    ar = _athena_read()
    snaps = snapshots(run_dir, problem_id)
    if not snaps:
        return None
    total = _columns(snaps[-1][1], ar)[0]
    per_orbit = len(snaps) / (total or 1)
    keep = [fn for i, (_, fn) in enumerate(snaps) if i / per_orbit <= max_orbit]

    t0, face0, _, x, y, _ = _columns(keep[0], ar)
    vmax = None
    for fn in keep[::max(1, len(keep) // 12)]:      # scale from the brightest frame
        v = _columns(fn, ar)[1].max()
        vmax = v if vmax is None else max(vmax, v)
    vmin = vmax * 1e-4

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    im = _draw(ax, face0.T, [x[0], x[-1], y[0], y[-1]], vmin, vmax, ("$x$", "$y$"))
    fig.colorbar(im, ax=ax, label="$\\log_{10}\\Sigma$")
    title = ax.set_title("")

    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, out_path, dpi=130):
        for fn in keep:
            t, face, _, _, _, _ = _columns(fn, ar)
            im.set_data(np.log10(np.maximum(face.T, vmin)))
            title.set_text(f"t = {t:.1f} $T_{{\\rm orb}}$")
            writer.grab_frame()
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories; the first is the primary")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--orbits", default="0,10,15,20,25,40")
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()

    orbits = [float(v) for v in args.orbits.split(",")]
    out = args.out or os.path.join(args.runs[0], "figs")
    os.makedirs(out, exist_ok=True)
    made = []

    m = montage(args.runs[0], orbits, os.path.join(out, "montage.png"),
                title="face-on column density")
    made.append(m)

    if len(args.runs) > 1:
        made.append(compare(args.runs[0], args.runs[1], orbits,
                            os.path.join(out, "compare.png")))

    labelled = [(os.path.basename(os.path.normpath(r)), r) for r in args.runs]
    made.append(growth(labelled, os.path.join(out, "growth.png")))

    if not args.no_movie:
        try:
            made.append(movie(args.runs[0], os.path.join(out, "growth.mp4")))
        except Exception as exc:                      # ffmpeg missing, codec, ...
            print(f"movie skipped: {exc}", file=sys.stderr)

    for f in made:
        if f:
            print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
