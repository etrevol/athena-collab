"""Azimuthal mode analysis of the self-gravitating torus runs.

    python3 torus_modes.py <run_dir> [-o figs] [--id sg_torus_m1]

Reproduces, for the hydrodynamic torus, the diagnostics Bannikova et al. (2026,
arXiv:2604.11528) apply to their N-body torus:

    their figure   what it shows                         made here by
    ------------------------------------------------------------------------
    Fig. 2, 9      density maps, face-on and edge-on      density_maps()
    Fig. 3         central-mass / barycentre trajectory   trajectory()
    Fig. 4         virial quantity and the displacement   virial()
    Fig. 7         A_m(t) for m = 1, 2, 3 on log time     mode_amplitudes()
    Fig. 8         a_m(t) together with x_c(t)            mode_coefficients()
    Fig. 18        the ratios A_1/A_2 and A_2/A_3         amplitude_ratios()
    Table 2        saturated A_m, k = A_2/A_1, Q_3        saturation_table()

Nearly all of it comes from the history file rather than from snapshots: the pgen
writes the Fourier coefficients a_m, b_m as per-cycle volume sums, so the mode
evolution is sampled hundreds of times per orbit instead of twice. Only the density
maps need the .athdf output.

One normalisation note that makes the comparison legitimate. The paper counts particles
per azimuthal sector, so its A_m are numbers of particles; here they are masses. The
amplitudes therefore cannot be compared directly, but every quantity the paper actually
draws conclusions from - the ratio k = A_2/A_1, the geometric-scaling diagnostic
Q_3 = (A_3/A_1)/(A_2/A_1)^2, and the normalised amplitude A_1/Sigma_0 - is a ratio, and
ratios carry over unchanged. A_m here is normalised by the torus mass, so A_1 is exactly
the paper's Ã_1 = A_1/Sigma_0 of their Eq. (13).
"""

import argparse
import re
import os
import sys

import numpy as np

# matplotlib without a display, the same way the vis scripts do it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


M_MAX = 5


def _load_athena_read():
    """Import Athena++'s own reader from vis/python, wherever the repo root is."""
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


# -- history ------------------------------------------------------------------

def read_hst(path):
    """Tolerant reader for Athena++ .hst files.

    athena_read.hst is strict, and a run that is interrupted - a crash, a kill, a laptop
    losing power mid-write - leaves a final line that is truncated or padded with NULs,
    which makes the strict reader throw away the whole file. Since the point of the
    history file is to survive exactly those events, incomplete trailing rows are dropped
    with a warning rather than treated as a fatal error.
    """
    names, rows, bad = None, [], 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                if "[1]=" in line:      # the column-name header
                    names = re.findall(r"\[\d+\]=(\S+)", line)
                continue
            if "\x00" in line:
                bad += 1
                continue
            parts = line.split()
            if names is None or len(parts) != len(names):
                bad += 1
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                bad += 1
    if names is None:
        raise ValueError(f"no column header found in {path}")
    if bad:
        print(f"note: dropped {bad} incomplete row(s) from {os.path.basename(path)} "
              "- the run did not shut down cleanly", file=sys.stderr)
    if not rows:
        raise ValueError(f"no usable rows in {path}")
    arr = np.array(rows)
    return {n: arr[:, i] for i, n in enumerate(names)}


def load_history(path):
    """Read the .hst file and derive everything the figures need.

    Returns a dict with, besides the raw columns:
        A[m]     normalised amplitude of harmonic m (m = 1..5), = A_m / M_tor
        Phi[m]   its phase
        r_tb     torus barycentre, (t, 3)
        r_c      central-mass displacement = -(M_tor/M_c) r_tb, from the centre-of-mass
                 relation the paper uses in its Sect. 3.1
        virial   2 E_kin + E_pot
    """
    h = read_hst(path)

    t = np.asarray(h["time"], dtype=float)
    mass = np.asarray(h["tor_mass"], dtype=float)
    safe = np.where(mass > 0.0, mass, np.nan)

    r_tb = np.stack([np.asarray(h["tor_m" + c], dtype=float) / safe
                     for c in ("x", "y", "z")], axis=1)

    A, Phi = {}, {}
    for m in range(1, M_MAX + 1):
        a = np.asarray(h[f"a{m}"], dtype=float) / safe
        b = np.asarray(h[f"b{m}"], dtype=float) / safe
        A[m] = np.hypot(a, b)
        Phi[m] = np.arctan2(b, a)

    E_kin = np.asarray(h["E_kin"], dtype=float)
    E_pot = (np.asarray(h["E_grav_c"], dtype=float)
             + np.asarray(h["E_grav_self"], dtype=float))

    # The grid pins the point mass at the origin, so its displacement is not simulated
    # directly; it follows from momentum conservation, exactly as in the paper.
    M_c = 1.0
    return dict(t=t, mass=mass, r_tb=r_tb, r_c=-(mass / M_c)[:, None] * r_tb,
                A=A, Phi=Phi, a={m: np.asarray(h[f"a{m}"]) / safe
                                 for m in range(1, M_MAX + 1)},
                E_kin=E_kin, E_pot=E_pot, virial=2.0 * E_kin + E_pot,
                E_int=np.asarray(h["E_int"], dtype=float),
                orbits=t / (2.0 * np.pi))


# -- figures ------------------------------------------------------------------

_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}


def mode_amplitudes(d, ax=None):
    """A_m(t) for m = 1..3 on a logarithmic time axis. Their Fig. 7."""
    ax = ax or plt.gca()
    for m in (1, 2, 3):
        ax.plot(d["orbits"], d["A"][m], color=_COLORS[m], lw=1.0, label=f"$A_{m}$")
    ax.set_xscale("log")
    ax.set_xlabel("$t\\,/\\,T_{\\rm orb}$")
    ax.set_ylabel("$A_m\\,/\\,M_{\\rm tor}$")
    ax.legend(frameon=False)
    ax.set_title("mode amplitudes")
    return ax


def mode_coefficients(d, ax=None):
    """a_m(t) for m = 1..3 with the central-mass x-coordinate. Their Fig. 8."""
    ax = ax or plt.gca()
    for m in (1, 2, 3):
        ax.plot(d["orbits"], d["a"][m], color=_COLORS[m], lw=0.9, label=f"$a_{m}$")
    ax.set_xlabel("$t\\,/\\,T_{\\rm orb}$")
    ax.set_ylabel("$a_m\\,/\\,M_{\\rm tor}$")
    ax.legend(frameon=False, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(d["orbits"], d["r_c"][:, 0], color="0.35", lw=0.9, ls="--")
    ax2.set_ylabel("$x_c$  (dashed)")
    ax.set_title("mode coefficients and the central-mass offset")
    return ax


def amplitude_ratios(d, ax=None):
    """A_1/A_2 and A_2/A_3. Their Fig. 18; the paper's k is the inverse of the first."""
    ax = ax or plt.gca()
    with np.errstate(divide="ignore", invalid="ignore"):
        ax.plot(d["orbits"], d["A"][1] / d["A"][2], color=_COLORS[1], lw=1.0,
                label="$A_1/A_2$")
        ax.plot(d["orbits"], d["A"][2] / d["A"][3], color=_COLORS[2], lw=1.0,
                label="$A_2/A_3$")
    ax.axhline(1.0 / 0.22, color="0.6", lw=0.8, ls=":",
               label="$1/k$, $k = 0.22$ (their Table 2)")
    ax.set_ylim(0, 20)
    ax.set_xlabel("$t\\,/\\,T_{\\rm orb}$")
    ax.set_ylabel("amplitude ratio")
    ax.legend(frameon=False)
    ax.set_title("geometric harmonic scaling")
    return ax


def run_parameters(run_dir):
    """q_rot and eps_soft from the run's own athinput, for the corotation radius."""
    fn = os.path.join(run_dir, "athinput")
    out = {"q_rot": 0.0, "eps_soft": 0.0}
    if os.path.isfile(fn):
        txt = open(fn).read()
        for k in out:
            m = re.search(rf"^{k}\s*=\s*([0-9.eE+-]+)", txt, re.M)
            if m:
                out[k] = float(m.group(1))
    return out


def corotation_radius(omega_p, q_rot=0.0, eps_soft=0.0):
    """Where the gas rotates at the pattern speed.

    The rotation law is l = l_0 R^q with l_0^2 = (1 + eps^2)^(-3/2), so
    Omega(R) = l/R^2 = l_0 R^(q-2) -- NOT R^(-3/2) unless q = 0. Using the Keplerian
    form regardless of q would misplace the corotation radius, which is the one number
    that decides whether a mode is the Papaloizou-Pringle instability or a slow mode.
    """
    if omega_p <= 0.0:
        return float("nan")
    l0 = (1.0 + eps_soft**2) ** -0.75
    return (l0 / omega_p) ** (1.0 / (2.0 - q_rot))


def pattern_speed(d, smooth=51):
    """Omega_p = d Phi_1 / dt, in units of the orbital frequency at R_tor.

    This is the discriminator the collisionless study never needed. A gas torus with
    l = const is subject to the Papaloizou-Pringle instability, whose fastest growing
    mode is also m = 1 and which requires no self-gravity at all. The two are told
    apart by how fast the pattern turns: the PP instability corotates with the fluid
    somewhere inside the torus, so Omega_p is of order the orbital frequency, whereas
    the slow mode of the paper is secular, Omega_p << Omega. A measured Omega_p near 1
    means the hydrodynamic instability, not the paper's mechanism, is what grew.
    """
    phi = np.unwrap(d["Phi"][1])
    t = d["t"]
    om = np.gradient(phi, t)                      # Omega at R_tor is 1 in these units
    if smooth > 1 and len(om) > smooth:
        kern = np.ones(smooth) / smooth
        om = np.convolve(om, kern, mode="same")
    return om


def pattern_speed_figure(d, ax=None):
    """Omega_p(t) against the two hypotheses it separates."""
    ax = ax or plt.gca()
    ax.plot(d["orbits"], pattern_speed(d), color=_COLORS[1], lw=1.0)
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--",
               label="$\\Omega(R_{\\rm tor})$: Papaloizou-Pringle instability")
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":", label="secular slow mode")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("$t\\,/\\,T_{\\rm orb}$")
    ax.set_ylabel("$\\Omega_p$")
    ax.legend(frameon=False)
    ax.set_title("pattern speed of the $m=1$ mode")
    return ax


def trajectory(d, ax=None):
    """The barycentre and the central mass, in anti-phase. Their Fig. 3."""
    ax = ax or plt.gca()
    ax.plot(d["mass"] * d["r_tb"][:, 0], d["mass"] * d["r_tb"][:, 1],
            color=_COLORS[1], lw=0.7, label="$M_{\\rm tor}\\,r_{\\rm tb}$")
    ax.plot(d["r_c"][:, 0], d["r_c"][:, 1], color=_COLORS[2], lw=0.7,
            label="$M_c\\,r_c$")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(frameon=False)
    ax.set_title("barycentre and central mass")
    return ax


def virial(d, ax=None):
    """2 E_kin + E_pot against the growth of the displacement. Their Fig. 4.

    The paper's point is the ordering: the displacement only starts growing after the
    virial quantity has settled, which is what makes the mode spontaneous rather than
    a transient response to the initial conditions.
    """
    ax = ax or plt.gca()
    ax.plot(d["orbits"], d["virial"], color=_COLORS[2], lw=1.0,
            label="$2E_{\\rm kin}+E_{\\rm pot}$")
    ax.set_xscale("log")
    ax.set_xlabel("$t\\,/\\,T_{\\rm orb}$")
    ax.set_ylabel("virial quantity")
    ax2 = ax.twinx()
    ax2.plot(d["orbits"], np.linalg.norm(d["r_c"], axis=1), color=_COLORS[1], lw=1.0)
    ax2.set_ylabel("$|r_c|$")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("virialisation, then the mode")
    return ax


def saturation_table(d, last_fraction=0.25):
    """Time-averaged amplitudes over the final stretch of the run. Their Table 2."""
    n = len(d["t"])
    sl = slice(int((1.0 - last_fraction) * n), n)
    A = {m: float(np.nanmean(d["A"][m][sl])) for m in range(1, M_MAX + 1)}
    k = A[2] / A[1] if A[1] > 0 else np.nan
    Q3 = (A[3] / A[1]) / k**2 if A[1] > 0 and k > 0 else np.nan
    om_p = float(np.nanmean(pattern_speed(d)[sl]))
    return dict(A=A, k=k, Q3=Q3, omega_p=om_p,
                window=(float(d["orbits"][sl][0]), float(d["orbits"][-1])))


def format_table(s, pars=None):
    verdict = ("secular slow mode" if abs(s["omega_p"]) < 0.2 else
               "FAST: this looks like the Papaloizou-Pringle instability, "
               "not the secular mode")
    L = [f"saturated state, averaged over orbits "
         f"{s['window'][0]:.0f}-{s['window'][1]:.0f}",
         "  " + "  ".join(f"A{m}={s['A'][m]:.4f}" for m in range(1, M_MAX + 1)),
         f"  k = A2/A1 = {s['k']:.3f}   (their runs: 0.20-0.25)",
         f"  Q3 = (A3/A1)/k^2 = {s['Q3']:.3f}   (geometric spectrum: 1)",
         f"  Omega_p = {s['omega_p']:.3f}   ({verdict})"]
    if pars is not None:
        rco = corotation_radius(s["omega_p"], **pars)
        L.append(f"  corotation R = {rco:.2f}   "
                 f"(Omega = l0 R^(q-2) with q = {pars['q_rot']}, not Keplerian)")
    return "\n".join(L)


# -- snapshots ----------------------------------------------------------------

def density_maps(run_dir, problem_id, index, out_path):
    """Face-on and edge-on column density of one snapshot. Their Fig. 2."""
    athena_read = _load_athena_read()
    fn = os.path.join(run_dir, f"{problem_id}.out1.{index:05d}.athdf")
    if not os.path.isfile(fn):
        return None
    data = athena_read.athdf(fn, quantities=["rho"])
    rho = data["rho"]                      # (z, y, x)
    x, y, z = data["x1v"], data["x2v"], data["x3v"]
    dz = z[1] - z[0]
    dy = y[1] - y[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    face = rho.sum(axis=0) * dz            # integrated over z
    edge = rho.sum(axis=1) * dy            # integrated over y
    for ax, img, ext, lab in (
            (axes[0], face, [x[0], x[-1], y[0], y[-1]], ("$x$", "$y$")),
            (axes[1], edge, [x[0], x[-1], z[0], z[-1]], ("$x$", "$z$"))):
        pos = img > 0
        vmax = img[pos].max() if pos.any() else 1.0
        im = ax.imshow(np.log10(np.maximum(img, vmax * 1e-6)).T, origin="lower",
                       extent=ext, cmap="inferno", aspect="equal")
        ax.plot(0, 0, "o", ms=3, color="cyan")
        ax.set_xlabel(lab[0])
        ax.set_ylabel(lab[1])
        fig.colorbar(im, ax=ax, label="$\\log_{10}\\Sigma$")
    axes[0].set_title(f"face-on, t = {data['Time']:.1f} = "
                      f"{data['Time'] / (2 * np.pi):.1f} $T_{{\\rm orb}}$")
    axes[1].set_title("edge-on")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# -- driver -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None, help="figure directory")
    ap.add_argument("--id", default="sg_torus_m1", help="problem_id")
    ap.add_argument("--last", type=float, default=0.25,
                    help="fraction of the run treated as saturated")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run_dir, "figs")
    os.makedirs(out, exist_ok=True)

    hst = os.path.join(args.run_dir, f"{args.id}.hst")
    if not os.path.isfile(hst):
        sys.exit(f"no history file at {hst}")
    d = load_history(hst)

    panels = [("modes", mode_amplitudes), ("coefficients", mode_coefficients),
              ("ratios", amplitude_ratios), ("trajectory", trajectory),
              ("virial", virial), ("pattern_speed", pattern_speed_figure)]
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        fn(d, ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=140)
        plt.close(fig)

    # the last snapshot that exists
    import glob
    snaps = sorted(glob.glob(os.path.join(args.run_dir, f"{args.id}.out1.*.athdf")))
    if snaps:
        idx = int(os.path.basename(snaps[-1]).split(".")[-2])
        density_maps(args.run_dir, args.id, idx, os.path.join(out, "density.png"))

    print(format_table(saturation_table(d, args.last), run_parameters(args.run_dir)))
    print(f"\n{len(panels) + bool(snaps)} figures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
