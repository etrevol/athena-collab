"""Write the README.md that says what a run was and what came out of it.

    python3 describe_run.py <run_dir> [--purpose "one line"]

Called automatically by sg_campaign.sh, once when a run starts and again when it
finishes, so every run directory explains itself months later without anyone having to
remember what "thin" meant. Run it by hand to refresh a description after re-analysing.

Everything except the purpose line is derived from files already in the directory - the
athinput for the configuration, the .hst for the outcome - so the description cannot
drift away from the run it describes. The purpose line is the one thing a machine cannot
infer; it is preserved across regenerations once written.

The last section is the useful one. A run can complete without being worth anything: the
torus can drain away, the floors can start doing physics, the mode can never leave its
seed. Those are checked here and stated at the top of the file rather than left for
someone to rediscover.
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "analysis"))


def read_athinput(path):
    """Flat dict of every parameter in the file; later blocks win on name clashes."""
    out, block = {}, ""
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("<"):
            block = line.strip("<>")
            continue
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            out[f"{block}/{k}"] = v
            out.setdefault(k, v)
    return out


def _f(d, key, default=None):
    try:
        return float(d[key])
    except (KeyError, ValueError, TypeError):
        return default


def geometry(p):
    """r_in, r_out, z_max and the paper's equivalent inclination, from the parameters."""
    C = _f(p, "C_prime")
    q = _f(p, "q_rot", 0.0)
    eps = _f(p, "eps_soft", 0.0)
    if C is None:
        return {}
    l0sq = (1.0 + eps * eps) ** -1.5

    def Psi(R, z):
        return (1.0 / np.sqrt(R * R + z * z + eps * eps)
                - l0sq * R ** (2 * q - 2) / (2 - 2 * q) - C)

    R = np.linspace(1e-3, 60.0, 200000)
    s = Psi(R, 0.0) > 0
    if not s.any():
        return {}
    r_in, r_out = R[s][0], R[s][-1]
    Rz = np.linspace(r_in, r_out, 4000)
    zmax = 0.0
    for r in Rz:
        z = np.linspace(0.0, 30.0, 3000)
        pz = Psi(r, z)
        if (pz > 0).any():
            zmax = max(zmax, z[pz > 0][-1])
    e_max = 0.5                                    # their canonical eccentricity spread
    s_i = zmax / (1.0 + 0.5 * e_max)
    return dict(r_in=r_in, r_out=r_out, z_max=zmax, Psi_c=float(Psi(1.0, 0.0)),
                i_max=float(np.degrees(np.arcsin(s_i))) if s_i <= 1.0 else float("nan"))


def outcome(run_dir, problem_id):
    """Measured quantities, or None when the run has not produced a history file."""
    hst = os.path.join(run_dir, f"{problem_id}.hst")
    if not os.path.isfile(hst):
        return None
    from torus_modes import (read_hst, load_history, pattern_speed,
                             corotation_radius, run_parameters)
    from torus_figures import linear_window

    h = read_hst(hst)
    d = load_history(hst)
    o, A = d["orbits"], d["A"]
    m = np.asarray(h["mass"])
    om = pattern_speed(d)

    sigma = float("nan")
    try:
        w = linear_window(o, A[1])
        if w.sum() > 3:
            sigma = float(np.polyfit(o[w], np.log(A[1][w]), 1)[0])
    except Exception:
        pass

    late = o > 0.7 * o[-1]
    k = (np.mean(A[2][late]) / np.mean(A[1][late])) if np.mean(A[1][late]) > 0 else np.nan
    om_late = float(np.mean(om[late]))
    return dict(
        orbits=float(o[-1]), mass_drift=float(m[-1] / m[0] - 1.0),
        sigma=sigma, peak_A1=float(A[1].max()), peak_orbit=float(o[np.argmax(A[1])]),
        late_A1=float(np.mean(A[1][late])), late_sd=float(np.std(A[1][late])),
        omega_p=om_late,
        R_co=float(corotation_radius(om_late, **run_parameters(run_dir))),
        k=float(k), r_tb=float(np.mean(np.linalg.norm(d["r_tb"], axis=1)[late])),
        n_dfloor=float(h["n_dfloor"][-1]) if "n_dfloor" in h else 0.0,
        v_max=float(np.max(h["v_max"])) if "v_max" in h else float("nan"),
    )


def trust(p, geo, res):
    """What would make this run unusable. Empty list means nothing was detected."""
    warn = []
    if res is None:
        return ["not finished yet - no history file"]
    if abs(res["mass_drift"]) > 0.10:
        warn.append(f"lost {abs(res['mass_drift'])*100:.0f}% of its mass; the torus did "
                    "not survive, so late-time numbers describe the remains")
    if res["n_dfloor"] > 0:
        warn.append(f"density floor fired {res['n_dfloor']:.2e} times; check where "
                    "before trusting anything (in the vacuum it is harmless)")
    alpha = _f(p, "alpha", 0.0)
    if alpha and geo:
        gam = _f(p, "gamma", 5 / 3.)
        n = 1.0 / (gam - 1.0)
        nu = alpha * gam * geo["Psi_c"] / (n + 1.0)
        dr = np.sqrt(nu * res["orbits"] * 2 * np.pi)
        L = _f(p, "x1max", 0.0)
        if geo["r_out"] + dr > L:
            warn.append(f"viscous spreading reaches r = {geo['r_out']+dr:.2f} against a "
                        f"box half-width of {L:.2f}; mass leaves through the faces and "
                        "the mode cannot be told apart from the loss of the torus")
    if res["orbits"] < 50:
        warn.append(f"only {res['orbits']:.0f} orbits; 50 is the screening minimum and "
                    "100 the reference, see docs/WORKFLOW.md")
    if res["late_A1"] < 3.0 * _f(p, "pert_mode_amp", 0.0) + 1e-12 and res["orbits"] > 20:
        warn.append("the m = 1 amplitude never rose far above its seed - no mode grew")
    return warn


def render(run_dir, purpose, problem_id="sg_torus_m1"):
    p = read_athinput(os.path.join(run_dir, "athinput"))
    geo = geometry(p)
    res = outcome(run_dir, problem_id)
    name = os.path.basename(os.path.normpath(run_dir))
    L = [f"# Run `{name}`", ""]
    if purpose:
        L += [f"**Purpose.** {purpose}", ""]

    warn = trust(p, geo, res)
    if warn:
        L += ["> **Read before using these numbers**"] + \
             [f"> - {w}" for w in warn] + [""]

    sg = "on" if _f(p, "self_grav", 1) else "off"
    alpha = _f(p, "alpha", 0.0)
    L += ["## Configuration", "",
          "| parameter | value | meaning |",
          "|---|---|---|",
          f"| `q_rot` | {p.get('q_rot','0')} | rotation law `l ~ R^q`; 0 is constant "
          "angular momentum, 0.5 Keplerian |",
          f"| `C_prime` | {p.get('C_prime','?')} | thickness parameter |",
          f"| `M_tor` | {p.get('M_tor','?')} | torus mass in units of the central mass |",
          f"| self-gravity | **{sg}** | Poisson solver; off reproduces their Appendix C |",
          f"| `alpha` | {alpha:g} | viscosity; 0.01 is one cloud collision per ~16 orbits |",
          f"| seed | white {p.get('pert_amp','0')}, coherent "
          f"{p.get('pert_mode_amp','0')} | coherent seeds m = 1..5 with random phases |",
          f"| grid | {p.get('mesh/nx1','?')}^3, box +-{p.get('x1max','?')} | "
          f"eps_soft = {p.get('eps_soft','?')} |",
          ""]
    if geo:
        L += ["## Geometry", "",
              f"- mid-plane edges `r_in`..`r_out` = {geo['r_in']:.3f} .. "
              f"{geo['r_out']:.3f}",
              f"- half-thickness `z_max` = {geo['z_max']:.3f}, equivalent to "
              f"`i_max` = {geo['i_max']:.0f} deg in their model "
              "(strong at 60, weak at 45, absent at 30)",
              f"- `Psi_c` = {geo['Psi_c']:.4f}  (pressure support; must stay positive)",
              ""]
    if res:
        L += ["## Outcome", "",
              "| quantity | value |",
              "|---|---|",
              f"| orbits completed | {res['orbits']:.0f} |",
              f"| mass drift | {res['mass_drift']*100:+.1f}% |",
              f"| growth rate | {res['sigma']:.3f} per orbit |",
              f"| peak `A_1` | {res['peak_A1']:.4f} at orbit {res['peak_orbit']:.0f} |",
              f"| late `A_1` | {res['late_A1']:.4f} +- {res['late_sd']:.4f} |",
              f"| pattern speed | {res['omega_p']:.3f}, corotation at R = "
              f"{res['R_co']:.2f} |",
              f"| `k = A_2/A_1` | {res['k']:.3f}  (theirs: 0.20-0.25) |",
              f"| barycentre `r_tb` | {res['r_tb']:.4f}  (theirs: 0.24) |",
              f"| max speed | {res['v_max']:.2f} |",
              "",
              "Figures and the four-panel animation are in `figs/`.", ""]
    L += ["---", "",
          "*Generated by `scripts/torus/run/describe_run.py`; edits outside the Purpose "
          "line will be overwritten.*"]
    return "\n".join(L) + "\n"


def existing_purpose(path):
    if not os.path.isfile(path):
        return None
    m = re.search(r"^\*\*Purpose\.\*\*\s*(.+)$", open(path).read(), re.M)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--purpose", default=None)
    ap.add_argument("--id", default="sg_torus_m1")
    args = ap.parse_args()

    out = os.path.join(args.run_dir, "README.md")
    purpose = args.purpose or existing_purpose(out)
    open(out, "w").write(render(args.run_dir, purpose, args.id))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
