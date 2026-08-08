"""The verification and validation checks themselves.

Each function takes the suite (for case directories) and returns a list of `Check` plus
whatever arrays the report wants to plot. Nothing here runs a simulation; nothing here
prints. The split matters: a verdict has exactly one definition, and it lives here.

Grouping follows the V&V plan:

  static       algebra of the equilibrium, no solver involved
  mapping      the C++ generator and scripts/theory agree on every derived number
  residual     order of accuracy of the discrete equilibrium residual
  convergence  grid convergence of the solution, plus its GCI
  temporal     timestep convergence
  invariance   things that must not change the answer at all
  sensitivity  parameters of numerical origin that must not change the answer either
  viscosity    the alpha-law as the solver applies it, and what it transports
  ppi          Papaloizou-Pringle growth rates against linear theory
"""

from __future__ import annotations

import re

import numpy as np

from . import analysis, ppi, ring
from .core import Check, OK, WARN, FAIL, INFO, band, grid_2d


# --------------------------------------------------------------------------- 0. health


def health_checks(suite, model, records):
    """Every run's timestep and floor history, in one place.

    A collapsed timestep is the failure mode this project has been bitten by most, and it
    is invisible in the usual places: model time freezes while cycles keep running, so
    .hst stops growing and its last recorded dt still looks healthy (ZVIT 2). Reading the
    ratio of the smallest recorded dt to the first one turns that into a number, and the
    fraction of the requested run that was actually completed turns a hang into a result.
    """
    out, data = [], {}
    rows = []
    for cid in sorted(records):
        cdir = suite.case_dir_path(cid)
        h = analysis.hst(cdir)
        if h is None or "dt" not in h or len(h["dt"]) < 3:
            continue
        dt0 = float(h["dt"][1])
        dt_min = float(np.min(h["dt"][1:]))
        ratio = dt_min / dt0 if dt0 > 0 else np.nan
        floors = analysis.max_floor_hits(h)
        nrow = np.asarray(h.get("n_dfloor", [0])) + np.asarray(h.get("n_pfloor", [0]))
        rows_hit = int((nrow > 0).sum())
        first_hit = (float(h["time"][np.argmax(nrow > 0)]) / model.P_orb
                     if rows_hit else np.nan)
        reached = float(h["time"][-1])
        want = None
        for a in records[cid].get("argv", []):
            if a.startswith("time/tlim="):
                want = float(a.split("=", 1)[1])
        frac = reached / want if want and want < 1e8 else 1.0
        rows.append((cid, records[cid]["status"], ratio, floors, frac,
                     rows_hit, len(nrow), first_hit,
                     "unstable" in records[cid].get("tags", [])))
    data["health_rows"] = rows

    stalled = [r for r in rows if r[4] < 0.98]
    floored = [r for r in rows if np.isfinite(r[3]) and r[3] > 0]

    # A floor activation is worth failing on even when it heals: it means the floor, not
    # the physics, set the state somewhere. But how long it lasted decides what to do
    # about it, so say that rather than just the peak count. Checking per cycle rather
    # than per snapshot is what makes brief activations visible at all.
    # Runs that are deliberately driven unstable are expected to reach states the floors
    # have to catch, once the seeded mode goes non-linear. Failing on those would fail on
    # the physics the test exists to produce, so they are separated and reported, while
    # the verdict rests on the runs that are supposed to stay smooth.
    steady = [r for r in floored if not r[8]]
    unstable = [r for r in floored if r[8]]

    def describe(rs):
        return ", ".join(f"{c} ({f:.0f} cells, {rh}/{tot} rows, from {fh:.2f} orbits)"
                         for c, _, _, f, _, rh, tot, fh, _ in rs)

    out.append(Check("health", "floors", OK if not steady else FAIL,
                     (f"{len(rows) - len(unstable)} steady runs, {len(steady)} with "
                      f"floor activations: " + describe(steady)) if steady else
                     f"no cell reached dfloor or pfloor at any cycle in any of the "
                     f"{len(rows) - len(unstable)} steady runs",
                     float(len(steady)), "= 0 runs"))
    if unstable:
        out.append(Check("health", "floors_unstable", INFO,
                         f"deliberately destabilised runs, where the seeded mode goes "
                         f"non-linear and the funnel empties: " + describe(unstable),
                         float(len(unstable))))

    # Same split as the floors: a run driven deliberately non-linear slows down as the
    # mode steepens, and stopping short of tlim there costs nothing - the growth rate is
    # fitted well inside the completed part, below the linearity ceiling. A steady run
    # that stops short is the collapsed-timestep failure and is a defect.
    stalled_steady = [r for r in stalled if not r[8]]
    if stalled_steady:
        worst = min(stalled_steady, key=lambda r: r[4])
        out.append(Check("health", "completion", FAIL,
                         f"{len(stalled_steady)} of {len(rows) - sum(r[8] for r in rows)}"
                         f" steady runs did not reach their tlim; worst is {worst[0]} at "
                         f"{100 * worst[4]:.0f}% with the timestep down to "
                         f"{worst[2]:.1e} of its initial value. This is the collapsed-dt "
                         f"failure mode, not a crash",
                         float(len(stalled_steady)), "= 0 runs"))
    else:
        out.append(Check("health", "completion", OK,
                         f"every steady run reached its time limit; smallest dt/dt_0 "
                         f"over the suite is "
                         f"{min((r[2] for r in rows), default=np.nan):.2e}",
                         min((r[2] for r in rows), default=np.nan), "all complete"))
    if stalled:
        for r in sorted(stalled, key=lambda r: r[4]):
            out.append(Check("health", f"stall_{r[0]}", INFO,
                             f"reached {100 * r[4]:.1f}% of tlim, dt fell to "
                             f"{r[2]:.2e} of its initial value before .hst stopped "
                             f"being written - the last recorded dt always looks "
                             f"healthy, which is why this is measured here", r[4]))
    return out, data


# --------------------------------------------------------------------------- 1. static


def static_checks(model):
    out, data = [], {}
    g = "static"

    r = np.linspace(model.r_in * 1.001, model.r_out * 0.999, 4000)
    # Bernoulli integral (n+1) p/rho + Phi_eff = const is what the torus IS; if it does
    # not hold to round-off the profile functions disagree with the derivation.
    l0sq = model.beta * model.r_center
    bern = (model.n_poly + 1) * model.p_disk(r) / model.rho_disk(r) \
        - model.beta / r + l0sq / (2 * r**2)
    dev = np.max(np.abs(bern - bern.mean())) / max(abs(bern.mean()), 1e-300)
    out.append(Check(g, "bernoulli", OK if dev < 1e-12 else FAIL,
                     f"(n+1)p/rho + Phi_eff constant across the torus to {dev:.2e}",
                     dev, "< 1e-12"))

    # specific angular momentum must be exactly constant inside the torus
    body = model.rho_disk(r) > 0.05
    v2 = (model.beta / r + (r / model.rho_disk(r))
          * np.gradient(model.p_disk(r), r))
    lr = r * np.sqrt(np.clip(v2, 0.0, None))
    dl = np.max(np.abs(lr[body] / np.sqrt(l0sq) - 1.0))
    out.append(Check(g, "l_const", OK if dl < 5e-3 else WARN,
                     f"r*v_phi within {dl:.2e} of sqrt(beta*r_c) in the body "
                     f"(finite-difference dp/dr, so this is a discretisation floor)",
                     dl, "< 5e-3"))

    # surfaces and the shape function
    fr = max(abs(model.f(model.r_in)), abs(model.f(model.r_out)))
    out.append(Check(g, "surfaces", OK if fr < 1e-14 else FAIL,
                     f"f(r_in) = f(r_out) = 0 to {fr:.2e}", fr, "< 1e-14"))

    hr_id = abs(model.H_over_r(model.r_center)
                - np.sqrt((model.gamma - 1) * (0.5 - model.C_prime)))
    out.append(Check(g, "H_over_r", OK if hr_id < 1e-12 else FAIL,
                     f"H/r = sqrt((gamma-1)(0.5-C')) = "
                     f"{np.sqrt((model.gamma - 1) * (0.5 - model.C_prime)):.4f} "
                     f"to {hr_id:.2e}", hr_id, "< 1e-12"))

    # T_0 and mu are a choice of units: no physical result may depend on them
    from disk_model import DiskModel
    ref = DiskModel(**{**{k: v for k, v in model.as_dict().items()
                          if k in ("M_bh", "mu", "chi", "rho_0", "gamma",
                                   "r_center", "C_prime", "alpha", "rho_atm",
                                   "t_atm_frac")}})
    spread = []
    for t0 in (5e3, 5e4, 5e5):
        m2 = DiskModel(**{**{k: getattr(ref, k) for k in
                             ("M_bh", "mu", "chi", "rho_0", "gamma", "r_center",
                              "C_prime", "alpha", "rho_atm", "t_atm_frac")}, "T_0": t0})
        spread.append((m2.P_orb * m2.T_scale, m2.pr_mid * m2.cs0**2))
    spread = np.array(spread)
    rel = np.max(np.abs(spread / spread[0] - 1.0))
    data["T0_invariance"] = spread
    out.append(Check(g, "units_T0", OK if rel < 1e-12 else FAIL,
                     f"physical P_orb and (p/rho) unchanged over T_0 = 5e3..5e5 K "
                     f"to {rel:.2e}: T_0 and mu set the unit of velocity, nothing else",
                     rel, "< 1e-12"))

    out.append(Check(g, "regime", INFO,
                     f"p_rad/p_gas = {model.p_rad_over_gas:.2e} at the ideal-gas "
                     f"T_mid = {model.T_mid:.3e} K; the consistent reading is "
                     f"p = aT^4/3 with T = {model.T_mid_rad:.3e} K. gamma = "
                     f"{model.gamma:.4f} is the radiation value, n_poly = "
                     f"{model.n_poly:.3f} (PP84 used 3)",
                     float(model.p_rad_over_gas)))
    out.append(Check(g, "newtonian", OK if model.v_K_over_c < 0.1 else WARN,
                     f"v_K/c = {model.v_K_over_c:.4f} at r_center: relativistic "
                     f"corrections are O(v^2/c^2) = {model.v_K_over_c**2:.1e}",
                     float(model.v_K_over_c), "< 0.1"))
    return out, data


# -------------------------------------------------------------------------- 2. mapping

_BANNER = {
    "beta": (r"Calculated Gravity \(beta\):\s+(\S+)", "beta"),
    "r_in": (r"Inner Geometric Boundary:\s+(\S+)", "r_in"),
    "r_out": (r"Outer Geometric Boundary:\s+(\S+)", "r_out"),
    "cs0": (r"Calculated Sound Speed \(cs_0\):\s+(\S+)", "cs0"),
    "r_g": (r"Gravitational radius \(r_g\):\s+(\S+)", "r_g"),
    "L_0": (r"Length scale \(L_0\):\s+(\S+)", "L_0"),
    "T_scale": (r"Time scale \(T_0\):\s+(\S+)", "T_scale"),
    "mass_scale": (r"Mass scale:\s+(\S+)", "mass_scale"),
    "mdot_scale": (r"Mdot scale:\s+(\S+)", "mdot_scale"),
    "n_poly": (r"Polytropic Index \(n\):\s+(\S+)", "n_poly"),
    "pr_mid": (r"\(p/rho\) at r_center \[code\]:\s+(\S+)", "pr_mid"),
    "T_mid": (r"T_mid, ideal-gas reading:\s+(\S+)", "T_mid"),
    "p_rad_over_gas": (r"p_rad/p_gas at that T_mid:\s+(\S+)", "p_rad_over_gas"),
}


def mapping_checks(suite, model):
    """The C++ generator and scripts/theory hold independent copies of these formulas.

    Nothing else in the project compares them, so a value corrected in one and not the
    other would survive every other test in this suite.
    """
    out = []
    log = suite.case_dir_path("header") / "run.log"
    if not log.exists():
        return [Check("mapping", "banner", FAIL, "header case produced no log")], {}
    text = log.read_text()

    worst_rel, worst_key = 0.0, ""
    missing = []
    for key, (pattern, attr) in _BANNER.items():
        hit = re.search(pattern, text)
        if not hit:
            missing.append(key)
            continue
        got = float(hit.group(1))
        want = float(getattr(model, attr))
        rel = abs(got - want) / max(abs(want), 1e-300)
        if rel > worst_rel:
            worst_rel, worst_key = rel, key
    if missing:
        out.append(Check("mapping", "banner", FAIL,
                         f"not printed by the generator: {', '.join(missing)}"))
    tol = 1e-10
    out.append(Check("mapping", "pgen_vs_theory",
                     OK if worst_rel < tol else FAIL,
                     f"{len(_BANNER) - len(missing)} derived quantities agree to "
                     f"{worst_rel:.2e} (worst: {worst_key or 'none'}); the banner is "
                     f"printed at 12 digits so this is a real comparison",
                     worst_rel, f"< {tol:g}"))
    return out, {}


# ------------------------------------------------------------------------- 3. residual


def residual_checks(suite, model, ladder):
    """Order of accuracy of the discrete equilibrium residual on the real problem.

    The initial condition balances analytically. The scheme balances it exactly only in
    the ambient (by construction) and to O(dr^2) inside the torus, and that leftover is
    what actually drives the run - so its convergence rate is the verification statement
    that matters, far more than a linear-wave test on a different problem.
    """
    out, data = [], {}
    # Four zones, because they answer different questions and averaging them together
    # hides all three answers. `core` is the smooth interior, where second order is the
    # claim; `surface` is where rho ~ (r-r_in)^n is only C^3 and the order must drop;
    # `outer` is the undisturbed balanced background, which should be exact; `funnel` is
    # ambient too, but sits inside the inner boundary's reach, so it is reported and not
    # judged.
    zones = {}
    for nx in ladder:
        cdir = suite.case_dir_path(f"resid_n{nx}")
        fr = analysis.read_frame(cdir, out_id=2, index=-1)
        pr = analysis.read_frame(cdir, out_id=1, index=-1)
        if fr is None or pr is None or "resid_r" not in fr:
            return [Check("residual", f"n{nx}", FAIL,
                          "no uov frame with fluxes (cycle > 0)")], {}
        r, _, gu = grid_2d(fr)
        _, _, gp = grid_2d(pr)
        rel = np.abs(gu["resid_r"]) / (gp["rho"] * model.beta / r[None, :] ** 2)
        rd = model.rho_disk(r)
        masks = {
            "core": rd > 0.3,
            "surface": (rd > 10 * model.rho_atm) & (rd <= 0.3),
            "outer": r > 1.1 * model.r_out,
            "funnel": r < 0.9 * model.r_in,
        }
        for k, msk in masks.items():
            zones.setdefault(k, []).append(float(rel[:, msk].mean()))
    data["resid_levels"] = list(ladder)
    data["resid_zones"] = zones
    data["resid_norms"] = zones["core"]
    data["resid_ambient"] = zones["outer"]

    for i, nx in enumerate(ladder):
        out.append(Check("residual", f"n{nx}", INFO,
                         f"relative residual: core {zones['core'][i]:.2e}, surface "
                         f"{zones['surface'][i]:.2e}, outer ambient "
                         f"{zones['outer'][i]:.2e}, inner funnel "
                         f"{zones['funnel'][i]:.2e}", zones["core"][i]))

    # The balanced background is an exact discrete equilibrium by construction, not an
    # approximate one (ZVIT 7.4). That is a much stronger statement than "small", and it
    # is the reason the disk does not drain: it holds to round-off at every resolution.
    worst_atm = max(zones["outer"])
    out.append(Check("residual", "ambient_exact",
                     OK if worst_atm < 1e-9 else FAIL,
                     f"undisturbed ambient residual <= {worst_atm:.2e} of |rho g| at "
                     f"every resolution - round-off, not truncation: the balanced "
                     f"background is exact in the discrete operators",
                     worst_atm, "< 1e-9"))
    out.append(Check("residual", "funnel", INFO,
                     f"inner funnel residual {zones['funnel'][-1]:.2e}: ambient too, but "
                     f"within reach of the inner boundary, so it is not exact - this is "
                     f"the size of the boundary's influence, measured",
                     zones["funnel"][-1]))

    def order(vals):
        if len(vals) < 3 or min(vals) <= 0:
            return np.nan
        return float(np.log(vals[-2] / vals[-1]) / np.log(2.0))

    p_core, p_surf = order(zones["core"]), order(zones["surface"])
    data["resid_order"] = (p_core, p_surf)
    out.append(band(p_core, 1.5, 2.5, "order_core", "residual",
                    f"observed order {p_core:.2f} in the smooth interior of the torus; "
                    f"PLM + VL2 is designed for 2"))
    out.append(Check("residual", "order_surface", INFO,
                     f"observed order {p_surf:.2f} at the torus surface, where "
                     f"rho ~ (r-r_in)^{model.n_poly:.0f} is C^{int(model.n_poly)} and "
                     f"not smooth. The loss there is a property of the solution, not of "
                     f"the scheme, and it is what limits the global rate",
                     p_surf))
    return out, data


# ---------------------------------------------------------------------- 4. convergence


def convergence_checks(suite, model, ladder):
    out, data = [], {}
    # Two mass functionals, and the difference between them is the point.
    #
    #   mass       Athena++'s own integral of rho over the whole domain. A finite-volume
    #              scheme conserves it exactly up to boundary fluxes, and it is a smooth
    #              functional of the solution, so Richardson extrapolation applies to it.
    #   disk_mass  the torus body only, selected by rho > 2 rho_atm. The threshold is what
    #              makes it a good physical diagnostic and a bad convergence target: cells
    #              cross that contour as the disk relaxes, and hop in and out of the sum.
    #              Measured, it reports a drift 15-26x larger than the true one and of the
    #              opposite sign, and its observed order collapses to 0.88 on the coarse
    #              grids where the unthresholded integral already gives 1.84.
    profiles, drifts, drifts_body, levels, radii = [], [], [], [], []
    for nx in ladder:
        cdir = suite.case_dir_path(f"conv_n{nx}")
        r, sig = analysis.surface_density(cdir)
        h = analysis.hst(cdir)
        if sig is None or h is None:
            return [Check("convergence", f"n{nx}", FAIL, "missing output")], {}
        levels.append(nx)
        profiles.append(sig)
        radii.append(r)
        drifts.append(analysis.relative_drift(h, "mass"))
        drifts_body.append(analysis.relative_drift(h, "disk_mass"))
    data["conv_levels"], data["conv_profiles"] = levels, profiles
    data["conv_radii"] = radii
    data["conv_drift"] = drifts
    data["conv_drift_body"] = drifts_body

    series = analysis.order_series(levels, profiles)
    data["conv_order_series"] = series
    p, e21, e32 = analysis.observed_order(levels, profiles)
    data["conv_order"], data["conv_errors"] = p, (e21, e32)
    out.append(band(p, 1.0, 2.5, "order", "convergence",
                    f"observed order {p:.2f} on Sigma(r) after 2 orbits, from the three "
                    f"finest grids {series[-1][0]} (L1 errors {e21:.3e} -> {e32:.3e})"))
    if len(series) > 1:
        out.append(Check("convergence", "order_trend", INFO,
                         "; ".join(f"{t} -> {pp:.2f}" for t, pp, _, _ in series)
                         + ". A rising order means the refinement is entering the "
                           "asymptotic range; a wandering one means it is not there yet",
                         series[-1][1]))

    g = analysis.gci(*drifts[-3:])
    data["conv_gci"] = g
    if len(drifts) > 3:
        data["conv_gci_series"] = [analysis.gci(*drifts[i:i + 3])
                                   for i in range(len(drifts) - 2)]
    if np.isfinite(g["gci_abs"]):
        # ABSOLUTE band. The exact mass drift is zero, so a relative band divides by a
        # number that is itself converging to nothing and diverges however good the
        # scheme is. The statement that means something is "the drift is this small,
        # give or take this much".
        val, band_abs = drifts[-1], g["gci_abs"]
        out.append(Check("convergence", "gci_mass",
                         OK if abs(val) + band_abs < 1e-6 else WARN,
                         f"total mass drift at 2 orbits = {val:+.3e} on {levels[-1]}^2, "
                         f"with a fine-grid GCI band of +/-{band_abs:.2e} (order "
                         f"{g['p']:.2f}). Mass is conserved to better than "
                         f"{abs(val) + band_abs:.1e} - quote that, not a percentage: "
                         f"the exact answer is zero, so a relative band is meaningless",
                         float(abs(val) + band_abs), "< 1e-6"))
    else:
        out.append(Check("convergence", "gci_mass", WARN,
                         f"GCI undefined: dm/m0 = {drifts} is not monotone in "
                         f"resolution, so the finest three levels are not in the "
                         f"asymptotic range"))
    if data.get("conv_gci_series"):
        out.append(Check("convergence", "gci_trend", INFO,
                         "; ".join(f"{levels[i:i + 3]} p={g2['p']:.2f} "
                                   f"band=+/-{g2['gci_abs']:.1e}"
                                   for i, g2 in enumerate(data["conv_gci_series"]))
                         + ". The band shrinking as the grids refine is what says the "
                           "extrapolation is becoming trustworthy",
                         data["conv_gci_series"][-1]["gci_abs"]))

    for nx, d, db in zip(levels, drifts, drifts_body):
        out.append(Check("convergence", f"drift_n{nx}", INFO,
                         f"total mass {d:+.3e}, torus body {db:+.3e} after 2 orbits", d))
    out.append(Check("convergence", "threshold_bias", INFO,
                     f"the thresholded disk_mass reports "
                     f"{abs(drifts_body[-1] / drifts[-1]):.0f}x the true drift and with "
                     f"the opposite sign at {levels[-1]}^2. It is the right diagnostic "
                     f"for tracking the torus and the wrong one for a convergence study, "
                     f"because rho > 2 rho_atm is not a smooth functional of the solution",
                     abs(drifts_body[-1] / drifts[-1])))
    return out, data


# ------------------------------------------------------------------------- 5. temporal


def temporal_checks(suite, model):
    out, data = [], {}
    cfls = (0.4, 0.2, 0.1)
    profs, drifts = [], []
    for c in cfls:
        cdir = suite.case_dir_path(f"cfl_{c}")
        r, sig = analysis.surface_density(cdir)
        if sig is None:
            return [Check("temporal", f"cfl{c}", FAIL, "missing output")], {}
        profs.append(sig)
        drifts.append(analysis.relative_drift(analysis.hst(cdir), "disk_mass"))
    e21 = float(np.abs(profs[1] - profs[0]).mean())
    e32 = float(np.abs(profs[2] - profs[1]).mean())
    data["cfl_values"], data["cfl_drift"] = list(cfls), drifts
    data["cfl_errors"] = (e21, e32)
    if e32 > 0:
        p = float(np.log(e21 / e32) / np.log(2.0))
        data["cfl_order"] = p
        out.append(band(p, 1.0, 3.0, "order", "temporal",
                        f"error falls as dt^{p:.2f} between cfl 0.4/0.2/0.1; VL2 is "
                        f"second order in time, but at these cfl numbers the spatial "
                        f"error is not negligible, so a value below 2 is expected"))
    for c, d in zip(cfls, drifts):
        out.append(Check("temporal", f"drift_cfl{c}", INFO,
                         f"dm/m0 = {d:+.3e} after 1 orbit", d))
    return out, data


# ------------------------------------------------------------------------ 6. invariance


def invariance_checks(suite, model):
    """Things that change nothing physical, and so must change nothing numerical."""
    out, data = [], {}
    base = suite.case_dir_path("inv_1block")

    d = analysis.field_difference(base, suite.case_dir_path("inv_repeat"))
    out.append(Check("invariance", "determinism", OK if d == 0.0 else FAIL,
                     f"the same run twice: max|d rho| = {d:.3e}", d, "= 0"))

    d = analysis.field_difference(base, suite.case_dir_path("inv_4block"))
    rho_scale = 1.0
    out.append(Check("invariance", "meshblocks",
                     OK if d < 1e-12 * rho_scale else (WARN if d < 1e-9 else FAIL),
                     f"1 block vs 4: max|d rho| = {d:.3e} after 1 orbit "
                     f"(round-off in a different summation order is expected; "
                     f"anything larger is a real block-boundary effect)",
                     d, "< 1e-12"))

    # A restart cannot be bitwise and it is wrong to ask for it: Athena++ clamps the last
    # step of a segment so that time lands exactly on tlim, so splitting a run inserts a
    # step the direct run never took. The right yardstick is the code's own timestep
    # error, which this suite already measures - halving the CFL number changes the
    # solution by this much, and a restart must disturb it by less.
    a = suite.case_dir_path("restart_direct")
    b = suite.case_dir_path("restart_split")
    scale = analysis.field_difference(suite.case_dir_path("cfl_0.4"),
                                      suite.case_dir_path("cfl_0.2"))
    if (b / "done.json").exists() and analysis.frame_indices(b):
        d = analysis.field_difference(a, b)
        ok = np.isfinite(scale) and d < scale
        out.append(Check("invariance", "restart",
                         OK if ok else (WARN if d < 10 * scale else FAIL),
                         f"2 orbits in one go vs 1+1 through a restart file: "
                         f"max|d rho| = {d:.3e}, against {scale:.3e} for halving the "
                         f"CFL number. The restart perturbs the solution by "
                         f"{d / scale:.2f}x the timestep error it already carries",
                         d / scale if np.isfinite(scale) else np.nan, "< 1"))
    else:
        out.append(Check("invariance", "restart", WARN,
                         "restart case produced no comparable frame"))

    # MPI. Same mesh, same decomposition, same binary; the only difference is whether a
    # ghost zone crosses a process boundary. Nothing in that path is allowed to touch a
    # value, so unlike the 1-vs-4-block test this one is held to bitwise equality.
    a = suite.case_dir_path("mpi_1rank")
    b = suite.case_dir_path("mpi_4rank")
    if analysis.frame_indices(a) and analysis.frame_indices(b):
        d = analysis.field_difference(a, b)
        out.append(Check("invariance", "mpi",
                         OK if d == 0.0 else (WARN if d < 1e-12 else FAIL),
                         f"4 MeshBlocks on 1 rank vs on 4 ranks: max|d rho| = {d:.3e} "
                         f"after 1 orbit. The exchange moves the same doubles either "
                         f"way, so this is bitwise or it is a defect",
                         d, "= 0"))
    else:
        out.append(Check("invariance", "mpi", INFO,
                         "MPI cases did not run; build the binary with "
                         "./scripts/vv/build_mpi.sh to include them"))
    return out, data


# ----------------------------------------------------------------------- 7. sensitivity


def sensitivity_checks(suite, model, gci_band):
    """Parameters with no physical meaning. A dependence here is a modelling artefact.

    The acceptance band is the discretisation band from the convergence study, not a
    number picked here: a shift smaller than the grid uncertainty is not a shift.
    """
    out, data = [], {}
    base_dir = suite.case_dir_path("conv_n128")
    base = analysis.relative_drift(analysis.hst(base_dir), "disk_mass")
    data["sens_base"] = base
    rows = []
    tol = max(gci_band if np.isfinite(gci_band) else 0.0, 1e-5) * 2.0

    variants = [("sens_rhoatm_0.001", "rho_atm = 1e-3"),
                ("sens_rhoatm_1e-05", "rho_atm = 1e-5"),
                ("sens_rhoatm_1e-06", "rho_atm = 1e-6"),
                ("sens_dfloor_1e-10", "dfloor = 1e-10"),
                ("sens_dfloor_1e-14", "dfloor = 1e-14"),
                ("sens_x1min_0.3", "x1min = 0.30"),
                ("sens_x1min_0.45", "x1min = 0.45")]
    for cid, label in variants:
        cdir = suite.case_dir_path(cid)
        h = analysis.hst(cdir)
        if h is None:
            out.append(Check("sensitivity", cid, FAIL, f"{label}: no .hst"))
            continue
        drift = analysis.relative_drift(h, "disk_mass")
        floors = analysis.max_floor_hits(h)
        shift = abs(drift - base)
        rows.append((label, drift, shift, floors))
        status = OK if shift <= tol else (WARN if shift <= 10 * tol else FAIL)
        out.append(Check("sensitivity", cid, status,
                         f"{label}: dm/m0 = {drift:+.3e}, shifted by {shift:.2e} "
                         f"from the baseline {base:+.3e}; floors hit {floors:.0f}",
                         shift, f"<= {tol:.1e} (2x the grid band)"))
    data["sens_rows"] = rows

    # The ambient medium is a numerical device, so the answer has to stop depending on it
    # as it is made thinner. One variant can only show that the answer moved; the ladder
    # shows whether the movement dies out, which is the statement that licenses the
    # baseline value. Successive differences, not distances from the baseline: a plateau
    # is about the slope going to zero.
    ladder = [("1e-3", "sens_rhoatm_0.001"), ("1e-4", None),
              ("1e-5", "sens_rhoatm_1e-05"), ("1e-6", "sens_rhoatm_1e-06")]
    drifts = []
    for label, cid in ladder:
        h = analysis.hst(base_dir if cid is None else suite.case_dir_path(cid))
        drifts.append(np.nan if h is None else analysis.relative_drift(h, "disk_mass"))
    steps = [abs(drifts[i + 1] - drifts[i]) for i in range(len(drifts) - 1)]
    data["sens_rhoatm_ladder"] = list(zip([l for l, _ in ladder], drifts))
    if all(np.isfinite(s) for s in steps):
        first, last = steps[0], steps[-1]
        decaying = last < first
        out.append(Check("sensitivity", "rhoatm_plateau",
                         OK if (decaying and last <= tol) else
                         (WARN if decaying else FAIL),
                         "dm/m0 over rho_atm = " +
                         ", ".join(f"{l}: {d:+.2e}" for (l, _), d in zip(ladder, drifts))
                         + f"; the step per decade falls {first:.2e} -> {last:.2e}, so "
                         f"the last decade moves the answer by "
                         f"{last / first if first else np.nan:.2f} of what the first one "
                         f"did", last, f"<= {tol:.1e} and falling"))
    else:
        out.append(Check("sensitivity", "rhoatm_plateau", WARN,
                         "the rho_atm ladder is incomplete; cannot say whether the "
                         "dependence plateaus"))
    return out, data


# -------------------------------------------------------------------------- 8. viscosity


def viscosity_checks(suite, model):
    out, data = [], {}
    from disk_model import DiskModel
    mv = DiskModel(**{**{k: v for k, v in model.as_dict().items()
                         if k in ("M_bh", "T_0", "mu", "chi", "rho_0", "gamma",
                                  "r_center", "C_prime", "rho_atm", "t_atm_frac")},
                      "alpha": 0.01})

    errs = {}
    for cfl in (0.4, 0.1):
        cdir = suite.case_dir_path(f"visc_cfl{cfl}")
        uov = analysis.read_frame(cdir, out_id=2, index=-1)
        prim = analysis.read_frame(cdir, out_id=1, index=-1)
        if uov is None or prim is None or "nu_applied" not in uov:
            out.append(Check("viscosity", f"alpha_law_cfl{cfl}", FAIL, "missing output"))
            continue
        r, _, gu = grid_2d(uov)
        _, _, gp = grid_2d(prim)
        cut = 10.0 * mv.rho_atm
        w = gp["rho"] ** 2 / (gp["rho"] ** 2 + cut**2)
        nu_a = (mv.alpha * mv.gamma / np.sqrt(mv.beta)) \
            * (gp["press"] / gp["rho"]) * r[None, :] ** 1.5 * w
        rel = np.abs(gu["nu_applied"] - nu_a) / np.maximum(nu_a, 1e-300)
        # Restrict to the smooth interior. The maximum over the whole disk lands on a
        # single cell at the torus surface, where the two runs have drifted apart by more
        # than the stage offset being measured, so it is noise rather than signal.
        core = model.rho_disk(r) > 0.3
        rel = rel[:, core]
        errs[cfl] = (float(rel.max()), float(np.median(rel)))
    data["alpha_law_err"] = errs

    if len(errs) == 2:
        e_coarse, e_fine = errs[0.4][0], errs[0.1][0]
        worst = max(e_coarse, e_fine)
        # The statement that can be defended is the agreement itself. Asking the residual
        # to scale with dt does not work here: in the smooth core the equilibrium barely
        # moves in one stage, so the offset sits at the 1e-6 level and is no longer
        # dominated by dt at all. Outside the core it is dominated by the surface, where
        # the two runs have genuinely diverged. Agreement to 1e-4 in the core verifies
        # the enrolled coefficient end to end, which is what this check exists for.
        out.append(Check("viscosity", "alpha_law",
                         OK if worst < 1e-4 else FAIL,
                         f"nu the solver actually held vs the analytic alpha-law, in "
                         f"the smooth core: max relative difference {worst:.2e} "
                         f"({e_coarse:.2e} at cfl 0.4, {e_fine:.2e} at cfl 0.1). The "
                         f"array is set one VL2 stage before the output primitives, so "
                         f"an exact match is not available at any dt",
                         worst, "< 1e-4"))
        out.append(Check("viscosity", "alpha_law_median", INFO,
                         f"median difference {errs[0.4][1]:.2e} (cfl 0.4) -> "
                         f"{errs[0.1][1]:.2e} (cfl 0.1); in the core the stage offset is "
                         f"already below the level where halving dt changes it",
                         errs[0.1][1]))

    # Recover alpha from the stress the solver actually applied.
    #
    #   T_rphi = rho nu (r dOmega/dr),   nu = alpha gamma (p/rho) / Omega_K
    #   =>  alpha = T_rphi Omega_K / (gamma p r dOmega/dr)
    #
    # inverted with no assumption about the rotation law. The textbook form,
    # T_rphi = -(3/2) alpha p, hides two things that are both wrong here: it assumes
    # Keplerian shear (r dOmega/dr = -3/2 Omega), while an l = const torus has -2 Omega,
    # and it drops the gamma that our alpha-law carries because it defines nu with the
    # isothermal sound speed. Using it gave a "measured" alpha 1.56x the input and looked
    # like a defect in the solver; it was a defect in the formula.
    cdir = suite.case_dir_path("visc_transport")
    uov = analysis.read_frame(cdir, out_id=2, index=-1)
    prim = analysis.read_frame(cdir, out_id=1, index=-1)
    if uov is not None and prim is not None and "T_rphi" in uov:
        r, _, gu = grid_2d(uov)
        _, _, gp = grid_2d(prim)
        omega = gp["vel2"].mean(axis=0) / r
        omega_k = np.sqrt(model.beta) * r**-1.5
        shear = r * np.gradient(omega, r)                  # r dOmega/dr
        t_mean = gu["T_rphi"].mean(axis=0)
        p_mean = gp["press"].mean(axis=0)
        body = (model.rho_disk(r) > 10 * model.rho_atm) & (np.abs(shear) > 0)
        alpha_r = t_mean * omega_k / (mv.gamma * p_mean * shear)
        alpha_eff = float(np.average(alpha_r[body], weights=p_mean[body]))
        data["alpha_eff"] = alpha_eff
        data["alpha_profile"] = (r[body], alpha_r[body])
        out.append(band(alpha_eff / mv.alpha, 0.9, 1.1, "alpha_eff", "viscosity",
                        f"alpha recovered from the applied stress, inverting "
                        f"T_rphi = rho nu r dOmega/dr with the rotation law taken from "
                        f"the data: {alpha_eff:.5f} against an input "
                        f"{mv.alpha:g}, ratio {alpha_eff / mv.alpha:.3f}. This is the "
                        f"viscous stress tensor itself, not just the coefficient"))

    # transport rate against the analytic viscous timescale
    h = analysis.hst(cdir)
    if h is not None:
        from disk_model import orbits_to_accrete
        n_orb = orbits_to_accrete(mv.alpha, mv.gamma, mv.C_prime)
        t_orb = h["time"][-1] / model.P_orb
        predicted = 1.0 - np.exp(-t_orb / n_orb)
        measured = -analysis.relative_drift(h, "disk_mass")
        data["visc_transport"] = (t_orb, measured, predicted)
        rel = abs(measured - predicted) / max(predicted, 1e-300)
        # Only a verdict if the run reached a useful fraction of a viscous time. Over a
        # couple of orbits out of a 159-orbit viscous time the exponential formula is
        # comparing two numbers that are both essentially zero, and it assumes the disk
        # is already transporting - it is not, at t = 0.
        judge = t_orb > 0.05 * n_orb
        out.append(Check("viscosity", "transport_rate",
                         (OK if rel < 0.5 else WARN) if judge else INFO,
                         f"mass lost over {t_orb:.1f} orbits: {100 * measured:.4f}% "
                         f"measured vs {100 * predicted:.4f}% from tau = R^2/nu "
                         f"(N = {n_orb:.0f} orbits)"
                         + ("" if judge else
                            f". Not a verdict: {t_orb:.1f} orbits is {t_orb / n_orb:.1%} "
                            f"of a viscous time, and the formula assumes transport is "
                            f"already established"),
                         rel, "< 0.5 once t > 0.05 tau_visc"))

    # numerical viscosity surrogate, from the inviscid run: any angular momentum the
    # alpha = 0 torus transports is the scheme's own viscosity.
    h0 = analysis.hst(suite.case_dir_path("conv_n128"))
    if h0 is not None and "disk_L" in h0:
        dl = abs(analysis.relative_drift(h0, "disk_L"))
        t_code = h0["time"][-1]
        # dL/L ~ (nu/R^2) t  ->  nu_num ~ (dL/L) R^2 / t ; alpha_num = nu sqrt(beta)/(gamma (p/rho) r^1.5)
        nu_num = dl * model.r_center**2 / max(t_code, 1e-300)
        alpha_num = nu_num * np.sqrt(model.beta) / (model.gamma * model.pr_mid
                                                    * model.r_center**1.5)
        data["alpha_num"] = alpha_num
        out.append(Check("viscosity", "numerical_viscosity",
                         OK if alpha_num < 0.1 * mv.alpha else WARN,
                         f"upper bound from the inviscid run: alpha_num <= "
                         f"{alpha_num:.2e} against the alpha = {mv.alpha:g} under test "
                         f"({alpha_num / mv.alpha:.1%} of it). This is a bound, not a "
                         f"measurement: it charges every source of dL to viscosity",
                         float(alpha_num), f"< {0.1 * mv.alpha:.1e}"))
    return out, data


# ------------------------------------------------------------------ 8b. LBP ring


def ring_checks(suite, model, cfg):
    """The viscous operator against an exact solution, and the scheme's own viscosity.

    Everything here rests on one fact: with nu = 0 there is nothing in the equations that
    can spread a ring. So the spreading that appears is the truncation error behaving as
    a diffusion, and fitting the analytic family to it turns that into a number instead
    of the upper bound a global budget gives.
    """
    out, data = [], {}
    norm = ring.normalisation(1.0, cfg.RING_TAU0, cfg.RING_R0)

    def profiles(cid):
        cdir = suite.case_dir_path(cid)
        idx = analysis.frame_indices(cdir, 1)
        rows = []
        for i in range(len(idx)):
            fr = analysis.read_frame(cdir, 1, i)
            if fr is None:
                continue
            r, _, g = grid_2d(fr)
            rows.append((fr["time"], r, g["rho"].mean(axis=0)))
        return rows

    def tau_series(cid):
        """Fitted tau at every output, and the run duration."""
        rows = profiles(cid)
        if len(rows) < 3:
            return None
        times = np.array([t for t, _, _ in rows])
        taus = np.array([ring.fit_tau(r, s, norm, cfg.RING_R0,
                                      background=cfg.RING_SIGMA_BG)[0]
                         for _, r, s in rows])
        return times, taus, rows

    # -- the viscous operator against the exact solution ------------------------------
    # The primary number is the VISCOSITY the spreading implies, not the distance between
    # profiles. tau advances as 12 nu t / R0^2 and nothing else, so recovering it from
    # the simulated profiles measures the viscous operator directly, with no sensitivity
    # to how well the rest of the setup matches LBP's idealisation.
    errs, levels, nus = [], [], []
    for nx in cfg.RING_LADDER:
        ser = tau_series(f"ring_visc_n{nx}")
        if ser is None:
            out.append(Check("ring", f"visc_n{nx}", FAIL,
                             "no output - has ./scripts/vv/build_ring.sh been run?"))
            return out, data
        times, taus, rows = ser
        nu_eff = (taus[-1] - taus[0]) * cfg.RING_R0**2 / (12.0 * (times[-1] - times[0]))
        nus.append(nu_eff)
        t, r, sig = rows[-1]
        tau_exact = ring.tau_of_time(t, cfg.RING_NU, cfg.RING_R0, cfg.RING_TAU0)
        e = ring.l1_error(r, sig, tau_exact, norm, cfg.RING_R0, cfg.RING_SIGMA_BG)
        errs.append(e)
        levels.append(nx)
        out.append(Check("ring", f"nu_eff_n{nx}", INFO,
                         f"{nx} cells: the ring spreads from tau = {taus[0]:.5f} to "
                         f"{taus[-1]:.5f}, implying nu = {nu_eff:.4e} against the "
                         f"{cfg.RING_NU:g} that was applied "
                         f"({100 * (nu_eff / cfg.RING_NU - 1):+.2f}%); profile distance "
                         f"{100 * e:.3f}% of the peak", nu_eff))
        if nx == cfg.RING_LADDER[-1]:
            data["ring_profile"] = dict(r=r, sim=sig, tau=tau_exact, norm=norm,
                                        nx=nx, t=t)
    data["ring_levels"], data["ring_errors"], data["ring_nu_eff"] = levels, errs, nus

    rel = abs(nus[-1] / cfg.RING_NU - 1.0)
    out.append(Check("ring", "viscous_operator", OK if rel < 0.02 else FAIL,
                     f"the viscosity recovered from an exact solution is within "
                     f"{100 * rel:.2f}% of the value applied, at {levels[-1]} cells. "
                     f"This is the viscous flux checked against LBP74, not against "
                     f"another copy of our own formula", rel, "< 2%"))
    out.append(Check("ring", "profile_floor", INFO,
                     f"the profile distance stops falling with resolution "
                     f"({', '.join(f'{100 * e:.3f}%' for e in errs)} at "
                     f"{levels}): below ~0.1% the difference is no longer the grid but "
                     f"the test's own idealisation - LBP assumes zero pressure, while "
                     f"this ring has a real edge, a background and viscous heating",
                     errs[-1]))

    # -- the scheme's own viscosity ---------------------------------------------------
    # With nu = 0 nothing can spread the ring, so any drift of tau is the truncation
    # error. In cylindrical coordinates aligned with the flow there is barely any: the
    # honest output is a bound set by the scatter of the fit, not a fitted slope through
    # noise. A slope is only quoted when the trend is actually significant.
    nums = {}
    for nx in cfg.RING_LADDER:
        ser = tau_series(f"ring_num_n{nx}")
        if ser is None:
            continue
        times, taus, _ = ser
        fit = ring.numerical_viscosity(times, taus, cfg.RING_R0)
        span = times[-1] - times[0]
        drift = abs(taus[-1] - taus[0])
        scatter = float(np.std(taus - np.polyval(np.polyfit(times, taus, 1), times)))
        bound = (drift + scatter) * cfg.RING_R0**2 / (12.0 * span)
        nums[nx] = dict(fit=fit, times=list(times), taus=list(taus),
                        bound=bound, drift=drift, scatter=scatter,
                        significant=bool(fit["r2"] > 0.9 and drift > 5 * scatter))
        a_eq = ring.alpha_equivalent(bound, model)
        out.append(Check("ring", f"nu_num_n{nx}", INFO,
                         f"{nx} cells, nu = 0: tau moves by {drift:.2e} over {span:.0f} "
                         f"time units, against a fit scatter of {scatter:.2e}. That "
                         f"bounds nu_num <= {bound:.2e}, the alpha-law equivalent of "
                         f"alpha = {a_eq:.1e}", bound))
    data["ring_numerical"] = nums

    if nums:
        finest = max(nums)
        d = nums[finest]
        a_eq = ring.alpha_equivalent(d["bound"], model)
        alpha_used = cfg.ALPHA_TEST
        out.append(Check("ring", "numerical_viscosity",
                         OK if a_eq < 0.01 * alpha_used else WARN,
                         f"numerical viscosity at {finest} cells is at most "
                         f"{d['bound']:.2e}, i.e. {d['bound'] / cfg.RING_NU:.1e} of the "
                         f"viscosity this test applies and the equivalent of alpha = "
                         f"{a_eq:.1e} against the alpha = {alpha_used:g} the torus runs "
                         f"use. Cylindrical cells aligned with the flow is why it is "
                         f"this small",
                         a_eq, f"< {0.01 * alpha_used:.1e}"))
        out.append(Check("ring", "nu_num_resolved",
                         INFO if not d["significant"] else OK,
                         "the drift is at the level of the fit scatter at every "
                         "resolution, so this is a bound and not a measurement: the "
                         "scheme is too clean here for a run of this length to resolve "
                         "its own viscosity"
                         if not d["significant"] else
                         f"the drift is significant, so nu_num = {d['fit']['nu']:.3e} "
                         f"is a measurement (R^2 = {d['fit']['r2']:.3f})",
                         d["fit"]["r2"]))
    return out, data


# --------------------------------------------------------------------------- 9. PPI


def ppi_checks(suite, model, n_eig=400):
    """Growth rates of the seeded modes against the linear eigenvalue problem.

    This is the one test that a scheme which quietly over-damps non-axisymmetric modes
    cannot pass: it conserves mass, energy and angular momentum perfectly and still gets
    this wrong.
    """
    out, data = [], {}
    theory = {}
    for m in (1, 2, 3):
        res = ppi.fastest_mode(model, m, n=n_eig)
        theory[m] = res
        efold = 1.0 / (res["rate"] * model.P_orb) if res["rate"] > 0 else np.inf
        status = OK if res["converged"] else WARN
        out.append(Check("ppi", f"theory_m{m}", status,
                         f"linear theory: growth {res['rate'] * model.P_orb:.4f} per "
                         f"orbit (e-fold {efold:.3f} orbits), pattern speed "
                         f"{res['pattern']:.2f}, corotation inside torus: "
                         f"{ppi.corotation_inside(model, res['pattern'])}, "
                         f"grid-converged: {res['converged']} "
                         f"({res['rel_change']:.1e} on halving N)",
                         res["rate"] * model.P_orb))
    data["ppi_theory"] = theory

    measured = {}
    for cid, m in (("ppi_m1", 1), ("ppi_m2", 2), ("ppi_m3", 3), ("ppi_m2_small", 2)):
        cdir = suite.case_dir_path(cid)
        idx = analysis.frame_indices(cdir, 1)
        if len(idx) < 6:
            out.append(Check("ppi", cid, FAIL, "too few frames"))
            continue
        times, amps = [], []
        for i in range(len(idx)):
            fr = analysis.read_frame(cdir, 1, i)
            r, phi, g = grid_2d(fr)
            a = ppi.mode_amplitudes(r, phi, g["rho"], m_max=4,
                                    r_lo=model.r_in, r_hi=model.r_out)
            times.append(fr["time"] / model.P_orb)
            amps.append(a)
        times = np.array(times)
        amps = np.array(amps)
        a_m = amps[:, m]
        # amps[:, 0] is the axisymmetric background; the fit is restricted to where the
        # perturbation is still small compared to it, i.e. where linear theory applies
        fit = ppi.best_growth_window(times, a_m, background=amps[:, 0])
        lo, hi = fit["window"]
        measured[cid] = dict(m=m, times=times, amps=amps, fit=fit, window=(lo, hi))

        th = theory[m]["rate"] * model.P_orb
        rel = abs(fit["rate"] - th) / max(th, 1e-300)
        label = "linearity check" if cid.endswith("small") else "growth rate"
        status = OK if (rel < 0.15 and fit["r2"] > 0.98) else (
            WARN if rel < 0.35 else FAIL)
        out.append(Check("ppi", f"sim_{cid}", status,
                         f"{label}, m = {m}: measured {fit['rate']:.4f} per orbit over "
                         f"{lo:.1f}-{hi:.1f} orbits (R^2 = {fit['r2']:.4f}, "
                         f"{fit['n']} points) vs {th:.4f} from linear theory - "
                         f"{100 * rel:.1f}% apart",
                         rel, "< 15%"))
    data["ppi_measured"] = measured

    # Snapshots of the pattern itself, normalised by the azimuthal mean so the mode is
    # visible at all. Kept as fields rather than as a number because "is it really an
    # m = 2 spiral" is a question a plot answers and a growth rate does not. All three
    # modes, because the azimuthal symmetry of each is the check that the seed did what
    # it was asked to.
    cases_ppi = [("ppi_m1", 1), ("ppi_m2", 2), ("ppi_m3", 3)]

    def pattern_row(cid, m, picks, labels=None):
        """Rows of normalised density perturbation at the given frame indices."""
        cdir = suite.case_dir_path(cid)
        row = {"m": m, "times": [], "resid": [], "peak": [], "labels": labels or [],
               "r": None, "phi": None}
        for i in picks:
            fr = analysis.read_frame(cdir, 1, i)
            if fr is None:
                continue
            r, phi, g = grid_2d(fr)
            # Only the torus body. Outside it the azimuthal mean is the ambient, and
            # dividing by a near-zero mean in the funnel produces a huge number that
            # would set the colour scale and hide the mode everywhere else.
            keep = (r >= model.r_in) & (r <= model.r_out)
            rho = g["rho"][:, keep]
            d = rho / rho.mean(axis=0)[None, :] - 1.0
            # Normalise by a MEDIAN-based width, not the RMS. The perturbation is far
            # stronger in the two or three cells at the inner edge than anywhere else,
            # so an RMS scale is set by them and leaves the pattern filling the disk -
            # the part being judged - almost invisible. 1.4826 x MAD is the standard
            # robust estimate, and ignores that handful of cells by construction.
            scale = 1.4826 * float(np.median(np.abs(d)))
            row["resid"].append(d / max(scale, 1e-300))
            row["peak"].append(float(np.abs(d).max()))
            row["times"].append(fr["time"] / model.P_orb)
            row["r"], row["phi"] = r[keep], phi
        return row

    # (a) each mode at a quarter, half and the end of its own run
    frames = []
    for cid, m in cases_ppi:
        idx = analysis.frame_indices(suite.case_dir_path(cid), 1)
        if len(idx) < 4:
            continue
        frames.append(pattern_row(cid, m, (len(idx) // 4, len(idx) // 2, len(idx) - 1)))
    if frames:
        data["ppi_frames"] = frames

    # (b) each mode after the SAME number of e-foldings. Equal elapsed time is not a fair
    # comparison of shape: m = 3 grows at 0.71 of the m = 1 rate, so at a common instant
    # it is simply younger. Comparing at equal growth puts every mode at the same stage of
    # its own development. The common budget is set by whichever mode achieves the fewest
    # e-foldings inside its run - here m = 3, which grows slowest and stopped earliest.
    budgets = {}
    for cid, m in cases_ppi:
        if cid not in measured:
            continue
        t = measured[cid]["times"]
        budgets[m] = (theory[m]["rate"] * model.P_orb) * (t[-1] - t[0]), t
    if budgets:
        n_common = min(v[0] for v in budgets.values())
        efold_rows = []
        for cid, m in cases_ppi:
            if m not in budgets:
                continue
            _, t = budgets[m]
            rate = theory[m]["rate"] * model.P_orb
            want = [t[0] + f * n_common / rate for f in (1 / 3, 2 / 3, 1.0)]
            picks = [int(np.argmin(np.abs(t - w))) for w in want]
            lab = [f"{f * n_common:.1f}" for f in (1 / 3, 2 / 3, 1.0)]
            efold_rows.append(pattern_row(cid, m, picks, labels=lab))
        if efold_rows:
            data["ppi_frames_efold"] = efold_rows
            data["ppi_efold_budget"] = n_common

    # purity: a single-m seed must stay single-m while it is linear
    if "ppi_m2" in measured:
        d = measured["ppi_m2"]
        lin = d["times"] <= d["window"][1]
        leak = float(np.max(d["amps"][lin][:, [1, 3]]) / np.max(d["amps"][lin][:, 2]))
        out.append(Check("ppi", "mode_purity", OK if leak < 5e-3 else WARN,
                         f"while linear, m = 1 and m = 3 stay {leak:.1e} of m = 2: the "
                         f"seed is clean and the scheme generates no spurious modes. "
                         f"The floor is set by the grid noise the modes grow from, not "
                         f"by the seed, so a few parts in a thousand is as clean as this "
                         f"measurement gets",
                         leak, "< 5e-3"))

    # linearity: a tenfold weaker seed must grow at the same rate
    if "ppi_m2" in measured and "ppi_m2_small" in measured:
        r1 = measured["ppi_m2"]["fit"]["rate"]
        r2 = measured["ppi_m2_small"]["fit"]["rate"]
        rel = abs(r1 - r2) / max(abs(r1), 1e-300)
        out.append(Check("ppi", "linearity", OK if rel < 0.1 else WARN,
                         f"seed amplitude 1e-3 vs 1e-4 gives {r1:.4f} vs {r2:.4f} per "
                         f"orbit ({100 * rel:.1f}% apart): the measurement is in the "
                         f"linear regime, so it is a property of the torus",
                         rel, "< 10%"))

    # the fastest mode must be m = 1, as PP84 and every later study find
    if all(k in theory for k in (1, 2, 3)):
        fastest = max(theory, key=lambda k: theory[k]["rate"])
        out.append(Check("ppi", "mode_hierarchy", OK if fastest == 1 else WARN,
                         f"fastest growing azimuthal mode is m = {fastest}; the "
                         f"literature finds m = 1 dominant for a constant-l torus",
                         float(fastest), "m = 1"))
    return out, data
