"""The Lynden-Bell & Pringle spreading ring: the analytic solution and what to do with it.

LBP74 (MNRAS 168, 603) give the exact surface density of a narrow ring of gas orbiting a
point mass with a constant kinematic viscosity:

    Sigma(x, tau) = M/(pi R0^2) * (1/(tau x^{1/4})) * I_{1/4}(2x/tau) * exp(-(1+x^2)/tau)
    x = R/R0,   tau = 12 nu t / R0^2

Written that way it cannot be evaluated in double precision at the small tau the test
starts from: the Bessel function overflows while the exponential underflows. Using the
exponentially scaled Bessel function and combining the exponents fixes it exactly,
since -(1+x^2) + 2x = -(1-x)^2. scipy provides the scaled function as `ive`.

Two measurements come out of it:

  nu > 0   the L1 distance between the simulated and analytic profile, and how it falls
           under refinement. This tests the viscous operator against an exact solution.

  nu = 0   nothing in the equations can spread the ring, so whatever spreading appears is
           the scheme's own. Fitting the analytic family to it gives an effective tau at
           each output, and the slope of tau against t gives nu_num as a NUMBER. A global
           angular-momentum budget can only bound it, because that budget also collects
           boundary fluxes and floor activations.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ive
from scipy.optimize import minimize_scalar


def sigma(r, tau, r0=1.0, mass_norm=1.0):
    """Analytic ring surface density, in the numerically stable form."""
    r = np.asarray(r, dtype=float)
    x = r / r0
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        out = (mass_norm / (tau * np.power(x, 0.25))
               * ive(0.25, 2.0 * x / tau) * np.exp(-((1.0 - x) ** 2) / tau))
    return np.where(np.isfinite(out), out, 0.0)


def normalisation(sigma_peak, tau0, r0=1.0):
    """M/(pi R0^2) such that the analytic peak at x = 1 equals sigma_peak.

    Fixed from the analytic value rather than from the grid, so the normalisation is the
    same at every resolution - which a convergence study needs it to be.
    """
    return sigma_peak / sigma(r0, tau0, r0, 1.0)


def tau_of_time(t, nu, r0=1.0, tau0=0.018):
    return tau0 + 12.0 * nu * t / (r0 * r0)


def fit_tau(r, sig, mass_norm, r0=1.0, bracket=(1e-3, 3.0), background=0.0):
    """The tau whose analytic profile best matches a measured one, in L1.

    L1 rather than L2 on purpose: the profile spans decades, and a least-squares fit
    would be decided entirely by the few cells at the peak.
    """
    s = np.asarray(sig, dtype=float) - background

    def cost(log_tau):
        model = sigma(r, np.exp(log_tau), r0, mass_norm)
        return float(np.abs(model - s).sum())

    res = minimize_scalar(cost, bounds=(np.log(bracket[0]), np.log(bracket[1])),
                          method="bounded", options={"xatol": 1e-6})
    return float(np.exp(res.x)), float(res.fun)


def l1_error(r, sig, tau, mass_norm, r0=1.0, background=0.0):
    """Mean |Sigma_sim - Sigma_analytic|, normalised by the analytic peak."""
    model = sigma(r, tau, r0, mass_norm)
    peak = model.max()
    if peak <= 0:
        return np.nan
    return float(np.abs((np.asarray(sig) - background) - model).mean() / peak)


def numerical_viscosity(times, taus, r0=1.0):
    """nu_num from the slope of tau against t: tau = tau0 + 12 nu t / R0^2.

    Returns the fitted nu, the intercept it implies, and the R^2 that says whether the
    spread really behaved like a viscosity at all - a scheme whose error is dispersive
    rather than diffusive would not give a straight line here, and reporting the slope
    without the R^2 would hide that.
    """
    t = np.asarray(times, dtype=float)
    ta = np.asarray(taus, dtype=float)
    good = np.isfinite(t) & np.isfinite(ta)
    if good.sum() < 3:
        return dict(nu=np.nan, tau0=np.nan, r2=np.nan, n=int(good.sum()))
    slope, icept = np.polyfit(t[good], ta[good], 1)
    pred = slope * t[good] + icept
    ss_tot = ((ta[good] - ta[good].mean()) ** 2).sum()
    r2 = 1.0 - ((ta[good] - pred) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return dict(nu=float(slope * r0 * r0 / 12.0), tau0=float(icept),
                r2=float(r2), n=int(good.sum()))


def alpha_equivalent(nu, model, r=1.0):
    """The alpha that would produce this nu at radius r in the torus problem.

    The ring test measures nu in its own units (GM = R0 = 1). Quoting it as an alpha is
    what makes it comparable to the alpha the torus runs actually use.
    """
    return nu * np.sqrt(model.beta) / (model.gamma * model.pr_mid * r ** 1.5)
