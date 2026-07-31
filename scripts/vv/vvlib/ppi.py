"""Linear stability of the 2D constant-l torus, and the mode content of a snapshot.

THEORY ARM. Papaloizou & Pringle (1984), eq. (4.1), with d/dz -> 0 for our 2D (r, phi)
problem. With W = p'/[rho (sigma + m Omega)] the perturbation equation is

    (1/r) d/dr ( rho r dW/dr ) - (m^2/r^2) rho W = - (sigma + m Omega)^2 rho^2 W / (gamma p)

which is quadratic in sigma. Writing it as PP84 (4.2)-(4.5) do,

    sigma^2 A W + sigma B W + C W = 0,
    A = rho^2/(gamma p),   B = 2 m Omega A,   C = L[.] + m^2 Omega^2 A,
    L[W] = (1/r)(rho r W')' - (m^2/r^2) rho W

a quadratic eigenvalue problem, linearised here into a 2N generalised one. Perturbations
go as exp[i(m phi + sigma t)], so a mode grows when Im(sigma) < 0, at rate -Im(sigma),
and its pattern speed is -Re(sigma)/m.

Two things make this tractable and trustworthy:

* rho vanishes at both torus surfaces, so on a cell-centred grid whose outer half-faces
  land exactly on r_in and r_out the conservative form of L has zero flux there by
  construction. That IS the regularity condition PP84 (4.25) states, imposed by the
  discretisation rather than bolted on, and it is why no explicit boundary condition
  appears below.
* the generalised form sigma*N*x = M*x is kept rather than inverting A. A ~ rho^(2-gamma)
  vanishes at the surfaces, so A^-1 is badly conditioned there; the generalised solver
  returns those directions as infinite eigenvalues instead of as noise.

The equilibrium here is the BARE torus - no ambient medium, unlike the simulation. That
difference is real and is why the comparison has a tolerance rather than being exact.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def _profiles(model, r):
    """rho, p and Omega of the bare torus on the given radii."""
    rho = model.rho_disk(r)
    p = model.p_disk(r)
    l0 = np.sqrt(model.beta * model.r_center)     # constant specific angular momentum
    omega = l0 / r**2
    return rho, p, omega


def spectrum(model, m, n=400):
    """Complex eigenfrequencies sigma for azimuthal mode m, on n cells."""
    r_in, r_out = model.r_in, model.r_out
    h = (r_out - r_in) / n
    r = r_in + (np.arange(n) + 0.5) * h            # cell centres
    rf = r_in + np.arange(n + 1) * h               # faces: rf[0]=r_in, rf[-1]=r_out

    rho, p, omega = _profiles(model, r)
    rho_f = model.rho_disk(rf)                     # exactly 0 at rf[0] and rf[-1]

    a = rho**2 / (model.gamma * p)                 # = rho / c_s^2

    # L[W] in conservative form. The i = 0 and i = n-1 rows lose their outer flux because
    # rho_f vanishes there, which is the regularity condition.
    coef = rho_f * rf / h**2
    lop = np.zeros((n, n))
    idx = np.arange(n)
    lop[idx, idx] = -(coef[:-1] + coef[1:]) / r - (m**2 / r**2) * rho
    lop[idx[:-1], idx[:-1] + 1] = coef[1:-1] / r[:-1]
    lop[idx[1:], idx[1:] - 1] = coef[1:-1] / r[1:]

    a_mat = np.diag(a)
    b_mat = np.diag(2.0 * m * omega * a)
    c_mat = lop + np.diag(m**2 * omega**2 * a)

    # sigma^2 A W + sigma B W + C W = 0  ->  sigma N x = M x,  x = [sigma W; W]
    zero, ident = np.zeros((n, n)), np.eye(n)
    big_m = np.block([[-b_mat, -c_mat], [ident, zero]])
    big_n = np.block([[a_mat, zero], [zero, ident]])

    sig = scipy.linalg.eig(big_m, big_n, right=False)
    return sig[np.isfinite(sig)]


def unstable_modes(model, m, n=400, min_rate=1e-6):
    """Growing modes, fastest first, as (growth_rate, pattern_speed) in code units."""
    sig = spectrum(model, m, n)
    grow = -sig.imag
    sel = grow > min_rate
    out = [(float(g), float(-s.real / m)) for g, s in zip(grow[sel], sig[sel])]
    return sorted(out, key=lambda t: -t[0])


def fastest_mode(model, m, n=400, n_check=None, rtol=0.05):
    """Fastest growing mode, cross-checked at a second resolution.

    PP84 ran two independent numerical methods for exactly this reason. Halving the grid
    is the cheap version of that: a rate that moves when the grid changes is a property
    of the discretisation, not of the torus, and is reported as unconverged.
    """
    n_check = n_check or n // 2
    fine = unstable_modes(model, m, n)
    coarse = unstable_modes(model, m, n_check)
    if not fine or not coarse:
        return dict(m=m, rate=0.0, pattern=np.nan, converged=bool(not fine and not coarse),
                    rel_change=0.0, n_modes=len(fine))
    rate, pattern = fine[0]
    rel = abs(rate - coarse[0][0]) / max(rate, 1e-300)
    return dict(m=m, rate=rate, pattern=pattern, converged=rel < rtol,
                rel_change=rel, n_modes=len(fine))


def corotation_inside(model, pattern_speed):
    """A genuine Papaloizou-Pringle mode corotates somewhere inside the torus."""
    l0 = np.sqrt(model.beta * model.r_center)
    return (l0 / model.r_out**2) <= pattern_speed <= (l0 / model.r_in**2)


# ---------------------------------------------------------------- snapshot diagnostics


def mode_amplitudes(r, phi, rho, m_max=8, r_lo=None, r_hi=None):
    """Amplitude of azimuthal mode m, m = 0 .. m_max, as an L2 norm over radius.

    The Fourier transform is taken at each radius FIRST, and only then combined across
    radii. Integrating over radius first - the obvious thing to do - lets the positive
    and negative radial lobes of the eigenfunction cancel, and as the pattern winds up
    with the differential rotation that cancellation oscillates. The effect grows with m,
    because higher modes have more radial structure: measured on these runs it left m = 3
    with R^2 = 0.87 and a rate 11% above theory, while m = 1 and m = 2 were unaffected.
    Norming after the transform removes it (m = 3: R^2 = 0.998, 1.9% below theory).
    """
    sel = np.ones_like(r, dtype=bool)
    if r_lo is not None:
        sel &= r >= r_lo
    if r_hi is not None:
        sel &= r <= r_hi
    rr, block = r[sel], rho[:, sel]
    amp_r = np.abs(np.fft.rfft(block, axis=0)) / block.shape[0]     # (m, r)
    return np.sqrt((amp_r[: m_max + 1] ** 2 * rr[None, :]).sum(axis=1))


def fit_growth(times, amps, t_lo, t_hi):
    """Least-squares d ln A / dt over a window, with the fit quality that justifies it."""
    sel = (times >= t_lo) & (times <= t_hi) & (amps > 0)
    if sel.sum() < 4:
        return dict(rate=np.nan, r2=np.nan, n=int(sel.sum()), window=(t_lo, t_hi))
    x, y = times[sel], np.log(amps[sel])
    slope, icept = np.polyfit(x, y, 1)
    resid = y - (slope * x + icept)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid**2).sum() / ss_tot if ss_tot > 0 else np.nan
    return dict(rate=float(slope), r2=float(r2), n=int(sel.sum()),
                window=(float(x[0]), float(x[-1])))


#: |A_m| / A_0 above which the perturbation is no longer small compared to the
#: axisymmetric background, and linear theory no longer describes it
LINEAR_CEILING = 0.02


def best_growth_window(times, amps, background=None, min_span=2.0, min_points=6,
                       ceiling=LINEAR_CEILING):
    """Find the window that is actually exponential, instead of assuming one.

    A seed placed in v_r takes a fraction of an orbit to imprint on the density, and the
    mode leaves the linear regime well before it stops growing. Both ends of a hand-picked
    window therefore contaminate the slope. Scanning every window of at least `min_span`
    orbits and keeping the straightest one finds the linear phase from the data, and the
    R^2 says whether such a phase existed at all.

    Straightness alone is not enough, which is why `background` (the m = 0 amplitude) is
    needed. A saturating exponential is at its straightest near the top, so the scan
    drifts into the weakly non-linear regime, where the curve is still smooth and the
    rate is already depressed. Measured: extending the runs from 6 to 9 orbits moved the
    m = 2 window from 2.5-6.0 to 4.5-6.5 orbits and the error from +0.6% to +1.9%, and
    the small-seed run from -1.8% to -4.2%. Capping |A_m|/A_0 restores both to under 0.5%.
    Longer runs do not improve this measurement; the ceiling does.
    """
    good = amps > 0
    if background is not None:
        good &= (amps / np.where(background > 0, background, np.inf)) < ceiling
    t, a = times[good], amps[good]
    best = dict(rate=np.nan, r2=-np.inf, n=0, window=(np.nan, np.nan))
    for i in range(len(t)):
        for j in range(i + min_points - 1, len(t)):
            if t[j] - t[i] < min_span:
                continue
            fit = fit_growth(t, a, t[i], t[j])
            # prefer the straighter fit, and among equals the longer one
            if fit["r2"] > best["r2"] + 1e-6 or (
                    abs(fit["r2"] - best["r2"]) <= 1e-6 and fit["n"] > best["n"]):
                best = fit
    return best
