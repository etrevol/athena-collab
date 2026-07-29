"""Papaloizou-Pringle torus: scalings, profiles and the grid they imply.

One source of truth for the dimensionless model, imported by the notebooks and by
make_athinput.py, so a number cannot be right in one place and wrong in another.

    python3 disk_model.py     # write inputs/hydro/athinput.acc_disk_visc and check it

That is the whole workflow: edit the RUN block below, run this file, and the input
file the simulation reads is regenerated from the model and verified in one step.
Use make_athinput.py directly only to pass one-off parameters without editing anything.

    from disk_model import DiskModel
    m = DiskModel(alpha=0.01, T_0=3e4)     # any physical input can be overridden
    m.beta, m.r_in, m.r_out, m.P_orb       # derived quantities as attributes
    m.rho(r), m.p(r), m.v_phi(r)           # radial profiles as the pgen builds them
    m.grid()                               # x1min/x1max that enclose the torus
    m.summary()                            # human-readable report

Reference: docs/astroformular/sections/{gravity,thermodynamics}_scaling.tex
"""

from dataclasses import dataclass, field, asdict
import pathlib
import sys

import numpy as np

# ============================================================================
# RUN CONFIGURATION - this is the block to edit, then `python3 disk_model.py`
# ============================================================================
# Physics: anything DiskModel accepts. Omitted keys keep the dataclass defaults.
RUN_PHYSICS = dict(
    alpha=0.001,        # Shakura-Sunyaev viscosity; H/r and N_orbits follow from it
)
# Resolution and run length.
RUN_SETUP = dict(
    orbits=100.0,       # run length, in orbits at r_center
    frames_per_orbit=10.0,
    nx1=176, nx2=128,
    meshblock=None,     # (n1, n2), or None for a single block
    cfl=0.4,
    fmt="tab",          # "tab" or "hdf5"
)
# Where the generated file goes, relative to the repository root.
RUN_OUTPUT = "inputs/hydro/athinput.acc_disk_visc"
# ============================================================================

# CGS
C_LIGHT = 2.99792458e10     # cm/s
K_B = 1.380649e-16          # erg/K
M_P = 1.67262192e-24        # g
G_GRAV = 6.67430e-8         # cm^3 g^-1 s^-2
M_SUN = 1.98841e33          # g
PC = 3.0856775814913673e18  # cm
YR = 3.15576e7              # s


@dataclass
class DiskModel:
    """Physical inputs; everything else is derived."""

    # target object
    M_bh: float = 4.5e7      # black hole mass [M_sun]
    T_0: float = 5.0e4       # reference temperature [K]
    mu: float = 0.6          # mean molecular weight; must match T_0, see __post_init__
    chi: float = 5.0e2       # r_0 / r_g
    rho_0: float = 1.0e-13   # reference density [g/cm^3]

    # torus geometry / thermodynamics
    gamma: float = 1.3       # adiabatic index
    r_center: float = 1.0    # density maximum, in units of r_0
    C_prime: float = 0.2     # thickness parameter, must be < 0.5

    # viscosity
    alpha: float = 0.0       # Shakura-Sunyaev alpha

    # ambient medium (a real equilibrium state, not the floor)
    rho_atm: float = 1.0e-4  # ambient density
    t_atm_frac: float = 1.0  # ambient p/rho in units of the torus mid-plane value

    _d: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.C_prime >= 0.5:
            raise ValueError(f"C_prime={self.C_prime} must be < 0.5 for a closed torus")
        if self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must exceed 1")
        # mu is not free: it is fixed by the ionization state at T_0, and it enters
        # cs0 ~ 1/sqrt(mu), hence beta and every code-to-cgs conversion. Molecular
        # (mu ~ 2.3) and fully ionized (mu ~ 0.6) are different physical regimes.
        if self.mu > 1.0 and self.T_0 > 1.0e4:
            raise ValueError(
                f"mu={self.mu} is molecular but T_0={self.T_0:.3g} K ionizes hydrogen; "
                "use mu ~ 0.6 above ~1e4 K, or mu ~ 2.3 below ~2e3 K")
        if self.mu < 1.0 and self.T_0 < 3.0e3:
            raise ValueError(
                f"mu={self.mu} assumes ionized gas but T_0={self.T_0:.3g} K is molecular")
        self._d = self._derive()

    # -- derived quantities -------------------------------------------------

    def _derive(self):
        cs0_sq = self.gamma * K_B * self.T_0 / (self.mu * M_P)
        cs0 = np.sqrt(cs0_sq)
        n = 1.0 / (self.gamma - 1.0)

        r_g = 2.0 * G_GRAV * self.M_bh * M_SUN / C_LIGHT**2   # Schwarzschild
        L_0 = self.chi * r_g
        beta = C_LIGHT**2 / (2.0 * self.chi * cs0_sq)

        disc = 1.0 - 2.0 * self.C_prime
        r_in = self.r_center * (1.0 - np.sqrt(disc)) / (2.0 * self.C_prime)
        r_out = self.r_center * (1.0 + np.sqrt(disc)) / (2.0 * self.C_prime)

        # orbital period at the density maximum; there v_phi is Keplerian
        P_orb = 2.0 * np.pi * np.sqrt(self.r_center**3 / beta)

        f_c = 0.5 - self.C_prime
        cs2_atm = self.t_atm_frac * beta * f_c / (self.r_center * (n + 1.0))

        # nu = alpha*gamma/sqrt(beta) * (p/rho) * r^(3/2), evaluated at r_center
        nu = (self.alpha * self.gamma * np.sqrt(beta) * f_c
              / (n + 1.0) * np.sqrt(self.r_center))
        tau_visc = self.r_center**2 / nu if nu > 0 else np.inf

        return dict(
            cs0=cs0, n_poly=n, beta=beta, r_g=r_g, L_0=L_0,
            V_0=cs0, T_scale=L_0 / cs0,
            mass_scale=self.rho_0 * L_0**3 / M_SUN,
            mdot_scale=self.rho_0 * L_0**2 * cs0 * YR / M_SUN,
            r_in=r_in, r_out=r_out, P_orb=P_orb, f_center=f_c,
            cs2_atm=cs2_atm, p_atm=self.rho_atm * cs2_atm,
            nu_iso=nu, tau_visc=tau_visc,
            N_orbits_visc=tau_visc / P_orb if nu > 0 else np.inf,
        )

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_d")[name]
        except KeyError:
            raise AttributeError(name) from None

    # -- radial profiles ----------------------------------------------------

    def f(self, r):
        """Torus shape function; positive inside the torus."""
        x = self.r_center / np.asarray(r, dtype=float)
        return x - 0.5 * x * x - self.C_prime

    def rho_disk(self, r):
        """Torus density, zero outside the surface."""
        f = self.f(r)
        return np.where(f > 0, np.power(np.clip(f, 0, None) / self.f_center,
                                        self.n_poly), 0.0)

    def p_disk(self, r):
        """Torus pressure, zero outside the surface."""
        f = self.f(r)
        norm = self.beta / (self.r_center * (self.n_poly + 1.0)
                            * self.f_center**self.n_poly)
        return np.where(f > 0, norm * np.power(np.clip(f, 0, None),
                                               self.n_poly + 1.0), 0.0)

    def rho(self, r):
        """Total density: torus plus ambient medium, as the pgen builds it."""
        return self.rho_disk(r) + self.rho_atm

    def p(self, r):
        """Total pressure: torus plus ambient medium."""
        return self.p_disk(r) + self.p_atm

    def v_phi(self, r):
        """Rotation from exact radial force balance, v^2 = beta/r + (r/rho) dp/dr.

        Reduces to the l = const torus inside and to Keplerian in the ambient.
        """
        r = np.asarray(r, dtype=float)
        f = self.f(r)
        fp = -self.r_center / r**2 + self.r_center**2 / r**3
        norm = self.beta / (self.r_center * (self.n_poly + 1.0)
                            * self.f_center**self.n_poly)
        dp = np.where(f > 0, norm * (self.n_poly + 1.0)
                      * np.power(np.clip(f, 0, None), self.n_poly) * fp, 0.0)
        v2 = self.beta / r + (r / self.rho(r)) * dp
        return np.sqrt(np.clip(v2, 0, None))

    def H_over_r(self, r, disk_only=False):
        """Scale height ratio c_s/v_K; the thin-disk assumption needs this << 1.

        Uses the total p and rho, so past the torus surface it reports the AMBIENT
        medium (c_s there is constant while v_K keeps falling, which makes the ratio
        rise to a value that says nothing about the torus). Pass disk_only=True for
        the torus alone; it is NaN outside the surface.

        At the density maximum the torus value reduces to sqrt((gamma-1)*(0.5-C')),
        independent of mass, temperature, mu and beta.
        """
        r = np.asarray(r, dtype=float)
        if disk_only:
            with np.errstate(invalid="ignore", divide="ignore"):
                cs = np.sqrt(self.gamma * self.p_disk(r) / self.rho_disk(r))
        else:
            cs = np.sqrt(self.gamma * self.p(r) / self.rho(r))
        return cs / np.sqrt(self.beta / r)

    # -- grid ---------------------------------------------------------------

    def grid(self, pad_in=0.62, pad_out=1.25, nx1=176, nx2=128):
        """Radial domain that encloses the torus, plus the matching resolution.

        The torus surface (rho, p, c_s -> 0) has to be interior: a zero-gradient
        boundary placed on it is ill posed and drains the disk. Defaults leave
        ~7 cells inside r_in and ~38 outside r_out at nx1=176.
        """
        x1min = round(self.r_in * pad_in, 2)
        x1max = round(self.r_out * pad_out, 2)
        return dict(x1min=x1min, x1max=x1max, nx1=nx1, nx2=nx2,
                    dr=(x1max - x1min) / nx1,
                    cells_inside=int((self.r_in - x1min) / ((x1max - x1min) / nx1)),
                    cells_outside=int((x1max - self.r_out) / ((x1max - x1min) / nx1)))

    # -- reporting ----------------------------------------------------------

    def summary(self):
        d = self._d
        L = []
        add = L.append
        add("physical inputs")
        add(f"  M_bh          {self.M_bh:.3e} M_sun    T_0   {self.T_0:.3e} K")
        add(f"  mu            {self.mu:<12.4g}      chi   {self.chi:.3e}")
        add(f"  rho_0         {self.rho_0:.3e} g/cm3   gamma {self.gamma}")
        add("dimensionless")
        add(f"  cs0           {d['cs0']:.6e} cm/s")
        add(f"  n_poly        {d['n_poly']:.6f}")
        add(f"  beta          {d['beta']:.6e}")
        add(f"  r_in / r_out  {d['r_in']:.6f} / {d['r_out']:.6f}")
        add(f"  P_orb         {d['P_orb']:.6e}")
        add("physical scales")
        add(f"  r_g           {d['r_g']:.6e} cm      L_0  {d['L_0']:.6e} cm"
            f"  ({d['L_0'] / PC:.4f} pc)")
        add(f"  T_scale       {d['T_scale']:.6e} s      ({d['T_scale'] / YR:.1f} yr)")
        add(f"  mass_scale    {d['mass_scale']:.4f} M_sun per code unit")
        add(f"  mdot_scale    {d['mdot_scale']:.6f} M_sun/yr per code unit")
        add("viscosity")
        if self.alpha > 0:
            add(f"  alpha         {self.alpha}")
            add(f"  nu_iso        {d['nu_iso']:.6e}")
            add(f"  tau_visc      {d['tau_visc']:.6e}  ({d['N_orbits_visc']:.1f} orbits)")
        else:
            add("  alpha         0  (inviscid; nu_iso must be 0 too)")
        return "\n".join(L)

    def as_dict(self):
        out = {k: v for k, v in asdict(self).items() if k != "_d"}
        out.update(self._d)
        return out


def orbits_to_accrete(alpha, gamma, C_prime):
    """N = 1 / (2 pi alpha (gamma-1) (0.5-C')) - independent of mass and temperature."""
    if alpha <= 0:
        return np.inf
    return 1.0 / (2.0 * np.pi * alpha * (gamma - 1.0) * (0.5 - C_prime))


def repo_root(start=None):
    """Walk up to the directory holding configure.py, so this survives being moved."""
    p = pathlib.Path(start or __file__).resolve()
    for d in [p] + list(p.parents):
        if (d / "configure.py").is_file():
            return d
    raise FileNotFoundError("no configure.py above " + str(p) + ": not inside the repo")


def report(m):
    """Everything worth seeing about a model, as one block of text."""
    g = m.grid()
    rr = np.linspace(m.r_in, m.r_out, 400)
    body = m.rho_disk(rr) > 10.0 * m.rho_atm
    hr_c = np.sqrt((m.gamma - 1.0) * (0.5 - m.C_prime))
    L = [m.summary(),
         "",
         f"grid  x1 [{g['x1min']}, {g['x1max']}]  nx1={g['nx1']}  dr={g['dr']:.6f}",
         f"      {g['cells_inside']} cells inside r_in, "
         f"{g['cells_outside']} outside r_out",
         f"H/r at r_center: {m.H_over_r(m.r_center):.4f}"
         f"  = sqrt((gamma-1)(0.5-C')) = {hr_c:.4f}",
         f"max H/r in the torus body: "
         f"{np.nanmax(m.H_over_r(rr[body], disk_only=True)):.4f}"
         f"   (past the surface the ratio reports the ambient, not the torus)"]
    return "\n".join(L)


def main():
    """Generate the athinput from the RUN_* block above, then verify it."""
    import make_athinput
    import verify_athinput

    m = DiskModel(**RUN_PHYSICS)
    out = repo_root() / RUN_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(make_athinput.build(m, **RUN_SETUP))

    print(report(m))
    print(f"\nwrote {out}\n")

    _, rows = verify_athinput.check(str(out))
    width = max(len(n) for _, n, _ in rows)
    n_fail = n_warn = 0
    for status, name, detail in rows:
        n_fail += status == verify_athinput.FAIL
        n_warn += status == verify_athinput.WARN
        mark = {"ok": "  ok  ", "warn": " warn ", "FAIL": " FAIL "}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")
    print("\n" + ("all checks passed" if not n_fail and not n_warn else
                  f"{n_fail} failure(s), {n_warn} warning(s)"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
