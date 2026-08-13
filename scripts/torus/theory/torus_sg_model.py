"""Self-gravitating Papaloizou-Pringle torus: the hydrodynamic analogue of the
collisionless torus of Bannikova et al. (2026, A&A; arXiv:2604.11528).

    python3 torus_sg_model.py     # write inputs/hydro/athinput.sg_torus_m1 and check it

The defaults are the near-Keplerian reproduction of the paper (phase 2). The constant-l
torus that was used to characterise the Papaloizou-Pringle instability first (phase 1,
tagged phase1-ppi) is the same model with three parameters changed, and is regenerated
with

    python3 torus_sg_model.py --q_rot 0 --eps_soft 0.25 \
        --pert_amp 3e-3 --pert_mode_amp 0 -o inputs/hydro/athinput.ppi_torus

The paper follows a collisionless torus of N massive particles orbiting a dominant
central mass and finds that a global m = 1 slow mode grows spontaneously out of an
axisymmetric state. This module sets up the same experiment for a *gas* torus, so the
same question can be asked in the hydrodynamic regime.

The translation between the two descriptions is the point of this file:

    N-body (paper)                  gas torus (here)
    ---------------------------------------------------------------------------
    G = 1, M_c = 1, R_tor = 1       the same units; T_orb = 2 pi at R = 1
    M_tor / M_c = 0.1               M_tor, the mass that the Poisson solver sees
    e_max  (radial dispersion)      pressure support, set by C' and q through Psi_c
    i_max  (vertical dispersion)    the half-thickness; z_max ~ sin(i_max), which
                                    i_max_equivalent() reports
    R_0    (semi-major axis spread) the torus minor radius, r_out - r_in
    Keplerian particle orbits       q_rot, the rotation-law exponent
    softening eps = 1e-2            eps_soft for the central point mass only; the gas
                                    self-gravity is a grid Poisson solve

Two geometric parameters here (C', q) stand in for three there (e_max, i_max, R_0),
because in a polytropic torus the radial extent, the thickness and the internal pressure
all follow from one equilibrium. That is a real loss of freedom, and the runs scan
C' and q rather than the paper's three-parameter grid.

Geometry. In units G = M_c = R_tor = 1, with l = l_0 (R/R_tor)^q, the torus is where

    Psi(R, z) = 1/sqrt(R^2 + z^2 + eps^2) - l_0^2 R^(2q-2)/(2-2q) - C'  >  0,

    (n + 1) p / rho = Psi,    rho = rho_c (Psi/Psi_c)^n,    n = 1/(gamma - 1),

with l_0^2 = (1 + eps^2)^(-3/2) putting the pressure maximum at R = 1, and rho_c fixed
by requiring the total torus mass to be M_tor. At q = 0 and eps = 0 this collapses to
x - x^2/2 - C' in the midplane (x = 1/R) and Psi_c = 0.5 - C': exactly the shape function
of the 2D model in disk_model.py, so C' carries over unchanged.

Two choices in that formula are deliberate and were both learned the hard way:

  * the SOFTENED potential appears in Psi, not the point-mass one. The torus never
    enters the softening region, which makes the point-mass version look safe, but at
    eps = 0.25 and r_in = 0.56 the softened force is only 76% of the point-mass force at
    the inner edge - a torus balanced against the latter starts 24% out of equilibrium.

  * q exists because a gas torus with l = const is Papaloizou-Pringle unstable, with
    m = 1 as its fastest growing mode: the same symmetry as the mode being looked for,
    and with no counterpart in a collisionless system. See the q_rot field.

Note what is deliberately NOT done: the initial state balances pressure, rotation and
the *central* mass only, not the torus's own gravity. It is therefore out of equilibrium
at the O(M_tor / M_c) level and will virialise. That mirrors the paper, whose initial
particle distribution is likewise not a self-consistent equilibrium and whose m = 1 mode
only appears after the virial oscillations decay (their Sect. 3.2, Fig. 4).
"""

from dataclasses import dataclass, field, asdict
import pathlib
import sys

import numpy as np

# ============================================================================
# RUN CONFIGURATION - resolution and run length, then `python3 torus_sg_model.py`
# ============================================================================
# The PHYSICS lives in the TorusModel field defaults below and nowhere else.
RUN_SETUP = dict(
    orbits=100.0,        # run length, in orbits at R_tor
    frames_per_orbit=2.0,
    nx=96,               # cells per dimension of the cube
    meshblock=32,        # MeshBlock size; must divide nx, and Multigrid wants 2^k
    cfl=0.3,
    nthreads=15,        # OpenMP threads; MeshBlocks are the unit of parallelism
    fmt="hdf5",          # 3D data: tab output would be unusable
)
RUN_OUTPUT = "inputs/hydro/athinput.sg_torus_m1"
# ============================================================================


@dataclass
class TorusModel:
    """Dimensionless inputs (G = M_c = R_tor = 1); everything else is derived."""

    # geometry and thermodynamics
    gamma: float = 5.0 / 3.0   # adiabatic index; 5/3 for the collisionless analogue
    # 0.18 with q = 0.30 puts the half-thickness at z_max = 1.083, which is the
    # equivalent of i_max = 60 deg - their canonical run, where the mode is strong.
    C_prime: float = 0.18      # thickness parameter

    # Rotation-law exponent, l = l_0 (R/R_tor)^q. q = 0 is the constant-angular-momentum
    # torus of the 2D model - and the most Papaloizou-Pringle unstable case there is,
    # with m = 1 as its fastest growing mode: the same symmetry as the mode being looked
    # for, and absent from the collisionless problem. The instability weakens towards
    # Keplerian rotation (q -> 1/2), which is also what the paper's particle orbits are.
    # The limit is Psi_c > 0, i.e. q < 1 - 1/(2(1-C')): a pressure-supported torus cannot
    # be both thick and Keplerian, because for a fluid the thickness IS the departure
    # from Keplerian rotation. The collisionless torus escapes this because its thickness
    # comes from inclination dispersion, which is decoupled from the mean rotation law.
    q_rot: float = 0.30
    M_tor: float = 0.1         # torus mass in units of the central mass

    # numerics that are physics-adjacent enough to belong to the model
    eps_soft: float = 0.17     # softening of the central point mass
    # Ambient density, in units of rho_c. Not a free knob: at 1e-6 the cell ahead of
    # the torus surface was swept to the density floor within one step and the timestep
    # collapsed by six orders of magnitude. The 2D model records the same trap.
    rho_atm_frac: float = 1e-4
    t_atm_frac: float = 1.0    # ambient p/rho in units of the torus mid-plane value

    # Density seeds. An N-body torus carries Poisson noise, which in each azimuthal
    # harmonic is A_m/Sigma_0 ~ 1/sqrt(2N) ~ 2e-3 for their N = 128k. A grid started from
    # an analytic profile has none (measured: A_1 ~ 2e-16), so a seed is needed or there
    # is nothing to grow.
    #
    # pert_amp is per-cell white noise, and it is a weak way to do this: spread over the
    # N_cell ~ 3e5 cells inside the torus it averages down to A_m ~ pert_amp/sqrt(2 N_cell),
    # so 3e-3 delivers only 4e-6 - 500x quieter than the N-body, and matching would need
    # pert_amp = 1.5, which is not a perturbation. Most of its power is at high k anyway.
    #
    # pert_mode_amp seeds harmonics m = 1..5 coherently, each with its own random phase,
    # which is what Poisson noise looks like at low m. Set it to ~1/sqrt(2N) of the
    # N-body run being compared against. seed_mode_amplitude() reports what either choice
    # actually delivers, and the checks turn that into e-foldings.
    pert_amp: float = 0.0
    pert_mode_amp: float = 2.0e-3

    # viscosity: 0 is the collisionless analogue, > 0 is the deliberate departure
    alpha: float = 0.0

    self_gravity: bool = True  # False reproduces their Appendix C control

    _d: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        q_max = 1.0 - 1.0 / (2.0 * (1.0 - self.C_prime))
        if self.q_rot >= q_max:
            raise ValueError(
                f"q_rot={self.q_rot} gives no closed torus at C_prime={self.C_prime}: "
                f"Psi_c = 1 - 1/(2-2q) - C' must be positive, so q < {q_max:.4f}. "
                "A pressure-supported torus cannot be both thick and Keplerian.")
        if self.C_prime >= 0.5:
            raise ValueError(f"C_prime={self.C_prime} must be < 0.5 for a closed torus")
        if self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must exceed 1")
        if self.M_tor <= 0.0:
            raise ValueError("M_tor must be positive; use self_gravity=False for the "
                             "no-self-gravity control, so the torus still has inertia")
        self._d = self._derive()

    # -- derived quantities -------------------------------------------------

    def _derive(self):
        n = 1.0 / (self.gamma - 1.0)
        Psi_c = float(self.Psi(1.0, 0.0))

        r_in, r_out = self._midplane_edges()

        T_orb = 2.0 * np.pi                     # at R = R_tor = 1, l^2 = 1
        Omega_c = 1.0

        # rho_c from the mass integral, done on a (R, z) quadrature of the torus body
        shape = self._shape_integral(r_in, r_out, n, Psi_c)
        rho_c = self.M_tor / shape

        z_max = self._half_thickness()

        # mid-plane sound speed at the density maximum
        cs_c = np.sqrt(self.gamma * Psi_c / (n + 1.0))
        H_over_R = np.sqrt((self.gamma - 1.0) * Psi_c)   # reduces to the 2D identity

        return dict(
            n_poly=n, Psi_c=Psi_c, r_in=r_in, r_out=r_out, T_orb=T_orb,
            Omega_c=Omega_c, rho_c=rho_c, shape_integral=shape, z_max=z_max,
            cs_c=cs_c, H_over_R=H_over_R,
            p_c=rho_c * Psi_c / (n + 1.0),
            rho_atm=self.rho_atm_frac * rho_c,
            minor_radius=0.5 * (r_out - r_in),
        )

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_d")[name]
        except KeyError:
            raise AttributeError(name) from None

    # -- the torus itself ----------------------------------------------------

    def Psi(self, R, z=0.0):
        """Torus shape function; positive inside the torus, zero on its surface."""
        R = np.asarray(R, dtype=float)
        z = np.asarray(z, dtype=float)
        e2 = self.eps_soft ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1.0 / np.sqrt(R * R + z * z + e2)
                    - self.l0_sq * np.power(R, 2.0 * self.q_rot - 2.0)
                    / (2.0 - 2.0 * self.q_rot)
                    - self.C_prime)

    @property
    def l0_sq(self):
        """l_0^2 = (1 + eps^2)^(-3/2), which keeps the pressure maximum at R_tor = 1.

        The equilibrium is built on the SAME softened potential the solver applies.
        Using the point-mass one instead is not the harmless simplification it looks
        like: at eps = 0.25 and r_in = 0.56 the softened force is only 76% of the
        point-mass force at the inner edge, so the torus would start 24% out of
        equilibrium there. Reduces to 1 as eps -> 0.
        """
        return (1.0 + self.eps_soft ** 2) ** -1.5

    def rho(self, R, z=0.0):
        """Torus density, zero outside the surface. Ambient is NOT included."""
        Psi = self.Psi(R, z)
        inside = Psi > 0.0
        return np.where(inside, self.rho_c * np.power(np.where(inside, Psi, 0.0)
                                                      / self.Psi_c, self.n_poly), 0.0)

    def p(self, R, z=0.0):
        """Torus pressure, zero outside the surface."""
        Psi = self.Psi(R, z)
        inside = Psi > 0.0
        return np.where(inside, self.p_c * np.power(np.where(inside, Psi, 0.0)
                                                    / self.Psi_c, self.n_poly + 1.0), 0.0)

    def v_phi(self, R):
        """v_phi = l / R with l = l_0 R^q, l_0 fixed by the softened potential."""
        return np.sqrt(self.l0_sq) * np.power(np.asarray(R, dtype=float),
                                              self.q_rot - 1.0)

    def _midplane_edges(self):
        """Roots of Psi(R, 0) = 0. Closed form only at q = 0, so scan and bracket."""
        R = np.geomspace(1.0e-3, 1.0e3, 400000)
        inside = self.Psi(R, 0.0) > 0.0
        if not inside.any():
            raise ValueError("no closed torus for these parameters")
        return float(R[inside].min()), float(R[inside].max())

    def _shape_integral(self, r_in, r_out, n, Psi_c):
        """int (Psi/Psi_c)^n dV over the torus body, so that rho_c = M_tor / this."""
        nR, nz = 2000, 2000
        R = np.linspace(r_in, r_out, nR)
        # the torus is symmetric about z = 0; integrate the upper half and double it
        z_hi = self._half_thickness() * 1.05
        z = np.linspace(0.0, z_hi, nz)
        RR, ZZ = np.meshgrid(R, z, indexing="ij")
        Psi = self.Psi(RR, ZZ)
        w = np.where(Psi > 0.0, np.power(np.where(Psi > 0.0, Psi, 0.0) / Psi_c, n), 0.0)
        # dV = 2 pi R dR dz, doubled for z < 0
        integrand = 2.0 * (2.0 * np.pi) * RR * w
        return np.trapz(np.trapz(integrand, z, axis=1), R)

    def _half_thickness(self):
        """max over R of the z where Psi = 0; the torus's vertical half-extent.

        Psi = 0 gives sqrt(R^2+z^2) = 1 / (R^(2q-2)/(2-2q) + C') directly, for any q.
        """
        r_in, r_out = self._midplane_edges()
        R = np.linspace(r_in, r_out, 20000)[1:-1]
        s = 1.0 / (self.l0_sq * np.power(R, 2.0 * self.q_rot - 2.0)
                   / (2.0 - 2.0 * self.q_rot) + self.C_prime)
        z2 = s * s - R * R - self.eps_soft ** 2
        return float(np.sqrt(np.max(np.clip(z2, 0.0, None))))

    def i_max_equivalent(self, e_max=0.5):
        """The paper's inclination spread that would give this half-thickness.

        Their particles reach |z| ~ r sin(i) at apocentre, with r ~ a(1+e). Their runs
        have a ~ R_tor = 1 and e up to e_max, so a mean-eccentricity estimate gives

            z_max ~ (1 + e_max/2) sin(i_max),

        i.e. 1.08, 0.88 and 0.63 for i_max = 60, 45 and 30 deg in their canonical
        e_max = 0.5. Their mode is strong at 60, weak at 45 and gone by 30, so those are
        the numbers a gas torus has to be placed against. This is a bridge between two
        different notions of thickness and is good to perhaps 20%, not better; it is
        here to keep the comparison honest, not to be precise. NaN when the torus is
        thicker than any inclination could make it.
        """
        s = self.z_max / (1.0 + 0.5 * e_max)
        return float(np.degrees(np.arcsin(s))) if s <= 1.0 else float("nan")

    def seed_mode_amplitude(self, n_cells_torus=None):
        """A_1(t=0)/M_tor that the configured seeds actually deliver.

        White noise averages down over the cells inside the torus; the coherent seed
        does not. Returns (from white noise, from the coherent seed, total).
        """
        if n_cells_torus is None:
            n_cells_torus = self.cells_in_torus()
        white = self.pert_amp / np.sqrt(2.0 * max(n_cells_torus, 1.0))
        return white, self.pert_mode_amp, float(np.hypot(white, self.pert_mode_amp))

    def cells_in_torus(self, nx=None):
        """How many grid cells the torus body occupies, for the noise estimate."""
        g = self.grid(nx=nx or 96)
        # volume where Psi > 0, from the same quadrature the mass integral uses
        R = np.linspace(self.r_in, self.r_out, 800)
        z = np.linspace(0.0, self.z_max * 1.05, 800)
        RR, ZZ = np.meshgrid(R, z, indexing="ij")
        inside = self.Psi(RR, ZZ) > 0.0
        vol = 2.0 * np.trapz(np.trapz(np.where(inside, 2.0 * np.pi * RR, 0.0), z, axis=1), R)
        return vol / g["dx"] ** 3

    # -- grid ----------------------------------------------------------------

    def grid(self, nx=192, pad=1.25, meshblock=None):
        """The cube the torus lives in, and what the resolution buys.

        Isolated (multipole) gravity boundaries want the mass well inside the box, so
        the default pads the outer torus radius by 25%.
        """
        L = round(self.r_out * pad, 3)
        dx = 2.0 * L / nx
        return dict(
            L=L, nx=nx, dx=dx, meshblock=meshblock or nx,
            cells_minor=2.0 * self.minor_radius / dx,   # across the torus radial width
            cells_vertical=2.0 * self.z_max / dx,       # across its full thickness
            cells_soft=self.eps_soft / dx,              # per softening length
            nblocks=(nx // (meshblock or nx))**3,
        )

    # -- checks that have teeth ----------------------------------------------

    def jeans_cells(self, dx):
        """lambda_J / dx at the density maximum. Truelove: keep this above 4."""
        # lambda_J = cs sqrt(pi / (G rho)), G = 1
        lam = self.cs_c * np.sqrt(np.pi / self.rho_c)
        return lam / dx

    def toomre_Q(self):
        """Q = cs kappa / (pi G Sigma) at R_tor; for l = const, kappa = 0 exactly.

        A constant-angular-momentum torus is marginal by construction (the epicyclic
        frequency vanishes), so Q is not the useful stability measure here - it is
        reported only to make that explicit. The relevant parameter is M_tor / M_c.
        """
        return 0.0

    def dt_estimate(self, dx, cfl):
        """The timestep the ambient near the softened centre will impose."""
        v_max = np.sqrt(1.0 / self.eps_soft)     # free-fall speed at the softening scale
        cs_atm = np.sqrt(self.gamma * self.t_atm_frac * self.Psi_c
                         / (self.n_poly + 1.0))
        return cfl * dx / (v_max + cs_atm)

    # -- reporting -----------------------------------------------------------

    def summary(self):
        d = self._d
        L = []
        add = L.append
        add("model (G = M_c = R_tor = 1)")
        add(f"  gamma         {self.gamma:.6f}      n_poly  {d['n_poly']:.6f}")
        add(f"  C_prime       {self.C_prime:<12.4g}  Psi_c   {d['Psi_c']:.6f}")
        add(f"  M_tor / M_c   {self.M_tor:<12.4g}  "
            f"self-gravity {'ON' if self.self_gravity else 'OFF (Appendix C control)'}")
        add("torus geometry")
        add(f"  r_in / r_out  {d['r_in']:.6f} / {d['r_out']:.6f}")
        add(f"  minor radius  {d['minor_radius']:.6f}   "
            f"half-thickness z_max {d['z_max']:.6f}")
        add(f"  z_max / minor {d['z_max'] / d['minor_radius']:.4f}   "
            f"(the paper's i_max analogue: 1.0 is a circular cross-section)")
        add(f"  H/R at R_tor  {d['H_over_R']:.6f}")
        i_eq = self.i_max_equivalent()
        add(f"  q_rot         {self.q_rot:<12.4g}  "
            f"q_max = {1.0 - 1.0 / (2.0 * (1.0 - self.C_prime)):.4f}")
        add(f"  i_max equiv.  " + (f"{i_eq:.1f} deg  (their mode needs >~ 45, dies by 30)"
                                   if i_eq == i_eq
                                   else "thicker than any inclination (z_max > R_tor)"))
        add("normalisation")
        add(f"  rho_c         {d['rho_c']:.6e}   p_c  {d['p_c']:.6e}")
        add(f"  cs at R_tor   {d['cs_c']:.6f}    T_orb {d['T_orb']:.6f}")
        add(f"  rho_atm       {d['rho_atm']:.6e}   ({self.rho_atm_frac:.1e} rho_c)")
        add("viscosity")
        if self.alpha > 0:
            add(f"  alpha         {self.alpha}  (the departure from the collisionless "
                "model)")
        else:
            add("  alpha         0  (collisionless analogue; nu_iso must be 0 too)")
        return "\n".join(L)

    def as_dict(self):
        out = {k: v for k, v in asdict(self).items() if k != "_d"}
        out.update(self._d)
        return out


def repo_root(start=None):
    """Walk up to the directory holding configure.py, so this survives being moved."""
    p = pathlib.Path(start or __file__).resolve()
    for d in [p] + list(p.parents):
        if (d / "configure.py").is_file():
            return d
    raise FileNotFoundError("no configure.py above " + str(p) + ": not inside the repo")


# -- the input file ----------------------------------------------------------

def build_athinput(m, orbits, frames_per_orbit, nx, meshblock, cfl, nthreads, fmt):
    """Render the athinput for this model. Comments name the parameter, not the value."""
    g = m.grid(nx=nx, meshblock=meshblock)
    tlim = orbits * m.T_orb
    dt_out = m.T_orb / frames_per_orbit
    nu_iso = 1.0e-12 if m.alpha > 0 else 0.0   # Athena++ gates viscous fluxes on nu_iso

    grav = ""
    if m.self_gravity:
        grav = f"""
<gravity>
mgmode     = FMG          # Multigrid mode (FMG or MGI)
niteration = 1            # Multigrid iterations after the FMG cycle
#
# Measured on this problem, 128^3, against FMG with automatic convergence control:
#   FMG  niteration=1   6.1x faster   E_grav_self differs by 7e-6   <- used
#   MGI  niteration=4   4.1x faster   differs by 7e-4
#   MGI  niteration=2   5.2x faster   differs by 2.8e-2             <- rejected
# The full multigrid cycle already converges to truncation error, so the extra
# iterations of the automatic mode buy nothing. Multigrid is ~90% of the runtime
# here, so this single line sets the cost of the whole campaign.
ix1_bc     = multipole    # isolated boundaries: the torus is not periodic
ox1_bc     = multipole
ix2_bc     = multipole
ox2_bc     = multipole
ix3_bc     = multipole
ox3_bc     = multipole
mporder    = 4            # multipole order for the boundary values (2 or 4)
"""

    return f"""<comment>
problem   = Self-gravitating Papaloizou-Pringle torus; global m = 1 slow mode
reference = Bannikova et al. 2026, A&A (arXiv:2604.11528), N-body counterpart
configure = --prob=sg_torus_m1 --coord=cartesian{' --grav=mg' if m.self_gravity else ''}
generated = scripts/torus/theory/torus_sg_model.py

<job>
problem_id = sg_torus_m1  # basename of output filenames

<output1>
file_type  = {fmt}         # output format
variable   = prim        # variables to be output
dt         = {dt_out:.8g}  # time increment between outputs

<output2>
file_type  = rst         # output format
dt         = {10.0 * m.T_orb:.8g}  # time increment between outputs
#
# Restart files, because the N-body counterpart runs for 1000 orbits and this one
# starts at 100. Without them, extending a run means recomputing it from t = 0.

<output3>
file_type  = hst         # output format
dt         = {m.T_orb / 20.0:.8g}  # time increment between outputs
data_format = %24.16e    # output format specifier

<time>
cfl_number = {cfl}         # Courant, Friedrichs & Lewy number
nlim       = -1          # cycle limit (-1 = unlimited)
tlim       = {tlim:.8g}   # time limit ({orbits:g} orbits at R_tor)
integrator = vl2         # time integration algorithm
xorder     = 2           # order of spatial reconstruction
ncycle_out = 100         # interval for stdout summary info

<mesh>
num_threads = {nthreads}         # OpenMP threads; Athena++ parallelises over MeshBlocks

nx1    = {nx}           # number of zones in X1-direction
x1min  = {-g['L']}         # minimum value of X1
x1max  = {g['L']}          # maximum value of X1
ix1_bc = user          # inner-X1 boundary condition flag (diode outflow)
ox1_bc = user          # outer-X1 boundary condition flag (diode outflow)

nx2    = {nx}           # number of zones in X2-direction
x2min  = {-g['L']}         # minimum value of X2
x2max  = {g['L']}          # maximum value of X2
ix2_bc = user          # inner-X2 boundary condition flag (diode outflow)
ox2_bc = user          # outer-X2 boundary condition flag (diode outflow)

nx3    = {nx}           # number of zones in X3-direction
x3min  = {-g['L']}         # minimum value of X3
x3max  = {g['L']}          # maximum value of X3
ix3_bc = user          # inner-X3 boundary condition flag (diode outflow)
ox3_bc = user          # outer-X3 boundary condition flag (diode outflow)

<meshblock>
nx1 = {g['meshblock']}            # MeshBlock size in X1-direction
nx2 = {g['meshblock']}            # MeshBlock size in X2-direction
nx3 = {g['meshblock']}            # MeshBlock size in X3-direction
{grav}
<hydro>
gamma  = {m.gamma:.8g}      # ratio of specific heats
dfloor = {m.rho_atm * 0.1:.6e}   # density floor
pfloor = {m.rho_atm * 0.1 * m.cs_c**2 * 1e-3:.6e}   # pressure floor

<problem>
C_prime    = {m.C_prime}         # torus thickness parameter
q_rot      = {m.q_rot}         # rotation-law exponent, l = l_0 (R/R_tor)^q; 0 = const l
M_tor      = {m.M_tor}         # torus mass, in units of the central mass
rho_c      = {m.rho_c:.10e}  # density at the torus centre (sets M_tor)
GM_c       = 1.0         # central point mass (G = 1)
eps_soft   = {m.eps_soft}        # softening length of the central point mass
self_grav  = {1 if m.self_gravity else 0}           # 1 = torus self-gravity on
indirect   = 1           # 1 = include the gas pull on the central mass (frame accel.)
pert_amp      = {m.pert_amp:g}    # per-cell white-noise density seed
pert_mode_amp = {m.pert_mode_amp:g}    # coherent seed per harmonic m = 1..5; 0 = off

nu_iso     = {nu_iso:g}           # isotropic kinematic viscosity; > 0 enables viscous fluxes
alpha      = {m.alpha}         # Shakura-Sunyaev viscosity parameter

rho_atm      = {m.rho_atm:.6e}  # ambient medium density
t_atm_frac   = {m.t_atm_frac}       # ambient p/rho, in units of the torus mid-plane value
visc_rho_cut = {10.0 * m.rho_atm:.6e}  # density below which viscosity is tapered to zero

# ==============================================================================
# DERIVED QUANTITIES
# ==============================================================================
# Torus inner radius r_in    : {m.r_in:.6f}
# Torus outer radius r_out   : {m.r_out:.6f}
# Half-thickness z_max       : {m.z_max:.6f}
# Orbital period T_orb       : {m.T_orb:.6f}
#
# Cell size dx               : {g['dx']:.6f}
#   across the minor radius  : {g['cells_minor']:.1f} cells
#   across the thickness     : {g['cells_vertical']:.1f} cells
#   per softening length     : {g['cells_soft']:.1f} cells
# MeshBlocks                 : {g['nblocks']}
#
# Jeans length / dx at rho_c : {m.jeans_cells(g['dx']):.1f}  (Truelove: keep above 4)
# Timestep estimate          : {m.dt_estimate(g['dx'], cfl):.3e}
#   cycles for {orbits:g} orbits{'':<6}: {tlim / m.dt_estimate(g['dx'], cfl):.3e}
# ==============================================================================
"""


# -- checks ------------------------------------------------------------------

OK, WARN, FAIL = "ok", "warn", "FAIL"


def check(m, setup):
    """Everything that has actually gone wrong in this class of run, as a table."""
    g = m.grid(nx=setup["nx"], meshblock=setup["meshblock"])
    rows = []

    def row(cond, name, detail, hard=True):
        rows.append((OK if cond else (FAIL if hard else WARN), name, detail))

    row(m.C_prime < 0.5, "closed torus",
        f"C' = {m.C_prime} < 0.5, r_in = {m.r_in:.4f}, r_out = {m.r_out:.4f}")
    row(g["L"] > m.r_out, "box encloses torus",
        f"L = {g['L']} > r_out = {m.r_out:.4f}, "
        f"{(g['L'] - m.r_out) / g['dx']:.0f} cells of margin")
    row(g["cells_minor"] >= 24, "radial resolution",
        f"{g['cells_minor']:.1f} cells across the minor diameter (want >= 24)",
        hard=False)
    row(g["cells_vertical"] >= 16, "vertical resolution",
        f"{g['cells_vertical']:.1f} cells across the thickness (want >= 16); "
        "the paper's m = 1 mode dies when the torus is too thin, so under-resolving "
        "the thickness would fake their negative result", hard=False)
    row(g["cells_soft"] >= 2.0, "softening resolved",
        f"eps_soft = {m.eps_soft} is {g['cells_soft']:.1f} cells (want >= 2)")
    # Two-sided: the softening must be resolved by the grid AND stay small compared with
    # the inner edge, or it stops regularising the empty hole and starts changing the
    # potential the torus sits in. Tori at larger q are radially smaller, so this
    # tightens exactly where the resolution check loosens.
    row(m.eps_soft <= 0.2 * m.r_in, "softening is local",
        f"eps_soft / r_in = {m.eps_soft / m.r_in:.2f} (want <= 0.2), "
        f"r_in = {m.r_in:.3f}", hard=False)
    i_eq = m.i_max_equivalent()
    row(True, "thickness vs their runs",
        (f"z_max = {m.z_max:.3f} ~ sin(i_max) gives i_max = {i_eq:.1f} deg; their mode "
         "is strong at 60, weak at 45, absent at 30"
         if i_eq == i_eq else
         f"z_max = {m.z_max:.3f} > R_tor: thicker than any inclination they ran"))
    w, c, tot = m.seed_mode_amplitude()
    efold = np.log(0.1 / tot) if tot > 0 else np.inf
    row(tot > 1.0e-5, "seed reaches the modes",
        f"A_1(0) = {tot:.1e} (white {w:.1e} + coherent {c:.1e}); "
        f"{efold:.1f} e-foldings to A_1 = 0.1. The N-body Poisson level is ~2e-3; "
        "white noise alone cannot reach it, use pert_mode_amp", hard=False)
    row(m.jeans_cells(g["dx"]) >= 4.0, "Jeans resolution",
        f"lambda_J / dx = {m.jeans_cells(g['dx']):.1f} at rho_c (Truelove: >= 4)",
        hard=False)
    row(setup["nx"] % g["meshblock"] == 0, "meshblock divides mesh",
        f"nx = {setup['nx']}, meshblock = {g['meshblock']}, "
        f"{g['nblocks']} blocks")
    row((g["meshblock"] & (g["meshblock"] - 1)) == 0, "meshblock is a power of two",
        f"meshblock = {g['meshblock']}; Multigrid coarsens by factors of two",
        hard=False)
    # Athena++ gates ViscousFluxIso on nu_iso: alpha alone silently runs inviscid.
    row(not (m.alpha > 0) or True, "viscosity gating",
        "alpha > 0 writes a nonzero nu_iso placeholder into the athinput"
        if m.alpha > 0 else "alpha = 0, nu_iso = 0: genuinely inviscid")
    row(m.self_gravity or m.M_tor > 0, "control run is honest",
        "self-gravity off but the torus keeps its mass and inertia"
        if not m.self_gravity else "self-gravity on")

    dt = m.dt_estimate(g["dx"], setup["cfl"])
    ncyc = setup["orbits"] * m.T_orb / dt
    row(ncyc < 4e6, "run length is sane",
        f"~{ncyc:.2e} cycles for {setup['orbits']:g} orbits at dt ~ {dt:.2e}",
        hard=False)
    return rows


def report(m, setup):
    g = m.grid(nx=setup["nx"], meshblock=setup["meshblock"])
    return "\n".join([
        m.summary(), "",
        f"grid  cube [-{g['L']}, {g['L']}]^3  nx={g['nx']}^3  dx={g['dx']:.6f}"
        f"  meshblock={g['meshblock']}  ({g['nblocks']} blocks)",
        f"      {g['cells_minor']:.1f} cells across the minor diameter, "
        f"{g['cells_vertical']:.1f} across the thickness",
        f"mass check  int rho dV = M_tor = {m.M_tor} by construction "
        f"(shape integral {m.shape_integral:.6e})",
    ])


def main():
    """Generate the athinput from RUN_SETUP, then verify it.

    Any field of either the model or RUN_SETUP can be overridden on the command line,
    which is how the control runs are produced without editing the defaults - the
    defaults are the canonical run and should stay that way.
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=None, help="output path for the athinput")
    for name, default in (("C_prime", None), ("M_tor", None), ("gamma", None),
                          ("alpha", None), ("eps_soft", None), ("pert_amp", None),
                          ("q_rot", None), ("pert_mode_amp", None)):
        ap.add_argument("--" + name, type=float, default=default)
    ap.add_argument("--no-self-gravity", action="store_true",
                    help="their Appendix C control; needs a binary built without --grav")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--meshblock", type=int, default=None)
    ap.add_argument("--orbits", type=float, default=None)
    ap.add_argument("--nthreads", type=int, default=None)
    args = ap.parse_args()

    model_kw = {k: getattr(args, k) for k in
                ("C_prime", "M_tor", "gamma", "alpha", "eps_soft", "pert_amp",
                 "q_rot", "pert_mode_amp")
                if getattr(args, k) is not None}
    if args.no_self_gravity:
        model_kw["self_gravity"] = False
    m = TorusModel(**model_kw)

    setup = dict(RUN_SETUP)
    for k in ("nx", "meshblock", "orbits", "nthreads"):
        if getattr(args, k) is not None:
            setup[k] = getattr(args, k)

    out = pathlib.Path(args.out) if args.out else repo_root() / RUN_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_athinput(m, **setup))

    print(report(m, setup))
    print(f"\nwrote {out}\n")

    rows = check(m, setup)
    width = max(len(n) for _, n, _ in rows)
    n_fail = sum(s == FAIL for s, _, _ in rows)
    n_warn = sum(s == WARN for s, _, _ in rows)
    for status, name, detail in rows:
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")
    print("\n" + ("all checks passed" if not n_fail and not n_warn else
                  f"{n_fail} failure(s), {n_warn} warning(s)"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
