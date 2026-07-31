"""The run matrix: one place that says what is computed, and roughly what it costs.

Everything here is derived from `disk_model.DiskModel`, so a resolution or a run length
cannot mean one thing here and another in the athinput. Case ids are stable strings -
they are what the report cites, and what a cached directory is matched against.

Cost model, measured on this machine at 128^2: ~8 s per orbit solo, and the cost of a run
scales as N^3 (two extra cells in each direction, plus the shorter timestep they force).
Two corrections were measured rather than guessed, after the first suite ran three times
longer than predicted: ~13 concurrent runs on 16 cores cost ~3x their solo time, because
the bottleneck is memory bandwidth and not cores, and a viscous run costs ~2x an inviscid
one at the same size. The `est` numbers only order the queue, longest first, so that one
long case cannot start last.

Two profiles:
  quick  the default - three grid levels, everything that answers a question cheaply
  full   adds a fourth grid level (512^2), which alone costs more than the rest together
"""

from __future__ import annotations

from .core import add_theory_to_path
from .runner import Case

add_theory_to_path()
from disk_model import DiskModel                                     # noqa: E402

#: single source of truth for the physics under test
MODEL = DiskModel()
GRID = MODEL.grid()
P = MODEL.P_orb

#: baseline resolution: every test that is not itself a resolution study uses this
NX = 128
#: the resolution ladder, coarsest first
LADDER_QUICK = (64, 128, 256)
LADDER_FULL = (64, 128, 256, 512)

#: numerical viscosity, if comparable to the physical alpha, would make an alpha this
#: small meaningless; the viscous cases use a value the surrogate measurement can defend
ALPHA_TEST = 0.01

#: seed amplitude for the Papaloizou-Pringle runs, in units of the local sound speed
PERT_AMP = 1.0e-3
#: azimuthal modes to seed
PERT_MODES = (1, 2, 3)
#: e-foldings each seeded mode should complete. A fixed number of ORBITS is the wrong
#: budget: m = 3 grows at 0.71 of the m = 1 rate, so the same run length leaves it with
#: no clean exponential segment while m = 1 has already saturated. Deriving the length
#: from the theoretical rate gives every mode the same amount of growth to fit.
#: 8 rather than more: the fit is capped at |A_m|/A_0 < 0.02 anyway (ppi.LINEAR_CEILING),
#: so anything past that only buys a non-linear tail that trips the floors.
PPI_EFOLDS = 8.0
#: hard cap on the run length. Past the linearity ceiling the mode steepens and the
#: timestep drops, so the last orbits cost the most and contribute nothing to the fit.
PPI_MAX_ORBITS = 7.0
#: samples per orbit of the mode amplitude
PPI_FRAMES_PER_ORBIT = 4.0


def ppi_orbits(m):
    """Run length for mode m, from the growth rate the eigenvalue problem predicts."""
    from . import ppi
    rate = ppi.fastest_mode(MODEL, m, n=200)["rate"] * P     # per orbit
    return min(PPI_MAX_ORBITS, max(4.0, PPI_EFOLDS / rate)) if rate > 0 else 6.0

_RST_BLOCK = """
<output4>
file_type = rst
dt        = 1e9
"""


#: measured slowdown when ~13 runs share 16 cores. The cores are there, the memory
#: bandwidth is not, so a case costs roughly three times its solo time in a full suite.
#: Ignoring this made the first suite's estimate wrong by exactly that factor.
_CONTENTION = 3.0
#: viscous runs carry the diffusion flux and a tighter timestep on top of that
_VISCOUS = 2.0


def _cost(nx, orbits, viscous=False):
    """Serial seconds under contention, from 8 s per orbit at 128^2 and N^3 scaling."""
    c = 8.0 * orbits * (nx / 128.0) ** 3 * _CONTENTION
    return max(1.5, c * (_VISCOUS if viscous else 1.0))


#: Lynden-Bell & Pringle ring: radial resolutions, viscosity, and the run length
RING_LADDER = (128, 256, 512)
RING_NU = 1.0e-4
RING_TAU0 = 0.018
RING_TAU_END = 0.10
RING_R0 = 1.0
RING_SIGMA_BG = 1.0e-6
RING_NX2 = 32
#: t = (tau_end - tau0) R0^2 / (12 nu); 10.9 orbits at R0 for these numbers
RING_TLIM = (RING_TAU_END - RING_TAU0) * RING_R0**2 / (12.0 * RING_NU)
RING_FRAMES = 10


def _ring_base(nx, nu):
    return {
        "mesh/nx1": nx, "mesh/nx2": RING_NX2,
        "meshblock/nx1": nx, "meshblock/nx2": RING_NX2,
        "time/tlim": f"{RING_TLIM:.10g}",
        "time/ncycle_out": 1000000,
        "problem/nu_iso": f"{nu:.10g}",
        "problem/tau0": RING_TAU0,
        "problem/r_ring": RING_R0,
        "problem/sigma_bg": RING_SIGMA_BG,
        "output1/dt": f"{RING_TLIM / RING_FRAMES:.10g}",
        "output2/dt": f"{RING_TLIM / 50:.10g}",
    }


def _ring_cost(nx):
    """Measured: ~35 s at nx1 = 128 under contention; cost goes as nx^2 (cells x steps)."""
    return 35.0 * (nx / 128.0) ** 2


def _base(nx=NX, orbits=1.0, frames=1, alpha=0.0, **extra):
    """Common overrides. Outputs are suppressed unless a test asks for them: at 256^2 a
    single .tab snapshot is already 8 MB, and most tests only need the final state."""
    ov = {
        "mesh/nx1": nx, "mesh/nx2": nx,
        "meshblock/nx1": nx, "meshblock/nx2": nx,     # one block: one file per output
        "time/tlim": f"{orbits * P:.10g}",
        "time/ncycle_out": 1000000,
        "problem/alpha": alpha,
        "problem/nu_iso": 1.0 if alpha > 0 else 0.0,
        "output1/dt": f"{orbits * P / frames:.10g}" if frames else "1e9",
        "output2/dt": "1e9",
        "output3/dt": f"{orbits * P / 50:.10g}",      # .hst is cheap; sample it densely
    }
    ov.update(extra)
    return ov


def build(profile="quick"):
    """The full list of cases for a profile."""
    ladder = LADDER_FULL if profile == "full" else LADDER_QUICK
    cases = []

    # -- pgen <-> theory agreement. No timestep at all; only the banner is read. -------
    cases.append(Case("header", _base(orbits=1e-9, frames=0, **{"time/nlim": 0}),
                      ("static",), est=2))

    # -- discrete equilibrium residual vs resolution ----------------------------------
    # Two cycles: enough for the flux arrays to hold real fluxes (they are empty before
    # the first one, and the pgen writes NaN there rather than a plausible wrong number).
    # The dump is triggered by dcycle, not by dt: Athena++'s forced final MakeOutputs
    # only applies to restart files, so a run that stops on the cycle limit writes no
    # final .tab at all and index -1 would silently be the t = 0 frame.
    for nx in ladder:
        cases.append(Case(
            f"resid_n{nx}",
            _base(nx=nx, orbits=1e-9, frames=0,
                  **{"time/nlim": 2, "time/tlim": "1e9",
                     # dt and dcycle are mutually exclusive in Athena++
                     "output1/dt": -1, "output2/dt": -1,
                     "output1/dcycle": 2, "output2/dcycle": 2}),
            ("resid",), est=_cost(nx, 0.02) + 1.5))

    # -- grid convergence of the solution itself --------------------------------------
    for nx in ladder:
        cases.append(Case(f"conv_n{nx}", _base(nx=nx, orbits=2.0, frames=1),
                          ("convergence",), est=_cost(nx, 2.0)))

    # -- temporal convergence ---------------------------------------------------------
    for cfl in (0.4, 0.2, 0.1):
        cases.append(Case(f"cfl_{cfl}", _base(orbits=1.0, frames=1,
                                              **{"time/cfl_number": cfl}),
                          ("temporal",), est=_cost(NX, 1.0) * (0.4 / cfl)))

    # -- invariance -------------------------------------------------------------------
    cases.append(Case("inv_1block", _base(orbits=1.0, frames=1), ("invariance",),
                      est=_cost(NX, 1.0)))
    cases.append(Case("inv_4block",
                      _base(orbits=1.0, frames=1,
                            **{"meshblock/nx1": NX // 2, "meshblock/nx2": NX // 2}),
                      ("invariance",), est=_cost(NX, 1.0)))
    cases.append(Case("inv_repeat", _base(orbits=1.0, frames=1), ("invariance",),
                      est=_cost(NX, 1.0)))
    # restart: same end state whether the run is done in one go or in two
    cases.append(Case("restart_direct",
                      _base(orbits=2.0, frames=1), ("invariance",),
                      est=_cost(NX, 2.0), extra_input=_RST_BLOCK))
    cases.append(Case("restart_split",
                      _base(orbits=1.0, frames=1), ("invariance",),
                      est=_cost(NX, 2.0), extra_input=_RST_BLOCK,
                      # Athena++ renames the last restart dump to .final (outputs.cpp:
                      # wtflag only forces output for rst, and changes the suffix), so
                      # the predictable name is this one, not an index.
                      post_argv=("-r", "acc_disk_visc.final.rst",
                                 f"time/tlim={2.0 * P:.10g}")))

    # -- parameters of numerical origin: none of these may move the answer -------------
    for rho_atm in (1e-3, 1e-5):
        cases.append(Case(f"sens_rhoatm_{rho_atm:g}",
                          _base(orbits=2.0, frames=1,
                                **{"problem/rho_atm": rho_atm,
                                   "problem/visc_rho_cut": 10 * rho_atm}),
                          ("sensitivity",), est=_cost(NX, 2.0)))
    for dfloor in (1e-10, 1e-14):
        cases.append(Case(f"sens_dfloor_{dfloor:g}",
                          _base(orbits=2.0, frames=1, **{"hydro/dfloor": dfloor}),
                          ("sensitivity",), est=_cost(NX, 2.0)))
    for x1min in (0.30, 0.45):
        cases.append(Case(f"sens_x1min_{x1min:g}",
                          _base(orbits=2.0, frames=1, **{"mesh/x1min": x1min}),
                          ("sensitivity",), est=_cost(NX, 2.0)))

    # -- viscosity --------------------------------------------------------------------
    # The alpha-law check needs two timesteps: nu_applied lags the output primitives by
    # one stage, so the residual offset must SHRINK with dt rather than vanish at fixed dt.
    for cfl in (0.4, 0.1):
        cases.append(Case(f"visc_cfl{cfl}",
                          _base(orbits=0.5, frames=1, alpha=ALPHA_TEST,
                                **{"time/cfl_number": cfl, "output2/dt": f"{0.5 * P:.10g}"}),
                          ("viscosity",), est=_cost(NX, 0.5, viscous=True) * (0.4 / cfl)))
    # longer viscous run for the transport rate and alpha_eff
    cases.append(Case("visc_transport",
                      _base(orbits=4.0, frames=4, alpha=ALPHA_TEST,
                            **{"output2/dt": f"{P:.10g}"}),
                      ("viscosity",), est=_cost(NX, 4.0, viscous=True),
                      # The viscous front running into the funnel collapses dt (ZVIT 6,
                      # 12). That is a known limitation, so the case is kept and given a
                      # ceiling: where it stops is the measurement.
                      timeout=420.0))

    # -- Papaloizou-Pringle instability -----------------------------------------------
    # Run length per mode, from the theoretical rate: every mode gets PPI_EFOLDS of
    # growth to fit, instead of a fixed number of orbits that suits only the fastest one.
    for m in PERT_MODES:
        orb = ppi_orbits(m)
        cases.append(Case(f"ppi_m{m}",
                          _base(orbits=orb, frames=int(orb * PPI_FRAMES_PER_ORBIT),
                                **{"problem/pert_amp": PERT_AMP,
                                   "problem/pert_kind": "single", "problem/pert_m": m}),
                          ("ppi", "unstable"), est=_cost(NX, orb)))
    # Linearity: a tenfold smaller seed must grow at the same rate. It starts ln(10) =
    # 2.3 e-foldings further back, so it needs that much extra run to reach the same
    # amplitude window.
    orb2 = ppi_orbits(2) * (1.0 + 2.3 / PPI_EFOLDS)
    cases.append(Case("ppi_m2_small",
                      _base(orbits=orb2, frames=int(orb2 * PPI_FRAMES_PER_ORBIT),
                            **{"problem/pert_amp": PERT_AMP / 10,
                               "problem/pert_kind": "single", "problem/pert_m": 2}),
                      ("ppi", "unstable"), est=_cost(NX, orb2)))

    # -- Lynden-Bell & Pringle spreading ring -----------------------------------------
    # Its own binary and its own input; see build_ring.sh. Azimuthally the problem is
    # trivial, so nx2 stays at 32 and all the resolution goes where the physics is.
    for nx in RING_LADDER:
        cases.append(Case(f"ring_visc_n{nx}",
                          _ring_base(nx, nu=RING_NU),
                          ("ring",), est=_ring_cost(nx),
                          binary="athena_visc_ring",
                          athinput="athinput.visc_ring"))
    # nu = 0: nothing in the equations spreads the ring, so what spreads it is the scheme
    for nx in RING_LADDER:
        cases.append(Case(f"ring_num_n{nx}",
                          _ring_base(nx, nu=0.0),
                          ("ring",), est=_ring_cost(nx),
                          binary="athena_visc_ring",
                          athinput="athinput.visc_ring"))

    return cases
