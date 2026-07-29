#!/usr/bin/env python3
"""Write athinput.acc_disk_visc from the physical parameters.

Grid bounds, tlim, output cadence and nu_iso are derived by disk_model.py instead of
being copied by hand, which is how x1min once ended up at r_inner.

For the normal workflow run `python3 disk_model.py` instead - it edits nothing, writes
the input file to the right place and verifies it in one step. Use this script when you
want one-off parameters on the command line without editing disk_model.py.

    make_athinput.py                                   # defaults, to stdout
    make_athinput.py -o inputs/hydro/athinput.acc_disk_visc
    make_athinput.py --alpha 0.01 --orbits 200 -o inputs/hydro/athinput.acc_disk_visc
    make_athinput.py --T0 1.5e3 --mu 2.3 --nx1 352 --nx2 256   # molecular regime

Physics flags: --M-bh --T0 --mu --chi --rho0 --gamma --C-prime --alpha --rho-atm
Run flags:     --orbits --frames-per-orbit --nx1 --nx2 --meshblock --cfl --format
"""

import argparse
import sys

from disk_model import DiskModel, orbits_to_accrete

TEMPLATE = """\
<comment>
problem   = Thermally-scaled Papaloizou-Pringle accretion disk with alpha viscosity
configure = --prob=acc_disk_visc --coord=cylindrical
generated = scripts/theory/make_athinput.py

<job>
problem_id = acc_disk_visc  # basename of output filenames

<output1>
file_type  = {fmt:<12}# output format
variable   = prim        # variables to be output
dt         = {dt_out:<12.6g}# time increment between outputs
dcycle     = -1          # cycle increment between outputs (-1 = unused)

<output2>
file_type  = {fmt:<12}# output format
variable   = uov         # variables to be output
dt         = {dt_out:<12.6g}# time increment between outputs
dcycle     = -1          # cycle increment between outputs (-1 = unused)

<output3>
file_type  = hst         # output format
dt         = {dt_out:<12.6g}# time increment between outputs
dcycle     = -1          # cycle increment between outputs (-1 = unused)
data_format = %24.16e    # output format specifier

<time>
cfl_number = {cfl:<12}# Courant, Friedrichs & Lewy number
nlim       = -1          # cycle limit (-1 = unlimited)
tlim       = {tlim:<12.6g}# time limit
integrator = vl2         # time integration algorithm
xorder     = 2           # order of spatial reconstruction
ncycle_out = 50          # interval for stdout summary info

<mesh>
nx1    = {nx1:<14}# number of zones in X1-direction
x1min  = {x1min:<14.6g}# minimum value of X1
x1max  = {x1max:<14.6g}# maximum value of X1
x1rat  = 1.0           # ratio of adjacent cell widths in X1
ix1_bc = user          # inner-X1 boundary condition flag
ox1_bc = user          # outer-X1 boundary condition flag

nx2    = {nx2:<14}# number of zones in X2-direction
x2min  = 0.0           # minimum value of X2
x2max  = 6.28318530718 # maximum value of X2
ix2_bc = periodic      # inner-X2 boundary condition flag
ox2_bc = periodic      # outer-X2 boundary condition flag

nx3    = 1             # number of zones in X3-direction
x3min  = -0.5          # minimum value of X3
x3max  = 0.5           # maximum value of X3
ix3_bc = periodic      # inner-X3 boundary condition flag
ox3_bc = periodic      # outer-X3 boundary condition flag

<meshblock>
nx1 = {mb1:<14}# MeshBlock size in X1-direction
nx2 = {mb2:<14}# MeshBlock size in X2-direction
nx3 = 1             # MeshBlock size in X3-direction

<hydro>
gamma  = {gamma:<14}# ratio of specific heats
dfloor = {dfloor:<14.6g}# density floor
pfloor = {pfloor:<14.6g}# pressure floor

<problem>
r_center = {r_center:<12}# radius of the torus density maximum
C_prime  = {C_prime:<12}# torus thickness parameter (0 < C_prime < 0.5)
nu_iso   = {nu_iso:<12.6g}# isotropic kinematic viscosity; > 0 enables viscous fluxes
alpha    = {alpha:<12.6g}# Shakura-Sunyaev viscosity parameter

rho_atm      = {rho_atm:<8.6g}# ambient medium density
t_atm_frac   = {t_atm_frac:<8.6g}# ambient p/rho, in units of the torus mid-plane value
visc_rho_cut = {visc_cut:<8.6g}# density below which viscosity is tapered to zero

M_bh     = {M_bh:<12.6g}# black hole mass [M_sun]
rho_0    = {rho_0:<12.6g}# reference density [g/cm^3]
T_0      = {T_0:<12.6g}# reference temperature [K]
mu       = {mu:<12.6g}# mean molecular weight
chi      = {chi:<12.6g}# spatial scaling factor r_0 / r_g
"""

THEORY_BLOCK = """
# ==============================================================================
# DERIVED QUANTITIES
# ==============================================================================
# Sound speed cs0            : {cs0:.6e} cm/s
# Polytropic index n         : {n_poly:.6f}
# Gravity parameter beta     : {beta:.6e}
# Torus inner radius r_in    : {r_in:.6f}
# Torus outer radius r_out   : {r_out:.6f}
# Orbital period P_orb       : {P_orb:.6e}
#
# Torus geometry             : R_in {R_in_pc:.4f} pc, R_out {R_out_pc:.4f} pc
# Orbital period             : {P_yr:.2f} yr
#
# Schwarzschild radius r_g   : {r_g:.6e} cm
# Length scale L_0           : {L_0:.6e} cm ({L_0_pc:.4f} pc)
# Time scale t_0             : {T_scale:.6e} s ({T_scale_yr:.1f} yr)
# Mass scale                 : {mass_scale:.4f} M_sun per code unit
# Mdot scale                 : {mdot_scale:.6f} M_sun/yr per code unit
#
# Viscosity (alpha = {alpha:g})
#   nu_iso                   : {nu_iso:.6e}
#   tau_visc                 : {tau_visc:.6e} ({N_orbits:.1f} orbits, {tau_yr:.1f} yr)
#   N_orbits = 1/(2*pi*alpha*(gamma-1)*(0.5-C')) is independent of mass and T_0
# ==============================================================================
"""


def build(model, orbits=100.0, frames_per_orbit=10.0, nx1=176, nx2=128,
          meshblock=None, cfl=0.4, fmt="tab", dfloor=1e-12, pfloor=1e-10,
          visc_rho_cut=None):
    g = model.grid(nx1=nx1, nx2=nx2)
    mb1, mb2 = (meshblock if meshblock else (nx1, nx2))
    visc_cut = visc_rho_cut if visc_rho_cut is not None else 10.0 * model.rho_atm

    body = TEMPLATE.format(
        fmt=fmt, dt_out=model.P_orb / frames_per_orbit, fpo=frames_per_orbit,
        cfl=cfl, tlim=orbits * model.P_orb, nx1=nx1, nx2=nx2, mb1=mb1, mb2=mb2, dr=g["dr"],
        x1min=g["x1min"], x1max=g["x1max"], gamma=model.gamma, dfloor=dfloor, pfloor=pfloor,
        r_center=model.r_center, C_prime=model.C_prime,
        nu_iso=(1.0 if model.alpha > 0 else 0.0), alpha=model.alpha,
        rho_atm=model.rho_atm, t_atm_frac=model.t_atm_frac, visc_cut=visc_cut,
        M_bh=model.M_bh, rho_0=model.rho_0, T_0=model.T_0,
        mu=model.mu, chi=model.chi)

    from disk_model import PC, YR
    theory = THEORY_BLOCK.format(
        cs0=model.cs0, n_poly=model.n_poly, beta=model.beta,
        r_in=model.r_in, r_out=model.r_out, P_orb=model.P_orb,
        r_g=model.r_g, L_0=model.L_0, L_0_pc=model.L_0 / PC,
        R_in_pc=model.r_in * model.L_0 / PC, R_out_pc=model.r_out * model.L_0 / PC,
        P_yr=model.P_orb * model.T_scale / YR,
        T_scale=model.T_scale, T_scale_yr=model.T_scale / YR,
        mass_scale=model.mass_scale, mdot_scale=model.mdot_scale,
        alpha=model.alpha, nu_iso=model.nu_iso, tau_visc=model.tau_visc,
        N_orbits=orbits_to_accrete(model.alpha, model.gamma, model.C_prime),
        tau_yr=model.tau_visc * model.T_scale / YR if model.alpha > 0 else float("inf"))
    return body + theory


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default=None, help="write here (default: stdout)")
    ap.add_argument("--M-bh", type=float, default=4.5e7, help="black hole mass [M_sun]")
    ap.add_argument("--T0", type=float, default=5.0e4, help="reference temperature [K]")
    ap.add_argument("--mu", type=float, default=0.6,
                    help="mean molecular weight (0.6 ionized, 2.3 molecular; must match T0)")
    ap.add_argument("--chi", type=float, default=5.0e2, help="r_0 / r_g")
    ap.add_argument("--rho0", type=float, default=1.0e-13, help="reference density [g/cm^3]")
    ap.add_argument("--gamma", type=float, default=1.3, help="adiabatic index")
    ap.add_argument("--C-prime", type=float, default=0.2, help="thickness parameter (<0.5)")
    ap.add_argument("--alpha", type=float, default=0.001, help="Shakura-Sunyaev alpha")
    ap.add_argument("--rho-atm", type=float, default=1.0e-4, help="ambient density")
    ap.add_argument("--orbits", type=float, default=100.0, help="run length in orbits")
    ap.add_argument("--frames-per-orbit", type=float, default=10.0)
    ap.add_argument("--nx1", type=int, default=176)
    ap.add_argument("--nx2", type=int, default=128)
    ap.add_argument("--meshblock", type=int, nargs=2, default=None, metavar=("N1", "N2"))
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--format", default="tab", choices=["tab", "hdf5"],
                    dest="fmt", help="output file_type for out1/out2")
    a = ap.parse_args(argv)

    model = DiskModel(M_bh=a.M_bh, T_0=a.T0, mu=a.mu, chi=a.chi, rho_0=a.rho0,
                      gamma=a.gamma, C_prime=a.C_prime, alpha=a.alpha,
                      rho_atm=a.rho_atm)
    text = build(model, orbits=a.orbits, frames_per_orbit=a.frames_per_orbit,
                 nx1=a.nx1, nx2=a.nx2, meshblock=a.meshblock, cfl=a.cfl, fmt=a.fmt)

    if a.output:
        with open(a.output, "w") as fh:
            fh.write(text)
        print(f"wrote {a.output}", file=sys.stderr)
        print(model.summary(), file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
