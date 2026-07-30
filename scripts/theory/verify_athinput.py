#!/usr/bin/env python3
"""Check an athinput against the theory it is meant to implement.

Catches values that drifted from what disk_model.py derives, plus the traps that are
easy to fall into: a boundary sitting on the disk surface, floors above the ambient
medium, and the two nu_iso/alpha combinations that silently change the physics.

    verify_athinput.py inputs/hydro/athinput.acc_disk_visc
    verify_athinput.py inputs/hydro/athinput.acc_disk_visc -q   # only warnings and failures

Exit status is 1 if any check fails, so it can gate a run.
"""

import argparse
import sys

from disk_model import DiskModel, orbits_to_accrete

OK, WARN, FAIL = "ok", "warn", "FAIL"


def parse_athinput(path):
    out, block = {}, None
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[f"{block}/{k.strip()}"] = v.strip()
    return out


def num(d, key, default=None):
    try:
        return float(d[key])
    except (KeyError, ValueError):
        return default


def check(path):
    p = parse_athinput(path)
    try:
        m = DiskModel(
            M_bh=num(p, "problem/M_bh", 4.5e7), T_0=num(p, "problem/T_0", 5e4),
            mu=num(p, "problem/mu", 0.6), chi=num(p, "problem/chi", 500.0),
            rho_0=num(p, "problem/rho_0", 1e-13), gamma=num(p, "hydro/gamma", 1.3),
            r_center=num(p, "problem/r_center", 1.0),
            C_prime=num(p, "problem/C_prime", 0.2),
            alpha=num(p, "problem/alpha", 0.0),
            rho_atm=num(p, "problem/rho_atm", 1e-4),
            t_atm_frac=num(p, "problem/t_atm_frac", 1.0))
    except ValueError as e:
        # The inputs contradict each other, so there is no model to check against;
        # report that as the failure rather than dying with a traceback.
        return None, [(FAIL, "inputs", str(e))]

    rows = []

    def add(status, name, detail):
        rows.append((status, name, detail))

    # -- grid must enclose the disk ---------------------------------------
    x1min, x1max = num(p, "mesh/x1min"), num(p, "mesh/x1max")
    nx1 = num(p, "mesh/nx1")
    if x1min is None or x1max is None:
        add(FAIL, "grid", "mesh/x1min or mesh/x1max missing")
    else:
        dr = (x1max - x1min) / nx1
        n_in = (m.r_in - x1min) / dr
        n_out = (x1max - m.r_out) / dr
        if n_in < 1.0:
            add(FAIL, "x1min", f"{x1min:g} is within {n_in:.2f} cells of r_in "
                               f"{m.r_in:.6g}: the disk surface sits on the boundary, "
                               f"the outflow condition is ill posed and the disk "
                               f"drains (~1%/orbit measured)")
        elif n_in < 3:
            add(WARN, "x1min", f"{x1min:g} leaves only {n_in:.1f} cells inside "
                               f"r_in {m.r_in:.6g}; 5+ is comfortable")
        else:
            add(OK, "x1min", f"{x1min:g}, {n_in:.0f} cells inside r_in {m.r_in:.6g}")

        if n_out < 1.0:
            add(FAIL, "x1max", f"{x1max:g} is within {n_out:.2f} cells of r_out "
                               f"{m.r_out:.6g}: outer surface on the boundary")
        elif n_out < 3:
            add(WARN, "x1max", f"{x1max:g} leaves only {n_out:.1f} cells outside "
                               f"r_out {m.r_out:.6g}")
        else:
            add(OK, "x1max", f"{x1max:g}, {n_out:.0f} cells outside r_out {m.r_out:.6g}")

    # -- viscosity switches -------------------------------------------------
    alpha, nu_iso = num(p, "problem/alpha", 0.0), num(p, "problem/nu_iso", 0.0)
    if alpha > 0 and nu_iso <= 0:
        add(FAIL, "viscosity", "alpha > 0 with nu_iso = 0 runs INVISCID: Athena++ "
                               "gates ViscousFluxIso on nu_iso")
    elif alpha == 0 and nu_iso > 0:
        add(FAIL, "viscosity", f"alpha = 0 with nu_iso = {nu_iso:g} is NOT inviscid: "
                               f"a constant nu = {nu_iso:g} is applied everywhere")
    elif alpha > 0:
        n_orb = orbits_to_accrete(alpha, m.gamma, m.C_prime)
        add(OK, "viscosity", f"alpha = {alpha:g}, accretion over ~{n_orb:.0f} orbits")
    else:
        add(OK, "viscosity", "inviscid (alpha = 0, nu_iso = 0)")

    # -- floors must stay below the ambient medium --------------------------
    dfloor, pfloor = num(p, "hydro/dfloor", 0.0), num(p, "hydro/pfloor", 0.0)
    if dfloor >= m.rho_atm:
        add(FAIL, "dfloor", f"{dfloor:g} >= rho_atm {m.rho_atm:g}: the floor becomes "
                            f"the physical background")
    else:
        add(OK, "dfloor", f"{dfloor:g}, {m.rho_atm / dfloor:.0e}x below rho_atm")
    if pfloor >= m.p_atm:
        add(FAIL, "pfloor", f"{pfloor:g} >= p_atm {m.p_atm:g}")
    else:
        add(OK, "pfloor", f"{pfloor:g}, {m.p_atm / pfloor:.0e}x below p_atm")

    if alpha > 0 and m.rho_atm < 1e-4:
        add(WARN, "rho_atm", f"{m.rho_atm:g} < 1e-4 with viscosity on: the cell ahead "
                             f"of the spreading inner front lacks inertia and the "
                             f"timestep collapses (measured: 1e-6 -> t=0.004)")

    # -- run length against the viscous timescale ---------------------------
    tlim = num(p, "time/tlim")
    if tlim:
        orbits = tlim / m.P_orb
        add(OK, "tlim", f"{tlim:g} = {orbits:.0f} orbits")
        if alpha > 0:
            n_orb = orbits_to_accrete(alpha, m.gamma, m.C_prime)
            if orbits < 0.2 * n_orb:
                add(WARN, "tlim", f"{orbits:.0f} orbits is short against the "
                                  f"{n_orb:.0f}-orbit viscous time; little will happen")

    # -- output cadence -----------------------------------------------------
    dt_out = num(p, "output1/dt")
    if dt_out and tlim:
        add(OK, "output1/dt", f"{dt_out:g}, {tlim / dt_out:.0f} frames, "
                              f"{m.P_orb / dt_out:.1f} per orbit")

    # -- resolution at the disk surface ------------------------------------
    if x1min is not None:
        dr = (x1max - x1min) / nx1
        # Width over which rho_disk climbs from rho_atm to its first cell value.
        # Search the whole disk, not a fixed cell window: at large n_poly the rise
        # takes tens of cells, and a short window makes argmax return 0 on an
        # all-False array - reporting a razor edge when the truth is the opposite.
        import numpy as np
        rr = np.linspace(m.r_in, m.r_out, 20000)
        hit = m.rho_disk(rr) > m.rho_atm
        if not hit.any():
            add(FAIL, "surface", f"rho_disk never reaches rho_atm = {m.rho_atm:g}: "
                                 f"the disk is entirely below the ambient medium")
        else:
            w = rr[np.argmax(hit)] - m.r_in
            body = rr[hit][-1] - rr[hit][0]
            add(OK if w / dr > 0.15 else WARN, "surface",
                f"the disk edge spans {w / dr:.2f} cells; it is set by rho_atm, not by dr")
            add(OK if body / dr > 30 else WARN, "disk body",
                f"{body / dr:.0f} cells across the part that rises above rho_atm "
                f"({body / (m.r_out - m.r_in):.0%} of the geometric disk)")

    return m, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("athinput")
    ap.add_argument("-q", "--quiet", action="store_true", help="only warnings and failures")
    a = ap.parse_args(argv)

    m, rows = check(a.athinput)
    width = max(len(n) for _, n, _ in rows)
    n_fail = n_warn = 0
    print(f"checking {a.athinput}\n")
    for status, name, detail in rows:
        if status == FAIL:
            n_fail += 1
        elif status == WARN:
            n_warn += 1
        elif a.quiet:
            continue
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")

    print()
    if n_fail:
        print(f"{n_fail} failure(s), {n_warn} warning(s)")
    elif n_warn:
        print(f"no failures, {n_warn} warning(s)")
    else:
        print("all checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
