#!/usr/bin/env python3
"""Conservation and health summary for one or more runs, straight from the .hst files.

    analyze.py <rundir> [<rundir> ...]

Columns, and what each is actually good for:

  dm/m0     drift of `disk_mass` (the torus body, ambient subtracted). With alpha = 0
            this is the primary correctness test: no viscosity means no angular momentum
            transport, so there is nothing to drive accretion and the mass must hold.
  dL/L0     drift of `disk_L` = integral of rho*v_phi*r over the body -- the conserved
            angular momentum. NOT Athena++'s built-in `2-mom`, which is the integral of
            rho*v_phi with no lever arm and is not conserved in cylindrical coordinates
            even when nothing exerts a torque. Runs made before disk_L existed fall back
            to 2-mom and are flagged, because the number then means something else.
  dE/E0     drift of total energy.
  floors    largest number of cells sitting on dfloor/pfloor at any output. In a healthy
            run this is exactly 0; a nonzero value means the floor, not the physics, set
            the state somewhere. rho_min/dfloor is the margin that survived.
  dt_visc   smallest ratio dt/dt_visc. Near 1 means the diffusion limit set the timestep,
            which is the signature of the viscous front running away; << 1 means the CFL
            limit was in charge, as it should be.
  max_vr    largest radial speed reached anywhere, the earliest warning of that runaway.
"""
import sys
import os
import glob

import numpy as np


def read_hst(path):
    names = None
    with open(path) as f:
        for line in f:
            if line.startswith('#') and '[1]=' in line:
                toks = line.lstrip('#').split()
                names = [t.split(']=')[1] for t in toks if ']=' in t]
                break
    d = np.loadtxt(path, comments='#')
    if d.ndim == 1:
        d = d[None, :]
    return names, d


def read_floors(run_dir):
    """dfloor/pfloor from the athinput copied beside the run, if it is there."""
    out = {}
    pats = [os.path.join(run_dir, 'materials', 'athinput.*'),
            os.path.join(run_dir, 'athinput.*'),
            os.path.join(run_dir, '..', 'materials', 'athinput.*')]
    for pat in pats:
        for path in glob.glob(pat):
            for line in open(path):
                line = line.split('#')[0].strip()
                for key in ('dfloor', 'pfloor'):
                    if line.startswith(key) and '=' in line:
                        try:
                            out[key] = float(line.split('=', 1)[1])
                        except ValueError:
                            pass
            if out:
                return out
    return out


def report(run_dir):
    hst = glob.glob(os.path.join(run_dir, '*.hst'))
    if not hst:
        hst = glob.glob(os.path.join(run_dir, 'data', '*.hst'))
    if not hst:
        return None
    names, d = read_hst(hst[0])
    idx = {n: i for i, n in enumerate(names)}

    def col(name):
        return d[:, idx[name]] if name in idx else None

    m = col('disk_mass')
    if m is None:
        m = col('mass')
    L = col('disk_L')
    L_is_true = L is not None
    if L is None:
        L = col('2-mom')       # not angular momentum; reported with a warning
    return dict(name=os.path.basename(os.path.abspath(run_dir)),
                t=d[:, 0], m=m, e=col('tot-E'), L=L, L_is_true=L_is_true,
                ke1=col('1-KE'), ke2=col('2-KE'),
                dt=col('dt'), dt_visc=col('dt_visc'),
                n_dfloor=col('n_dfloor'), n_pfloor=col('n_pfloor'),
                rho_min=col('rho_min'), max_vr=col('max_vr'),
                floors=read_floors(run_dir), names=names, d=d, idx=idx)


def fmt_drift(x):
    if x is None or len(x) < 2 or x[0] == 0:
        return f"{'n/a':>10}"
    return f"{(x[-1] - x[0]) / abs(x[0]):>+10.3e}"


def main(dirs):
    hdr = (f"{'run':<28} {'t_end':>9} {'dm/m0':>10} {'dL/L0':>10} {'dE/E0':>10} "
           f"{'floors':>8} {'rho_min/df':>11} {'min dt/dtv':>10} {'max_vr':>10}")
    print(hdr)
    print('-' * len(hdr))
    rows, legacy_L = [], []
    for dd in dirs:
        r = report(dd)
        if r is None:
            print(f"{os.path.basename(os.path.abspath(dd)):<28}  (no .hst)")
            continue

        if r['n_dfloor'] is not None:
            nf = int(np.max(r['n_dfloor']) + np.max(r['n_pfloor']))
            floors = f"{nf:>8d}"
        else:
            floors = f"{'n/a':>8}"

        dfloor = r['floors'].get('dfloor')
        if r['rho_min'] is not None and dfloor:
            margin = f"{np.min(r['rho_min']) / dfloor:>11.2e}"
        else:
            margin = f"{'n/a':>11}"

        if r['dt_visc'] is not None and r['dt'] is not None:
            # dt_visc comes back as the Real max when no diffusion process is active
            live = np.isfinite(r['dt_visc']) & (r['dt_visc'] > 0) & (r['dt_visc'] < 1e100)
            ratio = (f"{np.max(r['dt'][live] / r['dt_visc'][live]):>10.2e}"
                     if live.any() else f"{'inviscid':>10}")
        else:
            ratio = f"{'n/a':>10}"

        vr = f"{np.max(r['max_vr']):>10.3e}" if r['max_vr'] is not None else f"{'n/a':>10}"

        print(f"{r['name']:<28} {r['t'][-1]:>9.4g} {fmt_drift(r['m'])} "
              f"{fmt_drift(r['L'])} {fmt_drift(r['e'])} {floors} {margin} {ratio} {vr}")
        if not r['L_is_true']:
            legacy_L.append(r['name'])
        rows.append(r)

    if legacy_L:
        print()
        print("NOTE: no disk_L column in " + ", ".join(legacy_L) + ".")
        print("      dL/L0 there is Athena++'s 2-mom = integral of rho*v_phi, which has")
        print("      no lever arm and is NOT the conserved angular momentum. Rerun with")
        print("      the current problem generator to get the real one.")
    return rows


if __name__ == '__main__':
    args = sys.argv[1:] or sorted(glob.glob('results/runs/*'))
    main([a for a in args if os.path.isdir(a)])
