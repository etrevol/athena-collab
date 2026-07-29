#!/usr/bin/env python3
"""Azimuthally averaged radial profiles and a mass budget by zone.

    profile.py <rundir>

Prints rho, p, v_r, v_phi against r for each output frame, plus the mass in the
inner third / middle / outer third and how it changed. Cells sitting exactly at
dfloor and pfloor are the signature of a floor-driven leak.
"""
import sys, glob, os
import numpy as np

def read_tab(path):
    """Return dict of column-name -> array, azimuthally averaged onto the r grid."""
    with open(path) as f:
        hdr = [l for l in f if l.startswith('#')]
    cols = None
    for l in hdr:
        if l.lstrip('#').split()[:1] == ['i']:
            cols = l.lstrip('#').split()
            break
    d = np.loadtxt(path, comments='#')
    if d.ndim == 1:
        d = d[None, :]
    return cols, d

def main(run):
    tabs = sorted((glob.glob(os.path.join(run, '*.out1.*.tab'))
            or glob.glob(os.path.join(run, 'data', '*.out1.*.tab'))))
    if not tabs:
        print('no out1 tab files in', run); return
    cols, d0 = read_tab(tabs[0])
    print('columns:', cols)
    # figure out index of x1v, rho, press, vel1, vel2
    def ci(*cands):
        for c in cands:
            if c in cols: return cols.index(c)
        return None
    i_r, i_d = ci('x1v'), ci('rho', 'dens')
    i_p, i_v1, i_v2 = ci('press', 'Etot'), ci('vel1'), ci('vel2')

    ref = None
    for t_i, path in enumerate(tabs):
        cols, d = read_tab(path)
        r = d[:, i_r]
        runiq = np.unique(r)
        # azimuthal average
        rho = np.array([d[r == rr, i_d].mean() for rr in runiq])
        vr  = np.array([d[r == rr, i_v1].mean() for rr in runiq])
        vp  = np.array([d[r == rr, i_v2].mean() for rr in runiq])
        pr  = np.array([d[r == rr, i_p].mean() for rr in runiq])
        dr = np.gradient(runiq)
        dm = rho * 2*np.pi*runiq*dr          # mass per radial shell (2D, unit height)
        M = dm.sum()
        if ref is None:
            ref = dm.copy(); Mref = M
        print(f"\n--- {os.path.basename(path)}   M_tot={M:.6e}  dM/M0={(M-Mref)/Mref:+.4e} ---")
        print(f"{'r':>8} {'rho':>11} {'p':>11} {'v_r':>11} {'v_phi':>10} {'dm':>11} {'dm-dm0':>11}")
        idx = list(range(0, 8)) + list(range(len(runiq)//2-1, len(runiq)//2+1)) + list(range(len(runiq)-8, len(runiq)))
        for i in idx:
            print(f"{runiq[i]:8.4f} {rho[i]:11.3e} {pr[i]:11.3e} {vr[i]:11.3e} "
                  f"{vp[i]:10.2f} {dm[i]:11.3e} {dm[i]-ref[i]:+11.3e}")
        # mass budget: inner third / middle / outer third
        n = len(runiq); a, b = n//3, 2*n//3
        print(f"  budget  inner1/3: {dm[:a].sum():.5e} ({dm[:a].sum()-ref[:a].sum():+.3e})"
              f"  mid: {dm[a:b].sum():.5e} ({dm[a:b].sum()-ref[a:b].sum():+.3e})"
              f"  outer1/3: {dm[b:].sum():.5e} ({dm[b:].sum()-ref[b:].sum():+.3e})")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/AI/runs/base_prof')
