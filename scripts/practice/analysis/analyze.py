#!/usr/bin/env python3
"""Compare mass conservation / energy health across runs in results/AI/runs."""
import sys, os, glob
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

def report(run_dir):
    hst = glob.glob(os.path.join(run_dir, '*.hst'))
    if not hst:
        return None
    names, d = read_hst(hst[0])
    idx = {n: i for i, n in enumerate(names)}
    t = d[:, 0]
    m = d[:, idx['mass']]
    e = d[:, idx['tot-E']]
    L = d[:, idx['2-mom']]        # angular momentum (rho*v_phi integrated)
    ke1 = d[:, idx['1-KE']]        # radial kinetic energy = wave/infall diagnostic
    return dict(name=os.path.basename(run_dir), t=t, m=m, e=e, L=L, ke1=ke1,
                names=names, d=d, idx=idx)

def main(dirs):
    hdr = (f"{'run':<26} {'t_end':>9} {'dm/m0':>10} {'dE/E0':>10} {'dL/L0':>10} "
           f"{'max KE_r/KE_phi0':>17} {'m_max/m0':>10}")
    print(hdr); print('-'*len(hdr))
    rows = []
    for dd in dirs:
        r = report(dd)
        if r is None:
            print(f"{os.path.basename(dd):<26}  (no .hst)"); continue
        m, e, L, t = r['m'], r['e'], r['L'], r['t']
        ke2_0 = r['d'][0, r['idx']['2-KE']]
        print(f"{r['name']:<26} {t[-1]:>9.4g} {(m[-1]-m[0])/m[0]:>+10.3e} "
              f"{(e[-1]-e[0])/abs(e[0]):>+10.3e} {(L[-1]-L[0])/abs(L[0]):>+10.3e} "
              f"{r['ke1'].max()/ke2_0:>17.3e} {m.max()/m[0]:>10.4f}")
        rows.append(r)
    return rows

if __name__ == '__main__':
    args = sys.argv[1:] or sorted(glob.glob('results/AI/runs/*'))
    main([a for a in args if os.path.isdir(a)])
