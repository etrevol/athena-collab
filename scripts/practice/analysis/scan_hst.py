#!/usr/bin/env python3
"""Scan all Athena++ .hst files under results/ and report mass-conservation health."""
import sys, glob, os
import numpy as np

def read_hst(path):
    names = None
    with open(path) as f:
        for line in f:
            if line.startswith('#') and '[1]=' in line:
                # parse "[1]=time  [2]=dt ..."
                toks = line.lstrip('#').split()
                names = []
                for t in toks:
                    if ']=' in t:
                        names.append(t.split(']=')[1])
                    elif names:
                        names[-1] += '_' + t
                break
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    return names, data

def main(paths):
    print(f"{'file':<62} {'N':>5} {'t_end':>10} {'m0':>11} {'m_end':>11} {'m_max/m0':>9} {'dm/m0':>9}")
    print('-'*130)
    for p in sorted(paths):
        try:
            names, d = read_hst(p)
        except Exception as e:
            print(f"{p:<62} ERR {e}")
            continue
        if d.size == 0 or names is None:
            continue
        t = d[:, 0]
        im = names.index('mass') if 'mass' in names else 2
        m = d[:, im]
        m0 = m[0]
        tag = os.path.relpath(p, 'results')
        print(f"{tag:<62} {len(t):>5} {t[-1]:>10.4g} {m0:>11.4e} {m[-1]:>11.4e} "
              f"{m.max()/m0:>9.3f} {(m[-1]-m0)/m0:>+9.3f}")

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        args = glob.glob('results/**/*.hst', recursive=True)
    main(args)
