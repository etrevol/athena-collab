"""Reading a case directory, and the numerics of a convergence study.

Kept separate from the checks so that the question "what does the data say" and the
question "does that pass" never get entangled: everything here returns numbers, and
nothing here decides a verdict.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from .core import read_hst, read_tab, grid_2d


# ------------------------------------------------------------------ case directory io


def hst(cdir):
    files = sorted(glob.glob(os.path.join(cdir, "*.hst")))
    return read_hst(files[0]) if files else None


def frame_indices(cdir, out_id=1):
    idx = set()
    for f in glob.glob(os.path.join(cdir, f"*.out{out_id}.*.tab")):
        idx.add(int(os.path.basename(f).split(f".out{out_id}.")[1].split(".")[0]))
    return sorted(idx)


def read_frame(cdir, out_id=1, index=-1, usecols=None):
    """One output frame as a flat dict, merging MeshBlock files if the run had several."""
    idx = frame_indices(cdir, out_id)
    if not idx:
        return None
    n = idx[index]
    files = sorted(glob.glob(os.path.join(cdir, f"*.out{out_id}.{n:05d}.tab")))
    parts = [read_tab(f) for f in files]
    out = {k: np.concatenate([p[k] for p in parts])
           for k in parts[0] if isinstance(parts[0][k], np.ndarray)}
    out["time"], out["cycle"], out["index"] = parts[0]["time"], parts[0]["cycle"], n
    return out


def azimuthal_profile(cdir, name, out_id=1, index=-1):
    """(r, <name>(r)) azimuthally averaged. None if the frame is not there."""
    fr = read_frame(cdir, out_id, index)
    if fr is None or name not in fr:
        return None, None
    r, _, f = grid_2d(fr)
    return r, f[name].mean(axis=0)


def surface_density(cdir, out_id=1, index=-1):
    """(r, Sigma(r)) with the uniform ambient removed, so this tracks the torus."""
    return azimuthal_profile(cdir, "rho", out_id, index)


# ------------------------------------------------------------------------ convergence


def restrict(values, factor):
    """Average blocks of `factor` cells: the fine-grid field seen on the coarse grid."""
    n = len(values)
    assert n % factor == 0, f"{n} not divisible by {factor}"
    return values.reshape(n // factor, factor).mean(axis=1)


def observed_order(levels, fields, ratio=2.0):
    """Observed order of accuracy from THREE grid levels, on their common coarse grid.

    levels  ascending cell counts, e.g. (64, 128, 256)
    fields  the same quantity on each, as a 1D array of that length

    Returns (p, e_coarse, e_fine). Given more than three levels the three FINEST are
    used, because those are the ones nearest the asymptotic range - taking the coarsest
    three would answer a question about a grid nobody is going to run on.

    Richardson assumes the three points sit in that asymptotic range; a p far from the
    design order is the signal that they do not, which is a result rather than a failure
    of the method.
    """
    levels, fields = list(levels)[-3:], list(fields)[-3:]
    n0 = levels[0]
    common = [restrict(f, n // n0) for n, f in zip(levels, fields)]
    e21 = np.abs(common[1] - common[0]).mean()
    e32 = np.abs(common[2] - common[1]).mean()
    if e32 <= 0 or e21 <= 0:
        return np.nan, e21, e32
    return float(np.log(e21 / e32) / np.log(ratio)), float(e21), float(e32)


def order_series(levels, fields, ratio=2.0):
    """Observed order for every consecutive triple, coarse to fine.

    One triple gives a number; a sequence of them says whether the refinement is
    ENTERING the asymptotic range (order climbing towards the design value) or wandering,
    which is exactly the question a single GCI cannot answer.
    """
    return [(tuple(levels[i:i + 3]),) + observed_order(levels[i:i + 3],
                                                       fields[i:i + 3], ratio)
            for i in range(len(levels) - 2)]


def gci(f_coarse, f_medium, f_fine, ratio=2.0, safety=1.25):
    """Roache's grid convergence index for a scalar, plus the order it implies.

    Both the absolute band and the relative one are returned, and which to quote depends
    on the quantity. The relative GCI divides by the fine-grid value, which is the right
    thing for a quantity with a finite exact value and meaningless for one whose exact
    value is zero: a conservation error that converges to zero produces a relative band
    that diverges no matter how good the scheme is. Mass drift is exactly that case, and
    reading its relative GCI as "the answer is uncertain by 500%" would be backwards -
    the number is small BECAUSE it is converging to nothing.
    """
    e21, e32 = f_medium - f_coarse, f_fine - f_medium
    if e32 == 0 or e21 == 0 or (e21 / e32) <= 0:
        return dict(p=np.nan, gci=np.nan, gci_abs=np.nan, extrapolated=np.nan)
    p = float(np.log(abs(e21 / e32)) / np.log(ratio))
    denom = ratio**p - 1.0
    band_abs = safety * abs(e32) / denom
    band_rel = band_abs / abs(f_fine) if f_fine != 0 else np.nan
    return dict(p=p, gci=float(band_rel), gci_abs=float(band_abs),
                extrapolated=float(f_fine + e32 / denom))


def relative_drift(h, key="disk_mass"):
    """(value_final - value_initial)/value_initial for a .hst column."""
    if h is None or key not in h or len(h[key]) < 2 or h[key][0] == 0:
        return np.nan
    return float((h[key][-1] - h[key][0]) / abs(h[key][0]))


def max_floor_hits(h):
    if h is None or "n_dfloor" not in h:
        return np.nan
    return float(np.max(h["n_dfloor"]) + np.max(h["n_pfloor"]))


def field_difference(cdir_a, cdir_b, name="rho", out_id=1, index=-1):
    """max|a - b| of one field between two runs on the same grid, after sorting."""
    fa, fb = read_frame(cdir_a, out_id, index), read_frame(cdir_b, out_id, index)
    if fa is None or fb is None:
        return np.nan
    ra, _, ga = grid_2d(fa)
    rb, _, gb = grid_2d(fb)
    if ga[name].shape != gb[name].shape:
        return np.nan
    return float(np.abs(ga[name] - gb[name]).max())
