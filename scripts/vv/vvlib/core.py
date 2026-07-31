"""Paths, criteria and Athena++ output readers - the pieces every check needs.

A criterion is a value plus a verdict, never a sentence in a notebook. `Check` objects
are what the checks return, what the report renders and what the exit status is computed
from, so the same statement cannot be green in one place and red in another.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "info"
_RANK = {FAIL: 3, WARN: 2, OK: 1, INFO: 0}


# ---------------------------------------------------------------------------- paths


def repo_root(start=None) -> pathlib.Path:
    """Walk up to the directory holding configure.py, so this survives being moved."""
    p = pathlib.Path(start or __file__).resolve()
    for d in [p] + list(p.parents):
        if (d / "configure.py").is_file():
            return d
    raise FileNotFoundError(f"no configure.py above {p}: not inside the repo")


def add_theory_to_path() -> None:
    """scripts/theory holds the single source of truth for the model; import from it."""
    theory = repo_root() / "scripts" / "theory"
    if str(theory) not in sys.path:
        sys.path.insert(0, str(theory))


# ---------------------------------------------------------------------------- criteria


@dataclasses.dataclass
class Check:
    """One machine-checkable statement about the model or the code.

    `value` is the number the statement is about; `detail` explains it in one line.
    `expected` records the acceptance band so a report reader never has to guess what
    "ok" meant. INFO is for measurements that are worth recording but carry no verdict.
    """

    group: str
    name: str
    status: str
    detail: str
    value: float | None = None
    expected: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def verdict(ok: bool, warn: bool = False) -> str:
    return FAIL if not ok and not warn else (WARN if warn else OK)


def band(value, lo, hi, name, group, detail, warn_factor=1.5):
    """Standard 'value must lie in [lo, hi]' criterion, with a soft outer band."""
    inside = lo <= value <= hi
    soft = (lo / warn_factor) <= value <= (hi * warn_factor)
    status = OK if inside else (WARN if soft else FAIL)
    return Check(group, name, status, detail, float(value), f"[{lo:g}, {hi:g}]")


def worst(checks) -> str:
    return max((c.status for c in checks), key=lambda s: _RANK[s], default=OK)


# ---------------------------------------------------------------------------- readers


def read_hst(path) -> dict:
    """Athena++ .hst as {column name: array}. Column names come from the header."""
    names = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") and "[1]=" in line:
                names = [t.split("]=")[1] for t in line.lstrip("#").split() if "]=" in t]
                break
    if names is None:
        raise ValueError(f"no column header in {path}")
    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:
        d = d[None, :]
    return {n: d[:, i] for i, n in enumerate(names)}


def read_tab(path) -> dict:
    """Athena++ 2D .tab as {column name: array} plus 'time' and 'cycle'.

    Athena++'s own vis/python/athena_read.py returns x1v/x2v as full (nx2, nx1) arrays
    rather than as the two axes, which is exactly the trap this reader avoids: here the
    columns stay flat and `grid_2d` below does the reshaping once, explicitly.
    """
    with open(path) as fh:
        head = fh.readline()
        cols = fh.readline().lstrip("#").split()
    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:
        d = d[None, :]
    out = {c: d[:, i] for i, c in enumerate(cols[: d.shape[1]])}
    out["time"] = float(head.split("time=")[1].split()[0])
    out["cycle"] = int(head.split("cycle=")[1].split()[0])
    return out


def grid_2d(tab):
    """(r, phi, {name: (nphi, nr) array}) from a flat .tab dict. Single MeshBlock only."""
    r = np.unique(tab["x1v"])
    phi = np.unique(tab["x2v"])
    nr, nphi = len(r), len(phi)
    order = np.lexsort((tab["x1v"], tab["x2v"]))       # phi outer, r inner
    fields = {k: v[order].reshape(nphi, nr)
              for k, v in tab.items()
              if isinstance(v, np.ndarray) and v.size == nr * nphi}
    return r, phi, fields


def azimuthal_mean(tab, name):
    r, _, f = grid_2d(tab)
    return r, f[name].mean(axis=0)


def cell_volumes(r_faces):
    """Cylindrical shell volumes per unit height and per radian: 0.5*(r+^2 - r-^2)."""
    return 0.5 * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)


def uniform_faces(r_centres):
    """Face radii of a uniform grid whose centroids are r_centres.

    Athena++'s x1v in cylindrical geometry is the VOLUME centroid, not the face midpoint,
    so recovering faces by averaging neighbours is wrong at the level we care about here.
    The grid is uniform in r, so the spacing follows from the extent instead.
    """
    n = len(r_centres)
    dr = (r_centres[-1] - r_centres[0]) / (n - 1)
    # centroid = r_mid * (1 + dr^2/(12 r_mid^2) + ...); invert to first order
    r_mid = r_centres - dr * dr / (12.0 * r_centres)
    return np.concatenate([[r_mid[0] - 0.5 * dr], r_mid + 0.5 * dr])
