#!/usr/bin/env python3
"""athvis - plots for an Athena++ run, in one file with no imports of its own.

Everything the four-script setup (vis2d/vis1d/vishst/visforces + athena_data)
does, in a single file that imports only numpy and matplotlib. Copy it into a
run directory, or anywhere else, and it works. pypalettes is used for colours
when installed.

    python3 athvis.py <command> [flags]
    python3 athvis.py <command> --help      # every flag of one command

Reads .tab only, single- or multi-meshblock. With no --data_dir it looks for
./data, then ../data, then . - so it runs both inside a run directory and one
level above it.


COMMANDS
--------
2d       heatmaps, radial and azimuthal profiles, polar view, three animations,
         velocity-vector overlay
1d       azimuthally averaged radial profiles, and their evolution animation
hst      the .hst history file as time series
forces   the radial force balance from the uov output (output id 2)


COMMON FLAGS
------------
--data_dir DIR      where the frames are      (default: ./data, ../data, .)
--output_dir DIR    where figures go          (default: figs_athvis_<command>)
--output_id N       which <outputN> block to read (2d/1d: 1, forces: 2)
--dpi N             raster resolution           (default 200; hst uses 150)
--title TEMPLATE    caption template over {time} {cycle} {frame} {base}
--palette NAME      pypalettes palette for line colours; measured for contrast
                    and colourblind separation, and reported if it is weak

2d, 1d and forces also take:

--start_frame N     first frame to use
--end_frame N       last frame to use
--num_workers N     processes sampling animation colour ranges (default: half
                    the cores)


2d
--
--mode MODE         all (default) | heatmaps | radial | azimuthal | polar |
                    animation | polar_animation | vector_animation
--frame N           frame for the still figures            (default: the last)
--fps N             animation frame rate                           (default 10)
--subsample N       use every Nth frame in animations              (default 1)
--r_min / --r_max        radial window
--phi_min / --phi_max    azimuthal window, in radians
--normalize MODE    none (default) | azimuthal - plot each field as its
                    deviation from the azimuthal mean, on a diverging map.
                    Non-axisymmetric structure is invisible without it: a
                    growing Papaloizou-Pringle mode is a percent-level
                    perturbation on a background spanning eight decades
--vmin_percentile P      lower colour-scale percentile, animations (default 2)
--vmax_percentile P      upper                                     (default 98)
--cmap_palette NAME pypalettes continuous map replacing the field colormaps

Vector overlay, all inert unless --add-vectors is given:

--add-vectors       draw the vector field over the maps
--vec_field F       velocity (default) | momentum
--vec_comp C        both (default) | radial | azimuthal
--vec_frame F       perturbation (default) | full. Orbital motion is ~100x
                    everything else, so `full` shows only the rotation
--vec_panels P      last (default) | all
--vec_style S       quiver (default) | stream
--vec_lattice L     polar (default) | square | hex - arrow placement on the
                    polar view
--vec_arrows N      arrows across the radial extent                (default 8)
--vec_stride N      r-phi panels: every Nth cell, instead of --vec_arrows
--vec_scale S       log (default) | sqrt | linear
--vec_clip P        percentile at which arrow length saturates     (default 92)
--vec_color C       arrow / streamline colour                  (default black)
--vec_density D     streamline density, --vec_style stream        (default 0.6)

Model annotation:

--info SPEC         none | min | default (default) | physics | full, or a
                    comma-separated list of: alpha, nu_iso, gamma, C_prime,
                    r_center, rho_atm, M_bh, T_0, mu, chi, grid, hr, N_orbits.
                    hr and N_orbits are derived from gamma and C_prime alone
--info_pos POS      box (default, top-left) | subtitle | footer
--info_on WHICH     polar (default) | all - which animations get annotated
--params PATH       athinput to read the model from (found automatically)


1d
--
--mode MODE         all (default) | profiles | animation
--frame N           frame for the still figure             (default: the last)
--fps N / --subsample N                     as for 2d
--r_min / --r_max / --phi_min / --phi_max   as for 2d
--normalize MODE    none (default) | azimuthal, as for 2d
--logscale          log y-axis for density and pressure   (default: linear)


hst
---
--mode MODE         default (disk_mass and mdot_in on shared axes) | all |
                    custom (--vars)
--vars A B ...      columns to plot, for --mode custom
--linear_scale VAR ...   columns to draw linearly instead of on a log axis
--linear            linear y-axis for every column

Columns go on a log axis by default, but only where one is usable: a column
that touches zero (mdot_in) or sits at the ~1e308 "viscosity off" sentinel
(dt_visc) stays linear, with a note saying which and why.
--figsize W H       per-figure size in inches                   (default 10 6)
--grid              draw a grid (off by default)
--style NAME        matplotlib style        (default seaborn-v0_8-darkgrid)

Beyond Athena++'s built-in columns, this project's generator adds disk_mass
(the disk body, ambient subtracted) and mdot_in (flux through the inner
boundary). Do not read the built-in 2-mom as angular momentum: it is
int rho v_phi dV, with no lever arm.


forces
------
--mode MODE         all (default) | frame | sum | animation | sum_animation
--frame N           frame for the still figures            (default: the last)
--fps N / --subsample N / --r_min / --r_max      as for 2d
--log               symlog y-axis, for the eight decades between disk and
                    ambient
--linthresh X       linear region of the symlog axis          (default 1e2)

The uov fields are f_grav, f_centr, f_press and f_sum, formed analytically by
the generator (-beta/r^2, v_phi^2/r, a difference of p on x1v). f_sum therefore
measures how well the analytic equilibrium holds, not the residual of the
scheme the solver steps with.


EXAMPLES
--------
    python3 athvis.py 2d
    python3 athvis.py 2d --mode polar --normalize azimuthal
    python3 athvis.py 2d --mode polar --add-vectors --vec_lattice hex
    python3 athvis.py 2d --mode vector_animation --vec_comp radial --fps 15
    python3 athvis.py 2d --mode polar_animation --info physics --subsample 2
    python3 athvis.py 1d --mode animation --start_frame 50
    python3 athvis.py hst --mode custom --vars disk_mass mdot_in
    python3 athvis.py forces --mode sum_animation --log
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool, cpu_count
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

# background for the vector panels: muted so black arrows stay legible on top of it.
# Plain "Greys" saturates to black exactly where the disk is, i.e. where the arrows are
VEC_BG_CMAP = LinearSegmentedColormap.from_list("vecbg", ["#ffffff", "#a8a8a8"])

# =============================================================================
# PALETTE - the same measurement athena_data.line_colors()/check_palette() do,
# inlined so this file has no import of that module. See its docstring for what
# the numbers mean: of 2707 pypalettes entries, none clears every gate for line
# plots on white, so a chosen palette is checked and reported, not trusted blind.
# =============================================================================
PALETTE_DEFAULT = "Bold"
PALETTE_FALLBACK = ["#6497B1", "#6A359C", "#FFB04F", "#679C35", "#CD1076",
                    "#0F7BA2", "#DD5129", "#43B284"]
_PALETTE_WARNED = set()


def _srgb_to_linear(hexc):
    h = hexc.lstrip("#")[:6]
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def _oklab(rgb):
    r, g, b = rgb
    cbrt = lambda v: max(v, 0.0) ** (1.0 / 3.0)
    l = cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


_CVD = {"protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
                   (-0.003882, -0.048116, 1.051998)),
        "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
                   (-0.011820, 0.042940, 0.968881))}


def _delta_e(a, b, kind=None):
    def prep(c):
        rgb = _srgb_to_linear(c)
        if kind:
            M = _CVD[kind]
            rgb = [min(1.0, max(0.0, sum(M[i][j] * rgb[j] for j in range(3))))
                   for i in range(3)]
        return _oklab(rgb)
    p, q = prep(a), prep(b)
    return 100.0 * math.dist(p, q)


def check_palette(colours):
    pairs = [(i, j) for i in range(len(colours)) for j in range(i + 1, len(colours))]
    if not pairs:
        return float("inf"), float("inf"), float("inf")
    cvd = min(min(_delta_e(colours[i], colours[j], "protan"),
                  _delta_e(colours[i], colours[j], "deutan")) for i, j in pairs)
    sep = min(_delta_e(colours[i], colours[j]) for i, j in pairs)
    con = min(1.05 / (0.2126 * r + 0.7152 * g + 0.0722 * b + 0.05)
              for r, g, b in (_srgb_to_linear(c) for c in colours))
    return cvd, sep, con


def line_colors(n, name=None, quiet=False):
    name = name or PALETTE_DEFAULT
    colours = None
    try:
        from pypalettes import load_cmap
        colours = [c[:7] for c in load_cmap(name).hex]
    except ImportError:
        if not quiet and "nolib" not in _PALETTE_WARNED:
            _PALETTE_WARNED.add("nolib")
            print("  note: pypalettes not installed; using the built-in palette")
    except Exception as exc:
        if not quiet and name not in _PALETTE_WARNED:
            _PALETTE_WARNED.add(name)
            print(f"  note: palette '{name}' not found ({exc}); using the built-in one")
    if not colours:
        colours = list(PALETTE_FALLBACK)
    if not quiet and name not in _PALETTE_WARNED:
        cvd, sep, con = check_palette(colours[:max(n, 2)])
        if cvd < 8.0 or sep < 15.0 or con < 1.5:
            _PALETTE_WARNED.add(name)
            print(f"  note: palette '{name}' is weak for line plots (colourblind dE "
                  f"{cvd:.1f}, need 8; separation {sep:.1f}, need 15; contrast "
                  f"{con:.1f}, need 1.5). Used anyway.")
    return [colours[i % len(colours)] for i in range(n)]


# =============================================================================
# DATA LAYER - .tab discovery, single/multi-block reading, no athena_read.py
# =============================================================================
def default_data_dir(start=None):
    here = start or os.getcwd()
    for cand in (os.path.join(here, "data"), os.path.join(here, "..", "data")):
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.out*.tab")):
            return cand
    return os.path.join(here, "data")


def discover(data_dir, output_id=1):
    """{'base':..., 'frames': {n: [paths]}, 'nblocks': k} for one output id."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"no such directory: {data_dir}")
    oid = int(output_id)
    tab_re = re.compile(
        r"^(?P<base>.+?)(?:\.block(?P<block>\d+))?\.out%d\.(?P<frame>\d+)\.tab$" % oid)
    frames, bases, blocks = {}, set(), set()
    for n in os.listdir(data_dir):
        m = tab_re.match(n)
        if not m:
            continue
        g = m.groupdict()
        frame = int(g["frame"])
        block = int(g["block"]) if g.get("block") is not None else 0
        bases.add(g["base"]); blocks.add(block)
        frames.setdefault(frame, []).append((block, os.path.join(data_dir, n)))
    if not frames:
        raise FileNotFoundError(
            f"no .tab files for out{oid} in {data_dir}\n"
            f"  (only .tab is supported - see the module docstring)")
    for f in frames:
        frames[f] = [p for _, p in sorted(frames[f])]
    return {"base": sorted(bases)[0], "frames": frames, "nblocks": len(blocks)}


def _canon(name):
    return {"rho": "density", "press": "pressure", "vel1": "vel_r",
            "vel2": "vel_phi", "vel3": "vel_z"}.get(name, name)


def _read_tab_file(path):
    """One .tab block. 1D and 2D header layouts (i,x1v[,j,x2v]) only."""
    with open(path) as fh:
        header1 = fh.readline()
        header2 = fh.readline()
    m = re.search(r"time=(\S+)\s+cycle=(\S+)\s+variables=(\S+)", header1)
    if not m:
        raise ValueError(f"{path}: could not parse the time/cycle header line")
    time, cycle = float(m.group(1)), int(m.group(2))
    headings = header2.split()[1:]

    data = np.loadtxt(path, skiprows=2)
    if data.ndim == 1:
        data = data[None, :]

    if headings[0] == "i" and headings[2] == "j":
        names = headings[1:2] + headings[3:]
        i_col, j_col = data[:, 0].astype(int), data[:, 2].astype(int)
        cols = np.concatenate([data[:, 1:2], data[:, 3:]], axis=1)
        ni, nj = i_col.max() - i_col.min() + 1, j_col.max() - j_col.min() + 1
        out = {"time": time, "cycle": cycle}
        for n, name in enumerate(names):
            out[_canon(name)] = cols[:, n].reshape(nj, ni)
        return out

    if headings[0] in ("i", "j", "k"):
        names = headings[1:]
        out = {"time": time, "cycle": cycle}
        for n, name in enumerate(names):
            out[_canon(name)] = data[:, n + 1]
        return out
    raise ValueError(f"{path}: unrecognised column header {header2!r}")


def read_frame(paths):
    """Read + join one frame's blocks. Returns time, cycle, r[, phi, R, Phi], vars."""
    joined_blocks = []
    for p in paths:
        d = _read_tab_file(p)
        x1 = np.asarray(d.pop("x1v"))
        x2 = np.asarray(d.pop("x2v")) if "x2v" in d else None
        if x1.ndim == 2:
            ax1, ax2 = x1[0, :], x2[:, 0]
        else:
            ax1, ax2 = x1.ravel(), None
        rec = {"x1v": ax1, "x2v": ax2, "time": d.pop("time"), "cycle": d.pop("cycle")}
        rec.update(d)
        joined_blocks.append(rec)

    if len(joined_blocks) == 1 and joined_blocks[0]["x2v"] is None:
        b = joined_blocks[0]
        out = {"time": b["time"], "cycle": b["cycle"], "r": b["x1v"]}
        for k, v in b.items():
            if k not in ("time", "cycle", "x1v", "x2v"):
                out[k] = np.asarray(v)
        return out

    varkeys = [k for k in joined_blocks[0] if k not in ("time", "cycle", "x1v", "x2v")]
    x1 = np.unique(np.concatenate([b["x1v"] for b in joined_blocks]))
    x2 = np.unique(np.concatenate([b["x2v"] for b in joined_blocks]))
    out = {"time": joined_blocks[0]["time"], "cycle": joined_blocks[0]["cycle"],
           "r": x1, "phi": x2}
    for k in varkeys:
        grid = np.zeros((len(x2), len(x1)))
        for b in joined_blocks:
            i0 = np.searchsorted(x1, b["x1v"][0]) if np.ndim(b["x1v"]) else 0
            iy = np.searchsorted(x2, b["x2v"])
            ix = np.searchsorted(x1, b["x1v"])
            grid[np.ix_(iy, ix)] = np.asarray(b[k]).reshape(len(iy), len(ix))
        out[k] = grid
    R, Phi = np.meshgrid(x1, x2)
    out["R"], out["Phi"] = R, Phi
    return out


def read_hst(data_dir):
    paths = glob.glob(os.path.join(data_dir, "*.hst"))
    if not paths:
        raise FileNotFoundError(f"no .hst file in {data_dir}")
    path = paths[0]
    with open(path) as fh:
        fh.readline()
        header = fh.readline()
    names = re.findall(r"\[\d+\]=(\S+)", header)
    d = np.loadtxt(path, skiprows=2)
    return path, {name: d[:, i] for i, name in enumerate(names)}


# =============================================================================
# MODEL-PARAMETER ANNOTATION - the same idea as athena_data.model_info(),
# reading athinput.in beside the data (or --params); omitted, not an error, if
# neither is found. Only the disk_model.py-derived quantities (H/r, orbits) are
# left out here, since importing scripts/theory/disk_model.py would break the
# no-cross-file-imports point of this script; the raw athinput values are not.
# =============================================================================
INFO_KEYS = ["alpha", "nu_iso", "gamma", "C_prime", "r_center", "rho_atm",
            "M_bh", "T_0", "mu", "chi", "grid", "hr", "N_orbits"]
INFO_PRESETS = {
    "none": [],
    "default": ["alpha", "gamma", "grid", "C_prime", "r_center"],
    "min": ["alpha", "nu_iso", "gamma", "grid"],
    "physics": ["alpha", "nu_iso", "gamma", "grid", "C_prime", "r_center", "rho_atm",
                "hr", "N_orbits"],
    "full": ["alpha", "nu_iso", "gamma", "grid", "C_prime", "r_center", "rho_atm",
            "hr", "N_orbits", "M_bh", "T_0", "mu", "chi"],
}


# CGS, for the one place a physical constant is unavoidable: the orbital period.
_K_B, _M_P, _C_LIGHT = 1.380649e-16, 1.67262192e-24, 2.99792458e10


def _derived_p_orb(params):
    """Orbital period at the density maximum, in code units, or None.

    beta = c^2 / (2 chi cs0^2) with cs0^2 = gamma k_B T_0 / (mu m_p), and
    P = 2 pi sqrt(r_c^3 / beta). Written out rather than imported from
    disk_model.py, which this file does not depend on.
    """
    try:
        gamma = float(params["hydro/gamma"])
        T_0 = float(params["problem/T_0"])
        mu = float(params["problem/mu"])
        chi = float(params["problem/chi"])
        r_c = float(params.get("problem/r_center", 1.0))
        cs0_sq = gamma * _K_B * T_0 / (mu * _M_P)
        beta = _C_LIGHT ** 2 / (2.0 * chi * cs0_sq)
        return 2.0 * math.pi * math.sqrt(r_c ** 3 / beta)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def frame_label(idx, total, time, p_orb=None):
    """Frame counter; adds orbits when P_orb could be derived. Code time on its
    own says nothing about how far the run has got."""
    text = f"Frame {idx:3d}/{total} | Time = {time:5.2f}"
    if p_orb:
        text += f"  ({time / p_orb:.1f} orbits)"
    return text


def _derived_hr(gamma, c_prime):
    """H/r at the density maximum = sqrt((gamma-1)(0.5-C')).

    Written out rather than imported from scripts/theory/disk_model.py, which
    would defeat this file's whole point. Worth knowing why the formula is this
    short: for the Papaloizou-Pringle torus p/rho is normalised to the local
    gravity, so beta - and with it M_bh, T_0, mu and chi - cancels out of the
    ratio entirely. Only gamma and C' survive.
    """
    if gamma is None or c_prime is None or gamma <= 1 or c_prime >= 0.5:
        return None
    return ((gamma - 1.0) * (0.5 - c_prime)) ** 0.5


def _derived_orbits_to_accrete(alpha, gamma, c_prime):
    """N = 1 / (2 pi alpha (gamma-1) (0.5-C')), i.e. 1/(2 pi alpha (H/r)^2).

    Independent of black hole mass and temperature, for the same reason as H/r.
    """
    if not alpha or alpha <= 0:
        return None
    hr = _derived_hr(gamma, c_prime)
    if not hr:
        return None
    return 1.0 / (2.0 * math.pi * alpha * hr * hr)
_ATH_LOOKUP = {"alpha": "problem/alpha", "nu_iso": "problem/nu_iso",
              "gamma": "hydro/gamma", "C_prime": "problem/C_prime",
              "r_center": "problem/r_center", "rho_atm": "problem/rho_atm",
              "M_bh": "problem/M_bh", "T_0": "problem/T_0", "mu": "problem/mu",
              "chi": "problem/chi"}


def find_athinput(data_dir, explicit=None):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    here = os.path.abspath(data_dir or ".")
    for _ in range(4):
        for cand in ("athinput.in", "athinput.txt"):
            p = os.path.join(here, cand)
            if os.path.isfile(p):
                return p
        # build.sh and sweep.sh keep the run's copy in materials/ beside data/,
        # so the athinput is never in the directory the frames are in
        hits = sorted(h for d in (here, os.path.join(here, "materials"))
                      for h in glob.glob(os.path.join(d, "athinput.*"))
                      if not h.endswith((".log", ".png", ".mp4")))
        if hits:
            return hits[0]
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def read_athinput(path):
    out, block = {}, None
    try:
        with open(path) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line:
                    continue
                if line.startswith("<") and line.endswith(">"):
                    block = line[1:-1]
                elif "=" in line:
                    k, v = line.split("=", 1)
                    out[f"{block}/{k.strip()}"] = v.strip()
    except OSError:
        return {}
    return out


def model_info_lines(params, spec):
    if spec in INFO_PRESETS:
        keys = INFO_PRESETS[spec]
    else:
        keys = [k.strip() for k in spec.split(",") if k.strip()]
        unknown = [k for k in keys if k not in INFO_KEYS]
        if unknown:
            print(f"  note: unknown --info key(s) {', '.join(unknown)}; known: "
                  f"{', '.join(INFO_KEYS)}")
            keys = [k for k in keys if k in INFO_KEYS]
    if not params or not keys:
        return []

    def g(k):
        try:
            return float(params[k])
        except (KeyError, ValueError):
            return None

    items = []
    for key in keys:
        if key == "grid":
            nx1, nx2 = g("mesh/nx1"), g("mesh/nx2")
            if nx1 and nx2:
                items.append(f"grid {int(nx1)}x{int(nx2)}")
            continue
        if key == "hr":
            hr = _derived_hr(g("hydro/gamma"), g("problem/C_prime"))
            if hr is not None:
                items.append(f"H/r = {hr:.3f}")
            continue
        if key == "N_orbits":
            n_orb = _derived_orbits_to_accrete(g("problem/alpha"), g("hydro/gamma"),
                                               g("problem/C_prime"))
            if n_orb is not None:
                items.append(f"{n_orb:.0f} orbits to accrete")
            continue
        v = g(_ATH_LOOKUP[key])
        if v is None:
            continue
        label = {"C_prime": "C'", "r_center": "$r_c$", "rho_atm": r"$\rho_{\rm atm}$",
                 "M_bh": r"$M_{\rm BH}$", "T_0": "$T_0$", "mu": r"$\mu$",
                 "chi": r"$\chi$", "alpha": r"$\alpha$", "nu_iso": r"$\nu_{\rm iso}$",
                 "gamma": r"$\gamma$"}[key]
        items.append(f"{label} = {v:.3g}")
    per_line = 5
    return ["   ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]


# =============================================================================
# VECTOR FIELD - lattice, magnitude shaping, clipping. Mirrors vis2d.py's
# implementation (verified against the same Cartesian ground truth: pure
# rotation and pure outflow, before being wired into a plot).
# =============================================================================
def vector_field(data, comp="both", frame="full"):
    u_r = np.asarray(data["vel_r"], dtype=float)
    u_phi = np.asarray(data["vel_phi"], dtype=float)
    if comp == "radial":
        u_phi = np.zeros_like(u_phi)
    elif comp == "azimuthal":
        u_r = np.zeros_like(u_r)
    degenerate, rel = False, 1.0
    if frame == "perturbation":
        full = float(np.abs(np.hypot(u_r, u_phi)).max())
        u_r = u_r - u_r.mean(axis=0)[None, :]
        u_phi = u_phi - u_phi.mean(axis=0)[None, :]
        pert = float(np.abs(np.hypot(u_r, u_phi)).max())
        rel = pert / full if full > 0 else 0.0
        degenerate = rel < 1e-8
    return u_r, u_phi, degenerate, rel


def _nearest(axis, values):
    """Index of the closest cell centre, not merely the insertion point."""
    idx = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
    left = np.abs(values - axis[idx - 1]) <= np.abs(axis[np.minimum(idx, len(axis) - 1)]
                                                    - values)
    return np.clip(np.where(left, idx - 1, idx), 0, len(axis) - 1)


def grid_lattice(r, phi, n, hexagonal=False):
    """Cartesian (or triangular) lattice clipped to the annulus.

    Both families are built as integer multiples of the step about the origin.
    Starting from -r_max and stepping - the obvious way - only lands on the
    centre when the diameter happens to be a whole number of steps; done that
    way the whole pattern sits off centre and overshoots the outer edge.
    `hexagonal` offsets alternate rows by half a step (still symmetric about
    x = 0) and compresses row spacing by sqrt(3)/2, the arrangement where every
    point is equidistant from all six of its neighbours.

    Returns (j, i, r_exact, phi_exact): the indices say which cell to read the
    field from, the last two say where to draw. Drawing at cell centres instead
    would quantise the lattice to the mesh - 2.8 degrees per cell at nphi = 128.
    """
    n = max(2, n)
    r0, r1 = float(r[0]), float(r[-1])
    d = (r1 - r0) / n
    if d <= 0:
        return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
    row_step = d * (np.sqrt(3.0) / 2.0 if hexagonal else 1.0)
    m = int(np.ceil(r1 / row_step))
    k_max = int(np.ceil(r1 / d)) + 1
    cols = np.arange(-k_max, k_max + 1, dtype=float)
    X, Y = [], []
    for row in range(-m, m + 1):
        xs = (cols + 0.5) * d if (hexagonal and row % 2) else cols * d
        X.append(xs)
        Y.append(np.full(xs.shape, row * row_step))
    X, Y = np.concatenate(X), np.concatenate(Y)
    rr = np.hypot(X, Y)
    keep = (rr >= r0) & (rr <= r1)
    if not keep.any():
        return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
    X, Y, rr = X[keep], Y[keep], rr[keep]
    pp = np.mod(np.arctan2(Y, X), 2.0 * np.pi)
    return _nearest(phi, pp), _nearest(r, rr), rr, pp


def polar_ring_lattice(r, phi, n_rings):
    """Lattice that is uniform in PHYSICAL space, not in index space.

    A fixed index stride puts the same number of arrows on every ring, so they
    crowd together towards the axis where the rings are short. Instead place
    rings a fixed distance d apart and space arrows d apart *along* each ring.
    Concentric by construction, so it is centred on the origin whatever the step
    works out to.

    Returns the same (j, i, r_exact, phi_exact) as grid_lattice().
    """
    r0, r1 = float(r[0]), float(r[-1])
    d = (r1 - r0) / max(2, n_rings)
    if d <= 0:
        return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
    js, iss, rr, pp = [], [], [], []
    for k in range(n_rings):
        rk = r0 + (k + 0.5) * d
        i = int(np.argmin(np.abs(r - rk)))
        n_arrow = max(3, int(round(2.0 * np.pi * rk / d)))
        angles = 2.0 * np.pi * (np.arange(n_arrow) + 0.5) / n_arrow
        js.append(_nearest(phi, angles)); iss.append(np.full(n_arrow, i))
        rr.append(np.full(n_arrow, rk)); pp.append(angles)
    return (np.concatenate(js), np.concatenate(iss),
            np.concatenate(rr), np.concatenate(pp))


def clip_length(U, V, pct=92.0, cap=None):
    """Shorten the longest arrows to a percentile cap. Pass `cap` to reuse one
    computed elsewhere (e.g. from a different frame) instead of measuring U, V."""
    mag = np.hypot(U, V)
    good = mag[np.isfinite(mag) & (mag > 0)]
    if cap is not None:
        if not np.isfinite(cap) or cap <= 0:
            return U, V, cap
        shrink = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
        return U * shrink, V * shrink, cap
    if not good.size:
        return U, V, 1.0
    cap = float(np.percentile(good, min(max(pct, 1.0), 100.0)))
    if not np.isfinite(cap) or cap <= 0:
        return U, V, float(good.max())
    if pct >= 100.0:
        return U, V, cap
    shrink = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
    return U * shrink, V * shrink, cap


def shape_length(U, V, cap, scale="log"):
    if scale == "linear" or cap <= 0:
        return U, V
    mag = np.hypot(U, V)
    t = np.clip(mag / cap, 0.0, 1.0)
    new = np.sqrt(t) if scale == "sqrt" else np.log1p(9.0 * t) / np.log(10.0)
    f = np.where(mag > 0, new * cap / np.maximum(mag, 1e-30), 0.0)
    return U * f, V * f


def overlay_stream(ax, data, a, polar=True, artists=None):
    """Streamlines. Ported from vis2d.py's _overlay_stream, verified there against
    a Cartesian ground truth (pure rotation, pure outflow) before being trusted;
    the two quirks handled below are not obvious and cost real debugging time:

      - streamplot demands an exactly uniform grid, but .tab stores coordinates
        to ~5 significant digits, so the recorded spacing wobbles by ~1e-3 of a
        cell and numpy's allclose (inside streamplot) rejects it as non-uniform.
        The mesh really is uniform, so it is rebuilt exactly with linspace.
      - on polar axes phi is periodic; without repeating the first ring at 2*pi,
        streamlines visibly break along the seam at phi = 0.

    matplotlib cannot update a StreamplotSet in place, so an animation has to
    tear down the old artists and redraw every frame (the `artists` argument).
    """
    r, phi = np.asarray(data["r"]), np.asarray(data["phi"])
    R = np.asarray(data["R"])
    u_r, u_phi, degenerate, rel = vector_field(data, a.vec_comp, a.vec_frame)
    if degenerate:
        ax.text(0.5, 0.5, f"axisymmetric\nto {rel:.0e}\n(no mode seeded)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color="crimson", fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.85))
        return None

    if artists is not None:
        artists.lines.remove()
        for art in list(ax.patches):
            art.remove()

    omega = u_phi / np.maximum(R, 1e-30)
    r_u = np.linspace(r[0], r[-1], len(r))
    phi_u = np.linspace(phi[0], phi[-1], len(phi))
    try:
        if polar:
            dphi = phi_u[1] - phi_u[0] if len(phi_u) > 1 else 0.0
            php = np.append(phi_u, phi_u[-1] + dphi)
            wp = np.vstack([omega, omega[:1]])
            vrp = np.vstack([u_r, u_r[:1]])
            sp = ax.streamplot(php, r_u, wp.T, vrp.T, density=a.vec_density,
                               color=a.vec_color, linewidth=0.7, arrowsize=0.7)
            ax.set_ylim(r_u.min(), r_u.max())
        else:
            sp = ax.streamplot(r_u, phi_u, u_r.T, omega.T, density=a.vec_density,
                               color=a.vec_color, linewidth=0.7, arrowsize=0.7)
            ax.set_xlim(r_u.min(), r_u.max()); ax.set_ylim(phi_u.min(), phi_u.max())
    except (ValueError, IndexError) as exc:
        print(f"    streamplot skipped: {exc}")
        return None
    return sp


def _quiver_uv(data, a, polar):
    """(X, Y, U, V, stride) for the quiver style, before clipping/shaping."""
    r, phi = np.asarray(data["r"]), np.asarray(data["phi"])
    R, Phi = np.asarray(data["R"]), np.asarray(data["Phi"])
    u_r, u_phi, degenerate, rel = vector_field(data, a.vec_comp, a.vec_frame)
    if degenerate:
        return None, rel
    if polar:
        lat = getattr(a, "vec_lattice", "polar")
        if lat == "polar":
            j, i, Rs, Phis = polar_ring_lattice(r, phi, a.vec_arrows)
        else:
            j, i, Rs, Phis = grid_lattice(r, phi, a.vec_arrows,
                                          hexagonal=(lat == "hex"))
        ur, up = u_r[j, i], u_phi[j, i]
        # true Cartesian components: on a polar axes matplotlib draws quiver's
        # U,V as screen offsets, and screen space there is Cartesian
        U = ur * np.cos(Phis) - up * np.sin(Phis)
        V = ur * np.sin(Phis) + up * np.cos(Phis)
        return (Phis, Rs, U, V, None), None
    # r-phi panels are a plain rectangular grid, so decimate each axis on its own
    # count - a single stride would give nphi/nx1 times too many arrows one way
    def stride(n):
        if getattr(a, "vec_stride", None):
            return max(1, int(a.vec_stride))
        return max(1, int(round(n / float(a.vec_arrows))))

    sl = (slice(None, None, stride(len(phi))), slice(None, None, stride(len(r))))
    Rs, Phis = R[sl], Phi[sl]
    U, V = u_r[sl], u_phi[sl] / np.maximum(Rs, 1e-30)
    return (Rs, Phis, U, V, stride(len(r))), None


def update_vectors(ax, data, a, polar, artists=None, scale_data=None, add_key=True):
    """Draw or update the --add-vectors overlay. Handles both the single-shot
    static case (artists=None every call) and animation (artists passed back in
    from the previous frame), because both failure modes bit once already:

      - matplotlib cannot update a StreamplotSet in place, so an animation must
        tear down the previous frame's lines before drawing new ones, or they
        accumulate frame over frame. Same story for repeatedly calling
        ax.quiver() fresh - each call adds a new artist rather than replacing
        the old one, so an animation must create it once and then use set_UVC.
      - quiver's `scale` is fixed at artist-creation time. Sizing it off frame 0
        is a real bug: this pgen's t=0 initial condition has v_r = 0 everywhere,
        so the first frame's own magnitude gives a degenerate scale and arrows
        end up comically over- or under-sized once real data arrives. `scale_data`
        decouples "what sizes the arrows" (a developed, non-zero frame) from
        "what the arrows currently show" (whichever frame is being drawn).

    Returns (cap_or_None, artists) - artists is a dict for quiver ({'q': artist}),
    a StreamplotSet for stream, or None if nothing was drawn (degenerate field).
    """
    if a.vec_style == "stream":
        return None, overlay_stream(ax, data, a, polar=polar, artists=artists)

    r = np.asarray(data["r"])
    xyuv, rel = _quiver_uv(data, a, polar)
    if xyuv is None:
        if artists is None:                        # only annotate once per axes
            ax.text(0.5, 0.5, f"axisymmetric\nto {rel:.0e}\n(no mode seeded)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9,
                    color="crimson", fontweight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.85))
        return None, artists

    X, Y, U, V, stride = xyuv
    ref_xyuv, _ = _quiver_uv(scale_data, a, polar) if scale_data is not None else (xyuv, None)
    _, _, rU, rV, _ = ref_xyuv if ref_xyuv is not None else xyuv
    _, _, cap = clip_length(rU, rV, a.vec_clip)
    U, V, _ = clip_length(U, V, cap=cap)
    U, V = shape_length(U, V, cap, a.vec_scale)

    if isinstance(artists, dict) and "q" in artists:
        artists["q"].set_UVC(U, V)
        return cap, artists

    kw = dict(color=a.vec_color, alpha=0.9, pivot="mid", width=0.005,
             headwidth=3.4, headlength=4.2, headaxislength=3.6, zorder=5)
    if polar:
        d = (float(r[-1]) - float(r[0])) / max(2, a.vec_arrows)
        q = ax.quiver(X, Y, U, V, scale_units="width",
                      scale=cap * 2.0 * float(r[-1]) / (0.85 * d), **kw)
    else:
        dx = (float(r[-1]) - float(r[0])) / max(len(r), 1) * stride
        q = ax.quiver(X, Y, U, V, scale_units="x", scale=cap / (0.85 * max(dx, 1e-30)), **kw)
        if add_key and np.isfinite(cap) and cap > 0:
            # 0.72/0.94: above the axes it ran through the panel title, at x=0.97
            # the shaft ran off the panel
            ax.quiverkey(q, 0.72, 0.94, cap, f"{cap:.3g}", labelpos="W",
                         coordinates="axes", labelsep=0.05, zorder=7,
                         fontproperties={"size": 8})
    return cap, {"q": q}


def overlay_vectors(ax, data, a, polar=True, add_key=True):
    """One-shot overlay for the static (single-frame) plots."""
    cap, _ = update_vectors(ax, data, a, polar=polar, add_key=add_key)
    return cap


def vector_label(a):
    sym = r"\rho\vec{v}" if a.vec_field == "momentum" else r"\vec{v}"
    base = f"${sym}$" if a.vec_frame == "full" else f"${sym}-\\langle{sym}\\rangle_\\phi$"
    return base if a.vec_comp == "both" else base + f" ({a.vec_comp} only)"


def vector_caption(a, cap):
    """Caption line. It carries the arrow scale for the polar views, which have no
    room for a quiverkey - all four corners of a polar axes hold an angle label."""
    tail = (f",  longest arrow = {cap:.3g}"
            if cap and a.vec_style != "stream" else "")
    if a.vec_scale != "linear" and tail:
        tail += f" ({a.vec_scale} length scale)"
    return f"arrows: {vector_label(a)}  ({a.vec_style}){tail}"


# =============================================================================
# 2D PLOTS (mirrors vis2d.py)
# =============================================================================
VARS_2D = {
    "density":  dict(cmap="inferno", log=True,  label=r"$\rho$ (Density)"),
    "pressure": dict(cmap="plasma",  log=True,  label=r"$P$ (Pressure)"),
    "vel_r":    dict(cmap="RdBu_r",  log=False, label=r"$v_r$ (Radial Velocity)"),
    "vel_phi":  dict(cmap="coolwarm", log=False, label=r"$v_\phi$ (Azimuthal Velocity)"),
}

# The heatmap panel adds v_z; the polar, radial and azimuthal figures do not, which
# is how the four-script setup lays them out. In a 2D (r, phi) run v_z is identically
# zero, so it is a check that the run is planar rather than a field to read.
VARS_HEATMAP = dict(VARS_2D,
                    vel_z=dict(cmap="PuOr", log=False,
                               label=r"$v_z$ (Vertical Velocity)"))


TITLE_FIELDS = ("time", "cycle", "frame", "base")


def format_title(default, a, **fields):
    """--title overrides the built-in caption. {time} {cycle} {frame} {base} are
    substituted.

    All four are always defined, so a template stays usable on the figures where
    one of them is meaningless - an animation's suptitle has no single time, and
    substitutes an empty string rather than dropping back to the default title.
    A placeholder outside the four does fall back, with a note: a bad template
    must not kill an animation half way through.
    """
    template = getattr(a, "title", None)
    if not template:
        return default
    known = {k: "" for k in TITLE_FIELDS}
    known["base"] = getattr(a, "_base", "") or ""
    if getattr(a, "_frame", None) is not None:
        known["frame"] = a._frame
    known.update(fields)
    try:
        return template.format(**known)
    except (KeyError, IndexError, ValueError) as exc:
        print(f"  note: --title {template!r} could not be formatted ({exc}); "
              f"available fields: {', '.join(TITLE_FIELDS)}. Using the default.")
        return default


def _norm(field, info):
    if info["log"] and np.all(field > 0):
        return LogNorm(vmin=field.min(), vmax=field.max())
    if info["log"]:
        pos = field[field > 0]
        return Normalize(vmin=pos.min() if pos.size else field.min(), vmax=field.max())
    vmax = np.abs(field).max()
    return Normalize(vmin=-vmax, vmax=vmax)


def _apply_bounds_raw(data, r_min, r_max, phi_min, phi_max):
    """apply_bounds() with plain arguments, so a multiprocessing worker can call
    it without shipping the argparse namespace across the process boundary."""
    r, phi = data["r"], data["phi"]
    rmask = np.ones_like(r, dtype=bool)
    if r_min is not None:
        rmask &= r >= r_min
    if r_max is not None:
        rmask &= r <= r_max
    pmask = np.ones_like(phi, dtype=bool)
    if phi_min is not None:
        pmask &= phi >= phi_min
    if phi_max is not None:
        pmask &= phi <= phi_max
    if rmask.all() and pmask.all():
        return data
    out = dict(data)
    out["r"], out["phi"] = r[rmask], phi[pmask]
    for k in ("R", "Phi", "density", "pressure", "vel_r", "vel_phi", "vel_z"):
        if k in data:
            out[k] = data[k][pmask, :][:, rmask]
    return out


def apply_bounds(data, a):
    return _apply_bounds_raw(data, a.r_min, a.r_max,
                             getattr(a, "phi_min", None), getattr(a, "phi_max", None))


def apply_normalize(data, mode):
    if mode != "azimuthal":
        return data
    out = dict(data)
    for k in ("density", "pressure", "vel_r", "vel_phi", "vel_z"):
        if k not in data:
            continue
        field = data[k]
        mean = field.mean(axis=0)
        if k in ("density", "pressure"):
            safe = np.where(np.abs(mean) > 0, mean, 1.0)
            out[k] = field / safe[None, :] - 1.0
        else:
            out[k] = field - mean[None, :]
    return out


def var_info_for(mode, base=None):
    base = base if base is not None else VARS_2D
    if mode != "azimuthal":
        return base
    out = {}
    for k, v in base.items():
        rel = k in ("density", "pressure")
        out[k] = dict(v, log=False, cmap="RdBu_r",
                      label=v["label"].split(" (")[0]
                      + (r"$/\langle\cdot\rangle_\phi - 1$" if rel
                         else r"$ - \langle\cdot\rangle_\phi$"))
    return out


def plot_heatmap(data, a, out_path):
    data = apply_normalize(apply_bounds(data, a), a.normalize)
    info_map = var_info_for(a.normalize, VARS_HEATMAP)
    # the vector field gets a panel of its own, over density: on the v_r and v_phi
    # panels arrows would only restate what the colour already says
    n_panels = len(info_map) + (1 if a.add_vectors else 0)
    ncols = 3 if n_panels > 4 else 2
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12), squeeze=False)
    fig.suptitle(format_title(f"2D Disk Structure (t={data['time']:.3f}, "
                              f"cycle={data['cycle']})", a,
                              time=data['time'], cycle=data['cycle']),
                fontsize=16, fontweight="bold")
    flat = list(axes.flat)
    for ax, (name, info) in zip(flat, info_map.items()):
        field = data[name]
        im = ax.pcolormesh(data["R"], data["Phi"], field, cmap=info["cmap"],
                           norm=_norm(field, info), shading="auto")
        ax.set_xlabel("r", fontsize=12); ax.set_ylabel(r"$\phi$ (rad)", fontsize=12)
        ax.set_title(info["label"], fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label=info["label"]); ax.grid(True, alpha=0.3)
    if a.add_vectors:
        axv = flat[len(info_map)]
        rho = data["density"]
        axv.pcolormesh(data["R"], data["Phi"], rho, cmap=VEC_BG_CMAP, shading="auto",
                       norm=(LogNorm(rho[rho > 0].min(), rho.max()) if np.all(rho > 0)
                             else Normalize(rho.min(), rho.max())))
        overlay_vectors(axv, data, a, polar=False)
        axv.set_xlabel("r", fontsize=12); axv.set_ylabel(r"$\phi$ (rad)", fontsize=12)
        axv.set_title(f"{vector_label(a)}  over $\\rho$", fontsize=13, fontweight="bold")
        axv.grid(True, alpha=0.3)
    for ax in flat[n_panels:]:          # a 2x3 grid holding 5 panels has a spare
        ax.remove()
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_polar(data, a, out_path):
    data = apply_normalize(apply_bounds(data, a), a.normalize)
    info_map = var_info_for(a.normalize)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(format_title(f"Polar View of Disk - t={data['time']:.3f}", a,
                              time=data['time'], cycle=data['cycle']),
                fontsize=16, fontweight="bold")
    last_cap = None
    keys = list(info_map)
    for idx, name in enumerate(keys):
        info = info_map[name]
        ax = plt.subplot(2, 2, idx + 1, projection="polar")
        field = data[name]
        im = ax.pcolormesh(data["Phi"], data["R"], field, cmap=info["cmap"],
                           norm=_norm(field, info), shading="auto")
        ax.set_title(info["label"], fontsize=13, fontweight="bold", pad=20)
        plt.colorbar(im, ax=ax, label=info["label"], pad=0.1)
        ax.grid(True, alpha=0.3)
        if a.add_vectors and (a.vec_panels == "all" or idx == len(keys) - 1):
            last_cap = overlay_vectors(ax, data, a, polar=True)
    if a.add_vectors:
        fig.text(0.5, 0.012, vector_caption(a, last_cap), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_radial(data, a, out_path):
    data = apply_normalize(apply_bounds(data, a), getattr(a, "normalize", "none"))
    info_map = var_info_for(getattr(a, "normalize", "none"))
    colors = line_colors(4, a.palette)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(format_title(f"Radial Profiles (averaged over \u03c6) - "
                              f"t={data['time']:.3f}", a,
                              time=data['time'], cycle=data['cycle']),
                fontsize=14, fontweight="bold")
    labels = {"density": r"$\langle\rho\rangle_\phi$", "pressure": r"$\langle P\rangle_\phi$",
             "vel_r": r"$\langle v_r\rangle_\phi$", "vel_phi": r"$\langle v_\phi\rangle_\phi$"}
    for ax, (name, info), color in zip(axes.flat, info_map.items(), colors):
        mean, std = data[name].mean(axis=0), data[name].std(axis=0)
        ax.plot(data["r"], mean, color=color, linewidth=2, label=labels[name])
        ax.fill_between(data["r"], mean - std, mean + std, alpha=0.25, color=color)
        ax.set_xlabel("r", fontsize=11); ax.set_ylabel(labels[name], fontsize=11)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        if info["log"] and not getattr(a, "linear", False) and (mean > 0).all():
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


VARS_1D = {
    "density":  dict(marker="o", label=r"Density $\rho$"),
    "pressure": dict(marker="s", label=r"Pressure $P$"),
    "vel_r":    dict(marker="^", label=r"Radial Velocity $v_r$"),
    "vel_phi":  dict(marker="d", label=r"Azimuthal Velocity $v_\phi$"),
}


def plot_profiles_1d(data, a, out_path):
    data = apply_bounds(data, a)
    colors = line_colors(4, a.palette)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(format_title(f"Radial Profiles (t={data['time']:.3f}, "
                              f"cycle={data['cycle']})", a,
                              time=data['time'], cycle=data['cycle']),
                 fontsize=16, fontweight="bold")
    for ax, (name, info), color in zip(axes.flat, VARS_1D.items(), colors):
        field = np.asarray(data[name])
        prof = field.mean(axis=0) if field.ndim == 2 else field
        ax.plot(data["r"], prof, color=color, linewidth=2, marker=info["marker"],
                markersize=4, linestyle="-", label=info["label"])
        ax.set_xlabel("Radius r", fontsize=12)
        ax.set_ylabel(info["label"], fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        if getattr(a, "logscale", False) and name in ("density", "pressure"):
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_azimuthal(data, a, out_path):
    data = apply_bounds(data, a)
    r, phi = data["r"], data["phi"]
    idxs = [len(r) // 4, len(r) // 2, 3 * len(r) // 4]
    colors = line_colors(3, a.palette)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(format_title(f"Azimuthal Profiles at Different Radii - "
                              f"t={data['time']:.3f}", a,
                              time=data['time'], cycle=data['cycle']),
                fontsize=14, fontweight="bold")
    labels = {"density": r"$\rho$", "pressure": "$P$", "vel_r": "$v_r$", "vel_phi": r"$v_\phi$"}
    for ax, (name, info) in zip(axes.flat, VARS_2D.items()):
        for ridx, color in zip(idxs, colors):
            ax.plot(phi, data[name][:, ridx], color=color, linewidth=2, alpha=0.7,
                   label=f"r={r[ridx]:.2f}")
        ax.set_xlabel(r"$\phi$ (rad)", fontsize=11); ax.set_ylabel(labels[name], fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        if info["log"]:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def _sample_one(args):
    """Worker for global_ranges(). Top-level so multiprocessing can pickle it."""
    paths, names, bounds, normalize = args
    try:
        d = _apply_bounds_raw(read_frame(paths), *bounds)
        d = apply_normalize(d, normalize)
        return {nm: np.asarray(d[nm]).ravel() for nm in names if nm in d}
    except Exception:
        return None


def global_ranges(frames, keys, a, names):
    """Percentile colour limits sampled across the animation, not from frame 0.

    Taking the scale from the first frame is wrong here in a specific way: frame
    0 is the initial condition, where v_r = 0 everywhere, so a diverging velocity
    map would be normalised against nothing and every later frame would clip.
    Sampling is capped at ~10 frames spread through the run - enough for a stable
    percentile, cheap enough not to double the animation's cost.
    """
    step = max(1, len(keys) // 10)
    sample_keys = keys[::step]
    bounds = (a.r_min, a.r_max, getattr(a, "phi_min", None), getattr(a, "phi_max", None))
    jobs = [(frames[k], names, bounds, getattr(a, "normalize", "none"))
            for k in sample_keys]

    workers = min(int(getattr(a, "num_workers", 1) or 1), len(jobs))
    if workers > 1:
        try:
            with Pool(processes=workers) as pool:
                results = pool.map(_sample_one, jobs)
        except Exception as exc:
            print(f"  note: parallel sampling unavailable ({exc}); using one process")
            results = [_sample_one(j) for j in jobs]
    else:
        results = [_sample_one(j) for j in jobs]

    info_map = var_info_for(getattr(a, "normalize", "none"))
    out = {}
    for name in names:
        chunks = [r[name] for r in results if r and name in r]
        if not chunks:
            continue
        allv = np.concatenate(chunks)
        info = info_map[name]
        lo_p = getattr(a, "vmin_percentile", 2.0)
        hi_p = getattr(a, "vmax_percentile", 98.0)
        if info["log"] and np.any(allv > 0):
            pos = allv[allv > 0]
            out[name] = LogNorm(vmin=np.percentile(pos, lo_p),
                                vmax=np.percentile(pos, hi_p))
        elif info["log"]:
            out[name] = Normalize(vmin=np.percentile(allv, lo_p),
                                  vmax=np.percentile(allv, hi_p))
        else:
            vmax = np.percentile(np.abs(allv), hi_p)
            out[name] = Normalize(vmin=-vmax, vmax=vmax)
    return out


def _draw_info_box(fig, lines, a=None, plot="polar"):
    """Model annotation, placed per --info_pos. --info_on restricts it to the
    polar animation by default, the one with room for it."""
    if not lines:
        return
    pos = getattr(a, "info_pos", "box") if a is not None else "box"
    on = getattr(a, "info_on", "polar") if a is not None else "polar"
    if on == "polar" and plot != "polar":
        return
    if pos == "subtitle":
        fig.text(0.5, 0.928, "\n".join(lines), ha="center", va="top",
                 fontsize=9, color="#3c3c3a")
        return
    if pos == "footer":
        fig.text(0.5, 0.012, "\n".join(lines), ha="center", va="bottom",
                 fontsize=9, color="#3c3c3a")
        return
    # box: one item per line, so it stays in the corner however many are asked for
    stacked = "\n".join(item for line in lines for item in line.split("   ") if item)
    fig.text(0.012, 0.988, stacked, ha="left", va="top", fontsize=8.5,
             color="#1a1a19", linespacing=1.5,
             bbox=dict(boxstyle="round", fc="white", ec="#b8b8b4", alpha=0.9))


def animate_heatmap(frames, keys, a, out_path, info_lines):
    keys = keys[::max(1, a.subsample)]
    first = apply_normalize(apply_bounds(read_frame(frames[keys[0]]), a), a.normalize)
    info_map = var_info_for(a.normalize)

    norms = global_ranges(frames, keys, a, list(info_map))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(format_title("Disk Evolution", a), fontsize=16, fontweight="bold")
    ims = []
    for ax, (name, info) in zip(axes.flat, info_map.items()):
        field = first[name]
        im = ax.pcolormesh(first["R"], first["Phi"], field, cmap=info["cmap"],
                           norm=norms.get(name) or _norm(field, info), shading="auto")
        ax.set_xlabel("r"); ax.set_ylabel(r"$\phi$"); ax.set_title(info["label"])
        plt.colorbar(im, ax=ax); ax.grid(True, alpha=0.3)
        ims.append(im)
    _draw_info_box(fig, info_lines, a, plot="cartesian")
    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=12, fontweight="bold")

    def update(k):
        d = apply_normalize(apply_bounds(read_frame(frames[keys[k]]), a), a.normalize)
        time_text.set_text(frame_label(k, len(keys) - 1, d["time"],
                                       getattr(a, "_p_orb", None)))
        for im, name in zip(ims, info_map):
            im.set_array(d[name].ravel())
        return ims + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=150)
    plt.close(fig)


def animate_polar(frames, keys, a, out_path, info_lines):
    keys = keys[::max(1, a.subsample)]
    first = apply_normalize(apply_bounds(read_frame(frames[keys[0]]), a), a.normalize)
    last = apply_normalize(apply_bounds(read_frame(frames[keys[-1]]), a), a.normalize)
    info_map = var_info_for(a.normalize)

    norms = global_ranges(frames, keys, a, list(info_map))
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(format_title("Polar View of Disk Evolution", a),
                 fontsize=16, fontweight="bold", y=0.98)
    axes, ims = [], []
    for i, (name, info) in enumerate(info_map.items()):
        ax = plt.subplot(2, 2, i + 1, projection="polar")
        field = first[name]
        im = ax.pcolormesh(first["Phi"], first["R"], field, cmap=info["cmap"],
                           norm=norms.get(name) or _norm(field, info), shading="auto")
        ax.set_title(info["label"], fontsize=13, fontweight="bold", pad=20)
        plt.colorbar(im, ax=ax, label=info["label"], pad=0.1)
        ax.grid(True, alpha=0.3)
        axes.append(ax); ims.append(im)

    vec_axes = [ax for idx, ax in enumerate(axes)
               if a.vec_panels == "all" or idx == len(axes) - 1]
    vec_state = {}
    if a.add_vectors:
        for ax in vec_axes:
            # sized off `last` (a developed frame), drawn from `first` - the
            # initial condition has v_r = 0 everywhere, a bad scale reference
            _, art = update_vectors(ax, first, a, polar=True, scale_data=last)
            vec_state[id(ax)] = art
    _draw_info_box(fig, info_lines, a, plot="polar")
    time_text = fig.text(0.5, 0.94, "", ha="center", fontsize=12, fontweight="bold")

    def update(k):
        d = apply_normalize(apply_bounds(read_frame(frames[keys[k]]), a), a.normalize)
        time_text.set_text(frame_label(k, len(keys) - 1, d["time"],
                                       getattr(a, "_p_orb", None)))
        for im, name in zip(ims, info_map):
            im.set_array(d[name].ravel())
        if a.add_vectors:
            for ax in vec_axes:
                _, vec_state[id(ax)] = update_vectors(
                    ax, d, a, polar=True, artists=vec_state[id(ax)], scale_data=last,
                    add_key=False)
        return ims + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    # leave room for the title and the frame counter, or the second row's panel
    # titles land on the first row's 270-degree tick label
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ani.save(out_path, fps=a.fps, dpi=150)
    plt.close(fig)


def animate_vector(frames, keys, a, out_path, info_lines):
    keys = keys[::max(1, a.subsample)]
    first = apply_bounds(read_frame(frames[keys[0]]), a)
    # scale reference: frame 0 is the initial condition, with v_r = 0 everywhere
    last = apply_bounds(read_frame(frames[keys[-1]]), a)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")
    fig.suptitle(format_title("Velocity Field", a), fontsize=16, fontweight="bold")
    rho = first["density"]
    # the background is context for the arrows, so its scale comes from the frame
    # it is drawn from rather than from a sample of the whole run
    lo = np.percentile(rho[rho > 0], a.vmin_percentile) if np.any(rho > 0) else None
    hi = np.percentile(rho, a.vmax_percentile)
    bg = ax.pcolormesh(first["Phi"], first["R"], rho, cmap=VEC_BG_CMAP, shading="auto",
                       norm=(LogNorm(vmin=lo, vmax=hi) if lo and lo > 0
                             else Normalize(vmin=rho.min(), vmax=hi)))
    plt.colorbar(bg, ax=ax, pad=0.1, fraction=0.04).set_label(r"$\rho$", fontsize=11)
    cap, vec_art = update_vectors(ax, first, a, polar=True, scale_data=last)
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.035, vector_caption(a, cap), ha="center", fontsize=12)
    _draw_info_box(fig, info_lines, a, plot="vector")
    time_text = fig.text(0.5, 0.93, "", ha="center", fontsize=12, fontweight="bold")

    def update(k):
        nonlocal vec_art
        d = apply_bounds(read_frame(frames[keys[k]]), a)
        time_text.set_text(frame_label(k, len(keys) - 1, d["time"],
                                       getattr(a, "_p_orb", None)))
        bg.set_array(d["density"].ravel())
        _, vec_art = update_vectors(ax, d, a, polar=True, artists=vec_art,
                                    scale_data=last, add_key=False)
        return [bg, time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=150)
    plt.close(fig)


def window_frames(keys, a):
    """Restrict to [--start_frame, --end_frame]. Subsampling is left to the
    animation functions, which apply it per animation."""
    lo = getattr(a, "start_frame", None)
    hi = getattr(a, "end_frame", None)
    out = [k for k in keys if (lo is None or k >= lo) and (hi is None or k <= hi)]
    if not out:
        raise SystemExit(f"no frames left after --start_frame/--end_frame "
                         f"({lo}..{hi}); available {keys[0]}..{keys[-1]}")
    if len(out) != len(keys):
        print(f"  frame window: {out[0]}..{out[-1]} ({len(out)} frames)")
    return out


def apply_cmap_palette(a):
    """Swap the 2D scalar colormaps for a continuous pypalettes map, on request.

    Off by default on purpose: inferno and plasma are perceptually uniform and
    RdBu_r/coolwarm are properly diverging, while most of the library's palettes
    are neither, so an arbitrary swap misrepresents the field rather than merely
    restyling it."""
    name = getattr(a, "cmap_palette", None)
    if not name:
        return
    try:
        from pypalettes import load_cmap
        cm = load_cmap(name, cmap_type="continuous")
        for info in VARS_2D.values():
            info["cmap"] = cm
        print(f"2D colormaps replaced by pypalettes '{name}'")
    except Exception as exc:
        print(f"  note: --cmap_palette '{name}' unusable ({exc}); keeping the defaults")


def cmd_2d(a):
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_athvis_2d"
    os.makedirs(out_dir, exist_ok=True)
    info = discover(data_dir, output_id=a.output_id)
    a._base = info.get("base", "")
    keys = sorted(info["frames"])
    print(f"{len(keys)} frame(s), {keys[0]}..{keys[-1]}, base '{info['base']}'")

    keys = window_frames(keys, a)
    apply_cmap_palette(a)

    # The athinput is read once, for the annotation and for P_orb - the latter is
    # wanted on the animations even when --info is off, since code time alone says
    # nothing about how far the run has got.
    info_lines = []
    ath = find_athinput(data_dir, a.params)
    a._p_orb = _derived_p_orb(read_athinput(ath)) if ath else None
    if a.info != "none":
        if ath:
            info_lines = model_info_lines(read_athinput(ath), a.info)
            print(f"Model info: {a.info} from {os.path.basename(ath)}")
        else:
            print("Model info requested, no athinput found; omitted")

    if a.mode in ("all", "animation"):
        out = os.path.join(out_dir, "disk_evolution_2d.mp4")
        animate_heatmap(info["frames"], keys, a, out, info_lines)
        print(f"Saved: {out}")
    if a.mode in ("all", "polar_animation"):
        out = os.path.join(out_dir, "disk_evolution_polar.mp4")
        animate_polar(info["frames"], keys, a, out, info_lines)
        print(f"Saved: {out}")
    if a.mode == "vector_animation":
        out = os.path.join(out_dir, "velocity_field.mp4")
        animate_vector(info["frames"], keys, a, out, info_lines)
        print(f"Saved: {out}")
    if a.mode in ("all", "heatmaps", "radial", "polar", "azimuthal"):
        frame = a.frame if a.frame is not None else keys[-1]
        a._frame = frame
        if frame not in info["frames"]:
            raise SystemExit(f"frame {frame} not found; available {keys[0]}..{keys[-1]}")
        data = read_frame(info["frames"][frame])
        if a.mode in ("all", "heatmaps"):
            out = os.path.join(out_dir, f"heatmap_{frame:05d}.png")
            plot_heatmap(data, a, out); print(f"Saved: {out}")
        if a.mode in ("all", "radial"):
            out = os.path.join(out_dir, f"radial_{frame:05d}.png")
            plot_radial(data, a, out); print(f"Saved: {out}")
        if a.mode in ("all", "polar"):
            out = os.path.join(out_dir, f"polar_{frame:05d}.png")
            plot_polar(data, a, out); print(f"Saved: {out}")
        if a.mode == "azimuthal":
            out = os.path.join(out_dir, f"azimuthal_{frame:05d}.png")
            plot_azimuthal(data, a, out); print(f"Saved: {out}")


# =============================================================================
# 1D (mirrors vis1d.py) - the radial profile evolution *line* animation
# =============================================================================
def animate_1d_evolution(frames, keys, a, out_path):
    """Radial profiles over time, on axes fixed by the whole window.

    Ranges come from every frame in the animation, not from the first: with
    per-frame autoscaling a curve that barely moves looks like it is thrashing,
    and one that does move goes off the top.
    """
    keys = keys[::max(1, a.subsample)]
    colors = line_colors(4, a.palette)
    logscale = getattr(a, "logscale", False)

    all_data = []
    for k in keys:
        d = apply_bounds(read_frame(frames[k]), a)
        prof = {"time": d["time"], "cycle": d["cycle"], "r": d["r"]}
        for name in VARS_1D:
            arr = np.asarray(d[name])
            prof[name] = arr.mean(axis=0) if arr.ndim == 2 else arr
        all_data.append(prof)

    ranges = {}
    for name in VARS_1D:
        vals = np.concatenate([d[name] for d in all_data])
        if logscale and name in ("density", "pressure"):
            pos = vals[vals > 0]
            ranges[name] = (pos.min() * 0.5, pos.max() * 2.0) if pos.size else (0.1, 1.0)
        else:
            margin = 0.1 * (vals.max() - vals.min())
            ranges[name] = (vals.min() - margin, vals.max() + margin)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(format_title("Radial Profile Evolution", a),
                 fontsize=16, fontweight="bold", y=0.98)
    r0 = all_data[0]["r"]
    lines = {}
    for ax, (name, info), color in zip(axes.flat, VARS_1D.items(), colors):
        (lines[name],) = ax.plot([], [], color=color, linewidth=2,
                                 marker=info["marker"], markersize=4, linestyle="-",
                                 label=info["label"])
        ax.set_xlabel("Radius r", fontsize=12)
        ax.set_ylabel(info["label"], fontsize=12)
        ax.set_title(info["label"], fontsize=13, fontweight="bold", pad=20)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_xlim(r0.min(), r0.max()); ax.set_ylim(ranges[name])
        if logscale and name in ("density", "pressure"):
            ax.set_yscale("log")
    time_text = fig.text(0.5, 0.94, "", ha="center", fontsize=12, fontweight="bold")

    def update(idx):
        d = all_data[idx]
        for name in VARS_1D:
            lines[name].set_data(d["r"], d[name])
        time_text.set_text(f"Frame {idx:3d}/{len(keys) - 1} | Time = {d['time']:5.2f}")
        return list(lines.values()) + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ani.save(out_path, fps=a.fps, dpi=150)
    plt.close(fig)


def cmd_1d(a):
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_athvis_1d"
    os.makedirs(out_dir, exist_ok=True)
    info = discover(data_dir, output_id=a.output_id)
    a._base = info.get("base", "")
    keys = window_frames(sorted(info["frames"]), a)
    print(f"{len(keys)} frame(s), {keys[0]}..{keys[-1]}")

    if a.mode in ("all", "profiles"):
        frame = a.frame if a.frame is not None else keys[-1]
        a._frame = frame
        data = read_frame(info["frames"][frame])
        out = os.path.join(out_dir, f"radial_profile_{frame:05d}.png")
        plot_profiles_1d(data, a, out)
        print(f"Saved: {out}")
    if a.mode in ("all", "animation"):
        out = os.path.join(out_dir, "radial_profile_evolution.mp4")
        animate_1d_evolution(info["frames"], keys, a, out)
        print(f"Saved: {out}")


# =============================================================================
# HST (mirrors vishst.py)
# =============================================================================
DEFAULT_HST_VARS = ["disk_mass", "mdot_in"]

# Fixed colours for the columns that have a meaning here; anything else falls back
# to the palette. Keeping disk_mass dark red across every figure is the point.
HST_INFO = {
    "time": ("Time", "black"), "dt": ("Timestep", "gray"),
    "mass": ("Total Mass", "blue"), "1-mom": ("Radial Momentum", "red"),
    "2-mom": ("Azimuthal Momentum", "green"), "3-mom": ("Vertical Momentum", "purple"),
    "1-KE": ("Radial Kinetic Energy", "orange"),
    "2-KE": ("Azimuthal Kinetic Energy", "cyan"),
    "3-KE": ("Vertical Kinetic Energy", "magenta"),
    "tot-E": ("Total Energy", "darkblue"),
    "disk_mass": (r"Disk Mass [$M_\odot$]", "darkred"),
    "mdot_in": (r"Accretion Rate [$M_\odot$/yr]", "darkgreen"),
}



def log_axis_ok(values):
    """(usable, reason) for putting `values` on a log y-axis.

    Two failures, both of which this project's .hst produces:
      - a column that touches zero or goes negative (mdot_in does). matplotlib
        masks those points and the curve silently stops.
      - dt_visc, which is ~7e307 when viscosity is off. That is finite and
        positive, so it passes the obvious check, but padding it by a decade
        overflows and the tick locator raises OverflowError - `--mode all` used
        to abort on any inviscid run.
    """
    v = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(v)):
        return False, "has non-finite samples"
    n_bad = int((v <= 0).sum())
    if n_bad:
        return False, f"{n_bad} of {v.size} samples are <= 0"
    if v.max() > 1e300:
        return False, "values are at the 'disabled' sentinel (~1e308)"
    return True, ""

def hst_style(name):
    """(label, colour) for a history column."""
    if name in HST_INFO:
        return HST_INFO[name]
    return name, line_colors(1, quiet=True)[0]


def _stats_box(ax, text, corner="top"):
    ax.text(0.02, 0.98 if corner == "top" else 0.02, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top" if corner == "top" else "bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat",
                  "alpha": 0.5 if corner == "top" else 0.7})


def plot_hst_variable(time, values, name, a, out_dir, linear=False):
    label, color = hst_style(name)
    figsize = tuple(getattr(a, "figsize", (10.0, 6.0)))
    fig, ax = plt.subplots(figsize=figsize, dpi=a.dpi)
    ax.plot(time, values, linewidth=2.0, alpha=0.8, color=color, label=label)
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(label, fontsize=12, fontweight="bold")
    ax.set_title(format_title(f"{label} Evolution", a), fontsize=14,
                 fontweight="bold", pad=15)
    if not linear:
        ok, why = log_axis_ok(values)
        if ok:
            ax.set_yscale("log")
        else:
            print(f"  note: {name} stays on a linear axis - it {why}")
    if getattr(a, "grid", False):
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    v = np.asarray(values, dtype=float)
    _stats_box(ax, f"Mean: {v.mean():.4e}\nStd: {v.std():.4e}\n"
                   f"Min: {v.min():.4e}\nMax: {v.max():.4e}")
    fig.tight_layout()
    out = os.path.join(out_dir, f"{name.replace('-', '_').replace(' ', '_')}_vs_time.png")
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    return out


def plot_hst_overview(time, series, names, a, out_dir, linear_vars=()):
    n = len(names)
    w, h = tuple(getattr(a, "figsize", (10.0, 6.0)))
    if n <= 2:
        nrows, ncols, figsize = (1, max(n, 1), (w, h))
    elif n <= 4:
        nrows, ncols, figsize = (2, 2, (w * 1.2, h * 1.5))
    elif n <= 6:
        nrows, ncols, figsize = (2, 3, (w * 1.5, h * 1.5))
    elif n <= 9:
        nrows, ncols, figsize = (3, 3, (w * 1.5, h * 2))
    else:
        nrows, ncols = -(-n // 3), 3
        figsize = (w * 1.5, h * (nrows * 0.8))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=a.dpi, squeeze=False)
    flat = list(axes.flat)
    for ax, name in zip(flat, names):
        label, color = hst_style(name)
        values = np.asarray(series[name], dtype=float)
        ax.plot(time, values, linewidth=1.6, alpha=0.8, color=color)
        ax.set_xlabel("Time", fontsize=10); ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        if name not in linear_vars and log_axis_ok(values)[0]:
            ax.set_yscale("log")
        if getattr(a, "grid", False):
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    for ax in flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(out_dir, "all_variables_overview.png")
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    return out


def plot_disk_mass_and_mdot(time, disk_mass, mdot_in, a, out_dir):
    """The two headline columns on shared axes - they differ by many decades, so
    a single y-axis would flatten one of them."""
    c1, c2 = line_colors(2, quiet=True)
    fig, ax1 = plt.subplots(figsize=tuple(getattr(a, "figsize", (10.0, 6.0))), dpi=a.dpi)
    ax1.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"Disk Mass [$M_\odot$]", fontsize=12, fontweight="bold", color=c1)
    l1 = ax1.plot(time, disk_mass, linewidth=2.0, alpha=0.8, color=c1, label="Disk Mass")
    ax1.tick_params(axis="y", labelcolor=c1)
    if getattr(a, "grid", False):
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Accretion Rate [$M_\odot$/yr]", fontsize=12, fontweight="bold",
                   color=c2)
    l2 = ax2.plot(time, mdot_in, linewidth=2.0, alpha=0.8, color=c2,
                  label=r"$\dot{M}_{in}$")
    ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title(format_title("Disk Mass and Accretion Rate Evolution", a),
                  fontsize=14, fontweight="bold", pad=15)
    lines = l1 + l2
    ax1.legend(lines, [str(l.get_label()) for l in lines], loc="best", framealpha=0.9)
    _stats_box(ax1,
               f"Mean Disk Mass: {np.mean(disk_mass):.4e} M$_\\odot$\n"
               f"Mean $\\dot{{M}}_{{in}}$: {np.mean(mdot_in):.4e} M$_\\odot$/yr",
               corner="bottom")
    fig.tight_layout()
    out = os.path.join(out_dir, "disk_mass_and_mdot.png")
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    return out


def apply_style(name):
    """Matplotlib style for the hst figures. vishst.py sets one and the other
    three scripts do not, so it is applied per command, not globally."""
    if not name or name == "none":
        return
    try:
        plt.style.use(name)
    except Exception:
        print(f"  note: style '{name}' is not available; using the default")
        plt.style.use("default")


def cmd_hst(a):
    apply_style(getattr(a, "style", None))
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_athvis_hst"
    os.makedirs(out_dir, exist_ok=True)
    path, series = read_hst(data_dir)
    a._base = os.path.basename(path).rsplit(".hst", 1)[0]
    time = series["time"]
    print(f"Reading: {path}")
    print(f"Available: {', '.join(series)}")

    if a.mode == "default":
        var_names = [v for v in DEFAULT_HST_VARS if v in series]
    elif a.mode == "all":
        var_names = [v for v in series if v not in ("time", "dt")]
    else:
        if not a.vars:
            raise SystemExit("--vars is required for --mode custom")
        missing = [v for v in a.vars if v not in series]
        if missing:
            raise SystemExit(f"not found: {', '.join(missing)}; available: "
                             f"{', '.join(series)}")
        var_names = a.vars

    linear_vars = set(a.linear_scale) if getattr(a, "linear_scale", None) else set()
    if getattr(a, "linear", False):
        linear_vars |= set(var_names)

    if a.mode == "default" and "disk_mass" in series and "mdot_in" in series:
        print(f"Saved: {plot_disk_mass_and_mdot(time, series['disk_mass'], series['mdot_in'], a, out_dir)}")
    if a.mode == "all":
        print(f"Saved: {plot_hst_overview(time, series, var_names, a, out_dir, linear_vars)}")

    for name in var_names:
        print(f"Saved: {plot_hst_variable(time, series[name], name, a, out_dir, name in linear_vars)}")

    if a.mode == "custom" and len(var_names) > 1:
        print(f"Saved: {plot_hst_overview(time, series, var_names, a, out_dir, linear_vars)}")


# =============================================================================
# FORCES (mirrors visforces.py) - uov output (default output_id=2)
# =============================================================================
FORCE_KEYS = ["f_grav", "f_centr", "f_press", "f_sum"]


def force_colors(a):
    """One colour per force, from the same 4-colour palette every script uses."""
    return dict(zip(FORCE_KEYS, line_colors(4, getattr(a, "palette", None), quiet=True)))
FORCE_LABELS = {
    "f_grav":  r"$f_\mathrm{grav} = -\beta/r^2$",
    "f_centr": r"$f_\mathrm{centr} = v_\phi^2/r$",
    "f_press": r"$f_\mathrm{press} = -(1/\rho)\,\partial P/\partial r$",
    "f_sum":   r"$f_\mathrm{sum} = f_\mathrm{grav}+f_\mathrm{centr}+f_\mathrm{press}$",
}


def read_uov_frame(paths):
    """Same shape convention as read_frame(), for the uov columns."""
    blocks = []
    for p in paths:
        d = _read_tab_file(p)
        x1 = np.asarray(d.pop("x1v")); x2 = np.asarray(d.pop("x2v"))
        ax1, ax2 = (x1[0, :], x2[:, 0]) if x1.ndim == 2 else (x1.ravel(), None)
        rec = {"x1v": ax1, "x2v": ax2, "time": d.pop("time"), "cycle": d.pop("cycle")}
        rec.update(d)
        blocks.append(rec)
    if len(blocks) == 1:
        b = blocks[0]
        out = {"time": b["time"], "cycle": b["cycle"], "r": b["x1v"]}
        for k in FORCE_KEYS:
            if k in b:
                out[k] = np.asarray(b[k])
        return out
    x1 = np.unique(np.concatenate([b["x1v"] for b in blocks]))
    x2 = np.unique(np.concatenate([b["x2v"] for b in blocks]))
    out = {"time": blocks[0]["time"], "cycle": blocks[0]["cycle"], "r": x1, "phi": x2}
    for k in FORCE_KEYS:
        if k not in blocks[0]:
            continue
        grid = np.zeros((len(x2), len(x1)))
        for b in blocks:
            iy, ix = np.searchsorted(x2, b["x2v"]), np.searchsorted(x1, b["x1v"])
            grid[np.ix_(iy, ix)] = np.asarray(b[k]).reshape(len(iy), len(ix))
        out[k] = grid
    return out


def clip_forces(data, a):
    """Radial window for the uov frames. They carry no phi bounds - the forces are
    plotted as azimuthal means - so this is deliberately narrower than
    apply_bounds()."""
    r = data["r"]
    lo, hi = getattr(a, "r_min", None), getattr(a, "r_max", None)
    if lo is None and hi is None:
        return data
    mask = np.ones_like(r, dtype=bool)
    if lo is not None:
        mask &= r >= lo
    if hi is not None:
        mask &= r <= hi
    out = dict(data)
    out["r"] = r[mask]
    for k in FORCE_KEYS:
        if k in data:
            out[k] = data[k][..., mask]
    return out


def phi_average(field):
    """(mean, std) over phi, or (field, 0) for a run that is already 1D."""
    arr = np.asarray(field)
    if arr.ndim == 2:
        return arr.mean(axis=0), arr.std(axis=0)
    return arr, np.zeros_like(arr)


def apply_symlog(ax, arrays, linthresh=None):
    """symlog y-axis. The forces change sign, so a plain log axis drops half the
    curve; linthresh sets the width of the linear region around zero and defaults
    to the 5th percentile of |f|, i.e. just below the smallest values on show."""
    if linthresh is None:
        allabs = np.abs(np.concatenate([np.asarray(x).ravel() for x in arrays]))
        nz = allabs[allabs > 0]
        linthresh = float(np.percentile(nz, 5)) if nz.size else 1.0
    ax.set_yscale("symlog", linthresh=linthresh)


def plot_forces_frame(data, a, out_path, show_sum=False):
    data = clip_forces(data, a)
    r = data["r"]
    keys = [k for k in FORCE_KEYS if k != "f_sum" and k in data]
    if show_sum and "f_sum" in data:
        keys.append("f_sum")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title(f"Radial Force Balance (\u03c6-averaged) \u2014 "
                              f"t = {data['time']:.4f}, cycle = {data['cycle']}", a,
                              time=data['time'], cycle=data['cycle']),
                 fontsize=13, fontweight="bold")
    means = []
    for key in keys:
        avg, std = phi_average(data[key])
        means.append(avg)
        color = force_colors(a)[key]
        ax.plot(r, avg, color=color, label=FORCE_LABELS[key],
                linestyle="--" if key == "f_sum" else "-",
                linewidth=1.5 if key == "f_sum" else 2.0)
        ax.fill_between(r, avg - std, avg + std, alpha=0.15, color=color)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$r$ (dimensionless)", fontsize=12)
    ax.set_ylabel(r"Force per unit mass", fontsize=12)
    ax.legend(fontsize=10, loc="best"); ax.grid(True, alpha=0.3)
    if a.log:
        apply_symlog(ax, means, a.linthresh)
    fig.tight_layout(); fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_forces_sum(data, a, out_path):
    data = clip_forces(data, a)
    r = data["r"]
    avg, std = phi_average(data["f_sum"])
    color = force_colors(a)["f_sum"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(format_title(f"Net Radial Force (\u03c6-averaged) \u2014 "
                              f"t = {data['time']:.4f}, cycle = {data['cycle']}", a,
                              time=data['time'], cycle=data['cycle']),
                 fontsize=13, fontweight="bold")
    ax.plot(r, avg, linewidth=2.0, color=color, label=FORCE_LABELS["f_sum"])
    ax.fill_between(r, avg - std, avg + std, alpha=0.2, color=color)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$r$ (dimensionless)", fontsize=12)
    ax.set_ylabel(r"$f_\mathrm{sum}$ (force per unit mass)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    if a.log:
        apply_symlog(ax, [avg], a.linthresh)
    fig.tight_layout(); fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def animate_forces(frames, keys, a, out_path, sum_only=False):
    """Animated phi-averaged force profiles, with the +/-std band.

    The y-limits are taken from a sample spread over the run, not from the first
    frame: at t = 0 the flux arrays are still empty, so frame 0's residual is not
    representative of anything that follows.
    """
    keys = keys[::max(1, a.subsample)]
    force_keys = ["f_sum"] if sum_only else ["f_grav", "f_centr", "f_press"]
    colors = force_colors(a)
    first = clip_forces(read_uov_frame(frames[keys[0]]), a)
    r = first["r"]

    step = max(1, len(keys) // 10)
    sampled = []
    for k in keys[::step]:
        try:
            d = clip_forces(read_uov_frame(frames[k]), a)
            sampled.extend(phi_average(d[key])[0] for key in force_keys)
        except Exception:
            pass
    if not sampled:
        ylim = (-0.1, 0.1) if sum_only else (-1.0, 1.0)
    elif sum_only:
        # the residual is centred on zero and should look it, so the limits are
        # symmetric rather than taken from where the sampled values happen to fall
        yabs = np.percentile(np.abs(np.concatenate(sampled)), 99)
        ylim = (-yabs * 1.2, yabs * 1.2)
    else:
        allv = np.concatenate(sampled)
        lo, hi = np.percentile(allv, 1), np.percentile(allv, 99)
        margin = 0.1 * (hi - lo)
        ylim = (lo - margin, hi + margin)

    figsize = (8, 5) if sum_only else (10, 6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(r"$r$ (dimensionless)", fontsize=12)
    ax.set_ylabel(r"$f_\mathrm{sum}$ (force per unit mass)" if sum_only
                  else r"Force per unit mass", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, alpha=0.3)
    if a.log:
        apply_symlog(ax, [phi_average(first[k])[0] for k in force_keys], a.linthresh)

    band = 0.2 if sum_only else 0.15
    lines, fills = {}, {}
    for key in force_keys:
        avg, std = phi_average(first[key])
        (lines[key],) = ax.plot(r, avg, linewidth=2.0, color=colors[key],
                                label=FORCE_LABELS[key])
        fills[key] = ax.fill_between(r, avg - std, avg + std, alpha=band,
                                     color=colors[key])
    ax.legend(fontsize=10, loc="best")
    title = ax.set_title("", fontsize=12, fontweight="bold")
    head = "Net Radial Force" if sum_only else "Radial Force Balance"

    def update(idx):
        d = clip_forces(read_uov_frame(frames[keys[idx]]), a)
        title.set_text(format_title(
            f"{head} (\u03c6-averaged) \u2014 t = {d['time']:.4f}  "
            f"[{idx + 1}/{len(keys)}]", a,
            time=d["time"], cycle=d["cycle"], frame=keys[idx]))
        for key in force_keys:
            avg, std = phi_average(d[key])
            lines[key].set_ydata(avg)
            fills[key].remove()
            fills[key] = ax.fill_between(d["r"], avg - std, avg + std,
                                         alpha=band, color=colors[key])
        return list(lines.values()) + [title]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=150)
    plt.close(fig)


def cmd_forces(a):
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_athvis_forces"
    os.makedirs(out_dir, exist_ok=True)
    info = discover(data_dir, output_id=a.output_id)
    a._base = info.get("base", "")
    keys = window_frames(sorted(info["frames"]), a)
    print(f"{len(keys)} uov frame(s), {keys[0]}..{keys[-1]}")

    if a.mode in ("frame", "all"):
        frame = a.frame if a.frame is not None else keys[-1]
        a._frame = frame
        data = read_uov_frame(info["frames"][frame])
        out = os.path.join(out_dir, f"forces_frame_{frame:05d}.png")
        plot_forces_frame(data, a, out); print(f"Saved: {out}")
    if a.mode in ("sum", "all"):
        frame = a.frame if a.frame is not None else keys[-1]
        a._frame = frame
        data = read_uov_frame(info["frames"][frame])
        out = os.path.join(out_dir, f"force_sum_frame_{frame:05d}.png")
        plot_forces_sum(data, a, out); print(f"Saved: {out}")
    if a.mode == "animation":
        out = os.path.join(out_dir, "forces_animation.mp4")
        animate_forces(info["frames"], keys, a, out); print(f"Saved: {out}")
    if a.mode == "sum_animation":
        out = os.path.join(out_dir, "forces_sum_animation.mp4")
        animate_forces(info["frames"], keys, a, out, sum_only=True); print(f"Saved: {out}")


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, output_id=1, dpi=200):
        sp.add_argument("--data_dir", default=None,
                        help="directory holding the frames (default: ./data, ../data, .)")
        sp.add_argument("--output_dir", default=None,
                        help="where the figures go (default: figs_athvis_<cmd>)")
        sp.add_argument("--output_id", type=int, default=output_id,
                        help="which <outputN> block to read")
        sp.add_argument("--dpi", type=int, default=dpi, help="raster resolution")
        sp.add_argument("--title", default=None,
                        help="caption template over {time} {cycle} {frame}")
        sp.add_argument("--palette", default=None,
                        help="pypalettes name for line colours")

    def frame_opts(sp):
        """Flags that only mean something where there is a sequence of frames."""
        sp.add_argument("--start_frame", type=int, default=None,
                        help="first frame number to use")
        sp.add_argument("--end_frame", type=int, default=None,
                        help="last frame number to use")
        sp.add_argument("--num_workers", type=int, default=max(1, cpu_count() // 2),
                        help="processes used to sample colour ranges")

    sp2d = sub.add_parser("2d", help="2D maps, polar views and animations")
    common(sp2d)
    frame_opts(sp2d)
    sp2d.add_argument("--mode", default="all",
                      choices=["all", "heatmaps", "radial", "polar", "azimuthal",
                              "animation", "polar_animation", "vector_animation"])
    sp2d.add_argument("--frame", type=int, default=None)
    sp2d.add_argument("--fps", type=int, default=10)
    sp2d.add_argument("--subsample", type=int, default=1)
    sp2d.add_argument("--r_min", type=float, default=None)
    sp2d.add_argument("--r_max", type=float, default=None)
    sp2d.add_argument("--phi_min", type=float, default=None)
    sp2d.add_argument("--phi_max", type=float, default=None)
    sp2d.add_argument("--normalize", default="none", choices=["none", "azimuthal"],
                      help="plot each field as its deviation from the azimuthal mean")
    sp2d.add_argument("--vmin_percentile", type=float, default=2.0,
                      help="lower colour-scale percentile for animations")
    sp2d.add_argument("--vmax_percentile", type=float, default=98.0,
                      help="upper colour-scale percentile for animations")
    sp2d.add_argument("--cmap_palette", default=None,
                      help="replace the 2D colormaps with a pypalettes continuous map")
    sp2d.add_argument("--add-vectors", dest="add_vectors", action="store_true",
                      help="overlay the velocity (or momentum) field")
    sp2d.add_argument("--vec_field", default="velocity", choices=["velocity", "momentum"],
                      help="which vector field to draw")
    sp2d.add_argument("--vec_comp", default="both", choices=["both", "radial", "azimuthal"],
                      help="keep both components, or zero one of them")
    sp2d.add_argument("--vec_frame", default="perturbation",
                      choices=["perturbation", "full"],
                      help="subtract the azimuthal mean, or draw the raw field")
    sp2d.add_argument("--vec_panels", default="last", choices=["last", "all"],
                      help="overlay on the last panel only, or on every panel")
    sp2d.add_argument("--vec_style", default="quiver", choices=["quiver", "stream"],
                      help="arrows, or integrated streamlines")
    sp2d.add_argument("--vec_lattice", default="polar",
                      choices=["polar", "square", "hex"],
                      help="where the arrows sit: rings of constant r, a Cartesian "
                           "lattice, or a triangular one")
    sp2d.add_argument("--vec_scale", default="log", choices=["linear", "sqrt", "log"],
                      help="how arrow length maps to magnitude")
    sp2d.add_argument("--vec_arrows", type=int, default=8,
                      help="arrows across the radial extent")
    sp2d.add_argument("--vec_stride", type=int, default=None,
                      help="r-phi panels: take every Nth cell instead of --vec_arrows")
    sp2d.add_argument("--vec_clip", type=float, default=92.0,
                      help="percentile above which arrow length saturates")
    sp2d.add_argument("--vec_color", default="black", help="arrow / streamline colour")
    sp2d.add_argument("--vec_density", type=float, default=0.6,
                      help="streamline density for --vec_style stream")
    sp2d.add_argument("--info", default="default",
                      help="model annotation: a preset (none/min/default/physics/full) "
                           "or a comma-separated list of keys")
    sp2d.add_argument("--info_pos", default="box",
                      choices=["box", "subtitle", "footer"],
                      help="where the annotation goes")
    sp2d.add_argument("--info_on", default="polar", choices=["polar", "all"],
                      help="annotate the polar animation only, or every animation")
    sp2d.add_argument("--params", default=None,
                      help="path to the athinput to read the model from")

    sp1d = sub.add_parser("1d", help="azimuthally averaged radial profiles")
    common(sp1d)
    frame_opts(sp1d)
    sp1d.add_argument("--mode", default="all", choices=["all", "profiles", "animation"])
    sp1d.add_argument("--frame", type=int, default=None)
    sp1d.add_argument("--fps", type=int, default=10)
    sp1d.add_argument("--subsample", type=int, default=1)
    sp1d.add_argument("--r_min", type=float, default=None)
    sp1d.add_argument("--r_max", type=float, default=None)
    sp1d.add_argument("--phi_min", type=float, default=None)
    sp1d.add_argument("--phi_max", type=float, default=None)
    sp1d.add_argument("--normalize", default="none", choices=["none", "azimuthal"],
                      help="profile the deviation from the azimuthal mean")
    sp1d.add_argument("--logscale", action="store_true",
                      help="log y-axis for density and pressure (default: linear)")
    sp1d.add_argument("--vmin_percentile", type=float, default=2.0)
    sp1d.add_argument("--vmax_percentile", type=float, default=98.0)

    sphst = sub.add_parser("hst", help="history (.hst) time series")
    common(sphst, dpi=150)
    sphst.add_argument("--mode", default="default", choices=["default", "all", "custom"],
                       help="the two headline columns, every column, or --vars")
    sphst.add_argument("--vars", nargs="+", default=None,
                       help="columns to plot, for --mode custom")
    sphst.add_argument("--linear_scale", nargs="+", default=[], metavar="VAR",
                       help="columns to draw on a linear y-axis instead of log")
    sphst.add_argument("--linear", action="store_true",
                       help="linear y-axis for every column")
    sphst.add_argument("--figsize", nargs=2, type=float, default=(10.0, 6.0),
                       metavar=("W", "H"), help="per-figure size in inches")
    sphst.add_argument("--grid", action="store_true",
                       help="draw a grid (off by default, as in vishst.py)")
    sphst.add_argument("--style", default="seaborn-v0_8-darkgrid",
                       help="matplotlib style for these figures ('none' to skip)")

    spf = sub.add_parser("forces", help="radial force balance from the uov output")
    common(spf, output_id=2)
    frame_opts(spf)
    spf.add_argument("--mode", default="all",
                     choices=["frame", "animation", "sum", "sum_animation", "all"])
    spf.add_argument("--frame", type=int, default=None)
    spf.add_argument("--fps", type=int, default=10)
    spf.add_argument("--subsample", type=int, default=1)
    spf.add_argument("--r_min", type=float, default=None)
    spf.add_argument("--r_max", type=float, default=None)
    spf.add_argument("--log", action="store_true",
                     help="symlog y-axis, for the eight decades between disk and ambient")
    spf.add_argument("--linthresh", type=float, default=None,
                     help="linear region of the symlog axis (default 1e2)")

    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    {"2d": cmd_2d, "1d": cmd_1d, "hst": cmd_hst, "forces": cmd_forces}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
