#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ VISUALIZATION - standalone, single-file, full feature set
=============================================================================

A self-contained alternative to vis1d.py / vis2d.py / vishst.py / visforces.py /
athena_data.py: this ONE file imports none of them. Copy it anywhere and it runs
with numpy + matplotlib on the path; pypalettes is used if present, with a
measured fallback palette if not.

Scope, so what's here vs. the four-script setup is explicit:
    - .tab only, no .athdf. Every run in this project produces .tab; the athdf
      reader lives in athena_read.py (upstream Athena++, not duplicated here).
    - single AND multi meshblock supported (blocks are joined onto one grid,
      same algorithm as athena_data.py's _read_tab_blocks).
    - every plot mode of vis2d.py: heatmaps, radial/azimuthal profiles, polar
      view, the three animations (cartesian, polar, vector-only), the full
      vector overlay (component/frame/quiver+stream/scale/clip), the model-
      parameter annotation, --normalize azimuthal, --cmap_palette.
    - vis1d.py's evolution animation (the radial profile *line* animated over
      time - distinct from vis2d's heatmap animations).
    - vishst.py's three modes (default/all/custom) over the .hst file.
    - visforces.py's four modes over the uov (output2) .tab files.
    - NOT reproduced: --vec_lattice square/hex (polar-ring lattice only, which
      is centred on the origin by construction and needs no square/hex fix at
      all - see athena_data.py's history for why the other two needed one),
      model-info's derived quantities (H/r, orbits-to-accrete - those need
      scripts/theory/disk_model.py, which this file does not import; raw
      athinput values are shown instead), parallel frame sampling for animation
      colour ranges (sequential here - slower on very long runs, far less
      code), and GPU array handling (the four scripts' own to_gpu hook was
      already a no-op).

USAGE:
    python3 vis_standalone.py <subcommand> [options]

SUBCOMMANDS (one per area of the four-script setup):
    2d       heatmaps / radial / azimuthal / polar / the three animations /
             vectors - everything vis2d.py does
    1d       radial profile + its evolution animation - vis1d.py
    hst      .hst time series - vishst.py
    forces   uov force-balance plots - visforces.py

Each subcommand's flags mirror the corresponding script's own as closely as the
single-file, no-multiprocessing scope allows; --help on any subcommand lists them.

EXAMPLES:
    python3 vis_standalone.py 2d --mode polar --add-vectors --vec_frame full
    python3 vis_standalone.py 2d --mode vector_animation --vec_comp radial
    python3 vis_standalone.py 2d --mode polar --info physics --info_pos box
    python3 vis_standalone.py 1d --mode animation
    python3 vis_standalone.py hst --mode all
    python3 vis_standalone.py forces --mode animation --log

=============================================================================
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
from matplotlib.colors import LogNorm, Normalize

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
            "M_bh", "T_0", "mu", "chi", "grid"]
INFO_PRESETS = {
    "none": [],
    "default": ["alpha", "gamma", "grid", "C_prime", "r_center"],
    "min": ["alpha", "nu_iso", "gamma", "grid"],
    "physics": ["alpha", "nu_iso", "gamma", "grid", "C_prime", "r_center", "rho_atm"],
    "full": ["alpha", "nu_iso", "gamma", "grid", "C_prime", "r_center", "rho_atm",
            "M_bh", "T_0", "mu", "chi"],
}
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
        hits = sorted(h for h in glob.glob(os.path.join(here, "athinput.*"))
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


def polar_ring_lattice(r, phi, n_rings):
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
        idx = np.clip(np.searchsorted(phi, angles), 0, len(phi) - 1)
        js.append(idx); iss.append(np.full(n_arrow, i))
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
        j, i, Rs, Phis = polar_ring_lattice(r, phi, a.vec_arrows)
        ur, up = u_r[j, i], u_phi[j, i]
        U = ur * np.cos(Phis) - up * np.sin(Phis)
        V = ur * np.sin(Phis) + up * np.cos(Phis)
        return (Phis, Rs, U, V, None), None
    stride = max(1, len(r) // a.vec_arrows)
    sl = (slice(None, None, stride), slice(None, None, stride))
    Rs, Phis = R[sl], Phi[sl]
    U, V = u_r[sl], u_phi[sl] / np.maximum(Rs, 1e-30)
    return (Rs, Phis, U, V, stride), None


def update_vectors(ax, data, a, polar, artists=None, scale_data=None):
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
    return cap, {"q": q}


def overlay_vectors(ax, data, a, polar=True):
    """One-shot overlay for the static (single-frame) plots."""
    cap, _ = update_vectors(ax, data, a, polar=polar)
    return cap


def vector_caption(a, cap):
    sym = r"\rho\vec{v}" if a.vec_field == "momentum" else r"\vec{v}"
    base = f"${sym}$" if a.vec_frame == "full" else f"${sym}-\\langle{sym}\\rangle_\\phi$"
    comp = "" if a.vec_comp == "both" else f" ({a.vec_comp} only)"
    tail = f",  longest arrow = {cap:.3g}" if cap else ""
    scale_note = f" ({a.vec_scale} length scale)" if a.vec_scale != "linear" and cap else ""
    return f"arrows: {base}{comp}  ({a.vec_style}){tail}{scale_note}"


# =============================================================================
# 2D PLOTS (mirrors vis2d.py)
# =============================================================================
VARS_2D = {
    "density":  dict(cmap="inferno", log=True,  label=r"$\rho$ (Density)"),
    "pressure": dict(cmap="plasma",  log=True,  label=r"$P$ (Pressure)"),
    "vel_r":    dict(cmap="RdBu_r",  log=False, label=r"$v_r$ (Radial Velocity)"),
    "vel_phi":  dict(cmap="coolwarm", log=False, label=r"$v_\phi$ (Azimuthal Velocity)"),
}


def _norm(field, info):
    if info["log"] and np.all(field > 0):
        return LogNorm(vmin=field.min(), vmax=field.max())
    if info["log"]:
        pos = field[field > 0]
        return Normalize(vmin=pos.min() if pos.size else field.min(), vmax=field.max())
    vmax = np.abs(field).max()
    return Normalize(vmin=-vmax, vmax=vmax)


def apply_bounds(data, a):
    r, phi = data["r"], data["phi"]
    rmask = np.ones_like(r, dtype=bool)
    if a.r_min is not None:
        rmask &= r >= a.r_min
    if a.r_max is not None:
        rmask &= r <= a.r_max
    pmask = np.ones_like(phi, dtype=bool)
    if a.phi_min is not None:
        pmask &= phi >= a.phi_min
    if a.phi_max is not None:
        pmask &= phi <= a.phi_max
    if rmask.all() and pmask.all():
        return data
    out = dict(data)
    out["r"], out["phi"] = r[rmask], phi[pmask]
    for k in ("R", "Phi", "density", "pressure", "vel_r", "vel_phi", "vel_z"):
        if k in data:
            out[k] = data[k][pmask, :][:, rmask]
    return out


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


def var_info_for(mode):
    if mode != "azimuthal":
        return VARS_2D
    out = {}
    for k, v in VARS_2D.items():
        rel = k in ("density", "pressure")
        out[k] = dict(v, log=False, cmap="RdBu_r",
                      label=v["label"].split(" (")[0]
                      + (r"$/\langle\cdot\rangle_\phi - 1$" if rel
                         else r"$ - \langle\cdot\rangle_\phi$"))
    return out


def plot_heatmap(data, a, out_path):
    data = apply_normalize(apply_bounds(data, a), a.normalize)
    info_map = var_info_for(a.normalize)
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle(f"2D Disk Structure (t={data['time']:.3f}, cycle={data['cycle']})",
                fontsize=16, fontweight="bold")
    for ax, (name, info) in zip(axes.flat, info_map.items()):
        field = data[name]
        im = ax.pcolormesh(data["R"], data["Phi"], field, cmap=info["cmap"],
                           norm=_norm(field, info), shading="auto")
        ax.set_xlabel("r"); ax.set_ylabel(r"$\phi$ (rad)"); ax.set_title(info["label"])
        plt.colorbar(im, ax=ax); ax.grid(alpha=0.3)
    ax6 = axes.flat[5]
    if a.add_vectors:
        rho = data["density"]
        ax6.pcolormesh(data["R"], data["Phi"], rho, cmap="Greys", shading="auto",
                       norm=Normalize(rho.min(), rho.max()))
        cap = overlay_vectors(ax6, data, a, polar=False)
        ax6.set_xlabel("r"); ax6.set_ylabel(r"$\phi$"); ax6.grid(alpha=0.3)
        ax6.set_title(vector_caption(a, cap), fontsize=10)
    else:
        ax6.remove()
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_polar(data, a, out_path):
    data = apply_normalize(apply_bounds(data, a), a.normalize)
    info_map = var_info_for(a.normalize)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Polar View of Disk - t={data['time']:.3f}",
                fontsize=16, fontweight="bold")
    last_cap = None
    keys = list(info_map)
    for idx, name in enumerate(keys):
        info = info_map[name]
        ax = plt.subplot(2, 2, idx + 1, projection="polar")
        field = data[name]
        im = ax.pcolormesh(data["Phi"], data["R"], field, cmap=info["cmap"],
                           norm=_norm(field, info), shading="auto")
        ax.set_title(info["label"], pad=20); plt.colorbar(im, ax=ax, pad=0.1)
        ax.grid(alpha=0.3)
        if a.add_vectors and (a.vec_panels == "all" or idx == len(keys) - 1):
            last_cap = overlay_vectors(ax, data, a, polar=True)
    if a.add_vectors:
        fig.text(0.5, 0.012, vector_caption(a, last_cap), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_radial(data, a, out_path):
    data = apply_bounds(data, a)
    colors = line_colors(4, a.palette)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Radial Profiles (averaged over phi) - t={data['time']:.3f}",
                fontsize=14, fontweight="bold")
    labels = {"density": r"$\langle\rho\rangle_\phi$", "pressure": r"$\langle P\rangle_\phi$",
             "vel_r": r"$\langle v_r\rangle_\phi$", "vel_phi": r"$\langle v_\phi\rangle_\phi$"}
    for ax, (name, info), color in zip(axes.flat, VARS_2D.items(), colors):
        mean, std = data[name].mean(axis=0), data[name].std(axis=0)
        ax.plot(data["r"], mean, color=color, linewidth=2, label=labels[name])
        ax.fill_between(data["r"], mean - std, mean + std, alpha=0.25, color=color)
        ax.set_xlabel("r"); ax.set_ylabel(labels[name]); ax.legend(); ax.grid(alpha=0.3)
        if info["log"]:
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
    fig.suptitle(f"Azimuthal Profiles at Different Radii - t={data['time']:.3f}",
                fontsize=14, fontweight="bold")
    labels = {"density": r"$\rho$", "pressure": "$P$", "vel_r": "$v_r$", "vel_phi": r"$v_\phi$"}
    for ax, (name, info) in zip(axes.flat, VARS_2D.items()):
        for ridx, color in zip(idxs, colors):
            ax.plot(phi, data[name][:, ridx], color=color, linewidth=2, alpha=0.8,
                   label=f"r={r[ridx]:.2f}")
        ax.set_xlabel(r"$\phi$ (rad)"); ax.set_ylabel(labels[name])
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        if info["log"]:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_info_box(fig, lines):
    if not lines:
        return
    stacked = "\n".join(item for line in lines for item in line.split("   ") if item)
    fig.text(0.012, 0.988, stacked, ha="left", va="top", fontsize=8.5,
             color="#1a1a19", linespacing=1.5,
             bbox=dict(boxstyle="round", fc="white", ec="#b8b8b4", alpha=0.9))


def animate_heatmap(frames, keys, a, out_path, info_lines):
    keys = keys[::max(1, a.subsample)]
    first = apply_normalize(apply_bounds(read_frame(frames[keys[0]]), a), a.normalize)
    info_map = var_info_for(a.normalize)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Disk Evolution", fontsize=16, fontweight="bold")
    ims = []
    for ax, (name, info) in zip(axes.flat, info_map.items()):
        field = first[name]
        im = ax.pcolormesh(first["R"], first["Phi"], field, cmap=info["cmap"],
                           norm=_norm(field, info), shading="auto")
        ax.set_xlabel("r"); ax.set_ylabel(r"$\phi$"); ax.set_title(info["label"])
        plt.colorbar(im, ax=ax); ax.grid(alpha=0.3)
        ims.append(im)
    _draw_info_box(fig, info_lines)
    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=12, fontweight="bold")

    def update(k):
        d = apply_normalize(apply_bounds(read_frame(frames[keys[k]]), a), a.normalize)
        time_text.set_text(f"Frame {k:3d}/{len(keys)-1} | Time = {d['time']:5.2f}")
        for im, name in zip(ims, info_map):
            im.set_array(d[name].ravel())
        return ims + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=130)
    plt.close(fig)


def animate_polar(frames, keys, a, out_path, info_lines):
    keys = keys[::max(1, a.subsample)]
    first = apply_normalize(apply_bounds(read_frame(frames[keys[0]]), a), a.normalize)
    last = apply_normalize(apply_bounds(read_frame(frames[keys[-1]]), a), a.normalize)
    info_map = var_info_for(a.normalize)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Polar View of Disk Evolution", fontsize=16, fontweight="bold")
    axes, ims = [], []
    for i, (name, info) in enumerate(info_map.items()):
        ax = plt.subplot(2, 2, i + 1, projection="polar")
        field = first[name]
        im = ax.pcolormesh(first["Phi"], first["R"], field, cmap=info["cmap"],
                           norm=_norm(field, info), shading="auto")
        ax.set_title(info["label"], pad=20); plt.colorbar(im, ax=ax, pad=0.1)
        ax.grid(alpha=0.3)
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
    _draw_info_box(fig, info_lines)
    time_text = fig.text(0.5, 0.94, "", ha="center", fontsize=12, fontweight="bold")

    def update(k):
        d = apply_normalize(apply_bounds(read_frame(frames[keys[k]]), a), a.normalize)
        time_text.set_text(f"Frame {k:3d}/{len(keys)-1} | Time = {d['time']:5.2f}")
        for im, name in zip(ims, info_map):
            im.set_array(d[name].ravel())
        if a.add_vectors:
            for ax in vec_axes:
                _, vec_state[id(ax)] = update_vectors(
                    ax, d, a, polar=True, artists=vec_state[id(ax)], scale_data=last)
        return ims + [time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=130)
    plt.close(fig)


def animate_vector(frames, keys, a, out_path, info_lines):
    keys = keys[::max(1, a.subsample)]
    first = read_frame(frames[keys[0]])
    last = read_frame(frames[keys[-1]])   # scale reference: frame 0 has v_r = 0
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")
    fig.suptitle("Velocity Field", fontsize=16, fontweight="bold")
    rho = first["density"]
    lo = rho[rho > 0].min() if np.any(rho > 0) else None
    bg = ax.pcolormesh(first["Phi"], first["R"], rho, cmap="Greys", shading="auto",
                       norm=(LogNorm(lo, rho.max()) if lo else Normalize(rho.min(), rho.max())))
    plt.colorbar(bg, ax=ax, pad=0.1, fraction=0.04).set_label(r"$\rho$")
    cap, vec_art = update_vectors(ax, first, a, polar=True, scale_data=last)
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.035, vector_caption(a, cap), ha="center", fontsize=12)
    _draw_info_box(fig, info_lines)
    time_text = fig.text(0.5, 0.93, "", ha="center", fontsize=12, fontweight="bold")

    def update(k):
        nonlocal vec_art
        d = read_frame(frames[keys[k]])
        time_text.set_text(f"Frame {k:3d}/{len(keys)-1} | Time = {d['time']:5.2f}")
        bg.set_array(d["density"].ravel())
        _, vec_art = update_vectors(ax, d, a, polar=True, artists=vec_art, scale_data=last)
        return [bg, time_text]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=130)
    plt.close(fig)


def cmd_2d(a):
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_standalone_2d"
    os.makedirs(out_dir, exist_ok=True)
    info = discover(data_dir, output_id=a.output_id)
    keys = sorted(info["frames"])
    print(f"{len(keys)} frame(s), {keys[0]}..{keys[-1]}, base '{info['base']}'")

    info_lines = []
    if a.info != "none":
        ath = find_athinput(data_dir, a.params)
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
    keys = keys[::max(1, a.subsample)]
    colors = line_colors(4, a.palette)
    labels = {"density": r"Density $\rho$", "pressure": r"Pressure $P$",
             "vel_r": r"Radial Velocity $v_r$", "vel_phi": r"Azimuthal Velocity $v_\phi$"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Radial Profile Evolution", fontsize=15, fontweight="bold")
    first = read_frame(frames[keys[0]])
    lines = {}
    for ax, (name, color) in zip(axes.flat, zip(VARS_2D, colors)):
        mean = first[name].mean(axis=0)
        (line,) = ax.plot(first["r"], mean, color=color, linewidth=2)
        ax.set_xlabel("r"); ax.set_ylabel(labels[name]); ax.grid(alpha=0.3)
        if VARS_2D[name]["log"]:
            ax.set_yscale("log")
        lines[name] = line
    time_text = fig.text(0.5, 0.965, "", ha="center", fontsize=11, fontweight="bold")

    def update(k):
        d = read_frame(frames[keys[k]])
        time_text.set_text(f"t = {d['time']:.3f}  (frame {k}/{len(keys)-1})")
        for name, line in lines.items():
            line.set_ydata(d[name].mean(axis=0))
        return list(lines.values()) + [time_text]

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=130)
    plt.close(fig)


def cmd_1d(a):
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_standalone_1d"
    os.makedirs(out_dir, exist_ok=True)
    info = discover(data_dir, output_id=a.output_id)
    keys = sorted(info["frames"])
    print(f"{len(keys)} frame(s), {keys[0]}..{keys[-1]}")

    if a.mode in ("all", "profiles"):
        frame = a.frame if a.frame is not None else keys[-1]
        data = read_frame(info["frames"][frame])
        out = os.path.join(out_dir, f"radial_profile_{frame:05d}.png")
        plot_radial(data, a, out)
        print(f"Saved: {out}")
    if a.mode in ("all", "animation"):
        out = os.path.join(out_dir, "radial_profile_evolution.mp4")
        animate_1d_evolution(info["frames"], keys, a, out)
        print(f"Saved: {out}")


# =============================================================================
# HST (mirrors vishst.py)
# =============================================================================
DEFAULT_HST_VARS = ["disk_mass", "mdot_in"]


def cmd_hst(a):
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_standalone_hst"
    os.makedirs(out_dir, exist_ok=True)
    path, series = read_hst(data_dir)
    time = series["time"]
    print(f"Reading: {path}")
    print(f"Available: {', '.join(series)}")

    if a.mode == "default":
        var_names = DEFAULT_HST_VARS
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

    colors = line_colors(len(var_names), a.palette)

    if a.mode == "default" and "disk_mass" in series and "mdot_in" in series:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        c1, c2 = line_colors(2, a.palette)
        ax1.plot(time, series["disk_mass"], color=c1, linewidth=2)
        ax1.set_ylabel("Disk Mass", color=c1); ax1.tick_params(axis="y", labelcolor=c1)
        ax2 = ax1.twinx()
        ax2.plot(time, series["mdot_in"], color=c2, linewidth=2)
        ax2.set_ylabel("Accretion Rate", color=c2); ax2.tick_params(axis="y", labelcolor=c2)
        ax1.set_xlabel("time"); ax1.grid(alpha=0.3)
        out = os.path.join(out_dir, "disk_mass_and_mdot.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved: {out}")
        var_names = [v for v in var_names if v not in ("disk_mass", "mdot_in")]
        colors = line_colors(max(len(var_names), 1), a.palette)

    for name, color in zip(var_names, colors):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(time, series[name], color=color, linewidth=1.8)
        ax.set_xlabel("time"); ax.set_ylabel(name); ax.grid(alpha=0.3)
        if not a.linear:
            ax.set_yscale("log")
        out = os.path.join(out_dir, f"{name}_vs_time.png")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved: {out}")


# =============================================================================
# FORCES (mirrors visforces.py) - uov output (default output_id=2)
# =============================================================================
FORCE_KEYS = ["f_grav", "f_centr", "f_press", "f_sum"]
FORCE_LABELS = {"f_grav": r"$f_{\rm grav}=-\beta/r^2$",
               "f_centr": r"$f_{\rm centr}=v_\phi^2/r$",
               "f_press": r"$f_{\rm press}=-(1/\rho)\partial P/\partial r$",
               "f_sum": r"$f_{\rm sum}$"}


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


def plot_forces_frame(data, a, out_path):
    r = data["r"]
    colors = line_colors(4, a.palette)
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color in zip(FORCE_KEYS, colors):
        if key not in data:
            continue
        field = data[key]
        mean = field.mean(axis=0) if field.ndim == 2 else field
        ax.plot(r, mean, color=color, linewidth=2, label=FORCE_LABELS[key])
    if a.log:
        ax.set_yscale("symlog", linthresh=a.linthresh or 1e2)
    ax.set_xlabel("r"); ax.set_ylabel("force / mass"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Force balance, t={data['time']:.3f}")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forces_sum(data, a, out_path):
    r = data["r"]
    color = line_colors(1, a.palette)[0]
    field = data["f_sum"]
    mean = field.mean(axis=0) if field.ndim == 2 else field
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, mean, color=color, linewidth=2)
    if a.log:
        ax.set_yscale("symlog", linthresh=a.linthresh or 1e2)
    ax.set_xlabel("r"); ax.set_ylabel(FORCE_LABELS["f_sum"]); ax.grid(alpha=0.3)
    ax.set_title(f"Net force, t={data['time']:.3f}")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def animate_forces(frames, keys, a, out_path, sum_only=False):
    keys = keys[::max(1, a.subsample)]
    first = read_uov_frame(frames[keys[0]])
    colors = line_colors(1 if sum_only else 4, a.palette)
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = {}
    force_keys = ["f_sum"] if sum_only else FORCE_KEYS
    for key, color in zip(force_keys, colors):
        field = first[key]
        mean = field.mean(axis=0) if field.ndim == 2 else field
        (line,) = ax.plot(first["r"], mean, color=color, linewidth=2,
                          label=FORCE_LABELS[key])
        lines[key] = line
    if a.log:
        ax.set_yscale("symlog", linthresh=a.linthresh or 1e2)
    ax.set_xlabel("r"); ax.set_ylabel("force / mass"); ax.legend(); ax.grid(alpha=0.3)
    title = fig.suptitle("", fontsize=12, fontweight="bold")

    def update(k):
        d = read_uov_frame(frames[keys[k]])
        title.set_text(f"t = {d['time']:.3f}")
        for key, line in lines.items():
            field = d[key]
            line.set_ydata(field.mean(axis=0) if field.ndim == 2 else field)
        return list(lines.values()) + [title]

    ani = animation.FuncAnimation(fig, update, frames=len(keys),
                                  interval=1000 / a.fps, blit=False)
    ani.save(out_path, fps=a.fps, dpi=130)
    plt.close(fig)


def cmd_forces(a):
    data_dir = a.data_dir or default_data_dir()
    out_dir = a.output_dir or "figs_standalone_forces"
    os.makedirs(out_dir, exist_ok=True)
    info = discover(data_dir, output_id=a.output_id)
    keys = sorted(info["frames"])
    print(f"{len(keys)} uov frame(s), {keys[0]}..{keys[-1]}")

    if a.mode in ("frame", "all"):
        frame = a.frame if a.frame is not None else keys[-1]
        data = read_uov_frame(info["frames"][frame])
        out = os.path.join(out_dir, f"forces_frame_{frame:05d}.png")
        plot_forces_frame(data, a, out); print(f"Saved: {out}")
    if a.mode in ("sum", "all"):
        frame = a.frame if a.frame is not None else keys[-1]
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

    def common(sp, output_id=1):
        sp.add_argument("--data_dir", default=None)
        sp.add_argument("--output_dir", default=None)
        sp.add_argument("--output_id", type=int, default=output_id)
        sp.add_argument("--dpi", type=int, default=150)
        sp.add_argument("--palette", default=None,
                        help="pypalettes name for line colours")

    sp2d = sub.add_parser("2d", help="vis2d.py's full feature set")
    common(sp2d)
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
    sp2d.add_argument("--normalize", default="none", choices=["none", "azimuthal"])
    sp2d.add_argument("--add-vectors", dest="add_vectors", action="store_true")
    sp2d.add_argument("--vec_field", default="velocity", choices=["velocity", "momentum"])
    sp2d.add_argument("--vec_comp", default="both", choices=["both", "radial", "azimuthal"])
    sp2d.add_argument("--vec_frame", default="perturbation",
                      choices=["perturbation", "full"])
    sp2d.add_argument("--vec_panels", default="last", choices=["last", "all"])
    sp2d.add_argument("--vec_style", default="quiver", choices=["quiver", "stream"])
    sp2d.add_argument("--vec_scale", default="log", choices=["linear", "sqrt", "log"])
    sp2d.add_argument("--vec_arrows", type=int, default=8)
    sp2d.add_argument("--vec_clip", type=float, default=92.0)
    sp2d.add_argument("--vec_color", default="black")
    sp2d.add_argument("--vec_density", type=float, default=0.6,
                      help="streamline density for --vec_style stream")
    sp2d.add_argument("--info", default="default")
    sp2d.add_argument("--params", default=None)

    sp1d = sub.add_parser("1d", help="vis1d.py")
    common(sp1d)
    sp1d.add_argument("--mode", default="all", choices=["all", "profiles", "animation"])
    sp1d.add_argument("--frame", type=int, default=None)
    sp1d.add_argument("--fps", type=int, default=10)
    sp1d.add_argument("--subsample", type=int, default=1)
    sp1d.add_argument("--r_min", type=float, default=None)
    sp1d.add_argument("--r_max", type=float, default=None)
    sp1d.add_argument("--phi_min", type=float, default=None)
    sp1d.add_argument("--phi_max", type=float, default=None)

    sphst = sub.add_parser("hst", help="vishst.py")
    common(sphst)
    sphst.add_argument("--mode", default="default", choices=["default", "all", "custom"])
    sphst.add_argument("--vars", nargs="+", default=None)
    sphst.add_argument("--linear", action="store_true",
                       help="linear y-scale (default: log)")

    spf = sub.add_parser("forces", help="visforces.py")
    common(spf, output_id=2)
    spf.add_argument("--mode", default="all",
                     choices=["frame", "animation", "sum", "sum_animation", "all"])
    spf.add_argument("--frame", type=int, default=None)
    spf.add_argument("--fps", type=int, default=10)
    spf.add_argument("--subsample", type=int, default=1)
    spf.add_argument("--log", action="store_true")
    spf.add_argument("--linthresh", type=float, default=None)

    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    {"2d": cmd_2d, "1d": cmd_1d, "hst": cmd_hst, "forces": cmd_forces}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
