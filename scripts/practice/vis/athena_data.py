"""Format-agnostic data layer for the Athena++ visualisation scripts.

The plotting scripts used to parse `.tab` files by column index. That tied them to
one output format and to a fixed column layout. This module puts a single reader in
front of them, so the same command works on `.athdf` and on `.tab`, and the calling
code no longer knows or cares which it got.

Reading itself is delegated to `vis/python/athena_read.py`, which ships with
Athena++ and already handles the awkward parts: mesh refinement, multiple
MeshBlocks, byte-string attributes, both file formats. There is no reason to
re-implement any of that here.

What this module adds on top is the bookkeeping the plotting scripts need:

  * discovering which frames exist in a directory, for a given output id;
  * choosing a format when both are present (`.athdf` wins - one file per frame
    instead of one per MeshBlock);
  * assembling multi-block `.tab` output into a single grid;
  * returning the exact dictionaries the existing plotting code expects, so the
    figures are unchanged.

Output ids follow the input file: out1 is normally `variable = prim`, out2 is
`variable = uov` (the user output variables). Pass `output_id` accordingly.
"""

import os
import re
import sys

import numpy as np

# athena_read.py lives in vis/python/ at the repository root
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.join(_HERE, "..", "..", "..", "vis", "python"),   # scripts/practice/vis -> repo
    os.path.join(_HERE, "..", "..", "vis", "python"),
    os.path.join(_HERE, "vis", "python"),
):
    _cand = os.path.abspath(_cand)
    if os.path.isfile(os.path.join(_cand, "athena_read.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
else:                                                          # pragma: no cover
    raise ImportError(
        "athena_read.py not found. It ships with Athena++ in vis/python/; this "
        "module expects to live in scripts/practice/vis/ inside the repository."
    )

import athena_read  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dec(x):
    """HDF5 attributes come back as bytes; make them ordinary strings."""
    if isinstance(x, bytes):
        return x.decode()
    return x


def _canon(name):
    """Map Athena++ variable names onto the names the plotting code uses."""
    return {
        "rho": "density",
        "press": "pressure",
        "vel1": "vel_r",
        "vel2": "vel_phi",
        "vel3": "vel_z",
    }.get(name, name)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover(data_dir, output_id=1, prefer=None):
    """Find the frames of one output id in a directory.

    Returns a dict:
        format   : 'athdf' or 'tab'
        base     : problem_id taken from the file names
        frames   : {frame number -> [paths]}  (one path for athdf, one per block for tab)
        nblocks  : number of MeshBlocks seen (1 for athdf, which is already joined)

    `prefer` forces a format instead of picking automatically.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"no such directory: {data_dir}")

    names = os.listdir(data_dir)
    oid = int(output_id)

    athdf_re = re.compile(r"^(?P<base>.+)\.out%d\.(?P<frame>\d+)\.athdf$" % oid)
    tab_re = re.compile(
        r"^(?P<base>.+?)(?:\.block(?P<block>\d+))?\.out%d\.(?P<frame>\d+)\.tab$" % oid)

    def _collect(regex, ext):
        frames, bases, blocks = {}, set(), set()
        for n in names:
            if not n.endswith(ext):
                continue
            m = regex.match(n)
            if not m:
                continue
            g = m.groupdict()
            frame = int(g["frame"])
            block = int(g["block"]) if g.get("block") is not None else 0
            bases.add(g["base"])
            blocks.add(block)
            frames.setdefault(frame, []).append((block, os.path.join(data_dir, n)))
        for f in frames:
            frames[f] = [p for _, p in sorted(frames[f])]
        return frames, bases, blocks

    a_frames, a_bases, _ = _collect(athdf_re, ".athdf")
    t_frames, t_bases, t_blocks = _collect(tab_re, ".tab")

    if prefer == "athdf" or (prefer is None and a_frames):
        if not a_frames:
            raise FileNotFoundError(
                f"no .athdf files for out{oid} in {data_dir}")
        return {"format": "athdf", "base": sorted(a_bases)[0],
                "frames": a_frames, "nblocks": 1}

    if not t_frames:
        raise FileNotFoundError(
            f"no .athdf or .tab files for out{oid} in {data_dir}.\n"
            f"        Enable HDF5 output in the input file:\n"
            f"            <output{oid}>\n"
            f"            file_type = hdf5\n"
            f"        and configure Athena++ with -hdf5.")
    return {"format": "tab", "base": sorted(t_bases)[0],
            "frames": t_frames, "nblocks": len(t_blocks)}


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _read_athdf(path):
    d = athena_read.athdf(path)
    varnames = [_dec(v) for v in d["VariableNames"]]
    out = {
        "time": float(d["Time"]),
        "cycle": int(d.get("NumCycles", 0)),
        "coordinates": _dec(d["Coordinates"]),
        "x1v": np.asarray(d["x1v"]),
        "x2v": np.asarray(d["x2v"]),
    }
    for v in varnames:
        # athdf arrays are (nx3, nx2, nx1); these runs are 2D at most
        out[_canon(v)] = np.asarray(d[v])[0]
    return out


def _read_tab_blocks(paths):
    """Read one frame from one or more .tab blocks and join them.

    Note on the layout athena_read.tab() returns: for a 2D dump `x1v` and `x2v`
    come back as full (nx2, nx1) arrays, not as the two 1D axes. Flattening them
    would give nx1*nx2 "coordinates" and silently corrupt the join, so the axes
    are taken as x1v[0, :] and x2v[:, 0].
    """
    blocks = []
    for p in paths:
        d = athena_read.tab(p)
        x1 = np.asarray(d["x1v"])
        x2 = np.asarray(d["x2v"]) if "x2v" in d else None
        if x1.ndim == 2:                       # 2D block
            ax1, ax2 = x1[0, :], x2[:, 0]
        else:                                  # 1D block
            ax1, ax2 = x1.ravel(), None
        rec = {"x1v": ax1, "x2v": ax2,
               "time": float(d["time"]), "cycle": int(d["cycle"])}
        for k, v in d.items():
            if k in ("time", "cycle", "variables", "x1v", "x2v", "x3v"):
                continue
            rec[_canon(k)] = np.asarray(v)
        blocks.append(rec)

    joined = {"time": blocks[0]["time"], "cycle": blocks[0]["cycle"],
              "coordinates": None}
    varkeys = [k for k in blocks[0]
               if k not in ("time", "cycle", "x1v", "x2v", "coordinates")]

    if blocks[0]["x2v"] is None:                      # 1D
        order = np.argsort(np.concatenate([b["x1v"] for b in blocks]))
        joined["x1v"] = np.concatenate([b["x1v"] for b in blocks])[order]
        joined["x2v"] = None
        for k in varkeys:
            joined[k] = np.concatenate(
                [np.asarray(b[k]).ravel() for b in blocks])[order]
        return joined

    # 2D: place each block into the global (nx2, nx1) grid. searchsorted + ix_
    # keeps this vectorised; the obvious double loop is far too slow at
    # production resolution.
    x1 = np.unique(np.concatenate([b["x1v"] for b in blocks]))
    x2 = np.unique(np.concatenate([b["x2v"] for b in blocks]))
    joined["x1v"], joined["x2v"] = x1, x2
    for k in varkeys:
        grid = np.zeros((len(x2), len(x1)))
        for b in blocks:
            arr = np.asarray(b[k])
            if arr.ndim == 1:
                arr = arr.reshape(len(b["x2v"]), len(b["x1v"]))
            rows = np.searchsorted(x2, b["x2v"])
            cols = np.searchsorted(x1, b["x1v"])
            grid[np.ix_(rows, cols)] = arr
        joined[k] = grid
    return joined


def read_frame(paths, fmt):
    """Read one frame. `paths` is what discover() put in frames[n]."""
    if fmt == "athdf":
        if len(paths) != 1:
            raise ValueError("expected exactly one .athdf file per frame")
        return _read_athdf(paths[0])
    return _read_tab_blocks(paths)


# ---------------------------------------------------------------------------
# shapes the plotting scripts expect
# ---------------------------------------------------------------------------

def frame_1d(paths, fmt):
    """1D radial profile: r, density, pressure, vel_r, vel_phi, vel_z."""
    d = read_frame(paths, fmt)
    out = {"time": d["time"], "cycle": d["cycle"], "r": np.asarray(d["x1v"]).ravel()}
    for k, v in d.items():
        if k in ("time", "cycle", "x1v", "x2v", "coordinates"):
            continue
        arr = np.asarray(v)
        # a 1D run may still come back with a length-1 second axis
        out[k] = arr.ravel() if arr.ndim == 1 else arr.mean(axis=0)
    return out


def frame_2d(paths, fmt, to_gpu=None):
    """2D (r, phi) frame in the layout the plotting code expects.

    Variable arrays are (nphi, nr), matching the meshgrid built from R and Phi.
    `to_gpu` is applied to the 2D arrays only, never to r/phi, which matplotlib
    needs on the host.
    """
    d = read_frame(paths, fmt)
    r = np.asarray(d["x1v"]).ravel()
    phi = np.asarray(d["x2v"]).ravel()
    R, Phi = np.meshgrid(r, phi)

    keep = to_gpu if to_gpu is not None else (lambda a: a)
    out = {"time": d["time"], "cycle": d["cycle"],
           "r": r, "phi": phi, "nr": len(r), "nphi": len(phi),
           "R": keep(R), "Phi": keep(Phi)}
    for k, v in d.items():
        if k in ("time", "cycle", "x1v", "x2v", "coordinates"):
            continue
        arr = np.asarray(v)
        if arr.shape != (len(phi), len(r)):
            arr = arr.reshape(len(phi), len(r))
        out[k] = keep(arr)
    return out


def describe(info):
    """One line for the scripts to print, so it is obvious what was read."""
    n = len(info["frames"])
    extra = "" if info["format"] == "athdf" else f", {info['nblocks']} block(s)"
    return f"{info['format']}: {n} frame(s), problem_id '{info['base']}'{extra}"


# module-level handle so the plotting scripts can reach the discovered format
INFO = None


def format_title(default, template, **fields):
    """Figure title, overridable from the command line.

    `template` is a format string; anything the caller passes in `fields` is
    available to it, typically time, cycle, frame and base. A malformed template
    should not kill a long animation run, so an unknown field falls back to the
    default title with a warning rather than raising.
    """
    if not template:
        return default
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:
        print(f"  Warning: --title {template!r} could not be formatted ({exc}); "
              f"available fields: {', '.join(sorted(fields))}. Using the default.")
        return default
