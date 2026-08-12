"""Format-agnostic data layer for the Athena++ visualisation scripts.

The plotting scripts used to parse `.tab` files by column index. That tied them to
one output format and to a fixed column layout. This module puts a single reader in
front of them, so the same command works on `.athdf` and on `.tab`, and the calling
code no longer knows or cares which it got.

.tab is what every run in this project actually produces, and reading it needs
nothing beyond numpy. .athdf support is kept but entirely lazy: locating and
importing `vis/python/athena_read.py` (Athena++'s own reader, not duplicated
here) only happens inside `_read_athdf`, the first time an .athdf file is
actually read. `import athena_data` and a pure-.tab session therefore never pay
for it - that lookup walks the directory tree and the import itself measured
~90ms, pure overhead when every file in sight ends in .tab.

What this module adds on top of parsing one file is the bookkeeping the plotting
scripts need:

  * discovering which frames exist in a directory, for a given output id;
  * choosing a format when both are present (`.athdf` wins - one file per frame
    instead of one per MeshBlock);
  * assembling multi-block `.tab` output into a single grid;
  * returning the exact dictionaries the existing plotting code expects, so the
    figures are unchanged.

Output ids follow the input file: out1 is normally `variable = prim`, out2 is
`variable = uov` (the user output variables). Pass `output_id` accordingly.
"""

import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_athena_read = None                     # populated by _load_athena_read(), once


def _load_athena_read():
    """Locate and import vis/python/athena_read.py, on first .athdf read only."""
    global _athena_read
    if _athena_read is not None:
        return _athena_read
    for start in (_HERE, os.getcwd()):
        p = start
        for _ in range(10):
            cand = os.path.join(p, "vis", "python")
            if os.path.isfile(os.path.join(cand, "athena_read.py")):
                found = cand
                break
            if os.path.isfile(os.path.join(p, "athena_read.py")):
                found = p
                break
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        else:
            continue
        break
    else:
        raise ImportError(
            "athena_read.py not found. It ships with Athena++ in vis/python/. Run "
            "from inside the repository, or copy athena_read.py next to this file. "
            "(.tab files need none of this - this is only reached for .athdf.)")
    if found not in sys.path:
        sys.path.insert(0, found)
    import athena_read as _ar_module
    _athena_read = _ar_module
    return _athena_read


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

    # The format check above is a directory listing plus two regexes - it happens
    # before anything else and costs nothing, whichever format turns out to be
    # present. What used to be expensive was importing athena_read.py, and that
    # is now lazy (see _load_athena_read): choosing "athdf" here does not pay
    # for it either, only actually reading an .athdf file does.
    #
    # Preference when both exist: tab. Every run in this project produces .tab;
    # .athdf only shows up if HDF5 output was enabled by hand, and there is no
    # reason to prefer the format this project does not use.
    if prefer == "tab" or (prefer is None and t_frames):
        if not t_frames:
            raise FileNotFoundError(f"no .tab files for out{oid} in {data_dir}")
        return {"format": "tab", "base": sorted(t_bases)[0],
                "frames": t_frames, "nblocks": len(t_blocks)}

    if prefer == "athdf" or a_frames:
        if not a_frames:
            raise FileNotFoundError(f"no .athdf files for out{oid} in {data_dir}")
        return {"format": "athdf", "base": sorted(a_bases)[0],
                "frames": a_frames, "nblocks": 1}

    raise FileNotFoundError(
        f"no .athdf or .tab files for out{oid} in {data_dir}.\n"
        f"        Enable HDF5 output in the input file:\n"
        f"            <output{oid}>\n"
        f"            file_type = hdf5\n"
        f"        and configure Athena++ with -hdf5.")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _read_athdf(path):
    d = _load_athena_read().athdf(path)
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


def _read_tab_file(path):
    """One .tab block, parsed with nothing but numpy - no athena_read.py involved.

    Deliberately narrower than athena_read.tab(): 1D and 2D only (3D never occurs
    in this project's 2D cylindrical setup), and it assumes what every file this
    pgen writes has - i fastest, j slower, i.e. row-major (nj, ni) once reshaped.
    Returns the same shape convention athena_read.tab() used to (x1v/x2v as full
    (nj, ni) grids for a 2D dump, redundant along the other axis), so the block-
    joining code below did not have to change, only where the bytes come from.
    """
    with open(path) as fh:
        header1 = fh.readline()
        header2 = fh.readline()
    m = re.search(r"time=(\S+)\s+cycle=(\S+)\s+variables=(\S+)", header1)
    if not m:
        raise ValueError(f"{path}: could not parse the time/cycle header line")
    time, cycle = float(m.group(1)), int(m.group(2))
    headings = header2.split()[1:]                    # drop the leading '#'

    data = np.loadtxt(path, skiprows=2)
    if data.ndim == 1:                                  # a single-row file
        data = data[None, :]

    if headings[0] == "i" and headings[2] == "j":
        names = headings[1:2] + headings[3:]
        i_col, j_col = data[:, 0].astype(int), data[:, 2].astype(int)
        cols = np.concatenate([data[:, 1:2], data[:, 3:]], axis=1)
        ni, nj = i_col.max() - i_col.min() + 1, j_col.max() - j_col.min() + 1
        out = {"time": time, "cycle": cycle}
        for n, name in enumerate(names):
            out[name] = cols[:, n].reshape(nj, ni)
        return out

    if headings[0] in ("i", "j", "k"):                  # 1D
        names = headings[1:]
        out = {"time": time, "cycle": cycle}
        for n, name in enumerate(names):
            out[name] = data[:, n + 1]
        return out

    raise ValueError(f"{path}: unrecognised column header {header2!r}")


def _read_tab_blocks(paths):
    """Read one frame from one or more .tab blocks and join them.

    Note on the layout _read_tab_file returns: for a 2D dump `x1v` and `x2v` come
    back as full (nx2, nx1) arrays, not as the two 1D axes. Flattening them would
    give nx1*nx2 "coordinates" and silently corrupt the join, so the axes are
    taken as x1v[0, :] and x2v[:, 0].
    """
    blocks = []
    for p in paths:
        d = _read_tab_file(p)
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


# ---------------------------------------------------------------------------
# helpers shared by the plotting scripts
# ---------------------------------------------------------------------------

try:                                            # optional dependency
    from tqdm import tqdm                        # noqa: F401
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        """Minimal stand-in so the scripts do not depend on tqdm."""
        total = kwargs.get("total") or (len(iterable) if hasattr(iterable, "__len__") else None)
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {desc} {i + 1}/{total}", end="", flush=True)
            yield item
        if total:
            print()


def add_common_args(parser, output_id=1, with_frames=True):
    """Attach the arguments every plotting script shares.

    Keeping them in one place is what stops --title existing on three scripts out
    of four, which is how they drifted apart before.
    """
    parser.add_argument("--data_dir", default=None,
                        help="directory with .athdf or .tab files (default: ../data)")
    parser.add_argument("--output_dir", default=None, help="where to write figures")
    parser.add_argument("--format", default=None, choices=["athdf", "tab"],
                        help="force an input format (default: auto, prefers .athdf)")
    parser.add_argument("--output_id", type=int, default=output_id,
                        help=f"Athena++ output block to read (default: {output_id})")
    parser.add_argument("--title", default=None,
                        help="figure title, centred at the top; a format string over "
                             "{time}, {cycle}, {frame}, {base}")
    if with_frames:
        parser.add_argument("--frame", type=int, default=None, help="single frame to plot")
        parser.add_argument("--start_frame", type=int, default=None)
        parser.add_argument("--end_frame", type=int, default=None)
        parser.add_argument("--subsample", type=int, default=1, help="take every Nth frame")
        parser.add_argument("--fps", type=int, default=10, help="animation frame rate")
    return parser


def select_frames(frames, first=None, last=None, stride=1):
    """Frame numbers after applying --start_frame / --end_frame / --subsample."""
    out = sorted(frames)
    if first is not None:
        out = [f for f in out if f >= first]
    if last is not None:
        out = [f for f in out if f <= last]
    return out[::max(1, stride)]


def clip_radius(data, r_min=None, r_max=None):
    """Restrict a frame to a radial range, in place on a copy."""
    if r_min is None and r_max is None:
        return data
    r = np.asarray(data["r"])
    keep = np.ones(r.shape, dtype=bool)
    if r_min is not None:
        keep &= r >= r_min
    if r_max is not None:
        keep &= r <= r_max
    out = dict(data)
    out["r"] = r[keep]
    for k, v in data.items():
        if k in ("time", "cycle", "r", "phi", "nr", "nphi"):
            continue
        a = np.asarray(v)
        if a.ndim == 1 and a.shape == r.shape:
            out[k] = a[keep]
        elif a.ndim == 2 and a.shape[1] == r.shape[0]:
            out[k] = a[:, keep]
    if "nr" in out:
        out["nr"] = int(keep.sum())
    return out


def default_data_dir(start=None):
    """Where the output most likely is, relative to where the script was started.

    Covers both layouts the run directories use: sweep.sh puts the plotting scripts
    next to a data/ subdirectory, build.sh puts them in materials/ beside it. Falls
    back to the current directory, which is where a bare `athena -i ...` writes.
    """
    base = start or os.getcwd()
    for cand in ("data", os.path.join("..", "data"), "."):
        p = os.path.abspath(os.path.join(base, cand))
        if os.path.isdir(p) and (
                glob.glob(os.path.join(p, "*.athdf")) or glob.glob(os.path.join(p, "*.tab"))):
            return p
    return os.path.abspath(os.path.join(base, "data"))


# =============================================================================
# COLOUR PALETTES  (pypalettes, https://python-graph-gallery.com/color-palette-finder/)
# =============================================================================
# Categorical LINE colours only. The 2D scalar maps stay on inferno/plasma/RdBu_r:
# those are perceptually uniform or properly diverging, and swapping them for an
# artistic palette would misrepresent the data rather than merely look different.
#
# Measured over all 2707 pypalettes entries, discrete sets of 4-6 colours:
#   colourblind separation (OKLab dE >= 8 under protanopia/deuteranopia)
#   plus mutual separation >= 15                          ->  124 pass
#   ... and no near-black or near-white member            ->    8 pass
#   ... and WCAG contrast >= 3 against white              ->    5 pass, all of them
#                                                              sports-team palettes
#   ... and inside the light-mode lightness band          ->    0 pass
# Sampling the continuous maps at 5 points also yields 0, which is expected: points
# taken along one ramp are similar to each other by construction.
#
# So the library has no drop-in categorical set for line plots on white. The default
# below is the best available compromise rather than a clean pass, and check_palette()
# reports where any chosen palette sits so a poor pick is visible instead of silent.
PALETTE_DEFAULT = "Bold"          # 5 colours, colourblind dE 9.0, separation 17.0
PALETTE_FALLBACK = ["#6497B1", "#6A359C", "#FFB04F", "#679C35", "#CD1076",
                    "#0F7BA2", "#DD5129", "#43B284"]
_PALETTE_WARNED = set()


def _srgb_to_linear(hex_colour):
    h = hex_colour.lstrip("#")[:6]
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def _oklab(rgb):
    r, g, b = rgb
    def cbrt(v):
        return max(v, 0.0) ** (1.0 / 3.0)
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
    return 100.0 * sum((p[i] - q[i]) ** 2 for i in range(3)) ** 0.5


def check_palette(colours):
    """(min colourblind dE, min separation, min contrast vs white) for a colour list."""
    pairs = [(i, j) for i in range(len(colours)) for j in range(i + 1, len(colours))]
    if not pairs:
        return (float("inf"),) * 3
    cvd = min(min(_delta_e(colours[i], colours[j], "protan"),
                  _delta_e(colours[i], colours[j], "deutan")) for i, j in pairs)
    sep = min(_delta_e(colours[i], colours[j]) for i, j in pairs)
    con = min(1.05 / (0.2126 * r + 0.7152 * g + 0.0722 * b + 0.05)
              for r, g, b in (_srgb_to_linear(c) for c in colours))
    return cvd, sep, con


def line_colors(n, name=None, quiet=False):
    """n categorical colours from a pypalettes palette, with the checks reported.

    Falls back to a built-in list when pypalettes is absent, so the scripts keep
    working without the dependency.
    """
    name = name or PALETTE_DEFAULT
    colours = None
    try:
        from pypalettes import load_cmap
        colours = [c[:7] for c in load_cmap(name).hex]
    except ImportError:
        if not quiet and "nolib" not in _PALETTE_WARNED:
            _PALETTE_WARNED.add("nolib")
            print("  note: pypalettes is not installed, using the built-in palette "
                  "(pip install pypalettes)")
    except Exception as exc:
        if not quiet and name not in _PALETTE_WARNED:
            _PALETTE_WARNED.add(name)
            print(f"  note: palette '{name}' not found ({exc}); using the built-in one")
    if not colours:
        colours = list(PALETTE_FALLBACK)

    if not quiet and name not in _PALETTE_WARNED:
        cvd, sep, con = check_palette(colours[:max(n, 2)])
        # Contrast is judged at 1.5, not the WCAG text figure of 3: these are 2px
        # lines rather than body text, and no palette in the library clears 3 anyway.
        if cvd < 8.0 or sep < 15.0 or con < 1.5:
            _PALETTE_WARNED.add(name)
            print(f"  note: palette '{name}' is weak for line plots "
                  f"(colourblind dE {cvd:.1f}, need 8; separation {sep:.1f}, need 15; "
                  f"contrast {con:.1f}, need 1.5). It will still be used.")
    if n <= len(colours):
        return colours[:n]
    return [colours[i % len(colours)] for i in range(n)]   # cycles; avoid if you can


# =============================================================================
# MODEL PARAMETERS FOR FIGURE ANNOTATION
# =============================================================================
# Sweep runs keep athinput.in beside the data; plain runs under results/runs keep
# only data/. So the search is a chain and a miss is not an error - the annotation
# is simply omitted.

def find_athinput(data_dir, explicit=None):
    """Path to the athinput governing this run, or None. Never raises."""
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
    """{'block/key': 'value'} from an athinput. Empty dict on any problem."""
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


def _num(params, key, default=None):
    try:
        return float(params[key])
    except (KeyError, ValueError, TypeError):
        return default


INFO_KEYS = {
    # key          label                     athinput lookup / derived
    "alpha":     (r"$\alpha$",              "problem/alpha"),
    "nu_iso":    (r"$\nu_{\rm iso}$",        "problem/nu_iso"),
    "gamma":     (r"$\gamma$",               "hydro/gamma"),
    "C_prime":   ("C'",                      "problem/C_prime"),
    "r_center":  (r"$r_c$",                  "problem/r_center"),
    "rho_atm":   (r"$\rho_{\rm atm}$",       "problem/rho_atm"),
    "M_bh":      (r"$M_{\rm BH}$",           "problem/M_bh"),
    "T_0":       (r"$T_0$",                  "problem/T_0"),
    "mu":        (r"$\mu$",                  "problem/mu"),
    "chi":       (r"$\chi$",                 "problem/chi"),
    "grid":      ("grid",                    None),      # nx1 x nx2
    "hr":        ("H/r",                     None),      # derived
    "N_orbits":  ("orbits to accrete",       None),      # derived
}

INFO_PRESETS = {
    "none":    [],
    "default": ["alpha", "gamma", "grid", "C_prime", "r_center"],
    "min":     ["alpha", "nu_iso", "gamma", "grid"],
    "orbits":  ["alpha", "nu_iso", "gamma", "grid"],
    "physics": ["alpha", "nu_iso", "gamma", "grid",
                "C_prime", "r_center", "rho_atm", "hr", "N_orbits"],
    "full":    ["alpha", "nu_iso", "gamma", "grid",
                "C_prime", "r_center", "rho_atm", "hr", "N_orbits",
                "M_bh", "T_0", "mu", "chi"],
}
# Presets that also want the frame counter expressed in orbits.
INFO_WITH_ORBITS = {"orbits", "physics", "full", "default"}
_PER_LINE = 5


def resolve_info_keys(spec):
    """A preset name or a comma-separated key list -> (keys, wants_orbits)."""
    spec = (spec or "none").strip()
    if spec in INFO_PRESETS:
        return list(INFO_PRESETS[spec]), spec in INFO_WITH_ORBITS
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in INFO_KEYS]
    if unknown:
        print(f"  note: unknown --info key(s) {', '.join(unknown)}; "
              f"known keys are {', '.join(sorted(INFO_KEYS))}")
        keys = [k for k in keys if k in INFO_KEYS]
    return keys, True


def model_info(params, level="default"):
    """Annotation lines for a figure.

    `level` is a preset name from INFO_PRESETS or a comma-separated list of
    INFO_KEYS. Returns (lines, P_orb); P_orb is None unless it could be derived,
    in which case the caller can also report time in orbits - usually the most
    informative thing on an animation, since code time means nothing on its own.
    """
    keys, wants_orbits = resolve_info_keys(level)
    if not params or not keys:
        return [], None

    def g(k, d=None):
        return _num(params, k, d)

    P_orb, derived = None, {}
    if wants_orbits or {"hr", "N_orbits"} & set(keys):
        try:                                    # disk_model lives in scripts/theory
            here = os.path.dirname(os.path.abspath(__file__))
            for _ in range(6):
                cand = os.path.join(here, "scripts", "theory")
                if os.path.isdir(cand):
                    sys.path.insert(0, cand)
                    break
                here = os.path.dirname(here)
            from disk_model import DiskModel, orbits_to_accrete
            m = DiskModel(M_bh=g("problem/M_bh", 4.5e7), T_0=g("problem/T_0", 5e4),
                          mu=g("problem/mu", 0.6), chi=g("problem/chi", 500.0),
                          rho_0=g("problem/rho_0", 1e-13),
                          gamma=g("hydro/gamma", 1.3),
                          r_center=g("problem/r_center", 1.0),
                          C_prime=g("problem/C_prime", 0.2),
                          alpha=g("problem/alpha", 0.0),
                          rho_atm=g("problem/rho_atm", 1e-4))
            P_orb = m.P_orb
            derived = {"hr": ((m.gamma - 1.0) * (0.5 - m.C_prime)) ** 0.5,
                       "N_orbits": orbits_to_accrete(m.alpha, m.gamma, m.C_prime)}
        except Exception:
            P_orb, derived = None, {}

    def fmt(key):
        label, path = INFO_KEYS[key]
        if key == "grid":
            nx1, nx2 = g("mesh/nx1"), g("mesh/nx2")
            return f"{label} {int(nx1)}x{int(nx2)}" if nx1 and nx2 else None
        if key == "hr":
            return f"{label} = {derived['hr']:.3f}" if "hr" in derived else None
        if key == "N_orbits":
            return (f"{derived['N_orbits']:.0f} {label}"
                    if "N_orbits" in derived else None)
        v = g(path)
        if v is None:
            return None
        if key == "M_bh":
            return rf"{label} = {v:.3g} $M_\odot$"
        if key == "T_0":
            return f"{label} = {v:.3g} K"
        return f"{label} = {v:.3g}"

    items = [t for t in (fmt(k) for k in keys) if t]
    return ["   ".join(items[i:i + _PER_LINE])
            for i in range(0, len(items), _PER_LINE)], P_orb
