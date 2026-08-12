#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ 2D DATA VISUALIZER
=============================================================================

Visualize 2D Athena++ simulation data with heatmaps, profiles, and animations
in both Cartesian and polar coordinates.

BASIC USAGE:
    python3 vis2d.py
        Creates all plots + 2 animations for the last frame

OPTIONS:
    --mode {all,heatmaps,radial,polar,azimuthal,animation,polar_animation,
            vector_animation}
        all              - all plots + both animations (default)
        heatmaps         - 2D heatmaps of all variables
        radial           - radial profiles (averaged over φ)
        polar            - polar coordinate plots
        azimuthal        - azimuthal profiles (at different radii)
        animation        - Cartesian animation
        polar_animation  - polar animation
        vector_animation - the vector field alone, one large polar panel over a
                           density background; implies --add-vectors
    
    --frame N        - process specific frame (default: last)
    --fps N          - FPS for animations (default: 10)
    --data_dir PATH  - directory with .tab files
    --output_dir PATH - output directory (default: ./figs_2d)
    --num_workers N  - number of parallel CPU workers (default: 4)
    --subsample N    - use every Nth frame for animation (default: 1)
    --start_frame N  - first frame for animation
    --end_frame N    - last frame for animation
    
    --r_min FLOAT    - minimum radius for plotting (filters data)
    --r_max FLOAT    - maximum radius for plotting (filters data)
    --phi_min FLOAT  - minimum azimuthal angle in radians (filters data)
    --phi_max FLOAT  - maximum azimuthal angle in radians (filters data)
    
    --vmin_percentile FLOAT - lower percentile for color scale (default: 2.0)
    --vmax_percentile FLOAT - upper percentile for color scale (default: 98.0)
                              Use percentiles to filter extreme outliers and
                              improve color contrast during normal evolution

MODEL ANNOTATION (animations):
    --info SPEC      - a preset (none|default|min|orbits|physics|full) or a comma
                       list of keys: alpha, nu_iso, gamma, grid, C_prime, r_center,
                       rho_atm, hr, N_orbits, M_bh, T_0, mu, chi.
                       Default: alpha,gamma,grid,C_prime,r_center
    --info_pos       - box (default, corner, one item per line) | footer | subtitle
    --info_on        - polar (default, only the polar animation) | all
    --params FILE    - athinput to read from. Default: search beside the data and
                       upwards; if none is found the annotation is silently omitted,
                       which is the normal case for runs under results/runs
    The frame counter also reports orbits whenever P_orb could be derived.

COLOUR:
    --palette NAME   - pypalettes name for categorical LINE colours. The 2D scalar
                       maps are untouched: inferno/plasma are perceptually uniform
                       and RdBu_r/coolwarm are properly diverging, while most library
                       palettes are neither. A weak choice is reported, not silently
                       accepted - of 2707 palettes only 124 clear the colourblind
                       gate and none clears every gate
    --cmap_palette NAME - opt in to replacing the 2D maps as well

VECTOR OVERLAY (entirely off unless --add-vectors is given, so every existing
command keeps producing exactly what it did before):
    --add-vectors    - draw the vector field. By default only on the bottom-right
                       panel (azimuthal velocity) of the polar views, plus a panel
                       of its own in the heatmap and the Cartesian animation
    --vec_panels {last,all}
                     - last: only that bottom-right panel. all: every polar panel
    --vec_field {velocity,momentum}
    --vec_comp {both,radial,azimuthal}
                     - both is the true direction. In this disk v_phi is ~130x v_r,
                       so a 'both' arrow is tilted off the tangent by well under a
                       degree and the radial flow is invisible; ask for it on its
                       own to see it, since each component is autoscaled separately
    --vec_frame {perturbation,full}
                     - perturbation subtracts the azimuthal mean. On an axisymmetric
                       run that leaves only round-off, and the script says so instead
                       of drawing noise
    --vec_style {quiver,stream}
    --vec_arrows N   - roughly how many arrows across the radial range (default 8)
    --vec_stride N   - fixed every-Nth-cell stride instead of --vec_arrows
    --vec_clip PCT   - cap arrow length at this percentile so fast cells do not draw
                       through their neighbours; direction is untouched (default 92)
    --vec_density F  - streamline density for --vec_style stream (default 0.6). This
                       one knob sets both line count and integrator step, so lowering
                       it for fewer lines also makes closed orbits polygonal

    On the polar view arrows sit on a lattice that is uniform in physical distance -
    rings a fixed distance apart, arrows that same distance apart along each ring -
    so the coverage looks as even as a Cartesian one rather than bunching towards
    the axis.

    The scale is reported as "longest arrow = ..." in the caption, and additionally
    as a reference arrow on the r-phi panels. Arrow length is capped and pinned to
    the sampling spacing, so arrows never overlap and never rescale between frames
    of an animation.

VECTOR EXAMPLES:
    python3 vis2d.py --mode polar --add-vectors --vec_frame full
    python3 vis2d.py --mode polar --add-vectors --vec_comp radial --vec_frame full
    python3 vis2d.py --mode polar --add-vectors --vec_panels all --vec_frame full
    python3 vis2d.py --mode vector_animation --vec_comp radial --vec_frame full
    python3 vis2d.py --mode polar --add-vectors --vec_style stream --vec_density 0.9

EXAMPLES:
    # Basic usage
    python3 vis2d.py --subsample 10
    python3 vis2d.py --mode heatmaps --frame 10
    python3 vis2d.py --fps 15
    python3 vis2d.py --mode polar_animation
    python3 vis2d.py --mode animation
    
    # Limit radial range (inner disk only)
    python3 vis2d.py --r_min 0.5 --r_max 5.0
    
    # Limit azimuthal range (first quadrant)
    python3 vis2d.py --phi_min 0 --phi_max 1.57
    
    # Combined bounds for animation
    python3 vis2d.py --mode animation --r_min 1.0 --r_max 3.0 --phi_min 0 --phi_max 3.14
    
    # Outer disk region only
    python3 vis2d.py --mode radial --r_min 2.0 --frame 500
    
    # Better color contrast (filter more extreme outliers)
    python3 vis2d.py --mode polar_animation --vmin_percentile 5 --vmax_percentile 95
    
    # Keep more dynamic range (include more extreme values)
    python3 vis2d.py --mode animation --vmin_percentile 1 --vmax_percentile 99

`import vis2d` is side-effect free - argv is read only when run as a script (or by
calling vis2d._setup() then vis2d.main() yourself, e.g. from a notebook).
=============================================================================
"""

import argparse

# Locate athena_data.py: next to this file when the script was copied into a run
# directory, otherwise in the repository it was copied from.
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
for _c in (_here, _os.getcwd()):
    _p = _c
    for _ in range(8):
        if _os.path.isfile(_os.path.join(_p, "athena_data.py")):
            _sys.path.insert(0, _p); break
        _cand = _os.path.join(_p, "scripts", "practice", "vis")
        if _os.path.isfile(_os.path.join(_cand, "athena_data.py")):
            _sys.path.insert(0, _cand); break
        _p = _os.path.dirname(_p)
    else:
        continue
    break

import athena_data as _ad
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

from athena_data import tqdm  # shared fallback when tqdm is absent

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'colormap': 'viridis',  # inferno, plasma, magma, viridis, coolwarm
    'dpi': 200,
    'figsize_single': (10, 8),
    'figsize_multi': (16, 12),
    'fps': 10,
    'log_scale_vars': ['density', 'pressure'],
    'radial_bins': 100,
    'polar_resolution': 200,
    'num_workers': max(1, cpu_count() - 2),  # Leave 2 cores for system
    'subsample': 1,
    'chunk_size': 100,  # Process frames in chunks to save memory
    'normalize': 'none',  # 'azimuthal' subtracts the axisymmetric part; see below
}

# Fields whose azimuthal mean is positive, so the deviation is worth showing as a
# fraction of it. The velocities are not on this list: <v_r> is close to zero, and
# dividing by it turns a small deviation into a large meaningless number.
RELATIVE_NORM_VARS = ['density', 'pressure']

# Background for the vector panels. Tops out at mid grey so black arrows stay legible
# without alpha - alpha on a pcolormesh leaks its cell edges and hatches the figure.
VEC_BG_CMAP = LinearSegmentedColormap.from_list('vecbg', ['#ffffff', '#a8a8a8'])

VARIABLE_INFO = {
    'density': {
        'label': r'$\rho$ (Density)',
        'cmap': 'inferno',
        'log': True,
    },
    'pressure': {
        'label': r'$P$ (Pressure)',
        'cmap': 'plasma',
        'log': True,
    },
    'vel_r': {
        'label': r'$v_r$ (Radial Velocity)',
        'cmap': 'RdBu_r',
        'log': False,
    },
    'vel_phi': {
        'label': r'$v_\phi$ (Azimuthal Velocity)',
        'cmap': 'coolwarm',
        'log': False,
    },
    'vel_z': {
        'label': r'$v_z$ (Vertical Velocity)',
        'cmap': 'PuOr',
        'log': False,
    },
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="2D Athena++ data visualization",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Usage examples:
  python vis2d.py
  python vis2d.py --mode all
  python vis2d.py --mode heatmaps --frame 10
  python vis2d.py --mode radial --animate
    """
)
parser.add_argument("--data_dir", default=None, 
                    help="Directory with .tab files (default: ../data)")
parser.add_argument("--output_id", type=int, default=1,
                    help="Athena++ output block id to read (default: 1 = prim)")
parser.add_argument("--title", default=None,
                    help="Figure title (centred, top). Supports {time}, {cycle}, {frame}, {base}, {plot}; e.g. --title \"PP disk, t={time:.3f}\"")
parser.add_argument("--output_dir", default=None, 
                    help="Output directory (default: ../figs_2d)")
parser.add_argument("--mode", default="all",
                    choices=['all', 'heatmaps', 'radial', 'polar', 'azimuthal',
                             'animation', 'polar_animation', 'vector_animation'],
                    help="Visualization mode")
parser.add_argument("--frame", type=int, default=None,
                    help="Frame number to visualize (default: last frame)")
parser.add_argument("--animate", action='store_true',
                    help="Create animation")
parser.add_argument("--fps", type=int, default=CONFIG['fps'],
                    help="FPS for animation")
parser.add_argument("--colormap", default=CONFIG['colormap'],
                    help="Colormap for visualization")
parser.add_argument("--num_workers", type=int, default=CONFIG['num_workers'],
                    help="Number of parallel CPU workers for frame processing")
parser.add_argument("--subsample", type=int, default=CONFIG['subsample'],
                    help="Use every Nth frame for animations (1=all frames)")
parser.add_argument("--start_frame", type=int, default=None,
                    help="First frame for animation")
parser.add_argument("--end_frame", type=int, default=None,
                    help="Last frame for animation")
parser.add_argument("--r_min", type=float, default=None,
                    help="Minimum radius for plotting")
parser.add_argument("--r_max", type=float, default=None,
                    help="Maximum radius for plotting")
parser.add_argument("--phi_min", type=float, default=None,
                    help="Minimum azimuthal angle for plotting (radians)")
parser.add_argument("--phi_max", type=float, default=None,
                    help="Maximum azimuthal angle for plotting (radians)")
parser.add_argument("--normalize", default="none", choices=["none", "azimuthal"],
                    help="azimuthal: plot the deviation from the azimuthal mean, which "
                         "is the only way non-axisymmetric modes are visible at all")
parser.add_argument("--vmin_percentile", type=float, default=2.0,
                    help="Lower percentile for color scale (default: 2.0, filters extreme low values)")
parser.add_argument("--vmax_percentile", type=float, default=98.0,
                    help="Upper percentile for color scale (default: 98.0, filters extreme high values)")
parser.add_argument("--add-vectors", dest="vectors", action="store_true",
                    help="Overlay the vector field (off by default)")
parser.add_argument("--palette", default=None,
                    help="pypalettes name for categorical LINE colours (default: %s). "
                         "The 2D scalar maps are unaffected" % _ad.PALETTE_DEFAULT)
parser.add_argument("--cmap_palette", default=None,
                    help="Optional: replace the 2D scalar colormaps with a continuous "
                         "pypalettes map. Off by default, because inferno/plasma are "
                         "perceptually uniform and most library palettes are not")
parser.add_argument("--info", default="default",
                    help="Model parameters annotated on animations: a preset "
                         "(none|default|min|orbits|physics|full) or a comma-separated "
                         "list of keys (%s). Default: %s"
                         % (",".join(sorted(_ad.INFO_KEYS)),
                            ",".join(_ad.INFO_PRESETS["default"])))
parser.add_argument("--info_pos", default="box",
                    choices=["footer", "subtitle", "box"],
                    help="Where that annotation goes (default: box)")
parser.add_argument("--info_on", default="polar", choices=["polar", "all"],
                    help="Which animations carry it. polar: only the polar animation, "
                         "which is the one with room for it. all: every animation")
parser.add_argument("--params", default=None,
                    help="athinput to read the parameters from (default: search beside "
                         "the data, then upwards)")
parser.add_argument("--vec_field", default="velocity", choices=["velocity", "momentum"],
                    help="Vector quantity: velocity (v_r, v_phi) or momentum (rho*v)")
parser.add_argument("--vec_comp", default="both",
                    choices=["both", "radial", "azimuthal"],
                    help="both: the true direction, radial and azimuthal at once. "
                         "radial / azimuthal: that component alone, autoscaled on its "
                         "own - the only way to see v_r, which is ~1/130 of v_phi here "
                         "and so tilts a 'both' arrow by well under a degree")
parser.add_argument("--vec_frame", default="perturbation",
                    choices=["perturbation", "full"],
                    help="perturbation: subtract the azimuthal mean, so rotation does not "
                         "swamp everything. full: raw field")
parser.add_argument("--vec_panels", default="last", choices=["last", "all"],
                    help="Which polar panels carry arrows. last: only the bottom-right "
                         "one (azimuthal velocity), which keeps the other three clean. "
                         "all: every panel (default: last)")
parser.add_argument("--vec_style", default="quiver", choices=["quiver", "stream"],
                    help="quiver: arrow length ~ magnitude. stream: streamlines")
parser.add_argument("--vec_lattice", default="polar",
                    choices=["square", "hex", "polar"],
                    help="Where arrows sit on the POLAR view. square: a true Cartesian "
                         "grid clipped to the annulus - equal spacing along rows and "
                         "columns. hex: triangular packing, every point equidistant "
                         "from six neighbours, the most even option. polar: rings of "
                         "equal radial spacing (spacing along each ring is only "
                         "approximate, so rows stagger). Default: polar")
parser.add_argument("--vec_scale", default="log", choices=["linear", "sqrt", "log"],
                    help="How magnitude maps to arrow LENGTH. sqrt/log compress the "
                         "range so weak cells stay visible while strong ones stop "
                         "dominating - useful when the field varies by decades")
parser.add_argument("--vec_arrows", type=int, default=8,
                    help="Roughly how many arrows across the radial range (default: 8)")
parser.add_argument("--vec_stride", type=int, default=None,
                    help="Override --vec_arrows with a fixed every-Nth-cell stride")
parser.add_argument("--vec_color", default="black",
                    help="Arrow/streamline colour")
parser.add_argument("--vec_density", type=float, default=0.6,
                    help="Streamline density for --vec_style stream. Trades line count against "
                         "smoothness in one knob: below ~0.6 closed orbits render as "
                         "polygons (default: 0.6)")
parser.add_argument("--vec_clip", type=float, default=92.0,
                    help="Percentile at which arrow LENGTH is capped, so the fastest "
                         "cells do not draw arrows through their neighbours. Direction "
                         "is never changed; 100 disables the cap (default: 92)")

def _setup(argv=None):
    """Parse argv, resolve paths/config, and discover the data.

    Only called from __main__ (see the bottom of this file), so
    `import vis2d` never touches argv, the filesystem, or stdout -
    every function below can be called directly once this has run.
    """
    global args, script_dir, data_dir, output_dir
    global frames_dict, base_name, num_blocks, is_multiblock
    global available_frames, num_frames
    global INFO_LINES, INFO_P_ORB

    args = parser.parse_args(argv)

    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir:
        data_dir = args.data_dir or _ad.default_data_dir()
        _output_base = script_dir
    else:
        _candidate = os.path.join(script_dir, "data")
        if not os.path.isdir(_candidate):
            _candidate = os.path.join(os.path.dirname(script_dir), "data")
            _output_base = os.path.dirname(script_dir)
        else:
            _output_base = script_dir
        data_dir = _candidate
    output_dir = args.output_dir if args.output_dir else os.path.join(_output_base, "figs_2d")
    os.makedirs(output_dir, exist_ok=True)

    CONFIG['fps'] = args.fps
    CONFIG['colormap'] = args.colormap
    CONFIG['num_workers'] = min(args.num_workers, cpu_count())
    CONFIG['subsample'] = args.subsample
    CONFIG['r_min'] = args.r_min
    CONFIG['r_max'] = args.r_max
    CONFIG['phi_min'] = args.phi_min
    CONFIG['phi_max'] = args.phi_max
    CONFIG['vmin_percentile'] = args.vmin_percentile
    CONFIG['vmax_percentile'] = args.vmax_percentile
    CONFIG['normalize'] = args.normalize
    CONFIG['vectors'] = args.vectors
    CONFIG['palette'] = args.palette
    CONFIG['info_pos'] = args.info_pos
    CONFIG['info_on'] = args.info_on
    CONFIG['vec_field'] = args.vec_field
    CONFIG['vec_comp'] = args.vec_comp
    CONFIG['vec_frame'] = args.vec_frame
    CONFIG['vec_panels'] = args.vec_panels
    CONFIG['vec_style'] = args.vec_style
    CONFIG['vec_lattice'] = args.vec_lattice
    CONFIG['vec_scale'] = args.vec_scale
    CONFIG['vec_arrows'] = args.vec_arrows
    CONFIG['vec_stride'] = args.vec_stride
    CONFIG['vec_color'] = args.vec_color
    CONFIG['vec_density'] = args.vec_density
    CONFIG['vec_clip'] = args.vec_clip

    if args.normalize == 'azimuthal':
        # The normalised fields are signed and centred on zero, so a sequential map on a log
        # scale would be both wrong and unreadable. One hue each side of a neutral middle.
        CONFIG['log_scale_vars'] = []
        for _v, _info in VARIABLE_INFO.items():
            _info['log'] = False
            _info['cmap'] = 'RdBu_r'
            _rel = _v in RELATIVE_NORM_VARS
            _info['label'] = (_info['label'].split(' (')[0]
                              + (r'$/\langle\cdot\rangle_\phi - 1$' if _rel
                                 else r'$ - \langle\cdot\rangle_\phi$'))

    print("="*80)
    print("ATHENA++ 2D VISUALIZER")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"CPU workers: {CONFIG['num_workers']} (parallel frame processing)")
    print(f"Frame subsampling: {CONFIG['subsample']}")
    if args.r_min is not None or args.r_max is not None:
        r_range = f"r: [{args.r_min if args.r_min is not None else 'auto'}:{args.r_max if args.r_max is not None else 'auto'}]"
        print(f"Radial range: {r_range}")
    if args.phi_min is not None or args.phi_max is not None:
        phi_range = f"φ: [{args.phi_min if args.phi_min is not None else 'auto'}:{args.phi_max if args.phi_max is not None else 'auto'}] rad"
        print(f"Azimuthal range: {phi_range}")
    print(f"Color scale percentiles: [{args.vmin_percentile}, {args.vmax_percentile}]")

    # Model annotation. A run with no athinput beside it simply gets none: results/runs
    # keeps only data/, and a missing file is not an error.
    INFO_LINES, INFO_P_ORB = [], None
    if args.info != "none":
        _ath = _ad.find_athinput(data_dir, args.params)
        if _ath:
            INFO_LINES, INFO_P_ORB = _ad.model_info(_ad.read_athinput(_ath), args.info)
            print(f"Model info: {args.info} ({args.info_pos}) from {os.path.basename(_ath)}")
        else:
            print("Model info: requested, but no athinput found next to the data; omitted")

    # The 2D colormaps are replaced only on request; see --cmap_palette.
    if args.cmap_palette:
        try:
            from pypalettes import load_cmap as _load_cmap
            _cm = _load_cmap(args.cmap_palette, cmap_type="continuous")
            for _vi in VARIABLE_INFO.values():
                _vi["cmap"] = _cm
            print(f"2D colormaps replaced by pypalettes '{args.cmap_palette}'")
        except Exception as _exc:
            print(f"  note: --cmap_palette '{args.cmap_palette}' unusable ({_exc}); "
                  f"keeping the defaults")


    frames_dict, base_name, num_blocks, is_multiblock = group_files_by_frame(data_dir)

    if not frames_dict:
        print(f"ERROR: No valid data files found in {data_dir}")
        print("Expected formats:")
        print("  Multi-block: <name>.block<N>.out1.<frame>.tab")
        print("  Single-block: <name>.out1.<frame>.tab")
        exit(1)

    available_frames = sorted(frames_dict.keys())
    num_frames = len(available_frames)

    if num_blocks > 1:
        print(f"Found {num_frames} frames with {num_blocks} blocks each")
    else:
        print(f"Found {num_frames} frames (single-block format)")
    print(f"Base name: {base_name}")
    print(f"Frame range: {available_frames[0]} to {available_frames[-1]}")




def info_footer_height(plot="polar"):
    """Vertical room the footer annotation needs, so captions can sit above it."""
    if (not INFO_LINES or CONFIG["info_pos"] != "footer"
            or (CONFIG["info_on"] == "polar" and plot != "polar")):
        return 0.0
    return 0.020 + 0.022 * len(INFO_LINES)


def draw_model_info(fig, below=0.0, plot="polar"):
    """Place INFO_LINES per --info_pos. Returns the bottom margin to reserve.

    `below` lifts the footer clear of whatever already occupies the bottom of the
    figure - the arrow caption sits at y=0.012 and the two collided otherwise."""
    # --info_on polar keeps it to the polar animation, the one with room for it.
    if not INFO_LINES or (CONFIG["info_on"] == "polar" and plot != "polar"):
        return 0.0
    txt = "\n".join(INFO_LINES)
    pos = CONFIG["info_pos"]
    if pos == "subtitle":
        fig.text(0.5, 0.928, txt, ha="center", va="top", fontsize=9, color="#3c3c3a")
        return 0.0
    if pos == "box":
        # One item per line. Joined into rows the box grows wide enough to reach the
        # centred title; stacked, it stays in the corner whatever the parameter set.
        stacked = "\n".join(item for line in INFO_LINES
                             for item in line.split("   ") if item)
        fig.text(0.012, 0.988, stacked, ha="left", va="top", fontsize=8.5,
                 color="#1a1a19", linespacing=1.5,
                 bbox=dict(boxstyle="round", fc="white", ec="#b8b8b4", alpha=0.9))
        return 0.0
    fig.text(0.5, 0.012 + below, txt, ha="center", va="bottom", fontsize=9,
             color="#3c3c3a")
    return 0.035 + below + 0.022 * (len(INFO_LINES) - 1)


def frame_label(idx, total, time):
    """Frame counter; adds orbits when P_orb could be derived."""
    base = f"Frame {idx:3d}/{total} | Time = {time:5.2f}"
    if INFO_P_ORB:
        base += f"  ({time / INFO_P_ORB:.1f} orbits)"
    return base

# =============================================================================
# FILE PARSING AND GROUPING
# =============================================================================
import re

def parse_filename(filename):
    """Parse Athena++ filename to extract base name, block number, and frame number
    
    Examples: 
        acc_disk.block5.out1.00123.tab -> ('acc_disk', 5, 123, True)
        pp_disk.block0.out1.00123.tab -> ('pp_disk', 0, 123, True)
        pp_disk.block0.out1.00123.tab -> ('pp_disk', None, 123, False)  # old format
    """
    # Try new format with blocks
    match = re.match(r'(.+)\.block(\d+)\.out1\.(\d+)\.tab$', filename)
    if match:
        base_name = match.group(1)
        block_num = int(match.group(2))
        frame_num = int(match.group(3))
        return base_name, block_num, frame_num, True
    
    # Try old format without explicit block number
    match = re.match(r'(.+)\.out1\.(\d+)\.tab$', filename)
    if match:
        base_name = match.group(1)
        frame_num = int(match.group(2))
        return base_name, None, frame_num, False
    
    return None, None, None, False

def group_files_by_frame(data_dir):
    """Discover frames, whatever the output format.

    Same return contract as the .tab-only version:
    (frames_dict, base_name, num_blocks, is_multiblock).
    """
    # Format check is a cheap directory listing, done inside discover() before
    # anything else; .tab is preferred when both are present (see athena_data.py).
    info = _ad.discover(data_dir, output_id=args.output_id)
    _ad.INFO = info
    print(f"  Input: {_ad.describe(info)}")
    return info["frames"], info["base"], info["nblocks"], info["nblocks"] > 1

# =============================================================================
# FIND AND GROUP FILES
# =============================================================================

# =============================================================================
# DATA READING AND PROCESSING FUNCTIONS
# =============================================================================
def to_cpu(array):
    """No-op; kept so the plotting code reads unchanged."""
    return array


def read_athena_2d_single_block(filename):
    """Read single block of 2D Athena++ file"""
    with open(filename, 'r') as f:
        header = f.readline()
        time_str = header.split('time=')[1].split()[0]
        cycle_str = header.split('cycle=')[1].split()[0]
        time = float(time_str)
        cycle = int(cycle_str)
    
    data = np.loadtxt(filename, skiprows=2)
    
    # Columns: i, x1v(r), j, x2v(phi), rho, press, vel1(r), vel2(phi), vel3(z)
    return {
        'time': time,
        'cycle': cycle,
        'r': data[:, 1],
        'phi': data[:, 3],
        'rho': data[:, 4],
        'press': data[:, 5],
        'vel_r': data[:, 6],
        'vel_phi': data[:, 7],
        'vel_z': data[:, 8],
    }

def read_athena_2d(file_list):
    """One 2D (r, phi) frame, from .athdf or .tab.

    Same keys as before: time, cycle, r, phi, nr, nphi, R, Phi, density,
    pressure, vel_r, vel_phi, vel_z. Variable arrays are (nphi, nr) and are
    moved to the GPU when asked; r and phi stay on the host for matplotlib.
    """
    if isinstance(file_list, str):
        file_list = [file_list]
    return _ad.frame_2d(file_list, _ad.INFO["format"])

def compute_statistics(data_array, axis=None):
    """Mean and standard deviation of a field."""
    return np.mean(data_array, axis=axis), np.std(data_array, axis=axis)


def filter_data_by_bounds(data, r_min=None, r_max=None, phi_min=None, phi_max=None):
    """Filter 2D data by radial and azimuthal bounds
    
    Args:
        data: dictionary with 2D arrays and r, phi coordinates
        r_min: minimum radius (None = no limit)
        r_max: maximum radius (None = no limit)
        phi_min: minimum azimuthal angle in radians (None = no limit)
        phi_max: maximum azimuthal angle in radians (None = no limit)
    
    Returns:
        filtered data dictionary with same structure
    """
    r = data['r']
    phi = data['phi']
    
    # Determine indices to keep
    r_mask = np.ones(len(r), dtype=bool)
    if r_min is not None:
        r_mask &= (r >= r_min)
    if r_max is not None:
        r_mask &= (r <= r_max)
    
    phi_mask = np.ones(len(phi), dtype=bool)
    if phi_min is not None:
        phi_mask &= (phi >= phi_min)
    if phi_max is not None:
        phi_mask &= (phi <= phi_max)
    
    # If no filtering needed, return original data
    if np.all(r_mask) and np.all(phi_mask):
        return normalize_azimuthal(data)

    # Create filtered data
    filtered_data = {
        'time': data['time'],
        'cycle': data['cycle'],
        'r': r[r_mask],
        'phi': phi[phi_mask],
        'nr': np.sum(r_mask),
        'nphi': np.sum(phi_mask),
    }
    
    # Filter 2D arrays
    for key in ['R', 'Phi', 'density', 'pressure', 'vel_r', 'vel_phi', 'vel_z']:
        if key in data:
            # Filter along both dimensions: [phi_indices, r_indices]
            filtered_array = data[key][phi_mask, :][:, r_mask]
            # Keep on same device (GPU or CPU)
            filtered_data[key] = filtered_array

    return normalize_azimuthal(filtered_data)


def normalize_azimuthal(data):
    """Replace each field by its deviation from the azimuthal mean.

    Non-axisymmetric structure - a growing Papaloizou-Pringle mode, a spiral arm - is a
    percent-level perturbation sitting on a background that spans many decades. On any
    colour scale wide enough to show the disk it is invisible; the polar view of a disk
    with a healthy m = 2 mode looks perfectly round. Dividing out the axisymmetric part
    is what makes it appear, and costs one line per field.

        density, pressure   f / <f>_phi - 1      (relative: <f> is positive)
        velocities          f - <f>_phi          (absolute: <v_r> is ~0)

    The result is signed and centred on zero, so the caller also switches to a diverging
    colour map and turns off log scaling.
    """
    # The arrow overlay must be built from the unnormalised field: it does its own
    # frame subtraction, and doing it twice would silently halve nothing and confuse
    # --vec_frame full. Stash the originals whether or not normalisation runs.
    data = dict(data)
    data['_raw'] = {k: data[k] for k in ('density', 'vel_r', 'vel_phi') if k in data}
    if CONFIG.get('normalize') != 'azimuthal':
        return data
    out = dict(data)
    for key in ['density', 'pressure', 'vel_r', 'vel_phi', 'vel_z']:
        if key not in data:
            continue
        field = to_cpu(data[key])
        mean = np.mean(field, axis=0)                     # average over phi, per radius
        if key in RELATIVE_NORM_VARS:
            safe = np.where(np.abs(mean) > 0.0, mean, 1.0)
            out[key] = field / safe[None, :] - 1.0
        else:
            out[key] = field - mean[None, :]
    return out

# =============================================================================
# VECTOR FIELD OVERLAY
# =============================================================================
# Why a frame choice exists at all: in this disk v_phi is ~130x v_r (medians 152 vs
# 1.15 in the reference sweep). Arrows built from the raw velocity are therefore a
# uniform azimuthal swirl in which nothing else is visible. Subtracting the azimuthal
# mean leaves the flow that actually carries the physics - spiral arms, the accretion
# stream, a growing Papaloizou-Pringle mode.
#
# Geometry, verified against a Cartesian ground truth for pure rotation and pure
# outflow before any of this was wired in:
#   polar axes   matplotlib treats quiver's U,V as screen offsets, and screen space
#                for a polar plot is Cartesian - so pass true (v_x, v_y).
#                streamplot integrates in the axes' data space (theta, r), so there
#                it wants (v_phi/r, v_r) instead, transposed to (nr, nphi).
#   r-phi axes   motion in that plane is (dr/dt, dphi/dt) = (v_r, v_phi/r).

_SYM = {'velocity': r"\vec{v}", 'momentum': r"\rho\vec{v}"}
_COMP = {'both': '', 'radial': r"$_r$", 'azimuthal': r"$_\phi$"}


def vector_caption():
    """Caption line. Carries the arrow scale for the polar views, which have no room
    for a quiverkey: all four corners of a polar axes already hold an angle label."""
    cap = CONFIG.get('_vec_cap')
    tail = f",  longest arrow = {cap:.3g}" if cap and CONFIG['vec_style'] != 'stream' else ""
    if CONFIG.get('vec_scale', 'log') != 'linear' and tail:
        tail += f" ({CONFIG['vec_scale']} length scale)"
    return f"arrows: {vector_label()}  ({CONFIG['vec_style']}){tail}"


def vector_label():
    s = _SYM[CONFIG['vec_field']]
    base = (f"${s}$" if CONFIG['vec_frame'] == 'full'
            else f"${s}-\\langle{s}\\rangle_\\phi$")
    comp = CONFIG.get('vec_comp', 'both')
    return base if comp == 'both' else base + f" ({comp} only)"


_DEGENERATE_WARNED = [False]


def vector_field(data, want_info=False):
    """(u_r, u_phi) on the full (nphi, nr) grid, per --vec_field and --vec_frame.

    In the perturbation frame an axisymmetric run leaves only round-off, and quiver
    autoscales it into what looks like real structure. Whenever the residual is a
    round-off-sized fraction of the full field we say so rather than drawing a
    convincing picture of nothing - the measured value in a quiet 100-orbit run is
    ~1e-14 of the field, which is exactly "no mode was ever seeded".
    """
    raw = data.get('_raw', data)
    u_r = to_cpu(raw['vel_r']).astype(float)
    u_phi = to_cpu(raw['vel_phi']).astype(float)
    if CONFIG['vec_field'] == 'momentum':
        rho = to_cpu(raw['density']).astype(float)
        u_r, u_phi = rho * u_r, rho * u_phi

    comp = CONFIG.get('vec_comp', 'both')
    if comp == 'radial':
        u_phi = np.zeros_like(u_phi)
    elif comp == 'azimuthal':
        u_r = np.zeros_like(u_r)

    info = {'degenerate': False, 'rel': 1.0}
    if CONFIG['vec_frame'] == 'perturbation':
        full = float(np.abs(np.hypot(u_r, u_phi)).max())
        u_r = u_r - u_r.mean(axis=0)[None, :]
        u_phi = u_phi - u_phi.mean(axis=0)[None, :]
        pert = float(np.abs(np.hypot(u_r, u_phi)).max())
        info['rel'] = pert / full if full > 0 else 0.0
        info['degenerate'] = info['rel'] < 1.0e-8
        if info['degenerate'] and not _DEGENERATE_WARNED[0]:
            _DEGENERATE_WARNED[0] = True
            print(f"  *** WARNING: the flow is axisymmetric to {info['rel']:.1e} of the "
                  f"full field.\n"
                  f"      Nothing seeded a non-axisymmetric mode, so the perturbation "
                  f"frame has only\n"
                  f"      round-off to show and the arrows would be noise. Use "
                  f"--vec_frame full, or\n"
                  f"      set <problem>/pert_amp > 0 in the athinput and rerun.")
    return (u_r, u_phi, info) if want_info else (u_r, u_phi)


def _stride(n, target=None):
    """Decimation step that leaves roughly `target` arrows along an axis."""
    if CONFIG.get('vec_stride'):
        return max(1, int(CONFIG['vec_stride']))
    target = target or CONFIG.get('vec_arrows', 8)
    return max(1, int(round(n / float(target))))


def _grid_lattice(r, phi, hexagonal=False):
    """Cell indices on a true Cartesian (or triangular) lattice clipped to the annulus.

    The polar-ring version below spaces rings evenly but has to round 2*pi*r/d to a
    whole number of arrows per ring, so the along-ring step drifts and neighbouring
    rings never line up - the eye reads that as a stagger, not a lattice. Laying the
    points out in (x, y) instead and keeping the ones that land on the annulus gives
    genuinely equal spacing along rows and columns. `hexagonal` offsets alternate rows
    by half a step and compresses row spacing by sqrt(3)/2, which is the arrangement
    where every point is equidistant from all six of its neighbours.

    Both families are built as integer multiples of the step about the origin. Starting
    from -r_max and stepping instead - the obvious way - only lands on the centre when
    the diameter happens to be a whole number of steps, so the whole pattern sat 0.30
    off centre and overshot the outer edge.

    Returns (j, i, r_exact, phi_exact) like _polar_lattice.
    """
    n = max(2, CONFIG.get('vec_arrows', 8))
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
        # odd hex rows sit at half-integer multiples, which is still symmetric about 0
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


def _nearest(axis, values):
    """Index of the closest cell centre, not merely the insertion point."""
    idx = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
    left = np.abs(values - axis[idx - 1]) <= np.abs(axis[np.minimum(idx, len(axis) - 1)]
                                                    - values)
    return np.clip(np.where(left, idx - 1, idx), 0, len(axis) - 1)


def _polar_lattice(r, phi):
    """Cell indices on a lattice that is uniform in PHYSICAL space, not in index space.

    A fixed index stride puts the same number of arrows on every ring, so they crowd
    together towards the axis where the rings are short. Instead place rings a fixed
    distance d apart and space arrows d apart *along* each ring, which is what makes a
    polar plot look as evenly covered as a Cartesian one. Concentric by construction,
    so it is centred on the origin whatever the step works out to.

    Returns (j, i, r_exact, phi_exact): the indices say which cell to read the field
    from, the last two say where to draw. Drawing at cell centres instead would quantise
    the lattice to the mesh - 2.8 degrees per cell at nphi=128, enough to see.
    """
    n_rings = max(2, CONFIG.get('vec_arrows', 8))
    r0, r1 = float(r[0]), float(r[-1])
    d = (r1 - r0) / n_rings
    if d <= 0:
        return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
    js, iss, rr, pp = [], [], [], []
    for k in range(n_rings):
        rk = r0 + (k + 0.5) * d
        i = int(np.argmin(np.abs(r - rk)))
        n_arrow = max(3, int(round(2.0 * np.pi * rk / d)))
        angles = 2.0 * np.pi * (np.arange(n_arrow) + 0.5) / n_arrow
        js.append(_nearest(phi, angles))
        iss.append(np.full(n_arrow, i))
        rr.append(np.full(n_arrow, rk))
        pp.append(angles)
    return (np.concatenate(js), np.concatenate(iss),
            np.concatenate(rr), np.concatenate(pp))


def _vectors_on_grid(data, polar):
    """(X, Y, U, V) ready for ax.quiver, on whichever grid the view needs."""
    r, phi = to_cpu(data['r']), to_cpu(data['phi'])
    R, Phi = to_cpu(data['R']), to_cpu(data['Phi'])
    u_r, u_phi = vector_field(data)
    if polar:
        lat = CONFIG.get('vec_lattice', 'polar')
        j, i, Rs, Phis = (_polar_lattice(r, phi) if lat == 'polar'
                          else _grid_lattice(r, phi, hexagonal=(lat == 'hex')))
        # field sampled at the owning cell, arrow drawn at the exact lattice point
        ur, up = u_r[j, i], u_phi[j, i]
        # true Cartesian components; matplotlib draws U,V in screen space here
        return (Phis, Rs,
                ur * np.cos(Phis) - up * np.sin(Phis),
                ur * np.sin(Phis) + up * np.cos(Phis))
    sl = (slice(None, None, _stride(len(phi))), slice(None, None, _stride(len(r))))
    Rs = R[sl]
    return Rs, Phi[sl], u_r[sl], u_phi[sl] / np.maximum(Rs, 1e-30)  # (dr/dt, dphi/dt)


def _shape_length(U, V, cap):
    """Remap magnitude to arrow length. Direction is never touched.

    linear is faithful but useless when the field spans decades: everything below the
    cap collapses into invisible stubs. sqrt and log compress the range so the weak
    cells stay readable, at the cost of length no longer being proportional - which is
    why the caption names the mapping.
    """
    mode = CONFIG.get('vec_scale', 'log')
    if mode == 'linear' or cap <= 0:
        return U, V
    mag = np.hypot(U, V)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.clip(mag / cap, 0.0, 1.0)
        new = np.sqrt(t) if mode == 'sqrt' else np.log1p(9.0 * t) / np.log(10.0)
        f = np.where(mag > 0, new * cap / np.maximum(mag, 1e-30), 0.0)
    return U * f, V * f


def _clip_length(U, V, cap=None):
    """Shorten the longest arrows to a percentile cap, keeping direction exactly.

    Returns (U, V, cap). --vec_clip 100 disables the shortening but still reports a cap
    so the key stays meaningful.
    """
    mag = np.hypot(U, V)
    good = mag[np.isfinite(mag) & (mag > 0)]
    pct = float(CONFIG.get('vec_clip', 92))
    if cap is None:
        if not good.size:
            return U, V, 1.0
        cap = float(np.percentile(good, min(max(pct, 1.0), 100.0)))
    if not np.isfinite(cap) or cap <= 0:
        return U, V, (float(good.max()) if good.size else 1.0)
    if pct >= 100.0:
        return U, V, cap
    with np.errstate(invalid='ignore', divide='ignore'):
        shrink = np.where(mag > cap, cap / np.maximum(mag, 1e-30), 1.0)
    return U * shrink, V * shrink, cap


def overlay_vectors(ax, data, polar, add_key=True, artists=None, scale_from=None):
    """Draw the vector overlay on `ax`.

    polar=True  -> ax has projection='polar' and pcolormesh(Phi, R, ...)
    polar=False -> ax is an r-phi rectangle, pcolormesh(R, Phi, ...)

    Returns a dict of artists for animation reuse; pass it back as `artists` to update
    in place. Streamlines cannot be updated, so they are torn down and redrawn.
    """
    if not CONFIG.get('vectors'):
        return None

    r = to_cpu(data['r'])
    phi = to_cpu(data['phi'])
    R = to_cpu(data['R'])
    Phi = to_cpu(data['Phi'])
    u_r, u_phi, vinfo = vector_field(data, want_info=True)
    style = CONFIG['vec_style']
    if vinfo['degenerate']:
        # Drawing round-off would be a picture of noise that looks like a result.
        ax.text(0.5, 0.5, f"axisymmetric\nto {vinfo['rel']:.0e}\n(no mode seeded)",
                transform=ax.transAxes, ha='center', va='center', fontsize=9,
                color='crimson', fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round', fc='white', ec='crimson', alpha=0.85))
        return None

    if style == 'stream':
        return _overlay_stream(ax, r, phi, R, u_r, u_phi, polar, artists)

    X, Y, U, V = _vectors_on_grid(data, polar)

    # Cap the length. |v_r| spans an order of magnitude across the disk, so on a single
    # linear scale the fastest arrows are long enough to run through their neighbours
    # and off the axes. Clipping keeps direction exactly and only shortens the tail of
    # the distribution; the key reports the cap so the scale is still readable.
    #
    # The cap comes from `scale_from` when the caller supplies one. An animation fixes
    # the quiver scale on the frame it creates the artist from, and frame 0 of these
    # runs has v = 0 everywhere - scaling off that produced arrows the width of the
    # panel once real data arrived. The animations therefore hand in a developed frame.
    _, _, rU, rV = (_vectors_on_grid(scale_from, polar) if scale_from is not None
                    else (None, None, U, V))
    _, _, cap = _clip_length(rU, rV)
    U, V, _ = _clip_length(U, V, cap=cap)
    U, V = _shape_length(U, V, cap)

    if artists and 'quiver' in artists:
        artists['quiver'].set_UVC(U, V)
        return artists

    ref = cap
    # Always pin the scale explicitly; never let quiver autoscale. Autoscaling reads
    # only the frame the artist is created from, which in an animation is frame 0 with
    # v = 0, and it also let the innermost polar ring draw arrows across the disk.
    if polar:
        # scale_units='width': an arrow of magnitude `scale` spans the axes, and the
        # axes spans 2*r_max of data. Longest arrow -> 0.85 of the lattice spacing.
        d = (float(r[-1]) - float(r[0])) / max(2, CONFIG.get('vec_arrows', 8))
        qkw = {'scale_units': 'width',
               'scale': cap * 2.0 * float(r[-1]) / (0.85 * d)}
    else:
        # scale_units='x': magnitude `scale` is one r-unit long. Longest arrow -> 0.85
        # of the sampling spacing in r.
        dx = (float(r[-1]) - float(r[0])) / max(len(r), 1) * _stride(len(r))
        qkw = {'scale_units': 'x', 'scale': cap / (0.85 * max(dx, 1e-30))}
    q = ax.quiver(X, Y, U, V, color=CONFIG['vec_color'], alpha=0.9, pivot='mid',
                  width=0.005, headwidth=3.4, headlength=4.2, headaxislength=3.6,
                  zorder=5, **qkw)
    if add_key:
        # Placing the key is fiddly and every obvious spot has failed once already:
        # above the axes it ran through the panel title, at x=0.97 the shaft ran off
        # the panel, and at 0.72 on a POLAR axes it landed on the disk itself where
        # dark text on a dark rim is invisible. A polar axes is square with the disk
        # inscribed, so its corner is the one reliably blank area; the r-phi rectangle
        # is data everywhere, so the key gets a backing box instead.
        if not polar:
            ax.quiverkey(q, 0.72, 0.94, ref, f"{ref:.3g}", labelpos='W',
                         coordinates='axes', labelsep=0.05, zorder=7,
                         fontproperties={'size': 8})
    CONFIG['_vec_cap'] = cap
    return {'quiver': q, 'cap': cap}


def _overlay_stream(ax, r, phi, R, u_r, u_phi, polar, artists):
    """Streamlines. matplotlib cannot update these, so old ones are removed first."""
    if artists and 'stream' in artists:
        artists['stream'].lines.remove()
        for art in list(ax.patches):
            art.remove()

    omega = u_phi / np.maximum(R, 1e-30)          # dphi/dt
    color = CONFIG['vec_color']
    dens = CONFIG['vec_density']

    # streamplot demands an exactly uniform grid, and .tab stores coordinates to only
    # ~5 significant digits, so the recorded spacing wobbles by ~1e-3 of a cell and
    # numpy's allclose rejects it. The mesh really is uniform (x1rat = 1), so rebuild
    # the axes exactly; the shift is far below one pixel.
    r = np.linspace(r[0], r[-1], len(r))
    phi = np.linspace(phi[0], phi[-1], len(phi))

    try:
        if polar:
            # phi is periodic: repeat the first row at 2*pi so streamlines do not
            # break along the seam at phi = 0.
            dphi = phi[1] - phi[0] if len(phi) > 1 else 0.0
            php = np.append(phi, phi[-1] + dphi)
            wp = np.vstack([omega, omega[:1]])
            vrp = np.vstack([u_r, u_r[:1]])
            # NOTE on --vec_density: streamplot steps in units of its internal mask,
            # which is 30*density cells across, NOT in units of the data grid. So the
            # knob trades line count against smoothness in one: below ~0.6 the closed
            # orbits of a rotating disk visibly become polygons, and refining the data
            # grid does not help because the integrator never sees it.
            sp = ax.streamplot(php, r, wp.T, vrp.T, density=dens, color=color,
                               linewidth=0.7, arrowsize=0.7)
            ax.set_ylim(r.min(), r.max())   # streamplot otherwise rescales the r axis
        else:
            sp = ax.streamplot(r, phi, u_r.T, omega.T, density=dens, color=color,
                               linewidth=0.7, arrowsize=0.7)
            ax.set_xlim(r.min(), r.max())
            ax.set_ylim(phi.min(), phi.max())
    except (ValueError, IndexError) as exc:
        print(f"    streamplot skipped: {exc}")
        return None
    return {'stream': sp}


# =============================================================================
# PARALLEL PROCESSING HELPERS (must be at module level for pickling)
# =============================================================================
def _read_frame_for_sampling(args_tuple):
    """Helper function for parallel frame reading (must be top-level for multiprocessing)"""
    frame, file_list, var = args_tuple
    try:
        data = read_athena_2d(file_list)  # Disable GPU in workers
        var_data = to_cpu(data[var]).ravel()
        return var_data
    except Exception as e:
        print(f"Warning: Failed to read frame {frame}: {e}")
        return None

def _compute_frame_stats(args_tuple):
    """Helper function for parallel statistics computation"""
    (file_list,) = args_tuple
    try:
        data = read_athena_2d(file_list)
        return data
    except Exception as e:
        print(f"Warning: Failed to process frame: {e}")
        return None

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_2d_heatmap(data, frame_num, output_dir):
    """Create 2D heatmaps for all variables"""
    print(f"  Creating 2D heatmaps for frame {frame_num}...")
    
    # Apply bounds filtering
    data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'], 
                                  CONFIG['phi_min'], CONFIG['phi_max'])
    
    fig, axes = plt.subplots(2, 3, figsize=CONFIG['figsize_multi'])
    fig.suptitle(_ad.format_title(f"2D Disk Structure (t={data['time']:.3f}, cycle={data['cycle']})", args.title,
                                  time=data['time'], cycle=data['cycle'], frame=locals().get('frame_num', 0),
                                  base=(_ad.INFO or {}).get("base", ""), plot='2d'),
                 fontsize=16, fontweight='bold')
    
    variables = ['density', 'pressure', 'vel_r', 'vel_phi', 'vel_z']
    
    for idx, var in enumerate(variables):
        if idx < 5:
            ax = axes.flatten()[idx]
            var_data = to_cpu(data[var])  # Transfer to CPU for matplotlib
            info = VARIABLE_INFO[var]
            
            # Choose normalization
            if info['log'] and np.all(var_data > 0):
                norm = LogNorm(vmin=var_data[var_data > 0].min(), vmax=var_data.max())
            else:
                vmax = np.abs(var_data).max()
                if not info['log']:
                    norm = Normalize(vmin=-vmax, vmax=vmax)
                else:
                    norm = Normalize(vmin=var_data.min(), vmax=var_data.max())
            
            # Draw heatmap
            R_cpu = to_cpu(data['R'])
            Phi_cpu = to_cpu(data['Phi'])
            im = ax.pcolormesh(R_cpu, Phi_cpu, var_data, 
                              cmap=info['cmap'], norm=norm, shading='auto')
            ax.set_xlabel('r', fontsize=12)
            ax.set_ylabel(r'$\phi$ (rad)', fontsize=12)
            ax.set_title(info['label'], fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax, label=info['label'])
            ax.grid(True, alpha=0.3)
    
    # The sixth slot is the vector field on its own, over density: on the v_r and
    # v_phi panels arrows would only restate what the colour already says.
    ax = axes.flatten()[5]
    if CONFIG.get('vectors'):
        rho = to_cpu(data['density'])
        # muted so the black arrows stay legible on top of it
        ax.pcolormesh(to_cpu(data['R']), to_cpu(data['Phi']), rho,
                      cmap=VEC_BG_CMAP, shading='auto',
                      norm=(LogNorm(vmin=rho[rho > 0].min(), vmax=rho.max())
                            if np.all(rho > 0) else Normalize()))
        overlay_vectors(ax, data, polar=False)
        ax.set_xlabel('r', fontsize=12)
        ax.set_ylabel(r'$\phi$ (rad)', fontsize=12)
        ax.set_title(f"{vector_label()}  over $\\rho$", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.remove()

    plt.tight_layout()
    filename = os.path.join(output_dir, f"heatmap_frame_{frame_num:05d}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def plot_radial_profiles(data, frame_num, output_dir):
    print(f"  Creating radial profiles for frame {frame_num}...")
    
    # Apply bounds filtering
    data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'], 
                                  CONFIG['phi_min'], CONFIG['phi_max'])
    
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['figsize_single'])
    fig.suptitle(_ad.format_title(f"Radial Profiles (averaged over φ) - t={data['time']:.3f}", args.title,
                                  time=data['time'], cycle=data['cycle'], frame=locals().get('frame_num', 0),
                                  base=(_ad.INFO or {}).get("base", ""), plot='radial'),
                 fontsize=14, fontweight='bold')
    
    r = data['r']
    
    # Average over phi
    _c = _ad.line_colors(4, CONFIG.get('palette'))
    variables = [
        ('density', axes[0, 0], _c[0], r'$\langle\rho\rangle_\phi$'),
        ('pressure', axes[0, 1], _c[1], r'$\langle P\rangle_\phi$'),
        ('vel_r', axes[1, 0], _c[2], r'$\langle v_r\rangle_\phi$'),
        ('vel_phi', axes[1, 1], _c[3], r'$\langle v_\phi\rangle_\phi$'),
    ]
    
    for var, ax, color, label in variables:
        # Use GPU for statistics if available
        var_avg, var_std = compute_statistics(data[var], axis=0)
        
        ax.plot(r, var_avg, color=color, linewidth=2, label=label)
        ax.fill_between(r, var_avg - var_std, var_avg + var_std,
                        alpha=0.25, color=color)
        ax.set_xlabel('r', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Log scale if needed
        if VARIABLE_INFO[var]['log']:
            ax.set_yscale('log')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"radial_profile_frame_{frame_num:05d}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def plot_polar(data, frame_num, output_dir):
    """Polar plots for disk"""
    print(f"  Creating polar plots for frame {frame_num}...")
    
    # Apply bounds filtering
    data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'], 
                                  CONFIG['phi_min'], CONFIG['phi_max'])
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(_ad.format_title(f"Polar View of Disk - t={data['time']:.3f}", args.title,
                                  time=data['time'], cycle=data['cycle'], frame=locals().get('frame_num', 0),
                                  base=(_ad.INFO or {}).get("base", ""), plot='polar'),
                 fontsize=16, fontweight='bold')
    
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']

    for idx, var in enumerate(variables):
        ax = plt.subplot(2, 2, idx+1, projection='polar')
        var_data = to_cpu(data[var])
        info = VARIABLE_INFO[var]
        
        # Normalization
        if info['log'] and np.all(var_data > 0):
            norm = LogNorm(vmin=var_data[var_data > 0].min(), vmax=var_data.max())
        else:
            vmax = np.abs(var_data).max()
            if not info['log']:
                norm = Normalize(vmin=-vmax, vmax=vmax)
            else:
                norm = Normalize(vmin=var_data.min(), vmax=var_data.max())
        
        # Polar plot
        Phi_cpu = to_cpu(data['Phi'])
        R_cpu = to_cpu(data['R'])
        im = ax.pcolormesh(Phi_cpu, R_cpu, var_data,
                          cmap=info['cmap'], norm=norm, shading='auto')
        ax.set_title(info['label'], fontsize=13, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label=info['label'], pad=0.1)
        ax.grid(True, alpha=0.3)
        if CONFIG['vec_panels'] == 'all' or idx == len(variables) - 1:
            overlay_vectors(ax, data, polar=True, add_key=True)

    if CONFIG.get('vectors'):
        fig.text(0.5, 0.012, vector_caption(),
                 ha='center', fontsize=11)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"polar_frame_{frame_num:05d}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def plot_azimuthal_profiles(data, frame_num, output_dir):
    """Azimuthal profiles at different radii"""
    print(f"  Creating azimuthal profiles for frame {frame_num}...")
    
    # Apply bounds filtering
    data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'], 
                                  CONFIG['phi_min'], CONFIG['phi_max'])
    
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['figsize_single'])
    fig.suptitle(_ad.format_title(f"Azimuthal Profiles at Different Radii - t={data['time']:.3f}", args.title,
                                  time=data['time'], cycle=data['cycle'], frame=locals().get('frame_num', 0),
                                  base=(_ad.INFO or {}).get("base", ""), plot='azimuthal'),
                 fontsize=14, fontweight='bold')
    
    phi = data['phi']
    r = data['r']
    
    # Select several radii
    r_indices = [len(r)//4, len(r)//2, 3*len(r)//4]
    colors = _ad.line_colors(3, CONFIG.get('palette'))
    
    variables = [
        ('density', axes[0, 0], r'$\rho$'),
        ('pressure', axes[0, 1], r'$P$'),
        ('vel_r', axes[1, 0], r'$v_r$'),
        ('vel_phi', axes[1, 1], r'$v_\phi$'),
    ]
    
    for var, ax, ylabel in variables:
        var_data = to_cpu(data[var])
        for r_idx, color in zip(r_indices, colors):
            profile = var_data[:, r_idx]
            ax.plot(phi, profile, color=color, linewidth=2, 
                   label=f'r={r[r_idx]:.2f}', alpha=0.7)
        
        ax.set_xlabel(r'$\phi$ (rad)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if VARIABLE_INFO[var]['log']:
            ax.set_yscale('log')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"azimuthal_profile_frame_{frame_num:05d}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def create_animation(available_frames, frames_dict, output_dir):
    """Create animation in Cartesian coordinates"""
    print(f"\nCreating Cartesian animation...")
    
    # Apply subsampling and frame range
    start_idx = 0 if args.start_frame is None else available_frames.index(args.start_frame)
    end_idx = len(available_frames) if args.end_frame is None else available_frames.index(args.end_frame) + 1
    frames_subset = available_frames[start_idx:end_idx:CONFIG['subsample']]
    
    print(f"  Total frames: {len(available_frames)}")
    print(f"  Animation frames: {len(frames_subset)} (subsampled by {CONFIG['subsample']})")
    
    # 2x3 so the vector field gets a panel of its own, matching the heatmap layout.
    _vec_on = CONFIG.get('vectors')
    fig, axes = plt.subplots(2, 3 if _vec_on else 2,
                             figsize=(17, 10) if _vec_on else (12, 10))
    fig.suptitle(_ad.format_title("Disk Evolution", args.title, time=0.0, cycle=0,
                                  frame=0, base=(_ad.INFO or {}).get("base", ""), plot="animation"),
                 fontsize=16, fontweight='bold')
    
    # Read first frame for initialization
    first_data = read_athena_2d(frames_dict[frames_subset[0]])
    
    # Apply bounds filtering
    first_data = filter_data_by_bounds(first_data, CONFIG['r_min'], CONFIG['r_max'],
                                        CONFIG['phi_min'], CONFIG['phi_max'])
    
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']
    ims = []
    
    # Compute global ranges for consistent colorscale
    print("  Computing global ranges...")
    global_ranges = {}
    sample_step = max(1, len(frames_subset)//10)
    frames_to_sample = frames_subset[::sample_step]
    
    for var in variables:
        # Prepare arguments for parallel processing
        args_list = [(frame, frames_dict[frame], var) 
                     for frame in frames_to_sample]
        
        # Use parallel processing if num_workers > 1
        if CONFIG['num_workers'] > 1:
            with Pool(processes=CONFIG['num_workers']) as pool:
                results = list(tqdm(
                    pool.imap(_read_frame_for_sampling, args_list),
                    total=len(args_list),
                    desc=f"  Sampling {var} (parallel)"
                ))
            # Filter out None results and concatenate
            all_data = np.concatenate([r for r in results if r is not None])
        else:
            # Sequential processing
            all_data = []
            for frame in tqdm(frames_to_sample, desc=f"  Sampling {var}"):
                data = read_athena_2d(frames_dict[frame])
                all_data.append(to_cpu(data[var]).ravel())
            all_data = np.concatenate(all_data)
        
        info = VARIABLE_INFO[var]
        if info['log'] and np.all(all_data > 0):
            # Use percentiles for log scale (only positive values)
            positive_data = all_data[all_data > 0]
            vmin = np.percentile(positive_data, CONFIG['vmin_percentile'])
            vmax = np.percentile(positive_data, CONFIG['vmax_percentile'])
            global_ranges[var] = (vmin, vmax)
            print(f"    {var}: [{vmin:.3e}, {vmax:.3e}] (log scale, {CONFIG['vmin_percentile']}-{CONFIG['vmax_percentile']} percentile)")
        else:
            if not info['log']:
                # For symmetric diverging colormaps (velocities)
                vmax = np.percentile(np.abs(all_data), CONFIG['vmax_percentile'])
                global_ranges[var] = (-vmax, vmax)
                print(f"    {var}: [{-vmax:.3e}, {vmax:.3e}] (symmetric, {CONFIG['vmax_percentile']} percentile)")
            else:
                # Linear scale with percentiles
                vmin = np.percentile(all_data, CONFIG['vmin_percentile'])
                vmax = np.percentile(all_data, CONFIG['vmax_percentile'])
                global_ranges[var] = (vmin, vmax)
                print(f"    {var}: [{vmin:.3e}, {vmax:.3e}] (linear, {CONFIG['vmin_percentile']}-{CONFIG['vmax_percentile']} percentile)")
    
    # Initialize plots
    for idx, var in enumerate(variables):
        ax = axes.flatten()[idx]
        info = VARIABLE_INFO[var]
        
        vmin, vmax = global_ranges[var]
        if info['log'] and vmin > 0:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        R_cpu = to_cpu(first_data['R'])
        Phi_cpu = to_cpu(first_data['Phi'])
        var_data = to_cpu(first_data[var])
        
        im = ax.pcolormesh(R_cpu, Phi_cpu, var_data,
                          cmap=info['cmap'], norm=norm, shading='auto')
        ax.set_xlabel('r')
        ax.set_ylabel(r'$\phi$')
        ax.set_title(info['label'])
        plt.colorbar(im, ax=ax)
        ax.grid(True, alpha=0.3)
        ims.append(im)

    vec_ax = vec_artists = vec_bg = None
    if _vec_on:
        flat = axes.flatten()
        vec_ax = flat[4]
        rho0 = to_cpu(first_data['density'])
        vec_bg = vec_ax.pcolormesh(
            to_cpu(first_data['R']), to_cpu(first_data['Phi']), rho0, cmap=VEC_BG_CMAP,
            shading='auto',
            norm=(LogNorm(*global_ranges['density']) if global_ranges['density'][0] > 0
                  else Normalize(*global_ranges['density'])))
        _scale_ref = filter_data_by_bounds(
            read_athena_2d(frames_dict[frames_subset[-1]]),
            CONFIG['r_min'], CONFIG['r_max'], CONFIG['phi_min'], CONFIG['phi_max'])
        vec_artists = overlay_vectors(vec_ax, first_data, polar=False,
                                      scale_from=_scale_ref)
        vec_ax.set_xlabel('r')
        vec_ax.set_ylabel(r'$\phi$')
        vec_ax.set_title(f"{vector_label()} over $\\rho$")
        vec_ax.grid(True, alpha=0.3)
        flat[5].remove()

    # Add frame number to text
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12, fontweight='bold')

    def update(frame_idx):
        frame = frames_subset[frame_idx]
        data = read_athena_2d(frames_dict[frame])

        # Apply bounds filtering
        data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'],
                                      CONFIG['phi_min'], CONFIG['phi_max'])

        # Update text with frame number
        time_text.set_text(frame_label(frame_idx, len(frames_subset)-1, data['time']))

        for idx, var in enumerate(variables):
            ims[idx].set_array(to_cpu(data[var]).ravel())

        nonlocal_out = ims + [time_text]
        if _vec_on:
            vec_bg.set_array(to_cpu(data['density']).ravel())
            update.artists = overlay_vectors(vec_ax, data, polar=False,
                                             artists=update.artists)
            nonlocal_out = nonlocal_out + [vec_bg]
        return nonlocal_out

    update.artists = vec_artists
    
    print("  Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_subset),
                                 interval=1000/CONFIG['fps'], blit=False)
    
    _margin = draw_model_info(fig, plot="cartesian")
    if _margin:
        fig.subplots_adjust(bottom=max(0.10, _margin + 0.06))
    output_video = os.path.join(output_dir, "disk_evolution_2d.mp4")
    ani.save(output_video, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved Cartesian animation: {output_video}")
    plt.close()

def create_polar_animation(available_frames, frames_dict, output_dir):
    """Create animation in polar coordinates with optimizations"""
    print(f"\nCreating POLAR animation...")
    
    # Apply subsampling and frame range
    start_idx = 0 if args.start_frame is None else available_frames.index(args.start_frame)
    end_idx = len(available_frames) if args.end_frame is None else available_frames.index(args.end_frame) + 1
    frames_subset = available_frames[start_idx:end_idx:CONFIG['subsample']]
    
    print(f"  Total frames: {len(available_frames)}")
    print(f"  Animation frames: {len(frames_subset)} (subsampled by {CONFIG['subsample']})")
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(_ad.format_title("Polar View of Disk Evolution", args.title,
                                  time=0.0, cycle=0, frame=0, base=(_ad.INFO or {}).get("base", ""),
                                  plot="polar_animation"),
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Read first frame to determine normalization
    first_data = read_athena_2d(frames_dict[frames_subset[0]])
    
    # Apply bounds filtering
    first_data = filter_data_by_bounds(first_data, CONFIG['r_min'], CONFIG['r_max'],
                                        CONFIG['phi_min'], CONFIG['phi_max'])
    
    # Calculate global min/max for all frames (for consistent color scale)
    print("  Computing global bounds for color scales...")
    global_ranges = {}
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']
    
    sample_step = max(1, len(frames_subset)//10)
    frames_to_sample = frames_subset[::sample_step]
    
    for var in variables:
        # Prepare arguments for parallel processing
        args_list = [(frame, frames_dict[frame], var) 
                     for frame in frames_to_sample]
        
        # Use parallel processing if num_workers > 1
        if CONFIG['num_workers'] > 1:
            with Pool(processes=CONFIG['num_workers']) as pool:
                results = list(tqdm(
                    pool.imap(_read_frame_for_sampling, args_list),
                    total=len(args_list),
                    desc=f"  Sampling {var} (parallel)"
                ))
            # Filter out None results and concatenate
            all_data = np.concatenate([r for r in results if r is not None])
        else:
            # Sequential processing
            all_data = []
            for frame in tqdm(frames_to_sample, desc=f"  Sampling {var}"):
                data = read_athena_2d(frames_dict[frame])
                all_data.append(to_cpu(data[var]).ravel())
            all_data = np.concatenate(all_data)
        
        info = VARIABLE_INFO[var]
        if info['log'] and np.all(all_data > 0):
            # Use percentiles for log scale (only positive values)
            positive_data = all_data[all_data > 0]
            vmin = np.percentile(positive_data, CONFIG['vmin_percentile'])
            vmax = np.percentile(positive_data, CONFIG['vmax_percentile'])
            global_ranges[var] = (vmin, vmax)
            print(f"    {var}: [{vmin:.3e}, {vmax:.3e}] (log scale, {CONFIG['vmin_percentile']}-{CONFIG['vmax_percentile']} percentile)")
        else:
            if not info['log']:
                # For symmetric diverging colormaps (velocities)
                vmax = np.percentile(np.abs(all_data), CONFIG['vmax_percentile'])
                global_ranges[var] = (-vmax, vmax)
                print(f"    {var}: [{-vmax:.3e}, {vmax:.3e}] (symmetric, {CONFIG['vmax_percentile']} percentile)")
            else:
                # Linear scale with percentiles
                vmin = np.percentile(all_data, CONFIG['vmin_percentile'])
                vmax = np.percentile(all_data, CONFIG['vmax_percentile'])
                global_ranges[var] = (vmin, vmax)
                print(f"    {var}: [{vmin:.3e}, {vmax:.3e}] (linear, {CONFIG['vmin_percentile']}-{CONFIG['vmax_percentile']} percentile)")
    
    # Create polar subplots
    axes = []
    ims = []
    cbars = []
    
    for idx, var in enumerate(variables):
        ax = plt.subplot(2, 2, idx+1, projection='polar')
        axes.append(ax)
        info = VARIABLE_INFO[var]
        
        var_data = to_cpu(first_data[var])
        vmin, vmax = global_ranges[var]
        
        if info['log'] and vmin > 0:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        Phi_cpu = to_cpu(first_data['Phi'])
        R_cpu = to_cpu(first_data['R'])
        im = ax.pcolormesh(Phi_cpu, R_cpu, var_data,
                          cmap=info['cmap'], norm=norm, shading='auto')
        ax.set_title(info['label'], fontsize=13, fontweight='bold', pad=20)
        cbar = plt.colorbar(im, ax=ax, label=info['label'], pad=0.1)
        ax.grid(True, alpha=0.3)
        
        ims.append(im)
        cbars.append(cbar)

    # Frame 0 is the initial condition with v = 0, so scale the arrows off the last
    # frame instead; otherwise quiver locks in a scale derived from nothing.
    _scale_ref = filter_data_by_bounds(read_athena_2d(frames_dict[frames_subset[-1]]),
                                       CONFIG['r_min'], CONFIG['r_max'],
                                       CONFIG['phi_min'], CONFIG['phi_max'])
    _last = len(axes) - 1
    vec_artists = [overlay_vectors(ax, first_data, polar=True, add_key=True,
                                   scale_from=_scale_ref)
                   if (CONFIG['vec_panels'] == 'all' or i == _last) else None
                   for i, ax in enumerate(axes)]

    # Add frame number to text
    time_text = fig.text(0.5, 0.94, '', ha='center', fontsize=12, fontweight='bold')
    if CONFIG.get('vectors'):
        fig.text(0.5, 0.015 + info_footer_height("polar"), vector_caption(),
                 ha='center', fontsize=11)

    def update(frame_idx):
        frame = frames_subset[frame_idx]
        data = read_athena_2d(frames_dict[frame])

        # Apply bounds filtering
        data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'],
                                      CONFIG['phi_min'], CONFIG['phi_max'])

        # Update text with frame number
        time_text.set_text(frame_label(frame_idx, len(frames_subset)-1, data['time']))

        for idx, var in enumerate(variables):
            ims[idx].set_array(to_cpu(data[var]).ravel())

        for i, ax in enumerate(axes):
            if CONFIG['vec_panels'] == 'all' or i == _last:
                vec_artists[i] = overlay_vectors(ax, data, polar=True, add_key=False,
                                                 artists=vec_artists[i])

        return ims + [time_text]
    
    print("  Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_subset),
                                 interval=1000/CONFIG['fps'], blit=False)
    
    _margin = draw_model_info(fig, plot="polar")
    plt.tight_layout(rect=(0, _margin, 1, 0.97))  # title, frame counter, model info
    output_video = os.path.join(output_dir, "disk_evolution_polar.mp4")
    ani.save(output_video, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved POLAR animation: {output_video}")
    plt.close()


def create_vector_animation(available_frames, frames_dict, output_dir):
    """The vector field alone, one large polar panel, over a muted density background.

    The four-panel views are for reading fields against each other; this one exists to
    actually look at the flow, so it gets the whole figure.
    """
    print("\nCreating VECTOR FIELD animation...")
    start_idx = 0 if args.start_frame is None else available_frames.index(args.start_frame)
    end_idx = (len(available_frames) if args.end_frame is None
               else available_frames.index(args.end_frame) + 1)
    frames_subset = available_frames[start_idx:end_idx:CONFIG['subsample']]
    print(f"  Animation frames: {len(frames_subset)} (subsampled by {CONFIG['subsample']})")

    bounds = (CONFIG['r_min'], CONFIG['r_max'], CONFIG['phi_min'], CONFIG['phi_max'])
    first = filter_data_by_bounds(read_athena_2d(frames_dict[frames_subset[0]]), *bounds)
    # frame 0 is the initial condition with v = 0; scale the arrows off a real one
    ref = filter_data_by_bounds(read_athena_2d(frames_dict[frames_subset[-1]]), *bounds)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    fig.suptitle(_ad.format_title("Velocity Field", args.title, time=0.0, cycle=0,
                                  frame=0, base=(_ad.INFO or {}).get("base", ""),
                                  plot="vector_animation"),
                 fontsize=16, fontweight='bold')

    rho = to_cpu(first['density'])
    lo = np.percentile(rho[rho > 0], CONFIG['vmin_percentile']) if np.any(rho > 0) else None
    hi = np.percentile(rho, CONFIG['vmax_percentile'])
    bg = ax.pcolormesh(to_cpu(first['Phi']), to_cpu(first['R']), rho, cmap=VEC_BG_CMAP,
                       shading='auto',
                       norm=(LogNorm(vmin=lo, vmax=hi) if lo and lo > 0
                             else Normalize(vmin=rho.min(), vmax=hi)))
    cb = plt.colorbar(bg, ax=ax, pad=0.1, fraction=0.04)
    cb.set_label(r'$\rho$', fontsize=11)
    art = overlay_vectors(ax, first, polar=True, add_key=True, scale_from=ref)
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.035 + info_footer_height("vector"), vector_caption(),
             ha='center', fontsize=12)
    time_text = fig.text(0.5, 0.93, '', ha='center', fontsize=12, fontweight='bold')

    def update(k):
        data = filter_data_by_bounds(read_athena_2d(frames_dict[frames_subset[k]]), *bounds)
        time_text.set_text(frame_label(k, len(frames_subset)-1, data['time']))
        bg.set_array(to_cpu(data['density']).ravel())
        update.art = overlay_vectors(ax, data, polar=True, add_key=False,
                                     artists=update.art)
        return [bg, time_text]

    update.art = art
    draw_model_info(fig, plot="vector")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_subset),
                                  interval=1000 / CONFIG['fps'], blit=False)
    out = os.path.join(output_dir, "velocity_field.mp4")
    ani.save(out, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved VECTOR animation: {out}")
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # Check if frames_dict is valid
    if frames_dict is None:
        print("ERROR: No frames loaded. Cannot proceed.")
        return
    
    # Determine which frame to process
    if args.frame is not None:
        if args.frame in available_frames:
            frames_to_process = [args.frame]
        else:
            print(f"WARNING: Frame {args.frame} not found. Available: {available_frames}")
            return
    else:
        frames_to_process = [available_frames[-1]]  # Last frame by default
    
    print(f"\nProcessing frames: {frames_to_process}")
    
    # Process frames
    for frame_num in frames_to_process:
        print(f"\nProcessing frame {frame_num}...")
        file_list = frames_dict[frame_num]
        if len(file_list) > 1:
            print(f"  Reading {len(file_list)} block(s)...")
        data = read_athena_2d(file_list)
        
        if args.mode == 'vector_animation':
            continue

        if args.mode in ['all', 'heatmaps']:
            plot_2d_heatmap(data, frame_num, output_dir)
        
        if args.mode in ['all', 'radial']:
            plot_radial_profiles(data, frame_num, output_dir)
        
        if args.mode in ['all', 'polar']:
            plot_polar(data, frame_num, output_dir)
        
        if args.mode in ['all', 'azimuthal']:
            plot_azimuthal_profiles(data, frame_num, output_dir)
    
    # Create animations
    # By default (mode='all') create both animations
    if args.animate or args.mode == 'animation':
        create_animation(available_frames, frames_dict, output_dir)
    
    if args.mode == 'all' or args.mode == 'polar_animation' or args.animate:
        create_polar_animation(available_frames, frames_dict, output_dir)

    if args.mode == 'vector_animation':
        if not CONFIG.get('vectors'):
            print("  (vector_animation implies --add-vectors)")
            CONFIG['vectors'] = True
        create_vector_animation(available_frames, frames_dict, output_dir)
    
    print("\n" + "="*80)
    print("DONE! All visualizations saved to:")
    print(f"   {output_dir}")
    print("="*80)

if __name__ == "__main__":
    _setup()
    main()
