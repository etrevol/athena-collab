#!/usr/bin/env python3
"""
=============================================================================
FORCE BALANCE VISUALIZER - Papaloizou-Pringle Disk
=============================================================================

Reads user-output-variable (UOV) .tab files containing per-cell radial
force magnitudes and animates the phi-averaged radial profiles over time.

Two force magnitudes are compared on the same plot at each frame:
  |a_grav|   = beta / r^2                    (gravitational acceleration)
  |a_press|  = |(1/rho) * dp/dr|             (pressure gradient acceleration)

Data is averaged over phi following the radial_profile_frame_*****.png
style from vis2d.py (mean ± 1σ shading).

BASIC USAGE:
    python3 visforces.py
    python3 visforces.py --subsample 5
    python3 visforces.py --fps 15 --log_scale
    python3 visforces.py --r_min 0.5 --r_max 4.5
    python3 visforces.py --out_num 2    # if UOV is <output2> in input file

OUTPUT FILES:
    figs_forces/force_profile_frame_*****.png   individual frames
    figs_forces/force_balance_animation.mp4     assembled video (needs imageio)

REQUIREMENTS (all optional except numpy/matplotlib):
    tqdm              - progress bars        pip install tqdm
    imageio           - MP4 assembly         pip install imageio imageio-ffmpeg

NOTES:
  - The UOV output block must use  variable = uov  in the Athena++ input file.
  - Column layout expected in the .tab file (2D cylindrical, nx3=1):
        col 0: i     col 1: r     col 2: j     col 3: phi
        col 4: grav_accel          col 5: press_accel
  - Adjust --out_num to match whichever <outputN> block contains UOV data.

=============================================================================
"""

import argparse
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Optional: progress bars
try:
    from tqdm import tqdm as _tqdm_impl
    TQDM_AVAILABLE = True
except ImportError:
    _tqdm_impl = None
    TQDM_AVAILABLE = False

def tqdm(iterable, desc="", **kwargs):
    if TQDM_AVAILABLE and _tqdm_impl is not None:
        return _tqdm_impl(iterable, desc=desc, ncols=80, ascii=True, **kwargs)
    return iterable

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'dpi': 200,
    'figsize': (10, 6),
    'style': 'seaborn-v0_8-darkgrid',
    'linewidth': 2.2,
    'alpha': 0.85,
    'grid': True,
    'log_scale': False,
}

# Column indices in UOV tab output and display metadata
FORCE_INFO = {
    'grav_accel': {
        'label': r'$|a_\mathrm{grav}|\ =\ \beta/r^2$',
        'color': 'crimson',
        'col_idx': 4,
    },
    'press_accel': {
        'label': r'$|a_{\nabla P}|\ =\ |(1/\rho)\,\partial p/\partial r|$',
        'color': 'royalblue',
        'col_idx': 5,
    },
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="Radial force balance animation from UOV .tab files",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python3 visforces.py
  python3 visforces.py --subsample 5 --fps 15
  python3 visforces.py --r_min 0.5 --r_max 4.5 --log_scale
  python3 visforces.py --out_num 2
    """,
)
parser.add_argument("--data_dir", default=None,
                    help="Directory with .tab files (default: auto-detect ./data or ../data)")
parser.add_argument("--output_dir", default=None,
                    help="Output directory for frames and video (default: figs_forces/)")
parser.add_argument("--out_num", type=int, default=3,
                    help="Output block number for UOV files, e.g. 3 for <output3> (default: 3)")
parser.add_argument("--subsample", type=int, default=1,
                    help="Render every Nth frame (default: 1 = all)")
parser.add_argument("--start_frame", type=int, default=None,
                    help="First frame number to include (0-based file index in available list)")
parser.add_argument("--end_frame", type=int, default=None,
                    help="Last frame number to include (inclusive)")
parser.add_argument("--fps", type=int, default=10,
                    help="Frames per second for output video (default: 10)")
parser.add_argument("--r_min", type=float, default=None,
                    help="Minimum radius to plot")
parser.add_argument("--r_max", type=float, default=None,
                    help="Maximum radius to plot")
parser.add_argument("--log_scale", action='store_true',
                    help="Logarithmic y-axis scale (default: linear, better for ratio inspection)")
parser.add_argument("--no_grid", action='store_true',
                    help="Disable plot grid (grid is shown by default)")
parser.add_argument("--dpi", type=int, default=CONFIG['dpi'],
                    help="DPI for saved frames (default: 200)")
parser.add_argument("--figsize", nargs=2, type=float, default=list(CONFIG['figsize']),
                    help="Figure size W H in inches (default: 10 6)")
parser.add_argument("--style", default=CONFIG['style'],
                    choices=['default', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-whitegrid',
                             'bmh', 'ggplot', 'classic'],
                    help="Matplotlib style")

args = parser.parse_args()

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if args.data_dir:
    data_dir = args.data_dir
    _output_base = os.path.dirname(os.path.abspath(args.data_dir))
else:
    _candidate = os.path.join(script_dir, "data")
    if not os.path.isdir(_candidate):
        _candidate = os.path.join(os.path.dirname(script_dir), "data")
        _output_base = os.path.dirname(script_dir)
    else:
        _output_base = script_dir
    data_dir = _candidate

output_dir = args.output_dir if args.output_dir else os.path.join(_output_base, "figs_forces")
os.makedirs(output_dir, exist_ok=True)

# --- Update CONFIG from args ---
CONFIG['dpi'] = args.dpi
CONFIG['figsize'] = tuple(args.figsize)
CONFIG['log_scale'] = args.log_scale
CONFIG['grid'] = not args.no_grid

try:
    plt.style.use(args.style)
except OSError:
    plt.style.use('default')

print("=" * 80)
print("FORCE BALANCE VISUALIZER")
print("=" * 80)
print(f"Data directory  : {data_dir}")
print(f"Output directory: {output_dir}")
print(f"UOV output block: out{args.out_num}")
print(f"Subsampling     : every {args.subsample} frame(s)")
print(f"Y-axis scale    : {'logarithmic' if CONFIG['log_scale'] else 'linear'}")
if args.r_min is not None or args.r_max is not None:
    print(f"Radial range    : [{args.r_min}, {args.r_max}]")

# =============================================================================
# FILE DISCOVERY
# =============================================================================

def _parse_uov_filename(filename, out_num):
    """Return (base_name, block_num, frame_num) or None if not a UOV tab file."""
    # Multi-block format: name.blockN.outM.NNNNN.tab
    m = re.match(rf'^(.+)\.block(\d+)\.out{out_num}\.(\d+)\.tab$', filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    # Single-block format: name.outM.NNNNN.tab
    m2 = re.match(rf'^(.+)\.out{out_num}\.(\d+)\.tab$', filename)
    if m2:
        return m2.group(1), 0, int(m2.group(2))
    return None


def group_uov_files_by_frame(data_dir, out_num):
    """Group UOV .tab files by frame number.

    Returns:
        frames_dict : {frame_int: [sorted list of full file paths]}
        base_name   : simulation base name string, or None on failure
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.tab')]
    parsed = []
    for f in all_files:
        result = _parse_uov_filename(f, out_num)
        if result is not None:
            base, block, frame = result
            parsed.append((base, block, frame, os.path.join(data_dir, f)))

    if not parsed:
        return None, None

    frames_dict = {}
    base_names = set()
    for base, block, frame, filepath in parsed:
        base_names.add(base)
        frames_dict.setdefault(frame, []).append((block, filepath))

    for frame in frames_dict:
        frames_dict[frame].sort(key=lambda x: x[0])
        frames_dict[frame] = [fp for _, fp in frames_dict[frame]]

    return frames_dict, (list(base_names)[0] if base_names else 'unknown')

# =============================================================================
# DATA READING
# =============================================================================

def _read_uov_block(filename):
    """Read a single UOV block .tab file.

    Expected column layout (2D cylindrical, nx3 = 1):
        0: i     1: r(x1v)     2: j     3: phi(x2v)
        4: grav_accel          5: press_accel
    """
    with open(filename, 'r', encoding='utf-8') as f:
        header = f.readline()
    time_val  = float(header.split('time=')[1].split()[0])
    cycle_val = int(header.split('cycle=')[1].split()[0])

    raw = np.loadtxt(filename, skiprows=2)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    return {
        'time':        time_val,
        'cycle':       cycle_val,
        'r':           raw[:, 1],
        'phi':         raw[:, 3],
        'grav_accel':  raw[:, FORCE_INFO['grav_accel']['col_idx']],
        'press_accel': raw[:, FORCE_INFO['press_accel']['col_idx']],
    }


def read_uov_frame(file_list):
    """Read and assemble all MeshBlock files for one time frame.

    Returns a dict with:
        time, cycle      : scalars
        r (nr,)          : unique radial coordinates
        phi (nphi,)      : unique azimuthal coordinates
        grav_2d  (nphi, nr)
        press_2d (nphi, nr)
    """
    if isinstance(file_list, str):
        file_list = [file_list]

    blocks = [_read_uov_block(f) for f in file_list]

    all_r     = np.concatenate([b['r']           for b in blocks])
    all_phi   = np.concatenate([b['phi']         for b in blocks])
    all_grav  = np.concatenate([b['grav_accel']  for b in blocks])
    all_press = np.concatenate([b['press_accel'] for b in blocks])

    unique_r   = np.unique(all_r)
    unique_phi = np.unique(all_phi)
    nr, nphi   = len(unique_r), len(unique_phi)

    r_to_idx   = {v: i for i, v in enumerate(unique_r)}
    phi_to_idx = {v: i for i, v in enumerate(unique_phi)}

    grav_2d  = np.zeros((nphi, nr))
    press_2d = np.zeros((nphi, nr))

    for k in range(len(all_r)):
        ri = r_to_idx[all_r[k]]
        pi = phi_to_idx[all_phi[k]]
        grav_2d[pi, ri]  = all_grav[k]
        press_2d[pi, ri] = all_press[k]

    return {
        'time':      blocks[0]['time'],
        'cycle':     blocks[0]['cycle'],
        'r':         unique_r,
        'phi':       unique_phi,
        'grav_2d':   grav_2d,
        'press_2d':  press_2d,
    }

# =============================================================================
# FRAME RENDERING
# =============================================================================

def render_frame(frame_idx, n_frames, r,
                 grav_avg, grav_std, press_avg, press_std,
                 time_val, cycle_val):
    """Render one force-balance radial profile frame.

    Style mirrors radial_profile_frame_*****.png from vis2d.py:
    phi-averaged profiles with ±1σ fill_between on the same axes.

    Returns the saved file path.
    """
    fig, ax = plt.subplots(figsize=CONFIG['figsize'], dpi=CONFIG['dpi'])

    grav_info  = FORCE_INFO['grav_accel']
    press_info = FORCE_INFO['press_accel']

    # --- gravity curve ---
    ax.plot(r, grav_avg,
            color=grav_info['color'], linewidth=CONFIG['linewidth'],
            alpha=CONFIG['alpha'], label=grav_info['label'])
    if CONFIG['log_scale']:
        grav_lo = np.maximum(grav_avg - grav_std, 1e-300)
    else:
        grav_lo = grav_avg - grav_std
    ax.fill_between(r, grav_lo, grav_avg + grav_std,
                    color=grav_info['color'], alpha=0.18)

    # --- pressure-gradient curve ---
    ax.plot(r, press_avg,
            color=press_info['color'], linewidth=CONFIG['linewidth'],
            alpha=CONFIG['alpha'], label=press_info['label'])
    if CONFIG['log_scale']:
        press_lo = np.maximum(press_avg - press_std, 1e-300)
    else:
        press_lo = press_avg - press_std
    ax.fill_between(r, press_lo, press_avg + press_std,
                    color=press_info['color'], alpha=0.18)

    if CONFIG['log_scale']:
        ax.set_yscale('log')

    ax.set_xlabel('r  [code units]', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Radial acceleration  $\langle\cdot\rangle_\phi$  [code units]',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        rf'Force Balance Profile  (t = {time_val:.4e},  cycle = {cycle_val})',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.legend(fontsize=12, loc='best', framealpha=0.9)

    if CONFIG['grid']:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    width = len(str(n_frames))
    ax.text(0.98, 0.02, f'Frame {frame_idx + 1:{width}d}/{n_frames}',
            transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom', color='gray')

    plt.tight_layout()
    out_file = os.path.join(output_dir, f'force_profile_frame_{frame_idx:05d}.png')
    plt.savefig(out_file, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    return out_file

# =============================================================================
# MAIN
# =============================================================================

frames_dict, base_name = group_uov_files_by_frame(data_dir, args.out_num)

if not frames_dict:
    print(f"\nERROR: No UOV tab files found in: {data_dir}")
    print(f"  Expected pattern: <name>.block<N>.out{args.out_num}.<frame>.tab")
    print(f"  → Check --out_num matches the <outputN> block in your input file.")
    print(f"  → Make sure the input has:  variable = uov")
    sys.exit(1)

available_frames = sorted(frames_dict.keys())
print(f"\nFound {len(available_frames)} frame(s)  (base: {base_name})")
print(f"Frame range: {available_frames[0]} to {available_frames[-1]}")

# --- apply start/end frame limits (by index into available_frames) ---
s_idx = 0 if args.start_frame is None else max(0, args.start_frame)
e_idx = len(available_frames) if args.end_frame is None else min(args.end_frame + 1, len(available_frames))
frames_subset = available_frames[s_idx:e_idx:args.subsample]
n_frames = len(frames_subset)

print(f"Rendering {n_frames} frame(s) (subsample={args.subsample})")
print()

# --- render loop ---
saved_frames  = []
report_every  = max(1, n_frames // 10)

for frame_idx, frame_num in enumerate(tqdm(frames_subset, desc="Rendering")):
    data = read_uov_frame(frames_dict[frame_num])

    r    = data['r']
    grav = data['grav_2d']
    pres = data['press_2d']

    # --- radial bounds ---
    mask = np.ones(len(r), dtype=bool)
    if args.r_min is not None:
        mask &= (r >= args.r_min)
    if args.r_max is not None:
        mask &= (r <= args.r_max)
    r    = r[mask]
    grav = grav[:, mask]
    pres = pres[:, mask]

    # --- phi-average (axis 0 = phi direction) ---
    grav_avg,  grav_std  = np.mean(grav, axis=0), np.std(grav, axis=0)
    press_avg, press_std = np.mean(pres, axis=0), np.std(pres, axis=0)

    out_file = render_frame(
        frame_idx, n_frames, r,
        grav_avg, grav_std, press_avg, press_std,
        data['time'], data['cycle'],
    )
    saved_frames.append(out_file)

    if not TQDM_AVAILABLE and (frame_idx + 1) % report_every == 0:
        w = len(str(n_frames))
        print(f"  [{frame_idx + 1:{w}d}/{n_frames}]  t = {data['time']:.4e}")

print(f"\nSaved {len(saved_frames)} frame(s) to: {output_dir}")

# --- assemble MP4 ---
video_file = os.path.join(output_dir, "force_balance_animation.mp4")
try:
    import imageio  # type: ignore
    print(f"Assembling video ({args.fps} fps): {video_file}")
    writer = imageio.get_writer(video_file, fps=args.fps)
    for frame_file in saved_frames:
        writer.append_data(imageio.imread(frame_file))
    writer.close()
    print(f"  ✓ Video saved: {video_file}")
except ImportError:
    print("  Note: install imageio + imageio-ffmpeg to generate MP4")
    print("        pip install imageio imageio-ffmpeg")
except Exception as exc:
    print(f"  Note: video assembly failed — {exc}")

print("\n" + "=" * 80)
print("DONE")
print(f"  Frames : {output_dir}/force_profile_frame_*****.png")
print(f"  Video  : {video_file}")
print("=" * 80)
