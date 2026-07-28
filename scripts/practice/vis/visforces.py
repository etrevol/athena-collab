#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ RADIAL FORCE BALANCE VISUALIZER
=============================================================================

Visualize the three forces governing Papaloizou-Pringle disk equilibrium,
averaged over the azimuthal direction (phi), from uov .tab output files.

Forces (per unit mass, radial direction):
  f_grav   = -beta / r^2                 (gravity, from NewtonianGravity source)
  f_centr  = v_phi^2 / r                 (centrifugal)
  f_press  = -(1/rho) * dP/dr            (pressure gradient)
  f_sum    = f_grav + f_centr + f_press  (net force, ~0 in equilibrium)

BASIC USAGE:
    python3 visforces.py
        Plots forces for the last available frame

OPTIONS:
    --mode {frame,animation,sum,sum_animation,all}
        frame           - radial profiles of 3 forces for a single frame
        animation       - animated 3-force radial profiles over all frames
        sum             - radial profile of net force (f_sum) for a single frame
        sum_animation   - animated net force radial profile
        all             - frame + sum for the last frame (default)
    --frame N        - specific frame number to plot (default: last)
    --fps N          - FPS for animation (default: 10)
    --subsample N    - use every Nth frame for animation (default: 1)
    --start_frame N  - first frame for animation
    --end_frame N    - last frame for animation
    --data_dir PATH  - directory with uov .tab files
    --output_dir PATH - output directory (default: ../figs_forces)
    --r_min FLOAT    - minimum radius for plotting
    --r_max FLOAT    - maximum radius for plotting
    --num_workers N  - parallel CPU workers for animation (default: auto)
    --log            - use symlog y-axis scale (handles negative forces)
    --linthresh F    - linear threshold for symlog (default: auto-detect from data)

EXAMPLES:
    python3 visforces.py
    python3 visforces.py --mode frame --frame 50
    python3 visforces.py --mode animation --subsample 5 --fps 15
    python3 visforces.py --mode all --r_min 0.5 --r_max 4.5
    python3 visforces.py --mode sum_animation --subsample 10
    python3 visforces.py --mode frame --log
    python3 visforces.py --mode animation --log --linthresh 1e4

NOTES:
    - Reads uov .tab files (output3 in athinput with variable=uov)
    - Column layout: i, r, j, phi, f_grav, f_centr, f_press, f_sum
    - Forces are phi-averaged (mean ± std shown as shaded band)
    - In equilibrium: f_grav + f_centr + f_press ≈ 0

=============================================================================
"""

import argparse

import athena_data as _ad
import os
import re
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# Try tqdm for progress bars
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
    'figsize_forces': (10, 6),
    'figsize_sum': (8, 5),
    'fps': 10,
    'subsample': 1,
    'num_workers': max(1, cpu_count() - 2),
    'r_min': None,
    'r_max': None,
    'log': False,
    'linthresh': None,
}

# Colors for the three forces (consistent throughout)
FORCE_COLORS = {
    'f_grav':  '#d62728',   # Red   — gravity (inward)
    'f_centr': '#1f77b4',   # Blue  — centrifugal (outward)
    'f_press': '#2ca02c',   # Green — pressure gradient
    'f_sum':   '#ff7f0e',   # Orange — net force
}

FORCE_LABELS = {
    'f_grav':  r'$f_\mathrm{grav} = -\beta/r^2$',
    'f_centr': r'$f_\mathrm{centr} = v_\phi^2/r$',
    'f_press': r'$f_\mathrm{press} = -(1/\rho)\,\partial P/\partial r$',
    'f_sum':   r'$f_\mathrm{sum} = f_\mathrm{grav}+f_\mathrm{centr}+f_\mathrm{press}$',
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="Papaloizou-Pringle disk radial force balance visualizer",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--data_dir", default=None,
                    help="Directory with uov .athdf or .tab files (default: auto-detect)")
parser.add_argument("--format", default=None, choices=["athdf", "tab"],
                    help="Force an input format (default: auto, prefers .athdf)")
parser.add_argument("--output_id", type=int, default=2,
                    help="Athena++ output block id holding the uov data (default: 2)")
parser.add_argument("--title", default=None,
                    help="Figure title (centred, top). Supports {time}, {cycle}, {frame}, {base}; e.g. --title \"Force balance, t={time:.4f}\"")
parser.add_argument("--output_dir", default=None,
                    help="Output directory (default: ../figs_forces)")
parser.add_argument("--mode", default="all",
                    choices=['frame', 'animation', 'sum', 'sum_animation', 'all'],
                    help="Visualization mode (default: all)")
parser.add_argument("--frame", type=int, default=None,
                    help="Frame number to visualize (default: last)")
parser.add_argument("--fps", type=int, default=CONFIG['fps'],
                    help="FPS for animation (default: 10)")
parser.add_argument("--subsample", type=int, default=CONFIG['subsample'],
                    help="Use every Nth frame for animation (default: 1)")
parser.add_argument("--start_frame", type=int, default=None,
                    help="First frame for animation")
parser.add_argument("--end_frame", type=int, default=None,
                    help="Last frame for animation")
parser.add_argument("--r_min", type=float, default=None,
                    help="Minimum radius for plotting")
parser.add_argument("--r_max", type=float, default=None,
                    help="Maximum radius for plotting")
parser.add_argument("--num_workers", type=int, default=CONFIG['num_workers'],
                    help="Parallel CPU workers for animation")
parser.add_argument("--log", action='store_true',
                    help="Use symlog y-axis scale (handles negative forces)")
parser.add_argument("--linthresh", type=float, default=None,
                    help="Linear threshold for symlog y-scale (default: auto-detect from data)")

args = parser.parse_args()

# Apply args to CONFIG
CONFIG['fps'] = args.fps
CONFIG['subsample'] = args.subsample
CONFIG['num_workers'] = min(args.num_workers, cpu_count())
CONFIG['r_min'] = args.r_min
CONFIG['r_max'] = args.r_max
CONFIG['log'] = args.log
CONFIG['linthresh'] = args.linthresh

# =============================================================================
# PATH RESOLUTION
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

if args.data_dir:
    data_dir = args.data_dir
    _output_base = script_dir
else:
    # Try to find 'data' relative to script location
    candidates = [
        os.path.join(script_dir, "data"),
        os.path.join(os.path.dirname(script_dir), "data"),
    ]
    data_dir = next((c for c in candidates if os.path.isdir(c)), candidates[0])
    _output_base = os.path.dirname(script_dir) if data_dir == candidates[1] else script_dir

output_dir = args.output_dir if args.output_dir else os.path.join(_output_base, "figs_forces")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("ATHENA++ RADIAL FORCE BALANCE VISUALIZER")
print("=" * 70)
print(f"Data directory:   {data_dir}")
print(f"Output directory: {output_dir}")
print(f"Mode:             {args.mode}")
if args.r_min is not None or args.r_max is not None:
    print(f"Radial range:     [{args.r_min}, {args.r_max}]")
if args.log:
    thresh_str = str(args.linthresh) if args.linthresh is not None else 'auto'
    print(f"Log scale:        symlog (linthresh={thresh_str})")
print()

# =============================================================================
# FILE DISCOVERY — uov .tab files
# =============================================================================
def parse_uov_filename(filename):
    """Parse Athena++ uov filename: <base>.block<N>.out2.<frame>.tab
    or <base>.out2.<frame>.tab (single-block).
    Returns (base, block_or_None, frame_int, is_multiblock) or None.
    """
    # Multi-block format
    m = re.match(r'(.+)\.block(\d+)\.out(\d+)\.(\d+)\.tab$', filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(4)), True
    # Single-block format
    m = re.match(r'(.+)\.out(\d+)\.(\d+)\.tab$', filename)
    if m:
        return m.group(1), None, int(m.group(3)), False
    return None


def discover_uov_files(data_dir):
    """Discover the user-output-variable frames, whatever the format.

    Same return contract as before: (frames_dict, base_name, num_blocks).
    """
    info = _ad.discover(data_dir, output_id=args.output_id, prefer=args.format)
    _ad.INFO = info
    print(f"  Input: {_ad.describe(info)}")
    return info["frames"], info["base"], info["nblocks"]


def _count_data_columns(filepath):
    """Count columns in the first data row (skipping 2 header lines)."""
    with open(filepath, 'r') as fh:
        fh.readline()  # header 1
        fh.readline()  # header 2 (column names)
        line = fh.readline()
    return len(line.split())


# =============================================================================
# DATA READING
# =============================================================================
def read_uov_single_block(filepath):
    """Read one uov .tab block file.
    
    Expected columns (cylindrical 2D):
        0: i
        1: x1v (r)
        2: j
        3: x2v (phi)
        4: f_grav
        5: f_centr
        6: f_press
        7: f_sum
    Returns dict with arrays.
    """
    with open(filepath, 'r') as fh:
        header = fh.readline()
        _col_header = fh.readline()  # skip column names line

    time_str = header.split('time=')[1].split()[0]
    cycle_str = header.split('cycle=')[1].split()[0]
    time = float(time_str)
    cycle = int(cycle_str)

    data = np.loadtxt(filepath, skiprows=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    if ncols < 8:
        raise ValueError(
            f"Expected >= 8 columns in uov file, got {ncols}.\n"
            f"File: {filepath}\n"
            "Check that acc_disk_visc.cpp has AllocateUserOutputVariables(4) "
            "and UserWorkBeforeOutput is implemented."
        )

    return {
        'time': time,
        'cycle': cycle,
        'r': data[:, 1],
        'phi': data[:, 3],
        'f_grav': data[:, 4],
        'f_centr': data[:, 5],
        'f_press': data[:, 6],
        'f_sum': data[:, 7],
    }


def read_uov_frame(file_list):
    """One uov frame, from .athdf or .tab.

    Keys as before: time, cycle, r, phi, nr, nphi, R, Phi,
    f_grav, f_centr, f_press, f_sum.
    """
    if isinstance(file_list, str):
        file_list = [file_list]
    return _ad.frame_2d(file_list, _ad.INFO["format"])


def filter_by_r(data, r_min=None, r_max=None):
    """Return a copy of data filtered to the specified radial range."""
    r = data['r']
    mask = np.ones(len(r), dtype=bool)
    if r_min is not None:
        mask &= (r >= r_min)
    if r_max is not None:
        mask &= (r <= r_max)
    if np.all(mask):
        return data
    result = {k: v for k, v in data.items() if k not in ('r', 'nr')}
    result['r'] = r[mask]
    result['nr'] = int(np.sum(mask))
    for key in ('f_grav', 'f_centr', 'f_press', 'f_sum'):
        if key in data:
            result[key] = data[key][:, mask]
    return result


# =============================================================================
# STATISTICS
# =============================================================================
def phi_average(arr2d):
    """Return (mean, std) averaged over phi-axis (axis=0)."""
    return np.mean(arr2d, axis=0), np.std(arr2d, axis=0)


def _apply_log_scale(ax, arrays, linthresh=None):
    """Apply symlog y-scale. Handles negative forces correctly.
    linthresh defines the linear region around zero; auto-detected if None."""
    if linthresh is None:
        all_abs = np.abs(np.concatenate([np.asarray(a).ravel() for a in arrays]))
        nonzero = all_abs[all_abs > 0]
        linthresh = float(np.percentile(nonzero, 5)) if len(nonzero) > 0 else 1.0
    ax.set_yscale('symlog', linthresh=linthresh)


# =============================================================================
# PLOTTING — single frame, 3 forces
# =============================================================================
def plot_forces_frame(data, frame_num, output_dir, show_sum=False):
    """Plot phi-averaged radial force profiles for one frame.
    
    Args:
        data: dict from read_uov_frame
        frame_num: integer frame number (for filename)
        output_dir: where to save
        show_sum: if True, also plot f_sum alongside the three forces
    """
    data = filter_by_r(data, CONFIG['r_min'], CONFIG['r_max'])
    r = data['r']

    fig, ax = plt.subplots(figsize=CONFIG['figsize_forces'])
    fig.suptitle(
        _ad.format_title(
            f"Radial Force Balance (φ-averaged) — t = {data['time']:.4f}, "
            f"cycle = {data['cycle']}",
            args.title, time=data['time'], cycle=data['cycle'],
            frame=locals().get('frame_num', 0), base=(_ad.INFO or {}).get("base", "")),
        fontsize=13, fontweight='bold'
    )

    forces = ['f_grav', 'f_centr', 'f_press']
    if show_sum:
        forces.append('f_sum')

    for key in forces:
        avg, std = phi_average(data[key])
        color = FORCE_COLORS[key]
        label = FORCE_LABELS[key]
        style = '--' if key == 'f_sum' else '-'
        lw = 1.5 if key == 'f_sum' else 2.0
        ax.plot(r, avg, linestyle=style, linewidth=lw, color=color, label=label)
        ax.fill_between(r, avg - std, avg + std, alpha=0.15, color=color)

    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel(r'$r$ (dimensionless)', fontsize=12)
    ax.set_ylabel(r'Force per unit mass', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    if CONFIG['log']:
        _apply_log_scale(ax, [phi_average(data[k])[0] for k in forces], CONFIG['linthresh'])

    plt.tight_layout()
    suffix = '_all' if show_sum else ''
    filename = os.path.join(output_dir, f"forces_frame_{frame_num:05d}{suffix}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# PLOTTING — single frame, net force only
# =============================================================================
def plot_sum_frame(data, frame_num, output_dir):
    """Plot phi-averaged net force (f_sum) for one frame."""
    data = filter_by_r(data, CONFIG['r_min'], CONFIG['r_max'])
    r = data['r']

    avg, std = phi_average(data['f_sum'])

    fig, ax = plt.subplots(figsize=CONFIG['figsize_sum'])
    fig.suptitle(
        _ad.format_title(
            f"Net Radial Force (φ-averaged) — t = {data['time']:.4f}, "
            f"cycle = {data['cycle']}",
            args.title, time=data['time'], cycle=data['cycle'],
            frame=locals().get('frame_num', 0), base=(_ad.INFO or {}).get("base", "")),
        fontsize=13, fontweight='bold'
    )

    color = FORCE_COLORS['f_sum']
    ax.plot(r, avg, linewidth=2.0, color=color, label=FORCE_LABELS['f_sum'])
    ax.fill_between(r, avg - std, avg + std, alpha=0.2, color=color)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

    ax.set_xlabel(r'$r$ (dimensionless)', fontsize=12)
    ax.set_ylabel(r'$f_\mathrm{sum}$ (force per unit mass)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if CONFIG['log']:
        _apply_log_scale(ax, [avg], CONFIG['linthresh'])

    plt.tight_layout()
    filename = os.path.join(output_dir, f"force_sum_frame_{frame_num:05d}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# ANIMATION HELPERS (module-level for multiprocessing pickling)
# =============================================================================
def _read_frame_worker(args_tuple):
    file_list, = args_tuple
    try:
        return read_uov_frame(file_list)
    except Exception as e:
        print(f"  Warning: failed to read frame: {e}")
        return None


# =============================================================================
# ANIMATION — 3 forces
# =============================================================================
def _get_animation_frames(available_frames, start_frame, end_frame, subsample):
    start_idx = 0
    if start_frame is not None and start_frame in available_frames:
        start_idx = available_frames.index(start_frame)
    end_idx = len(available_frames)
    if end_frame is not None and end_frame in available_frames:
        end_idx = available_frames.index(end_frame) + 1
    return available_frames[start_idx:end_idx:subsample]


def create_forces_animation(available_frames, frames_dict, output_dir,
                             start_frame=None, end_frame=None):
    """Animate phi-averaged profiles of f_grav, f_centr, f_press on one axes."""
    frames_subset = _get_animation_frames(
        available_frames, start_frame, end_frame, CONFIG['subsample']
    )
    print(f"\nCreating forces animation: {len(frames_subset)} frames "
          f"(subsampled by {CONFIG['subsample']})")

    # Read first frame to set up axes limits
    first_data = filter_by_r(
        read_uov_frame(frames_dict[frames_subset[0]]),
        CONFIG['r_min'], CONFIG['r_max']
    )
    r = first_data['r']

    # Compute global y-limits by sampling 10% of frames
    print("  Computing global y-axis limits...")
    sample_step = max(1, len(frames_subset) // 10)
    sampled = frames_subset[::sample_step]
    all_vals = []
    for frame in tqdm(sampled, desc="  Sampling"):
        try:
            d = filter_by_r(read_uov_frame(frames_dict[frame]),
                             CONFIG['r_min'], CONFIG['r_max'])
            for key in ('f_grav', 'f_centr', 'f_press'):
                avg, _ = phi_average(d[key])
                all_vals.append(avg)
        except Exception:
            pass
    if all_vals:
        all_vals = np.concatenate(all_vals)
        ymin = np.percentile(all_vals, 1)
        ymax = np.percentile(all_vals, 99)
        margin = 0.1 * (ymax - ymin)
        ylim = (ymin - margin, ymax + margin)
    else:
        ylim = (-1.0, 1.0)

    fig, ax = plt.subplots(figsize=CONFIG['figsize_forces'])
    ax.set_xlabel(r'$r$ (dimensionless)', fontsize=12)
    ax.set_ylabel(r'Force per unit mass', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, alpha=0.3)

    # Initialize lines and fill_between patches
    lines = {}
    fills = {}
    force_keys = ['f_grav', 'f_centr', 'f_press']
    if CONFIG['log']:
        _apply_log_scale(ax, [phi_average(first_data[k])[0] for k in force_keys], CONFIG['linthresh'])
    for key in force_keys:
        avg, std = phi_average(first_data[key])
        line, = ax.plot(r, avg, linewidth=2.0,
                        color=FORCE_COLORS[key], label=FORCE_LABELS[key])
        fill = ax.fill_between(r, avg - std, avg + std,
                               alpha=0.15, color=FORCE_COLORS[key])
        lines[key] = line
        fills[key] = fill

    ax.legend(fontsize=10, loc='best')
    title = ax.set_title('', fontsize=12, fontweight='bold')

    def update(frame_idx):
        nonlocal fills
        frame = frames_subset[frame_idx]
        try:
            d = filter_by_r(read_uov_frame(frames_dict[frame]),
                            CONFIG['r_min'], CONFIG['r_max'])
        except Exception as e:
            print(f"  Warning: frame {frame}: {e}")
            return list(lines.values()) + [title]

        title.set_text(
            f"Radial Force Balance (φ-averaged) — "
            f"t = {d['time']:.4f}  [{frame_idx+1}/{len(frames_subset)}]"
        )
        for key in force_keys:
            avg, std = phi_average(d[key])
            lines[key].set_ydata(avg)
            # Remove old fill and redraw
            fills[key].remove()
            fills[key] = ax.fill_between(d['r'], avg - std, avg + std,
                                          alpha=0.15, color=FORCE_COLORS[key])
        return list(lines.values()) + [title]

    print("  Rendering animation...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_subset),
        interval=1000 // CONFIG['fps'], blit=False
    )
    outpath = os.path.join(output_dir, "forces_animation.mp4")
    ani.save(outpath, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved: {outpath}")
    plt.close()


# =============================================================================
# ANIMATION — net force only
# =============================================================================
def create_sum_animation(available_frames, frames_dict, output_dir,
                          start_frame=None, end_frame=None):
    """Animate phi-averaged net force (f_sum)."""
    frames_subset = _get_animation_frames(
        available_frames, start_frame, end_frame, CONFIG['subsample']
    )
    print(f"\nCreating net-force animation: {len(frames_subset)} frames "
          f"(subsampled by {CONFIG['subsample']})")

    first_data = filter_by_r(
        read_uov_frame(frames_dict[frames_subset[0]]),
        CONFIG['r_min'], CONFIG['r_max']
    )
    r = first_data['r']

    # Global y-limits
    print("  Computing global y-axis limits...")
    sample_step = max(1, len(frames_subset) // 10)
    sampled = frames_subset[::sample_step]
    all_vals = []
    for frame in tqdm(sampled, desc="  Sampling"):
        try:
            d = filter_by_r(read_uov_frame(frames_dict[frame]),
                             CONFIG['r_min'], CONFIG['r_max'])
            avg, _ = phi_average(d['f_sum'])
            all_vals.append(avg)
        except Exception:
            pass
    if all_vals:
        all_vals = np.concatenate(all_vals)
        yabs = np.percentile(np.abs(all_vals), 99)
        ylim = (-yabs * 1.2, yabs * 1.2)
    else:
        ylim = (-0.1, 0.1)

    fig, ax = plt.subplots(figsize=CONFIG['figsize_sum'])
    ax.set_xlabel(r'$r$ (dimensionless)', fontsize=12)
    ax.set_ylabel(r'$f_\mathrm{sum}$ (force per unit mass)', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, alpha=0.3)
    if CONFIG['log']:
        _apply_log_scale(ax, [phi_average(first_data['f_sum'])[0]], CONFIG['linthresh'])

    color = FORCE_COLORS['f_sum']
    avg0, std0 = phi_average(first_data['f_sum'])
    line, = ax.plot(r, avg0, linewidth=2.0, color=color, label=FORCE_LABELS['f_sum'])
    fill_ref = [ax.fill_between(r, avg0 - std0, avg0 + std0, alpha=0.2, color=color)]
    ax.legend(fontsize=10)
    title = ax.set_title('', fontsize=12, fontweight='bold')

    def update(frame_idx):
        frame = frames_subset[frame_idx]
        try:
            d = filter_by_r(read_uov_frame(frames_dict[frame]),
                            CONFIG['r_min'], CONFIG['r_max'])
        except Exception as e:
            print(f"  Warning: frame {frame}: {e}")
            return [line, title]

        avg, std = phi_average(d['f_sum'])
        line.set_ydata(avg)
        fill_ref[0].remove()
        fill_ref[0] = ax.fill_between(d['r'], avg - std, avg + std,
                                       alpha=0.2, color=color)
        title.set_text(
            f"Net Radial Force (φ-averaged) — "
            f"t = {d['time']:.4f}  [{frame_idx+1}/{len(frames_subset)}]"
        )
        return [line, title]

    print("  Rendering animation...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_subset),
        interval=1000 // CONFIG['fps'], blit=False
    )
    outpath = os.path.join(output_dir, "forces_sum_animation.mp4")
    ani.save(outpath, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved: {outpath}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Discover uov files
    frames_dict, base_name, num_blocks = discover_uov_files(data_dir)
    if not frames_dict:
        print(f"ERROR: No uov .tab files found in: {data_dir}")
        print()
        print("Expected: files with 8 columns (i, r, j, phi, f_grav, f_centr, f_press, f_sum)")
        print("Make sure acc_disk_visc.cpp has AllocateUserOutputVariables(4) and is rebuilt.")
        sys.exit(1)

    available_frames = sorted(frames_dict.keys())
    num_frames = len(available_frames)
    print(f"Found {num_frames} uov frames, {num_blocks} block(s), base: '{base_name}'")
    print(f"Frame range: {available_frames[0]} — {available_frames[-1]}")
    print()

    # Determine target frame
    if args.frame is not None:
        if args.frame not in available_frames:
            print(f"ERROR: Frame {args.frame} not found. "
                  f"Available: {available_frames[0]}–{available_frames[-1]}")
            sys.exit(1)
        target_frame = args.frame
    else:
        target_frame = available_frames[-1]

    mode = args.mode  # frame | animation | sum | sum_animation | all

    # --- Static frame plots ---
    if mode in ('frame', 'all'):
        print(f"Plotting forces for frame {target_frame}...")
        data = read_uov_frame(frames_dict[target_frame])
        plot_forces_frame(data, target_frame, output_dir, show_sum=False)

    if mode in ('sum', 'all'):
        print(f"Plotting net force for frame {target_frame}...")
        data = read_uov_frame(frames_dict[target_frame])
        plot_sum_frame(data, target_frame, output_dir)

    # --- Animations ---
    if mode == 'animation':
        create_forces_animation(
            available_frames, frames_dict, output_dir,
            start_frame=args.start_frame, end_frame=args.end_frame
        )

    if mode == 'sum_animation':
        create_sum_animation(
            available_frames, frames_dict, output_dir,
            start_frame=args.start_frame, end_frame=args.end_frame
        )

    print()
    print("=" * 70)
    print(f"Done. Output saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
