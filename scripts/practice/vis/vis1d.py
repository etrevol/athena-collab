#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ 1D RADIAL PROFILE VISUALIZER
=============================================================================

Visualize 1D radial profiles from Athena++ simulations with static plots 
and animations showing temporal evolution.

BASIC USAGE:
    python3 vis1d.py
        Creates radial profile plot for last frame + evolution animation

OPTIONS:
    --mode {all,profiles,animation}
        all        - static profiles + animation (default)
        profiles   - only static radial profiles
        animation  - only evolution animation
    
    --frame N        - process specific frame (default: last)
    --fps N          - FPS for animations (default: 10)
    --data_dir PATH  - directory with .tab files
    --output_dir PATH - output directory (default: ./figs_1d)
    --subsample N    - use every Nth frame for animation (default: 1)
    --start_frame N  - first frame for animation
    --end_frame N    - last frame for animation
    --logscale       - use log scale for density and pressure
    
    --r_min FLOAT    - minimum radius for plotting
    --r_max FLOAT    - maximum radius for plotting

EXAMPLES:
    # Basic usage
    python3 vis1d.py
    python3 vis1d.py --mode profiles --frame 10
    python3 vis1d.py --mode animation --subsample 5
    python3 vis1d.py --logscale
    
    # Limit radial range
    python3 vis1d.py --r_min 0.1 --r_max 2.0

=============================================================================
"""

import os
import re
import sys
import argparse

import athena_data as _ad
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import warnings
warnings.filterwarnings('ignore')

# Progress bar support
try:
    from tqdm import tqdm as _tqdm_impl
    TQDM_AVAILABLE = True
except ImportError:
    _tqdm_impl = None
    TQDM_AVAILABLE = False
    print("✗ tqdm not found - no progress bars")
    print("  Install with: pip install tqdm")

def tqdm(iterable, desc="", **kwargs):
    """Wrapper for tqdm with default settings or passthrough if not available"""
    if TQDM_AVAILABLE and _tqdm_impl is not None:
        return _tqdm_impl(iterable, desc=desc, ncols=80, ascii=True, **kwargs)
    else:
        return iterable

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'dpi': 200,
    'figsize': (12, 10),
    'fps': 10,
    'subsample': 1,
    'logscale': False,
}

VARIABLE_INFO = {
    'density': {
        'label': r'Density $\rho$',
        'color': 'red',
        'marker': 'o',
        'linestyle': '-',
    },
    'pressure': {
        'label': r'Pressure $P$',
        'color': 'blue',
        'marker': 's',
        'linestyle': '-',
    },
    'vel_r': {
        'label': r'Radial Velocity $v_r$',
        'color': 'green',
        'marker': '^',
        'linestyle': '-',
    },
    'vel_phi': {
        'label': r'Azimuthal Velocity $v_\phi$',
        'color': 'purple',
        'marker': 'd',
        'linestyle': '-',
    },
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="1D Athena++ radial profile visualization",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--data_dir", default=None, 
                    help="Directory with .athdf or .tab files (default: ../data)")
parser.add_argument("--format", default=None, choices=["athdf", "tab"],
                    help="Force an input format (default: auto, prefers .athdf)")
parser.add_argument("--output_id", type=int, default=1,
                    help="Athena++ output block id to read (default: 1 = prim)")
parser.add_argument("--title", default=None,
                    help="Figure title (centred, top). Supports {time}, {cycle}, {frame}, {base}; e.g. --title \"PP disk, t={time:.3f}\"")
parser.add_argument("--output_dir", default=None, 
                    help="Output directory (default: ../figs_1d)")
parser.add_argument("--mode", default="all",
                    choices=['all', 'profiles', 'animation'],
                    help="Visualization mode")
parser.add_argument("--frame", type=int, default=None,
                    help="Frame number to visualize (default: last frame)")
parser.add_argument("--fps", type=int, default=CONFIG['fps'],
                    help="FPS for animation")
parser.add_argument("--subsample", type=int, default=CONFIG['subsample'],
                    help="Use every Nth frame for animations (1=all frames)")
parser.add_argument("--start_frame", type=int, default=None,
                    help="First frame for animation")
parser.add_argument("--end_frame", type=int, default=None,
                    help="Last frame for animation")
parser.add_argument("--logscale", action='store_true',
                    help="Use log scale for density and pressure")
parser.add_argument("--r_min", type=float, default=None,
                    help="Minimum radius for plotting")
parser.add_argument("--r_max", type=float, default=None,
                    help="Maximum radius for plotting")

args = parser.parse_args()

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = args.data_dir if args.data_dir else os.path.join(script_dir, "data")
output_dir = args.output_dir if args.output_dir else os.path.join(script_dir, "figs_1d")
os.makedirs(output_dir, exist_ok=True)

CONFIG['fps'] = args.fps
CONFIG['subsample'] = args.subsample
CONFIG['logscale'] = args.logscale
CONFIG['r_min'] = args.r_min
CONFIG['r_max'] = args.r_max

print("="*80)
print("ATHENA++ 1D RADIAL PROFILE VISUALIZER")
print("="*80)
print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")
print(f"Mode: {args.mode}")
print(f"Frame subsampling: {CONFIG['subsample']}")
print(f"Log scale: {'✓ ENABLED' if CONFIG['logscale'] else '✗ DISABLED'}")
if args.r_min is not None or args.r_max is not None:
    r_range = f"r: [{args.r_min if args.r_min is not None else 'auto'}:{args.r_max if args.r_max is not None else 'auto'}]"
    print(f"Radial range: {r_range}")

# =============================================================================
# FILE PARSING AND GROUPING
# =============================================================================

def parse_filename(filename):
    """Parse Athena++ filename to extract base name, block number, and frame number"""
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

    Kept for call compatibility: returns (frames_dict, base_name, num_blocks,
    is_multiblock) exactly as the .tab-only version did.
    """
    info = _ad.discover(data_dir, output_id=args.output_id, prefer=args.format)
    _ad.INFO = info
    print(f"  Input: {_ad.describe(info)}")
    return info["frames"], info["base"], info["nblocks"], info["nblocks"] > 1


# =============================================================================
# DATA READING FUNCTIONS
# =============================================================================

def read_athena_1d(file_list):
    """One frame as a radial profile, from .athdf or .tab.

    Returns the same keys as before: time, cycle, r, density, pressure,
    vel_r, vel_phi, vel_z.
    """
    if isinstance(file_list, str):
        file_list = [file_list]
    return _ad.frame_1d(file_list, _ad.INFO["format"])


def filter_data_by_radius(data, r_min=None, r_max=None):
    """Filter 1D data by radial bounds"""
    r = data['r']
    
    # Determine indices to keep
    mask = np.ones(len(r), dtype=bool)
    if r_min is not None:
        mask &= (r >= r_min)
    if r_max is not None:
        mask &= (r <= r_max)
    
    # If no filtering needed, return original data
    if np.all(mask):
        return data
    
    # Create filtered data
    filtered_data = {
        'time': data['time'],
        'cycle': data['cycle'],
        'r': r[mask],
        'density': data['density'][mask],
        'pressure': data['pressure'][mask],
        'vel_r': data['vel_r'][mask],
        'vel_phi': data['vel_phi'][mask],
        'vel_z': data['vel_z'][mask],
    }
    
    return filtered_data

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_radial_profiles(data, frame_num, output_dir):
    """Create radial profile plots for all variables"""
    print(f"  Creating radial profiles for frame {frame_num}...")
    
    # Apply bounds filtering
    data = filter_data_by_radius(data, CONFIG['r_min'], CONFIG['r_max'])
    
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['figsize'])
    fig.suptitle(_ad.format_title(
                     f"Radial Profiles (t={data['time']:.3f}, cycle={data['cycle']})",
                     args.title, time=data['time'], cycle=data['cycle'],
                     frame=frame_num, base=(_ad.INFO or {}).get("base", "")),
                 fontsize=16, fontweight='bold')
    
    r = data['r']
    
    variables = [
        ('density', axes[0, 0]),
        ('pressure', axes[0, 1]),
        ('vel_r', axes[1, 0]),
        ('vel_phi', axes[1, 1]),
    ]
    
    for var, ax in variables:
        info = VARIABLE_INFO[var]
        ax.plot(r, data[var], color=info['color'], linewidth=2, 
                marker=info['marker'], markersize=4, linestyle=info['linestyle'],
                label=info['label'])
        ax.set_xlabel('Radius r', fontsize=12)
        ax.set_ylabel(info['label'], fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Log scale for density and pressure if requested
        if CONFIG['logscale'] and var in ['density', 'pressure']:
            ax.set_yscale('log')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"radial_profile_frame_{frame_num:05d}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def create_radial_evolution_animation(available_frames, frames_dict, output_dir):
    """Create animation showing evolution of radial profiles over time"""
    print(f"\nCreating radial profile evolution animation...")
    
    # Apply subsampling and frame range
    start_idx = 0 if args.start_frame is None else available_frames.index(args.start_frame)
    end_idx = len(available_frames) if args.end_frame is None else available_frames.index(args.end_frame) + 1
    frames_subset = available_frames[start_idx:end_idx:CONFIG['subsample']]
    
    print(f"  Total frames: {len(available_frames)}")
    print(f"  Animation frames: {len(frames_subset)} (subsampled by {CONFIG['subsample']})")
    
    # Read all frames to determine global ranges
    print("  Reading all frames for global ranges...")
    all_data = []
    for frame in tqdm(frames_subset, desc="  Loading frames"):
        data = read_athena_1d(frames_dict[frame])
        data = filter_data_by_radius(data, CONFIG['r_min'], CONFIG['r_max'])
        all_data.append(data)
    
    # Compute global ranges
    print("  Computing global ranges...")
    global_ranges = {}
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']
    
    for var in variables:
        all_values = np.concatenate([d[var] for d in all_data])
        if CONFIG['logscale'] and var in ['density', 'pressure']:
            # Filter out zeros/negatives for log scale
            all_values = all_values[all_values > 0]
            global_ranges[var] = (all_values.min() * 0.5, all_values.max() * 2.0)
        else:
            vrange = all_values.max() - all_values.min()
            vmargin = vrange * 0.1
            global_ranges[var] = (all_values.min() - vmargin, all_values.max() + vmargin)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['figsize'])
    fig.suptitle(_ad.format_title("Radial Profile Evolution", args.title,
                                  time=0.0, cycle=0, frame=0, base=(_ad.INFO or {}).get("base", "")),
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Initialize plots
    lines = {}
    for idx, var in enumerate(variables):
        ax = axes.flatten()[idx]
        info = VARIABLE_INFO[var]
        
        # Initial empty plot
        line, = ax.plot([], [], color=info['color'], linewidth=2, 
                       marker=info['marker'], markersize=4, linestyle=info['linestyle'],
                       label=info['label'])
        lines[var] = line
        
        ax.set_xlabel('Radius r', fontsize=12)
        ax.set_ylabel(info['label'], fontsize=12)
        ax.set_title(info['label'], fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(all_data[0]['r'].min(), all_data[0]['r'].max())
        ax.set_ylim(global_ranges[var])
        
        if CONFIG['logscale'] and var in ['density', 'pressure']:
            ax.set_yscale('log')
    
    # Add time text
    time_text = fig.text(0.5, 0.94, '', ha='center', fontsize=12, fontweight='bold')
    
    def update(frame_idx):
        data = all_data[frame_idx]
        r = data['r']
        
        for var in variables:
            lines[var].set_data(r, data[var])
        
        time_text.set_text(f"Frame {frame_idx:3d}/{len(frames_subset)-1} | Time = {data['time']:5.2f}")
        
        return list(lines.values()) + [time_text]
    
    print("  Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_subset),
                                 interval=1000/CONFIG['fps'], blit=False)
    
    plt.tight_layout(rect=(0, 0, 1, 0.97))  # Leave space for title and text
    output_video = os.path.join(output_dir, "radial_profile_evolution.mp4")
    ani.save(output_video, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved animation: {output_video}")
    plt.close()

# =============================================================================
# FIND AND GROUP FILES
# =============================================================================
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

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # Safety check (should never happen due to earlier exit)
    if frames_dict is None:
        print(f"ERROR: No frames dictionary available!")
        exit(1)
    
    # Determine which frame to process for static plots
    if args.frame is not None:
        if args.frame not in available_frames:
            print(f"ERROR: Frame {args.frame} not found!")
            exit(1)
        frames_to_process = [args.frame]
    else:
        frames_to_process = [available_frames[-1]]  # Last frame
    
    print(f"\nProcessing frames for static plots: {frames_to_process}")
    
    # Process frames for static plots
    if args.mode in ['all', 'profiles']:
        for frame_num in frames_to_process:
            print(f"\nFrame {frame_num}:")
            data = read_athena_1d(frames_dict[frame_num])
            plot_radial_profiles(data, frame_num, output_dir)
    
    # Create animation
    if args.mode in ['all', 'animation']:
        create_radial_evolution_animation(available_frames, frames_dict, output_dir)
    
    print("\n" + "="*80)
    print("DONE! All visualizations saved to:")
    print(f"   {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
