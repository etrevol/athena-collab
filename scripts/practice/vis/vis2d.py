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
    --mode {all,heatmaps,radial,polar,azimuthal,animation,polar_animation}
        all             - all plots + both animations (default)
        heatmaps        - 2D heatmaps of all variables
        radial          - radial profiles (averaged over φ)
        polar           - polar coordinate plots
        azimuthal       - azimuthal profiles (at different radii)
        animation       - Cartesian animation
        polar_animation - polar animation
    
    --frame N        - process specific frame (default: last)
    --fps N          - FPS for animations (default: 10)
    --data_dir PATH  - directory with .tab files
    --output_dir PATH - output directory (default: ./figs_2d)
    --use_gpu        - enable GPU acceleration (auto-enabled if CuPy available)
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

EXAMPLES:
    # Basic usage
    python3 vis2d.py --subsample 10
    python3 vis2d.py --mode heatmaps --frame 10
    python3 vis2d.py --fps 15
    python3 vis2d.py --mode polar_animation
    python3 vis2d.py --mode animation --use_gpu --num_workers 8
    
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

=============================================================================
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("✗ CuPy not found - using CPU only")
    print("  Install with: pip install cupy-cuda12x  (adjust for your CUDA version)")

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
    'colormap': 'viridis',  # inferno, plasma, magma, viridis, coolwarm
    'dpi': 200,
    'figsize_single': (10, 8),
    'figsize_multi': (16, 12),
    'fps': 10,
    'log_scale_vars': ['density', 'pressure'],
    'radial_bins': 100,
    'polar_resolution': 200,
    'use_gpu': True,  # Auto-enable if CuPy available
    'num_workers': max(1, cpu_count() - 2),  # Leave 2 cores for system
    'subsample': 1,
    'chunk_size': 100,  # Process frames in chunks to save memory
}

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
parser.add_argument("--output_dir", default=None, 
                    help="Output directory (default: ../figs_2d)")
parser.add_argument("--mode", default="all",
                    choices=['all', 'heatmaps', 'radial', 'polar', 'azimuthal', 'animation', 'polar_animation'],
                    help="Visualization mode")
parser.add_argument("--frame", type=int, default=None,
                    help="Frame number to visualize (default: last frame)")
parser.add_argument("--animate", action='store_true',
                    help="Create animation")
parser.add_argument("--fps", type=int, default=CONFIG['fps'],
                    help="FPS for animation")
parser.add_argument("--colormap", default=CONFIG['colormap'],
                    help="Colormap for visualization")
parser.add_argument("--use_gpu", action='store_true',
                    help="Use GPU acceleration (requires CuPy)")
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
parser.add_argument("--vmin_percentile", type=float, default=2.0,
                    help="Lower percentile for color scale (default: 2.0, filters extreme low values)")
parser.add_argument("--vmax_percentile", type=float, default=98.0,
                    help="Upper percentile for color scale (default: 98.0, filters extreme high values)")

args = parser.parse_args()

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if args.data_dir:
    data_dir = args.data_dir
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
CONFIG['use_gpu'] = args.use_gpu and GPU_AVAILABLE
CONFIG['num_workers'] = min(args.num_workers, cpu_count())
CONFIG['subsample'] = args.subsample
CONFIG['r_min'] = args.r_min
CONFIG['r_max'] = args.r_max
CONFIG['phi_min'] = args.phi_min
CONFIG['phi_max'] = args.phi_max
CONFIG['vmin_percentile'] = args.vmin_percentile
CONFIG['vmax_percentile'] = args.vmax_percentile

print("="*80)
print("ATHENA++ 2D VISUALIZER")
print("="*80)
print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")
print(f"Mode: {args.mode}")
print(f"GPU acceleration: {'✓ ENABLED' if CONFIG['use_gpu'] else '✗ DISABLED'}")
print(f"CPU workers: {CONFIG['num_workers']} (parallel frame processing)")
print(f"Frame subsampling: {CONFIG['subsample']}")
if args.r_min is not None or args.r_max is not None:
    r_range = f"r: [{args.r_min if args.r_min is not None else 'auto'}:{args.r_max if args.r_max is not None else 'auto'}]"
    print(f"Radial range: {r_range}")
if args.phi_min is not None or args.phi_max is not None:
    phi_range = f"φ: [{args.phi_min if args.phi_min is not None else 'auto'}:{args.phi_max if args.phi_max is not None else 'auto'}] rad"
    print(f"Azimuthal range: {phi_range}")
print(f"Color scale percentiles: [{args.vmin_percentile}, {args.vmax_percentile}]")

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
    """Group all .tab files by frame number
    
    Returns:
        dict: {frame_number: [list of full file paths for all blocks]}
        str: base name of the simulation
        int: number of blocks
        bool: whether files use multi-block format
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.tab')]
    
    # Parse all files
    parsed_files = []
    has_blocks = False
    for f in all_files:
        base, block, frame, is_multiblock = parse_filename(f)
        if base is not None:
            parsed_files.append((base, block, frame, is_multiblock, f))
            if is_multiblock:
                has_blocks = True
    
    if not parsed_files:
        return None, None, 0, False
    
    # Group by frame
    frames_dict = {}
    base_names = set()
    max_block = 0
    
    for base, block, frame, is_multiblock, filename in parsed_files:
        base_names.add(base)
        if block is not None:
            max_block = max(max_block, block)
        
        if frame not in frames_dict:
            frames_dict[frame] = []
        frames_dict[frame].append((block if block is not None else 0, os.path.join(data_dir, filename)))
    
    # Sort files within each frame by block number
    for frame in frames_dict:
        frames_dict[frame].sort(key=lambda x: x[0])
        frames_dict[frame] = [filepath for _, filepath in frames_dict[frame]]
    
    base_name = list(base_names)[0] if base_names else 'unknown'
    num_blocks = max_block + 1 if has_blocks else 1
    
    return frames_dict, base_name, num_blocks, has_blocks

# =============================================================================
# FIND AND GROUP FILES
# =============================================================================
frames_dict, base_name, num_blocks, is_multiblock = group_files_by_frame(data_dir)

if not frames_dict:
    print(f"ERROR: No valid .tab files found in {data_dir}")
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
# DATA READING AND PROCESSING FUNCTIONS
# =============================================================================
def to_gpu(array):
    """Transfer array to GPU if GPU is enabled"""
    if CONFIG['use_gpu'] and cp is not None:
        return cp.asarray(array)
    return array

def to_cpu(array):
    """Transfer array back to CPU"""
    if CONFIG['use_gpu'] and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

def read_athena_2d_single_block(filename, use_gpu=False):
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

def read_athena_2d(file_list, use_gpu=False):
    """Read and combine 2D Athena++ files from multiple blocks
    
    Args:
        file_list: list of file paths (one per block) or single file path
        use_gpu: whether to use GPU acceleration
    
    Returns:
        dict with combined data from all blocks
    """
    # Handle both single file and list of files
    if isinstance(file_list, str):
        file_list = [file_list]
    
    # Read all blocks
    blocks_data = []
    for filename in file_list:
        block_data = read_athena_2d_single_block(filename, use_gpu=False)
        blocks_data.append(block_data)
    
    # Use time and cycle from first block (should be same for all)
    time = blocks_data[0]['time']
    cycle = blocks_data[0]['cycle']
    
    # Concatenate data from all blocks
    all_r = np.concatenate([bd['r'] for bd in blocks_data])
    all_phi = np.concatenate([bd['phi'] for bd in blocks_data])
    all_rho = np.concatenate([bd['rho'] for bd in blocks_data])
    all_press = np.concatenate([bd['press'] for bd in blocks_data])
    all_vel_r = np.concatenate([bd['vel_r'] for bd in blocks_data])
    all_vel_phi = np.concatenate([bd['vel_phi'] for bd in blocks_data])
    all_vel_z = np.concatenate([bd['vel_z'] for bd in blocks_data])
    
    # Determine unique r and phi values
    unique_r = np.unique(all_r)
    unique_phi = np.unique(all_phi)
    nr = len(unique_r)
    nphi = len(unique_phi)
    
    # Create mapping for r and phi to indices
    r_to_idx = {r_val: idx for idx, r_val in enumerate(unique_r)}
    phi_to_idx = {phi_val: idx for idx, phi_val in enumerate(unique_phi)}
    
    # Create 2D arrays
    R_2d = np.zeros((nphi, nr))
    Phi_2d = np.zeros((nphi, nr))
    rho_2d = np.zeros((nphi, nr))
    press_2d = np.zeros((nphi, nr))
    vel_r_2d = np.zeros((nphi, nr))
    vel_phi_2d = np.zeros((nphi, nr))
    vel_z_2d = np.zeros((nphi, nr))
    
    # Fill 2D arrays
    for i in range(len(all_r)):
        r_idx = r_to_idx[all_r[i]]
        phi_idx = phi_to_idx[all_phi[i]]
        
        R_2d[phi_idx, r_idx] = all_r[i]
        Phi_2d[phi_idx, r_idx] = all_phi[i]
        rho_2d[phi_idx, r_idx] = all_rho[i]
        press_2d[phi_idx, r_idx] = all_press[i]
        vel_r_2d[phi_idx, r_idx] = all_vel_r[i]
        vel_phi_2d[phi_idx, r_idx] = all_vel_phi[i]
        vel_z_2d[phi_idx, r_idx] = all_vel_z[i]
    
    # Transfer to GPU if requested
    def to_gpu_if_needed(arr):
        return to_gpu(arr) if use_gpu else arr
    
    result = {
        'time': time,
        'cycle': cycle,
        'r': unique_r,  # Keep on CPU for matplotlib
        'phi': unique_phi,  # Keep on CPU for matplotlib
        'nr': nr,
        'nphi': nphi,
        'R': to_gpu_if_needed(R_2d),
        'Phi': to_gpu_if_needed(Phi_2d),
        'density': to_gpu_if_needed(rho_2d),
        'pressure': to_gpu_if_needed(press_2d),
        'vel_r': to_gpu_if_needed(vel_r_2d),
        'vel_phi': to_gpu_if_needed(vel_phi_2d),
        'vel_z': to_gpu_if_needed(vel_z_2d),
    }
    
    return result

def compute_statistics_gpu(data_array, axis=0):
    """Compute mean and std using GPU if available"""
    if CONFIG['use_gpu'] and cp is not None:
        mean = cp.mean(data_array, axis=axis)
        std = cp.std(data_array, axis=axis)
        return to_cpu(mean), to_cpu(std)
    else:
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
        return data
    
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
    
    return filtered_data

# =============================================================================
# PARALLEL PROCESSING HELPERS (must be at module level for pickling)
# =============================================================================
def _read_frame_for_sampling(args_tuple):
    """Helper function for parallel frame reading (must be top-level for multiprocessing)"""
    frame, file_list, var, use_gpu = args_tuple
    try:
        data = read_athena_2d(file_list, use_gpu=False)  # Disable GPU in workers
        var_data = to_cpu(data[var]).ravel()
        return var_data
    except Exception as e:
        print(f"Warning: Failed to read frame {frame}: {e}")
        return None

def _compute_frame_stats(args_tuple):
    """Helper function for parallel statistics computation"""
    file_list, use_gpu = args_tuple
    try:
        data = read_athena_2d(file_list, use_gpu=False)
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
    fig.suptitle(f"2D Disk Structure (t={data['time']:.3f}, cycle={data['cycle']})", 
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
    
    # Remove last subplot
    axes.flatten()[5].remove()
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"heatmap_frame_{frame_num:05d}.png")
    plt.savefig(filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def plot_radial_profiles(data, frame_num, output_dir):
    """Radial profiles (averaged over phi) with GPU acceleration"""
    print(f"  Creating radial profiles for frame {frame_num}...")
    
    # Apply bounds filtering
    data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'], 
                                  CONFIG['phi_min'], CONFIG['phi_max'])
    
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['figsize_single'])
    fig.suptitle(f"Radial Profiles (averaged over φ) - t={data['time']:.3f}", 
                 fontsize=14, fontweight='bold')
    
    r = data['r']
    
    # Average over phi
    variables = [
        ('density', axes[0, 0], 'r-', r'$\langle\rho\rangle_\phi$'),
        ('pressure', axes[0, 1], 'g-', r'$\langle P\rangle_\phi$'),
        ('vel_r', axes[1, 0], 'b-', r'$\langle v_r\rangle_\phi$'),
        ('vel_phi', axes[1, 1], 'm-', r'$\langle v_\phi\rangle_\phi$'),
    ]
    
    for var, ax, color, label in variables:
        # Use GPU for statistics if available
        var_avg, var_std = compute_statistics_gpu(data[var], axis=0)
        
        ax.plot(r, var_avg, color, linewidth=2, label=label)
        ax.fill_between(r, var_avg - var_std, var_avg + var_std, 
                        alpha=0.3, color=color[0])
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
    fig.suptitle(f"Polar View of Disk - t={data['time']:.3f}", 
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
    fig.suptitle(f"Azimuthal Profiles at Different Radii - t={data['time']:.3f}", 
                 fontsize=14, fontweight='bold')
    
    phi = data['phi']
    r = data['r']
    
    # Select several radii
    r_indices = [len(r)//4, len(r)//2, 3*len(r)//4]
    colors = ['r', 'g', 'b']
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Disk Evolution", fontsize=16, fontweight='bold')
    
    # Read first frame for initialization
    first_data = read_athena_2d(frames_dict[frames_subset[0]], 
                                 use_gpu=CONFIG['use_gpu'])
    
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
        args_list = [(frame, frames_dict[frame], var, CONFIG['use_gpu']) 
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
                data = read_athena_2d(frames_dict[frame], use_gpu=CONFIG['use_gpu'])
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
    
    # Add frame number to text
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12, fontweight='bold')
    
    def update(frame_idx):
        frame = frames_subset[frame_idx]
        data = read_athena_2d(frames_dict[frame], 
                             use_gpu=CONFIG['use_gpu'])
        
        # Apply bounds filtering
        data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'],
                                      CONFIG['phi_min'], CONFIG['phi_max'])
        
        # Update text with frame number
        time_text.set_text(f"Frame {frame_idx:3d}/{len(frames_subset)-1} | Time = {data['time']:5.2f}")
        
        for idx, var in enumerate(variables):
            ims[idx].set_array(to_cpu(data[var]).ravel())
        
        return ims + [time_text]
    
    print("  Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_subset),
                                 interval=1000/CONFIG['fps'], blit=False)
    
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
    fig.suptitle("Polar View of Disk Evolution", fontsize=16, fontweight='bold', y=0.98)
    
    # Read first frame to determine normalization
    first_data = read_athena_2d(frames_dict[frames_subset[0]],
                                use_gpu=CONFIG['use_gpu'])
    
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
        args_list = [(frame, frames_dict[frame], var, CONFIG['use_gpu']) 
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
                data = read_athena_2d(frames_dict[frame], use_gpu=CONFIG['use_gpu'])
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
    
    # Add frame number to text
    time_text = fig.text(0.5, 0.94, '', ha='center', fontsize=12, fontweight='bold')
    
    def update(frame_idx):
        frame = frames_subset[frame_idx]
        data = read_athena_2d(frames_dict[frame],
                             use_gpu=CONFIG['use_gpu'])
        
        # Apply bounds filtering
        data = filter_data_by_bounds(data, CONFIG['r_min'], CONFIG['r_max'],
                                      CONFIG['phi_min'], CONFIG['phi_max'])
        
        # Update text with frame number
        time_text.set_text(f"Frame {frame_idx:3d}/{len(frames_subset)-1} | Time = {data['time']:5.2f}")
        
        for idx, var in enumerate(variables):
            ims[idx].set_array(to_cpu(data[var]).ravel())
        
        return ims + [time_text]
    
    print("  Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames_subset),
                                 interval=1000/CONFIG['fps'], blit=False)
    
    plt.tight_layout(rect=(0, 0, 1, 0.97))  # Leave space for title and text
    output_video = os.path.join(output_dir, "disk_evolution_polar.mp4")
    ani.save(output_video, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved POLAR animation: {output_video}")
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
        data = read_athena_2d(file_list, use_gpu=CONFIG['use_gpu'])
        
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
    
    print("\n" + "="*80)
    print("DONE! All visualizations saved to:")
    print(f"   {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
