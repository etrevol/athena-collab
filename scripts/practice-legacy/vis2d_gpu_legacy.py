#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ 2D DATA VISUALIZER (LEGACY GPU VERSION)
=============================================================================

DEPRECATED: Use ../vis2d.py instead - it has full multi-block support + GPU.

This legacy version only works with single-block format and lacks modern features.

BASIC USAGE:
    python3 vis2d_gpu_legacy.py
        Creates all plots + 2 animations for the last frame
        GPU-accelerated version (legacy single-block only)

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
    --output_dir PATH - output directory (default: ../figs_2d)
    --use_gpu        - enable GPU acceleration (requires CuPy)
    --num_workers N  - number of parallel workers (default: 4)
    --subsample N    - use every Nth frame for animation (default: 1)
    --start_frame N  - first frame for animation
    --end_frame N    - last frame for animation

EXAMPLES:
    python3 vis2d_gpu_legacy.py --mode heatmaps --frame 10
    python3 vis2d_gpu_legacy.py --fps 15
    python3 vis2d_gpu_legacy.py --mode polar_animation
    python3 vis2d_gpu_legacy.py --use_gpu --subsample 5
    python3 vis2d_gpu_legacy.py --mode animation --use_gpu --num_workers 8

NOTE: This is a legacy version. For modern multi-block support, use ../vis2d.py

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
    'use_gpu': False,
    'num_workers': 4,
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
  python vis2d_gpu_legacy.py
  python vis2d_gpu_legacy.py --mode all
  python vis2d_gpu_legacy.py --mode heatmaps --frame 10
  python vis2d_gpu_legacy.py --mode radial --animate
    """
)
parser.add_argument("--data_dir", default=None, 
                    help="Directory with .tab files (default: current directory)")
parser.add_argument("--output_dir", default=None, 
                    help="Output directory (default: ../figs_2d)")
parser.add_argument("--prefix", default="pp_disk.block0.out1.", 
                    help="File prefix")
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
                    help="Number of parallel workers for rendering")
parser.add_argument("--subsample", type=int, default=CONFIG['subsample'],
                    help="Use every Nth frame for animations (1=all frames)")
parser.add_argument("--start_frame", type=int, default=None,
                    help="First frame for animation")
parser.add_argument("--end_frame", type=int, default=None,
                    help="Last frame for animation")

args = parser.parse_args()

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = args.data_dir if args.data_dir else script_dir
output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(script_dir), "figs_2d")
os.makedirs(output_dir, exist_ok=True)

CONFIG['fps'] = args.fps
CONFIG['colormap'] = args.colormap
CONFIG['use_gpu'] = args.use_gpu and GPU_AVAILABLE
CONFIG['num_workers'] = min(args.num_workers, cpu_count())
CONFIG['subsample'] = args.subsample

print("="*80)
print("ATHENA++ 2D VISUALIZER")
print("="*80)
print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")
print(f"Mode: {args.mode}")
print(f"GPU acceleration: {'✓ ENABLED' if CONFIG['use_gpu'] else '✗ DISABLED'}")
print(f"Parallel workers: {CONFIG['num_workers']}")
print(f"Frame subsampling: {CONFIG['subsample']}")

# =============================================================================
# FIND FILES
# =============================================================================
tab_files = sorted([
    f for f in os.listdir(data_dir)
    if f.startswith(args.prefix) and f.endswith(".tab")
])

if not tab_files:
    print(f"ERROR: No .tab files found in {data_dir}")
    exit(1)

available_frames = sorted([int(f.split('.')[-2]) for f in tab_files])
num_frames = len(available_frames)
print(f"Found {num_frames} data files")

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

def read_athena_2d(filename, use_gpu=False):
    """Read 2D Athena++ file"""
    with open(filename, 'r') as f:
        header = f.readline()
        time_str = header.split('time=')[1].split()[0]
        cycle_str = header.split('cycle=')[1].split()[0]
        time = float(time_str)
        cycle = int(cycle_str)
    
    data = np.loadtxt(filename, skiprows=2)
    
    # Columns: i, x1v(r), j, x2v(phi), rho, press, vel1(r), vel2(phi), vel3(z)
    i_idx = data[:, 0].astype(int)
    r = data[:, 1]
    j_idx = data[:, 2].astype(int)
    phi = data[:, 3]
    rho = data[:, 4]
    press = data[:, 5]
    vel_r = data[:, 6]
    vel_phi = data[:, 7]
    vel_z = data[:, 8]
    
    # Determine grid dimensions
    unique_r = np.unique(r)
    unique_phi = np.unique(phi)
    nr = len(unique_r)
    nphi = len(unique_phi)
    
    # Reshape to 2D arrays
    def reshape_2d(arr):
        reshaped = arr.reshape((nphi, nr))
        return to_gpu(reshaped) if use_gpu else reshaped
    
    result = {
        'time': time,
        'cycle': cycle,
        'r': unique_r,  # Keep on CPU for matplotlib
        'phi': unique_phi,  # Keep on CPU for matplotlib
        'nr': nr,
        'nphi': nphi,
        'R': reshape_2d(r),
        'Phi': reshape_2d(phi),
        'density': reshape_2d(rho),
        'pressure': reshape_2d(press),
        'vel_r': reshape_2d(vel_r),
        'vel_phi': reshape_2d(vel_phi),
        'vel_z': reshape_2d(vel_z),
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

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_2d_heatmap(data, frame_num, output_dir):
    """Create 2D heatmaps for all variables"""
    print(f"  Creating 2D heatmaps for frame {frame_num}...")
    
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

def create_animation(available_frames, data_dir, output_dir, prefix):
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
    first_data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frames_subset[0]:05d}.tab"), 
                                 use_gpu=CONFIG['use_gpu'])
    
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']
    ims = []
    
    # Compute global ranges for consistent colorscale
    print("  Computing global ranges...")
    global_ranges = {}
    sample_step = max(1, len(frames_subset)//10)
    for var in variables:
        all_data = []
        frames_to_sample = frames_subset[::sample_step]
        
        for frame in tqdm(frames_to_sample, desc=f"  Sampling {var}"):
            data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frame:05d}.tab"), 
                                 use_gpu=CONFIG['use_gpu'])
            all_data.append(to_cpu(data[var]).ravel())
        
        all_data = np.concatenate(all_data)
        info = VARIABLE_INFO[var]
        if info['log'] and np.all(all_data > 0):
            global_ranges[var] = (all_data[all_data > 0].min(), all_data.max())
        else:
            vmax = np.abs(all_data).max()
            if not info['log']:
                global_ranges[var] = (-vmax, vmax)
            else:
                global_ranges[var] = (all_data.min(), all_data.max())
    
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
        data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frame:05d}.tab"), 
                             use_gpu=CONFIG['use_gpu'])
        
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

def create_polar_animation(available_frames, data_dir, output_dir, prefix):
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
    first_data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frames_subset[0]:05d}.tab"),
                                use_gpu=CONFIG['use_gpu'])
    
    # Calculate global min/max for all frames (for consistent color scale)
    print("  Computing global bounds for color scales...")
    global_ranges = {}
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']
    
    sample_step = max(1, len(frames_subset)//10)
    for var in variables:
        all_data = []
        frames_to_sample = frames_subset[::sample_step]
        
        for frame in tqdm(frames_to_sample, desc=f"  Sampling {var}"):
            data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frame:05d}.tab"),
                                 use_gpu=CONFIG['use_gpu'])
            all_data.append(to_cpu(data[var]).ravel())
        
        all_data = np.concatenate(all_data)
        
        info = VARIABLE_INFO[var]
        if info['log'] and np.all(all_data > 0):
            global_ranges[var] = (all_data[all_data > 0].min(), all_data.max())
        else:
            vmax = np.abs(all_data).max()
            if not info['log']:
                global_ranges[var] = (-vmax, vmax)
            else:
                global_ranges[var] = (all_data.min(), all_data.max())
    
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
        data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frame:05d}.tab"),
                             use_gpu=CONFIG['use_gpu'])
        
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
        filename = os.path.join(data_dir, f"{args.prefix}{frame_num:05d}.tab")
        data = read_athena_2d(filename, use_gpu=CONFIG['use_gpu'])
        
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
        create_animation(available_frames, data_dir, output_dir, args.prefix)
    
    if args.mode == 'all' or args.mode == 'polar_animation' or args.animate:
        create_polar_animation(available_frames, data_dir, output_dir, args.prefix)
    
    print("\n" + "="*80)
    print("DONE! All visualizations saved to:")
    print(f"   {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
