#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ 2D DATA VISUALIZER (LEGACY VERSION)
=============================================================================

Visualize 2D Athena++ simulation data with heatmaps, profiles, and animations
in both Cartesian and polar coordinates.

NOTE: This is the legacy version. Use vis2d.py for improved multi-block support.

BASIC USAGE:
    python3 vis2d_legacy.py
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
    --output_dir PATH - output directory (default: ../figs_2d)

EXAMPLES:
    python3 vis2d_legacy.py --mode heatmaps --frame 10
    python3 vis2d_legacy.py --fps 15
    python3 vis2d_legacy.py --mode polar_animation

=============================================================================
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

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
  python vis2d_legacy.py
  python vis2d_legacy.py --mode all
  python vis2d_legacy.py --mode heatmaps --frame 10
  python vis2d_legacy.py --mode radial --animate
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

args = parser.parse_args()

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = args.data_dir if args.data_dir else script_dir
output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(script_dir), "figs_2d")
os.makedirs(output_dir, exist_ok=True)

CONFIG['fps'] = args.fps
CONFIG['colormap'] = args.colormap

print("="*80)
print("ATHENA++ 2D VISUALIZER")
print("="*80)
print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")
print(f"Mode: {args.mode}")

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
def read_athena_2d(filename):
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
        return arr.reshape((nphi, nr))
    
    return {
        'time': time,
        'cycle': cycle,
        'r': unique_r,
        'phi': unique_phi,
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
            var_data = data[var]
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
            im = ax.pcolormesh(data['R'], data['Phi'], var_data, 
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
    """Radial profiles (averaged over phi)"""
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
        var_avg = np.mean(data[var], axis=0)
        var_std = np.std(data[var], axis=0)
        
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
        var_data = data[var]
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
        im = ax.pcolormesh(data['Phi'], data['R'], var_data, 
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
        for r_idx, color in zip(r_indices, colors):
            profile = data[var][:, r_idx]
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
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Disk Evolution", fontsize=16, fontweight='bold')
    
    # Read first frame for initialization
    first_data = read_athena_2d(os.path.join(data_dir, f"{prefix}{available_frames[0]:05d}.tab"))
    
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']
    ims = []
    
    for idx, var in enumerate(variables):
        ax = axes.flatten()[idx]
        info = VARIABLE_INFO[var]
        
        # Dummy plot for initialization
        var_data = first_data[var]
        if info['log'] and np.all(var_data > 0):
            norm = LogNorm(vmin=1e-6, vmax=1)
        else:
            norm = Normalize(vmin=-1, vmax=1)
        
        im = ax.pcolormesh(first_data['R'], first_data['Phi'], var_data,
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
        frame = available_frames[frame_idx]
        data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frame:05d}.tab"))
        
        # Update text with frame number
        time_text.set_text(f"Frame {frame_idx:3d}/{len(available_frames)-1} | Time = {data['time']:5.2f}")
        
        for idx, var in enumerate(variables):
            ims[idx].set_array(data[var].ravel())
        
        return ims + [time_text]
    
    ani = animation.FuncAnimation(fig, update, frames=len(available_frames),
                                 interval=1000/CONFIG['fps'], blit=False)
    
    output_video = os.path.join(output_dir, "disk_evolution_2d.mp4")
    ani.save(output_video, fps=CONFIG['fps'], dpi=150)
    print(f"  Saved Cartesian animation: {output_video}")
    plt.close()

def create_polar_animation(available_frames, data_dir, output_dir, prefix):
    """Create animation in polar coordinates"""
    print(f"\nCreating POLAR animation...")
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Polar View of Disk Evolution", fontsize=16, fontweight='bold', y=0.98)
    
    # Read first frame to determine normalization
    first_data = read_athena_2d(os.path.join(data_dir, f"{prefix}{available_frames[0]:05d}.tab"))
    
    # Calculate global min/max for all frames (for consistent color scale)
    print("  Computing global bounds for color scales...")
    global_ranges = {}
    variables = ['density', 'pressure', 'vel_r', 'vel_phi']
    
    for var in variables:
        all_data = []
        for frame in available_frames[::max(1, len(available_frames)//10)]:  # Sample every 10th frame
            data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frame:05d}.tab"))
            all_data.append(data[var])
        all_data = np.concatenate([d.ravel() for d in all_data])
        
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
        
        var_data = first_data[var]
        vmin, vmax = global_ranges[var]
        
        if info['log'] and vmin > 0:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        im = ax.pcolormesh(first_data['Phi'], first_data['R'], var_data,
                          cmap=info['cmap'], norm=norm, shading='auto')
        ax.set_title(info['label'], fontsize=13, fontweight='bold', pad=20)
        cbar = plt.colorbar(im, ax=ax, label=info['label'], pad=0.1)
        ax.grid(True, alpha=0.3)
        
        ims.append(im)
        cbars.append(cbar)
    
    # Add frame number to text
    time_text = fig.text(0.5, 0.94, '', ha='center', fontsize=12, fontweight='bold')
    
    def update(frame_idx):
        frame = available_frames[frame_idx]
        data = read_athena_2d(os.path.join(data_dir, f"{prefix}{frame:05d}.tab"))
        
        # Update text with frame number
        time_text.set_text(f"Frame {frame_idx:3d}/{len(available_frames)-1} | Time = {data['time']:5.2f}")
        
        for idx, var in enumerate(variables):
            ims[idx].set_array(data[var].ravel())
        
        return ims + [time_text]
    
    ani = animation.FuncAnimation(fig, update, frames=len(available_frames),
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
        data = read_athena_2d(filename)
        
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
