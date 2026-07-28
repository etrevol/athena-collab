#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ HISTORY FILE VISUALIZER
=============================================================================

Visualize time series data from Athena++ .hst (history) files.

BASIC USAGE:
    python3 vishst.py
        Creates plots for disk_mass and mdot_in with log scale (default)

OPTIONS:
    --mode {default,all,custom}
        default - plot disk_mass and mdot_in only (default)
        all     - plot all variables in the history file
        custom  - plot specific variables (use --vars)
    
    --vars VAR1 VAR2 ...     - specific variables to plot (for custom mode)
    --hst_file PATH          - path to .hst file (default: auto-detect in ../data)
    --output_dir PATH        - output directory (default: ../figs_hst)
    --linear_scale VAR1 VAR2 - variables to plot with linear scale (default: log)
    --grid                   - show grid on plots
    --dpi N               - DPI for saved figures (default: 150)
    --figsize W H         - figure size in inches (default: 10 6)
    --style {default,seaborn,bmh,ggplot}
                          - matplotlib style (default: seaborn-v0_8-darkgrid)

EXAMPLES:
    # Basic usage (disk_mass and mdot_in) - logarithmic scale by default
    python3 vishst.py
    
    # Plot all variables
    python3 vishst.py --mode all
    
    # Plot specific variables
    python3 vishst.py --mode custom --vars mass tot-E disk_mass
    
    # Custom styling
    python3 vishst.py --grid --dpi 200 --figsize 12 8
    
    # Linear scale for specific variables (default is log scale)
    python3 vishst.py --mode all --linear_scale disk_mass mass

=============================================================================
"""

import os
import sys
import argparse

import athena_data as _ad
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import re
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'dpi': 150,
    'figsize': (10, 6),
    'style': 'seaborn-v0_8-darkgrid',
    'linewidth': 2.0,
    'markersize': 3.0,
    'alpha': 0.8,
    'grid': False,
}

# Default variables for standard mode
DEFAULT_VARS = ['disk_mass', 'mdot_in']

# Variable display information
VARIABLE_INFO = {
    'time': {'label': 'Time', 'unit': '', 'color': 'black'},
    'dt': {'label': 'Timestep', 'unit': '', 'color': 'gray'},
    'mass': {'label': 'Total Mass', 'unit': '', 'color': 'blue'},
    '1-mom': {'label': 'Radial Momentum', 'unit': '', 'color': 'red'},
    '2-mom': {'label': 'Azimuthal Momentum', 'unit': '', 'color': 'green'},
    '3-mom': {'label': 'Vertical Momentum', 'unit': '', 'color': 'purple'},
    '1-KE': {'label': 'Radial Kinetic Energy', 'unit': '', 'color': 'orange'},
    '2-KE': {'label': 'Azimuthal Kinetic Energy', 'unit': '', 'color': 'cyan'},
    '3-KE': {'label': 'Vertical Kinetic Energy', 'unit': '', 'color': 'magenta'},
    'tot-E': {'label': 'Total Energy', 'unit': '', 'color': 'darkblue'},
    'disk_mass': {'label': r'Disk Mass [$M_\odot$]', 'unit': r'$M_\odot$', 'color': 'darkred'},
    'mdot_in': {'label': r'Accretion Rate [$M_\odot$/yr]', 'unit': r'$M_\odot$/yr', 'color': 'darkgreen'},
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="Athena++ history file visualization",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Usage examples:
  python vishst.py
  python vishst.py --mode all
  python vishst.py --mode custom --vars disk_mass mdot_in tot-E
  python vishst.py --linear_scale disk_mass mass
    """
)
parser.add_argument("--hst_file", default=None,
                    help="Path to .hst file (default: auto-detect in ../data)")
parser.add_argument("--output_dir", default=None,
                    help="Output directory (default: ../figs_hst)")
parser.add_argument("--mode", default="default",
                    choices=['default', 'all', 'custom'],
                    help="Visualization mode")
parser.add_argument("--vars", nargs='+', default=None,
                    help="Variables to plot (for custom mode)")
parser.add_argument("--linear_scale", nargs='+', default=[],
                    help="Variables to plot with linear scale (default: all use log scale)")
parser.add_argument("--grid", action='store_true',
                    help="Show grid on plots")
parser.add_argument("--dpi", type=int, default=CONFIG['dpi'],
                    help="DPI for saved figures")
parser.add_argument("--figsize", nargs=2, type=float, default=CONFIG['figsize'],
                    help="Figure size in inches (width height)")
parser.add_argument("--title", default=None,
                    help="Figure title (centred, top). Supports {label}, {var}, {file}; "
                         "e.g. --title \"PP disk — {label}\"")
parser.add_argument("--style", default=CONFIG['style'],
                    choices=['default', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-whitegrid', 
                             'bmh', 'ggplot', 'classic'],
                    help="Matplotlib style")

args = parser.parse_args()

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if args.hst_file:
    data_dir = os.path.join(script_dir, "data")
    _output_base = script_dir
else:
    _candidate = os.path.join(script_dir, "data")
    if not os.path.isdir(_candidate):
        _candidate = os.path.join(os.path.dirname(script_dir), "data")
        _output_base = os.path.dirname(script_dir)
    else:
        _output_base = script_dir
    data_dir = _candidate
output_dir = args.output_dir if args.output_dir else os.path.join(_output_base, "figs_hst")
os.makedirs(output_dir, exist_ok=True)

# Update config
CONFIG['dpi'] = args.dpi
CONFIG['figsize'] = tuple(args.figsize)
CONFIG['grid'] = args.grid
if args.style and args.style != 'default':
    CONFIG['style'] = args.style

# Apply matplotlib style
try:
    plt.style.use(CONFIG['style'])
except OSError:
    print(f"Warning: Style '{CONFIG['style']}' not available, using default")
    plt.style.use('default')

print("="*80)
print("ATHENA++ HISTORY FILE VISUALIZER")
print("="*80)
print(f"Output directory: {output_dir}")
print(f"Mode: {args.mode}")
print(f"Grid: {'✓ ENABLED' if CONFIG['grid'] else '✗ DISABLED'}")
print(f"DPI: {CONFIG['dpi']}")
print(f"Figure size: {CONFIG['figsize'][0]} x {CONFIG['figsize'][1]} inches")

# =============================================================================
# FILE DETECTION AND READING
# =============================================================================
def find_hst_file(data_dir):
    """Find .hst file in data directory"""
    hst_files = list(Path(data_dir).glob('*.hst'))
    if not hst_files:
        return None
    return str(hst_files[0])

def read_hst_file(filename):
    """Read Athena++ history file
    
    Returns:
        dict: {column_name: numpy_array}
    """
    # Read header to get column names
    with open(filename, 'r', encoding='utf-8') as f:
        _ = f.readline().strip()  # Skip first comment line
        second_line = f.readline().strip()
    
    # Parse column names from second line
    # Format: # [1]=time [2]=dt [3]=mass ...
    column_pattern = r'\[(\d+)\]=([^\s]+)'
    matches = re.findall(column_pattern, second_line)
    
    if not matches:
        raise ValueError(f"Could not parse column names from: {second_line}")
    
    # Create column name mapping
    col_indices = {}
    col_names = []
    for idx_str, name in matches:
        idx = int(idx_str) - 1  # Convert to 0-based indexing
        col_indices[name] = idx
        col_names.append(name)
    
    # Read data (skip first two comment lines)
    data = np.loadtxt(filename, skiprows=2)
    
    # Create dictionary of arrays
    data_dict = {}
    for name, idx in col_indices.items():
        data_dict[name] = data[:, idx]
    
    return data_dict, col_names

# Find and read history file
if args.hst_file:
    hst_file = args.hst_file
else:
    hst_file = find_hst_file(data_dir)

if not hst_file or not os.path.exists(hst_file):
    print(f"ERROR: No .hst file found in {data_dir}")
    print("Please specify with --hst_file")
    sys.exit(1)

print(f"Reading: {hst_file}")
data, all_vars = read_hst_file(hst_file)
print(f"Available variables: {', '.join(all_vars)}")
print(f"Time range: {data['time'][0]:.6e} to {data['time'][-1]:.6e}")
print(f"Number of timesteps: {len(data['time'])}")
print(f"Scale: {'Linear for: ' + ', '.join(args.linear_scale) if args.linear_scale else 'Logarithmic (default)'}")

# =============================================================================
# DETERMINE WHICH VARIABLES TO PLOT
# =============================================================================
vars_to_plot = []  # Initialize to avoid unbound variable warning

if args.mode == 'default':
    vars_to_plot = DEFAULT_VARS
    print(f"Plotting default variables: {', '.join(vars_to_plot)}")
elif args.mode == 'all':
    # Exclude 'time' and 'dt' from plotting
    vars_to_plot = [v for v in all_vars if v not in ['time', 'dt']]
    print(f"Plotting all {len(vars_to_plot)} variables")
elif args.mode == 'custom':
    if not args.vars:
        print("ERROR: --vars required for custom mode")
        sys.exit(1)
    vars_to_plot = args.vars
    # Check if all requested variables exist
    missing = [v for v in vars_to_plot if v not in data]
    if missing:
        print(f"ERROR: Variables not found: {', '.join(missing)}")
        print(f"Available: {', '.join(all_vars)}")
        sys.exit(1)
    print(f"Plotting custom variables: {', '.join(vars_to_plot)}")
else:
    # This should never happen due to argparse choices, but satisfies type checker
    print(f"ERROR: Unknown mode '{args.mode}'")
    sys.exit(1)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_variable(time, var_data, var_name, output_dir, linear_scale=False):
    """Plot single variable vs time"""
    fig, ax = plt.subplots(figsize=CONFIG['figsize'], dpi=CONFIG['dpi'])
    
    # Get variable info
    if var_name in VARIABLE_INFO:
        info = VARIABLE_INFO[var_name]
        label = info['label']
        color = info['color']
    else:
        label = var_name
        color = 'steelblue'
    
    # Plot
    ax.plot(time, var_data, 
            linewidth=CONFIG['linewidth'],
            alpha=CONFIG['alpha'],
            color=color,
            label=label)
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel(label, fontsize=12, fontweight='bold')
    ax.set_title(_ad.format_title(f'{label} Evolution', args.title,
                                  label=label, var=var, file=os.path.basename(hst_file)),
                 fontsize=14, fontweight='bold', pad=15)
    
    # Use log scale by default, unless explicitly set to linear
    if not linear_scale:
        ax.set_yscale('log')
    
    if CONFIG['grid']:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text box
    mean_val = float(np.mean(var_data))
    std_val = float(np.std(var_data))
    min_val = float(np.min(var_data))
    max_val = float(np.max(var_data))
    
    stats_text = f'Mean: {mean_val:.4e}\nStd: {std_val:.4e}\nMin: {min_val:.4e}\nMax: {max_val:.4e}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
    
    plt.tight_layout()
    
    # Save
    safe_name = var_name.replace('-', '_').replace(' ', '_')
    output_file = os.path.join(output_dir, f'{safe_name}_vs_time.png')
    plt.savefig(output_file, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_multiple_variables(time, data_dict, var_names, output_dir, linear_scale_vars=None):
    """Plot multiple variables in subplots"""
    if linear_scale_vars is None:
        linear_scale_vars = []
    n_vars = len(var_names)
    
    # Determine subplot layout
    if n_vars == 1:
        nrows, ncols = 1, 1
        figsize = (CONFIG['figsize'][0], CONFIG['figsize'][1])
    elif n_vars == 2:
        nrows, ncols = 1, 2
        figsize = (CONFIG['figsize'][0], CONFIG['figsize'][1])
    elif n_vars <= 4:
        nrows, ncols = 2, 2
        figsize = (CONFIG['figsize'][0] * 1.2, CONFIG['figsize'][1] * 1.5)
    elif n_vars <= 6:
        nrows, ncols = 2, 3
        figsize = (CONFIG['figsize'][0] * 1.5, CONFIG['figsize'][1] * 1.5)
    elif n_vars <= 9:
        nrows, ncols = 3, 3
        figsize = (CONFIG['figsize'][0] * 1.5, CONFIG['figsize'][1] * 2)
    else:
        nrows = int(np.ceil(n_vars / 3))
        ncols = 3
        figsize = (CONFIG['figsize'][0] * 1.5, CONFIG['figsize'][1] * (nrows * 0.8))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=CONFIG['dpi'])
    
    # Flatten axes array for easy iteration
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, var_name in enumerate(var_names):
        ax = axes[idx]
        var_data = data_dict[var_name]
        
        # Get variable info
        if var_name in VARIABLE_INFO:
            info = VARIABLE_INFO[var_name]
            label = info['label']
            color = info['color']
        else:
            label = var_name
            color = 'steelblue'
        
        # Plot
        ax.plot(time, var_data,
                linewidth=CONFIG['linewidth'] * 0.8,
                alpha=CONFIG['alpha'],
                color=color)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        
        # Use log scale by default, unless explicitly set to linear
        if var_name not in linear_scale_vars:
            ax.set_yscale('log')
        
        if CONFIG['grid']:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 'all_variables_overview.png')
    plt.savefig(output_file, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_disk_mass_and_mdot(time, disk_mass, mdot_in, output_dir):
    """Special plot for disk mass and accretion rate with dual y-axes"""
    fig, ax1 = plt.subplots(figsize=CONFIG['figsize'], dpi=CONFIG['dpi'])
    
    # Plot disk mass on left axis
    color1 = 'darkred'
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'Disk Mass [$M_\odot$]', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(time, disk_mass, 
                     linewidth=CONFIG['linewidth'],
                     alpha=CONFIG['alpha'],
                     color=color1,
                     label=r'Disk Mass')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    if CONFIG['grid']:
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Plot accretion rate on right axis
    ax2 = ax1.twinx()
    color2 = 'darkgreen'
    ax2.set_ylabel(r'Accretion Rate [$M_\odot$/yr]', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(time, mdot_in,
                     linewidth=CONFIG['linewidth'],
                     alpha=CONFIG['alpha'],
                     color=color2,
                     label=r'$\dot{M}_{in}$')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    ax1.set_title(_ad.format_title('Disk Mass and Accretion Rate Evolution', args.title,
                                   label='Disk Mass and Accretion Rate', var='disk_mass',
                                   file=os.path.basename(hst_file)), 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    lines = line1 + line2
    labels = [str(l.get_label()) for l in lines]  # Explicitly convert to str for type safety
    ax1.legend(lines, labels, loc='best', framealpha=0.9)
    
    # Statistics
    mass_mean = float(np.mean(disk_mass))
    mdot_mean = float(np.mean(mdot_in))
    # Use explicit string formatting to avoid LaTeX escaping issues in f-strings
    stats_text = (f'Mean Disk Mass: {mass_mean:.4e} M$_\\odot$\n'
                  f'Mean $\\dot{{{{M}}}}_{{{{in}}}}$: {mdot_mean:.4e} M$_\\odot$/yr')
    ax1.text(0.02, 0.02, stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.7})
    
    fig.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 'disk_mass_and_mdot.png')
    plt.savefig(output_file, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    return output_file

# =============================================================================
# MAIN PLOTTING
# =============================================================================
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

time = data['time']
linear_scale_vars = args.linear_scale

# Mode-specific plotting
if args.mode == 'default':
    # Special dual-axis plot for disk mass and mdot
    if 'disk_mass' in data and 'mdot_in' in data:
        print("\nCreating combined disk_mass and mdot_in plot...")
        output = plot_disk_mass_and_mdot(time, data['disk_mass'], data['mdot_in'], output_dir)
        print(f"  ✓ Saved: {output}")
    
    # Individual plots
    for var in vars_to_plot:
        if var in data:
            print(f"\nPlotting {var}...")
            use_linear = var in linear_scale_vars
            output = plot_variable(time, data[var], var, output_dir, linear_scale=use_linear)
            print(f"  ✓ Saved: {output}")
        else:
            print(f"  ✗ Warning: {var} not found in data")

elif args.mode == 'all':
    # Create overview plot with all variables
    print("\nCreating overview plot with all variables...")
    output = plot_multiple_variables(time, data, vars_to_plot, output_dir, linear_scale_vars)
    print(f"  ✓ Saved: {output}")
    
    # Individual plots for each variable
    print("\nCreating individual plots...")
    for var in vars_to_plot:
        use_linear = var in linear_scale_vars
        output = plot_variable(time, data[var], var, output_dir, linear_scale=use_linear)
        print(f"  ✓ {var}")

elif args.mode == 'custom':
    # Plot only requested variables
    for var in vars_to_plot:
        print(f"\nPlotting {var}...")
        use_linear = var in linear_scale_vars
        output = plot_variable(time, data[var], var, output_dir, linear_scale=use_linear)
        print(f"  ✓ Saved: {output}")
    
    # If more than one variable, create overview
    if len(vars_to_plot) > 1:
        print("\nCreating overview plot...")
        output = plot_multiple_variables(time, data, vars_to_plot, output_dir, linear_scale_vars)
        print(f"  ✓ Saved: {output}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"All plots saved to: {output_dir}")
print(f"Total plots created: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])}")
print("="*80)
