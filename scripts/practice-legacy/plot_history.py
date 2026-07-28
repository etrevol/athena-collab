#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ HISTORY DATA PLOTTER
=============================================================================

Visualization of parameters from Athena++ simulation .hst files.
Supports plotting any parameters from history files.

USAGE:
    python3 plot_history.py
        Plots all parameters from .hst file in current directory

OPTIONS:
    --file PATH          - path to .hst file (default: auto-search in script directory)
    --params P1 P2 ...   - list of parameters to plot (e.g.: M_dot mass tot-E)
    --all                - plot all parameters
    --output_dir PATH    - directory to save plots (default: ./figs_history)
    --dpi N              - plot resolution (default: 200)
    --style {default,dark,ggplot,seaborn}  - plot style

EXAMPLES:
    python3 plot_history.py --params M_dot mass tot-E
    python3 plot_history.py --all
    python3 plot_history.py --file simulation.hst --params M_dot
    python3 plot_history.py --params M_dot --style dark --dpi 300

=============================================================================
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'dpi': 200,
    'figsize': (10, 6),
    'grid': True,
    'linewidth': 2.0,
}

# Colors for different parameter types
PARAM_COLORS = {
    'M_dot': '#e74c3c',      # Red
    'mass': '#3498db',       # Blue
    'tot-E': '#2ecc71',      # Green
    'dt': '#f39c12',         # Orange
    '1-mom': '#9b59b6',      # Purple
    '2-mom': '#1abc9c',      # Turquoise
    '3-mom': '#34495e',      # Dark gray
    '1-KE': '#e67e22',       # Dark orange
    '2-KE': '#16a085',       # Dark turquoise
    '3-KE': '#8e44ad',       # Dark purple
}

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="Plot parameters from Athena++ .hst files",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
)
parser.add_argument("--file", default=None,
                    help="Path to .hst file (auto-search in script directory if not specified)")
parser.add_argument("--params", nargs='+', default=None,
                    help="List of parameters to plot")
parser.add_argument("--all", action='store_true',
                    help="Plot all parameters")
parser.add_argument("--output_dir", default=None,
                    help="Directory to save plots (default: figs_history in script directory)")
parser.add_argument("--dpi", type=int, default=200,
                    help="Plot resolution (DPI)")
parser.add_argument("--style", default="default",
                    choices=['default', 'dark', 'ggplot', 'seaborn'],
                    help="Matplotlib plot style")

args = parser.parse_args()

# Apply style
if args.style != 'default':
    if args.style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use(args.style)

# =============================================================================
# SEARCH FOR .HST FILE
# =============================================================================
def find_hst_file():
    """Automatic search for .hst file in script directory"""
    # Directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search in script directory
    hst_files = glob.glob(os.path.join(script_dir, "*.hst"))
    if hst_files:
        return sorted(hst_files)[-1]  # Return the last file
    
    return None

# Determine file to process
hst_file = args.file if args.file else find_hst_file()

if not hst_file or not os.path.exists(hst_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"❌ No .hst file found in script directory: {script_dir}")
    print("Use --file to specify file path")
    exit(1)

print(f"📊 Processing file: {os.path.basename(hst_file)}")

# =============================================================================
# READ .HST FILE
# =============================================================================
def parse_hst_file(filename):
    """Parse Athena++ .hst file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find line with parameter names
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('# [1]='):
            header_line = line
            data_start = i + 1
            break
    
    if header_line is None:
        raise ValueError("Header with parameter names not found")
    
    # Parse parameter names
    # Format: # [1]=time     [2]=dt       [3]=mass ...
    param_names = []
    parts = header_line.split('[')
    for part in parts[1:]:  # Skip first element (empty or '#')
        if ']=' in part:
            param_name = part.split(']=')[1].split()[0]
            param_names.append(param_name)
    
    # Read data
    data = np.loadtxt(filename, skiprows=data_start)
    
    # Create dictionary with data
    data_dict = {}
    for i, name in enumerate(param_names):
        if i < data.shape[1]:
            data_dict[name] = data[:, i]
    
    return data_dict, param_names

try:
    data_dict, param_names = parse_hst_file(hst_file)
    print(f"✓ Found {len(param_names)} parameters: {', '.join(param_names)}")
except Exception as e:
    print(f"❌ Error reading file: {e}")
    exit(1)

# =============================================================================
# DETERMINE PARAMETERS TO PLOT
# =============================================================================
if args.all:
    # Plot all parameters except 'time'
    params_to_plot = [p for p in param_names if p != 'time']
    print(f"📈 Plotting all parameters ({len(params_to_plot)} total)")
elif args.params:
    # Check if specified parameters exist
    params_to_plot = []
    for p in args.params:
        if p in param_names:
            params_to_plot.append(p)
        else:
            print(f"⚠️  Parameter '{p}' not found in file. Available: {', '.join(param_names)}")
    
    if not params_to_plot:
        print("❌ None of the specified parameters found!")
        exit(1)
else:
    # Default - M_dot if available, otherwise first 3 parameters (except time)
    if 'M_dot' in param_names:
        params_to_plot = ['M_dot']
    else:
        params_to_plot = [p for p in param_names if p != 'time'][:3]
    
    print(f"📈 Plotting parameters: {', '.join(params_to_plot)}")

# Створюємо вихідну директорію (в директорії скрипта якщо не вказано)
if args.output_dir is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_dir = os.path.join(script_dir, "figs_history")

os.makedirs(args.output_dir, exist_ok=True)

# =============================================================================
# PLOT GENERATION
# =============================================================================
def get_param_color(param_name):
    """Get color for parameter"""
    if param_name in PARAM_COLORS:
        return PARAM_COLORS[param_name]
    # Random color for unknown parameters
    import hashlib
    hash_val = int(hashlib.md5(param_name.encode()).hexdigest(), 16)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    return colors[hash_val % len(colors)]

def plot_parameter(time, values, param_name, output_dir, dpi):
    """Plot a single parameter"""
    plt.figure(figsize=CONFIG['figsize'])
    
    color = get_param_color(param_name)
    plt.plot(time, values, color=color, linewidth=CONFIG['linewidth'], label=param_name)
    
    plt.xlabel('Time', fontsize=12, fontweight='bold')
    plt.ylabel(param_name, fontsize=12, fontweight='bold')
    plt.title(f'Evolution of {param_name} over time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    
    if CONFIG['grid']:
        plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f"{param_name}.png")
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")

def plot_multiple_parameters(time, data_dict, param_names, output_dir, dpi):
    """Plot multiple parameters"""
    if len(param_names) == 1:
        return  # Один параметр вже побудовано окремо
    
    fig, axes = plt.subplots(len(param_names), 1, 
                            figsize=(10, 4*len(param_names)),
                            sharex=True)
    
    if len(param_names) == 1:
        axes = [axes]
    
    for i, param in enumerate(param_names):
        color = get_param_color(param)
        axes[i].plot(time, data_dict[param], color=color, 
                    linewidth=CONFIG['linewidth'], label=param)
        axes[i].set_ylabel(param, fontsize=11, fontweight='bold')
        axes[i].legend(loc='best', fontsize=9)
        axes[i].grid(True, alpha=0.3, linestyle='--')
    
    axes[-1].set_xlabel('Time', fontsize=12, fontweight='bold')
    fig.suptitle('Evolution of parameters over time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, "combined_parameters.png")
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved combined plot: {output_file}")

# Time is always the first parameter
time = data_dict['time']

# Будуємо окремі графіки
print("\n🎨 Створення графіків...")
for param in params_to_plot:
    plot_parameter(time, data_dict[param], param, args.output_dir, args.dpi)

# Будуємо комбінований графік, якщо параметрів більше одного
if len(params_to_plot) > 1:
    plot_multiple_parameters(time, data_dict, params_to_plot, 
                            args.output_dir, args.dpi)

print(f"\n✅ Готово! Графіки збережено в: {args.output_dir}")
print(f"📊 Побудовано графіків: {len(params_to_plot) + (1 if len(params_to_plot) > 1 else 0)}")
