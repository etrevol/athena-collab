#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# НАЛАШТУВАННЯ ДІАПАЗОНІВ ГРАФІКІВ (легко модифікувати)
# =============================================================================
PLOT_RANGES = {
    'density': {
        'xlim': (10, 200),
        'ylim': (0, 1.1)
    },
    'pressure': {
        'xlim': (10, 200),
        'ylim': (0, 16)
    },
    'velocity_r': {
        'xlim': (10, 200),
        'ylim': (-1, 50)
    },
    'velocity_phi': {
        'xlim': (10, 200),
        'ylim': (-1, 40)
    }
}

# =============================================================================
# АРГУМЕНТИ КОМАНДНОГО РЯДКА (з значеннями за замовчуванням)
# =============================================================================
parser = argparse.ArgumentParser(description="Універсальна візуалізація даних Athena")
parser.add_argument("--data_dir", default=None, 
                    help="Папка з .tab файлами (за замовчуванням: ../data)")
parser.add_argument("--output_dir", default=None, 
                    help="Папка для збереження графіків (за замовчуванням: ../figs)")
parser.add_argument("--prefix", default="pp_disk.block0.out1.", 
                    help="Префікс файлів")
parser.add_argument("--skip_rows", type=int, default=2, 
                    help="Кількість рядків заголовка для пропуску")
parser.add_argument("--fps", type=int, default=5, 
                    help="Кадрів на секунду для анімації")
args = parser.parse_args()

# Визначаємо шляхи
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = args.data_dir if args.data_dir else os.path.join(os.path.dirname(script_dir), "data")
output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(script_dir), "figs")

prefix = args.prefix
skip_rows = args.skip_rows
fps = args.fps

# Створюємо папку для результатів
os.makedirs(output_dir, exist_ok=True)

print(f"Папка з даними: {data_dir}")
print(f"Папка для результатів: {output_dir}")

# =============================================================================
# ЗНАХОДИМО ВСІ .TAB ФАЙЛИ
# =============================================================================
tab_files = sorted([
    f for f in os.listdir(data_dir)
    if f.startswith(prefix) and f.endswith(".tab")
])

if not tab_files:
    print(f"Не знайдено .tab файлів у {data_dir}")
    exit(1)

available_frames = sorted([int(f.split('.')[-2]) for f in tab_files])
num_frames = len(available_frames)
print(f"Знайдено {num_frames} файлів даних")

# =============================================================================
# ФУНКЦІЯ ДЛЯ ЗБЕРЕЖЕННЯ ГРАФІКІВ
# =============================================================================
def save_plot(x, y, xlabel, ylabel, title, filename, color, label, xlim=None, ylim=None):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Збережено: {filename}")

# =============================================================================
# ПОБУДОВА ГРАФІКІВ ДЛЯ ПОЧАТКОВОГО ФРЕЙМУ
# =============================================================================
initial_frame = available_frames[0]
initial_file = os.path.join(data_dir, f"{prefix}{initial_frame:05d}.tab")

if os.path.exists(initial_file):
    print(f"\nОбробка початкового фрейму: {initial_frame}")
    data = np.loadtxt(initial_file, skiprows=skip_rows)
    r_init = data[:, 1]
    rho_init = data[:, 2]
    P_init = data[:, 3]
    vx_init = data[:, 4]
    vy_init = data[:, 5]

    save_plot(r_init, rho_init, "r", r"$\rho$", "Initial Density Distribution",
              os.path.join(output_dir, "initial_density.png"), 'r-', "Initial Density",
              xlim=PLOT_RANGES['density']['xlim'], ylim=PLOT_RANGES['density']['ylim'])
    
    save_plot(r_init, P_init, "r", "P", "Initial Pressure Distribution",
              os.path.join(output_dir, "initial_pressure.png"), 'g-', "Initial Pressure",
              xlim=PLOT_RANGES['pressure']['xlim'], ylim=PLOT_RANGES['pressure']['ylim'])
    
    save_plot(r_init, vx_init, "r", r"$v_r$", "Initial Radial Velocity Distribution",
              os.path.join(output_dir, "initial_velocity_r.png"), 'b-', r"$v_r$",
              xlim=PLOT_RANGES['velocity_r']['xlim'], ylim=PLOT_RANGES['velocity_r']['ylim'])
    
    save_plot(r_init, vy_init, "r", r"$v_\phi$", "Initial Azimuthal Velocity Distribution",
              os.path.join(output_dir, "initial_velocity_phi.png"), 'm-', r"$v_\phi$",
              xlim=PLOT_RANGES['velocity_phi']['xlim'], ylim=PLOT_RANGES['velocity_phi']['ylim'])

# =============================================================================
# АНІМАЦІЯ ЕВОЛЮЦІЇ ПАРАМЕТРІВ
# =============================================================================
print("\nСтворення анімації...")
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax = ax.flatten()
line1, = ax[0].plot([], [], 'r-', label="Density")
line2, = ax[1].plot([], [], 'g-', label="Pressure")
line3, = ax[2].plot([], [], 'b-', label=r"$v_r$")
line4, = ax[3].plot([], [], 'm-', label=r"$v_\phi$")

# Налаштування осей з використанням PLOT_RANGES
ax[0].set_xlim(PLOT_RANGES['density']['xlim'])
ax[0].set_ylim(PLOT_RANGES['density']['ylim'])
ax[0].set_ylabel(r"$\rho$")
ax[0].set_title("Density Evolution")
ax[0].legend()
ax[0].grid(True)

ax[1].set_xlim(PLOT_RANGES['pressure']['xlim'])
ax[1].set_ylim(PLOT_RANGES['pressure']['ylim'])
ax[1].set_ylabel("P")
ax[1].set_title("Pressure Evolution")
ax[1].legend()
ax[1].grid(True)

ax[2].set_xlim(PLOT_RANGES['velocity_r']['xlim'])
ax[2].set_ylim(PLOT_RANGES['velocity_r']['ylim'])
ax[2].set_ylabel(r"$v_r$")
ax[2].set_title("Radial Velocity Evolution")
ax[2].legend()
ax[2].grid(True)

ax[3].set_xlim(PLOT_RANGES['velocity_phi']['xlim'])
ax[3].set_ylim(PLOT_RANGES['velocity_phi']['ylim'])
ax[3].set_ylabel(r"$v_\phi$")
ax[3].set_title("Azimuthal Velocity Evolution")
ax[3].legend()
ax[3].grid(True)

for a in ax:
    a.set_xlabel("r")

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4

def update(frame):
    filename = os.path.join(data_dir, f"{prefix}{frame:05d}.tab")
    if not os.path.exists(filename):
        return line1, line2, line3, line4
    data = np.loadtxt(filename, skiprows=skip_rows)
    r = data[:, 1]
    rho = data[:, 2]
    P = data[:, 3]
    vx = data[:, 4]
    vy = data[:, 5]
    line1.set_data(r, rho)
    line2.set_data(r, P)
    line3.set_data(r, vx)
    line4.set_data(r, vy)
    return line1, line2, line3, line4

ani = animation.FuncAnimation(fig, update, frames=available_frames, init_func=init, blit=True)
output_video = os.path.join(output_dir, "evolution.mp4")
ani.save(output_video, fps=fps)
print(f"Збережено: {output_video}")

print(f"\n✓ Всі графіки та анімація збережені у папку: {output_dir}")