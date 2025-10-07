import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Папка з вихідними даними
data_dir = "/home/etrevol/work/acc_diskk/data"
file_pattern = "acc_disk.block0.out1.{:05d}.tab"

# Папка для збереження результатів
output_dir = "/home/etrevol/work/acc_diskk/plots/prim"
os.makedirs(output_dir, exist_ok=True)

# Автоматично знаходимо доступні файли
all_files = sorted([
    f for f in os.listdir(data_dir)
    if f.startswith("acc_disk.block0.out1.") and f.endswith(".tab")
])
available_frames = sorted([int(f.split('.')[-2]) for f in all_files])
num_frames = len(available_frames)
skip_rows = 2  # Пропускаємо заголовки

# Шляхи для збереження
output_image_density = os.path.join(output_dir, "100op_cygnus_acc_disk_initial_density.png")
output_image_pressure = os.path.join(output_dir, "100op_cygnus_acc_disk_initial_pressure.png")
output_image_velocity_x = os.path.join(output_dir, "100op_cygnus_acc_disk_initial_velocity_r.png")
output_image_velocity_y = os.path.join(output_dir, "100op_cygnus_acc_disk_initial_velocity_phi.png")
output_video = os.path.join(output_dir, "100op_acc_disk.mp4")

# Створюємо фігуру для анімації
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax = ax.flatten()
line1, = ax[0].plot([], [], 'r-', label="Density")
line2, = ax[1].plot([], [], 'g-', label="Pressure")
line3, = ax[2].plot([], [], 'b-', label=r"$v_r$")
line4, = ax[3].plot([], [], 'm-', label=r"$v_\phi$")

# Налаштування осей
for a in ax:
    a.set_xlim(10, 200)
    a.set_xlabel("r")
    a.grid(True)
ax[0].set_ylim(0, 1.1)
ax[0].set_ylabel(r"$\rho$")
ax[0].set_title("Density Evolution")
ax[0].legend()
ax[1].set_ylim(0, 16)
ax[1].set_ylabel("P")
ax[1].set_title("Pressure Evolution")
ax[1].legend()
ax[2].set_ylim(-1, 50)
ax[2].set_ylabel(r"$v_r$")
ax[2].set_title("Radial Velocity Evolution")
ax[2].legend()
ax[3].set_ylim(-1, 40)
ax[3].set_ylabel(r"$v_\phi$")
ax[3].set_title("Azimuthal Velocity Evolution")
ax[3].legend()

# Вибір фрейму
initial_frame_number = int(input(f"Введіть номер фрейму для побудови графіків (0 - {available_frames[-1]}): "))
initial_file = os.path.join(data_dir, file_pattern.format(initial_frame_number))

def save_plot(x, y, xlabel, ylabel, title, filename, color, label):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.2)
    plt.close()

if os.path.exists(initial_file):
    data = np.loadtxt(initial_file, skiprows=skip_rows)
    r_init = data[:, 1]
    rho_init = data[:, 2]
    P_init = data[:, 3]
    vx_init = data[:, 4]
    vy_init = data[:, 5]

    save_plot(r_init, rho_init, "r", r"$\rho$", "Initial Density Distribution",
              output_image_density, 'r-', "Initial Density")
    save_plot(r_init, P_init, "r", "P", "Initial Pressure Distribution",
              output_image_pressure, 'g-', "Initial Pressure")
    save_plot(r_init, vx_init, "r", r"$v_r$", "Initial Radial Velocity Distribution",
              output_image_velocity_x, 'b-', r"$v_r$")
    save_plot(r_init, vy_init, "r", r"$v_\phi$", "Initial Azimuthal Velocity Distribution",
              output_image_velocity_y, 'm-', r"$v_\phi$")

# Функція ініціалізації
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4

# Функція оновлення
def update(frame):
    filename = os.path.join(data_dir, file_pattern.format(frame))
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

# Створення анімації
ani = animation.FuncAnimation(fig, update, frames=available_frames, init_func=init, blit=True)
ani.save(output_video, fps=5)

print(f"Графіки та анімація збережені у папку: {output_dir}")
