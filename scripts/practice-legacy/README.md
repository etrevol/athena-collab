# Athena++ Visualization Scripts

## Ваша конфігурація

- **CPU:** 16 cores (8 фізичних + hyperthreading)
- **GPU:** NVIDIA GeForce GTX 1650 (4GB VRAM)
- **RAM:** 7.5 GB
- **CUDA:** 13.0 ✓
- **CuPy:** 13.6.0 ✓

## Активні скрипти

### vis2d.py ⭐ (РЕКОМЕНДОВАНО)
Основний 2D візуалізатор з повною підтримкою multi-block формату та GPU прискорення.

**Базове використання:**
```bash
cd results/pp_disk_viscosity/sample-*/materials/
python3 vis2d.py
```

**Параметри:**
- `--mode {all,heatmaps,radial,polar,azimuthal,animation,polar_animation}`
- `--frame N` - номер кадру (за замовчуванням: останній)
- `--use_gpu` - увімкнути GPU прискорення (якщо є CuPy)
- `--num_workers N` - кількість CPU процесів для паралельної обробки
- `--fps N` - FPS для анімацій (за замовчуванням: 10)
- `--subsample N` - використовувати кожен N-й кадр (1 = всі кадри)
- `--start_frame N` / `--end_frame N` - діапазон кадрів для анімації

**Рекомендовані команди для вашого PC:**

```bash
# Швидка візуалізація одного кадру
python3 vis2d.py --mode heatmaps

# З GPU прискоренням
python3 vis2d.py --mode heatmaps --use_gpu

# Анімація з паралельною обробкою (використовує 8 CPU cores)
python3 vis2d.py --mode animation --num_workers 8 --subsample 5

# Все разом (GPU + багатопоточність)
python3 vis2d.py --mode all --use_gpu --num_workers 8

# Полярна анімація кадрів 100-500
python3 vis2d.py --mode polar_animation --start_frame 100 --end_frame 500 --subsample 10 --use_gpu
```

### vis1d.py
Візуалізація 1D радіальних профілів.

```bash
python3 vis1d.py --data_dir ../data --output_dir ../figs
```

### plot_history.py
Побудова історії симуляції (.hst файли).

```bash
python3 plot_history.py
```

### test_performance.py
Тест продуктивності CPU та GPU для визначення оптимальних параметрів.

```bash
python3 scripts/test_performance.py
```

## Розуміння параметрів

### --num_workers (CPU паралелізм) ✓ ПРАЦЮЄ
- Кількість **CPU процесів** для паралельної обробки кадрів
- У вас 16 cores → рекомендовано: **8-14 workers**
- **Прискорює:**
  - Sampling кадрів для визначення глобальних діапазонів (~8x швидше на 8 cores)
  - Читання та обробку великої кількості кадрів
- За замовчуванням: `cpu_count() - 2` (залишає 2 cores для системи)

```bash
--num_workers 1   # без паралелізму (для дебагу)
--num_workers 8   # оптимально для вашого PC ✓
--num_workers 14  # максимум (залишає 2 cores для системи)
```

### --use_gpu (GPU прискорення)
- Використовує **NVIDIA GPU** через CuPy/CUDA
- Прискорює статистичні розрахунки (mean, std)
- Ефективно для великих датасетів
- Ваша GTX 1650: ~1.1x прискорення (для малих даних може не відчуватися)

**Коли використовувати GPU:**
- ✓ Багато кадрів (>100)
- ✓ Великі сітки (>1000x1000)
- ✓ Багато статистичних обчислень
- ✗ Один кадр (CPU буде швидше)

### --subsample (прореджування кадрів)
- Використовує кожен N-й кадр для анімації
- Зменшує час рендерингу та розмір відео

```bash
--subsample 1   # всі кадри (повільно, великий файл)
--subsample 5   # кожен 5-й кадр (оптимально для попереднього перегляду)
--subsample 10  # кожен 10-й кадр (швидкий огляд)
```

## Оптимальні комбінації для вашого PC

### Швидкий попередній перегляд
```bash
python3 vis2d.py --mode animation --subsample 10 --num_workers 4 --fps 15
```

### Якісна анімація
```bash
python3 vis2d.py --mode polar_animation --subsample 3 --num_workers 8 --use_gpu --fps 10
```

### Максимальна якість (всі кадри)
```bash
python3 vis2d.py --mode all --subsample 1 --num_workers 14 --use_gpu --fps 10
```
*Увага: може зайняти багато часу та RAM!*

## Структура виводу

Скрипти створюють такі файли:
```
results/*/
├── data/           # результати симуляції (.athdf, .hst, .tab)
├── materials/      # вихідний код, інпути, скрипти
└── figs_2d/        # згенеровані візуалізації
    ├── heatmap_frame_*.png
    ├── radial_profile_frame_*.png
    ├── polar_frame_*.png
    ├── azimuthal_profile_frame_*.png
    ├── disk_evolution_2d.mp4
    └── disk_evolution_polar.mp4
```

## Troubleshooting

### "No .tab files found"
- Переконайтеся, що ви в папці `materials/`
- Або вкажіть `--data_dir ../data`

### "CuPy not found"
- GPU прискорення недоступне
- Встановіть: `pip install cupy-cuda12x`
- Або працюйте без GPU (все одно працюватиме)

### "Memory error"
- Зменшіть `--num_workers`
- Збільшіть `--subsample`
- Закрийте інші програми

### Повільна обробка
- Використовуйте `--subsample 5` або більше
- Додайте `--use_gpu` (якщо є CuPy)
- Збільшіть `--num_workers` до 8-12

## Посилання

- [Legacy scripts](legacy/README.md) - застарілі версії
- Документація Athena++: https://github.com/PrincetonUniversity/athena
