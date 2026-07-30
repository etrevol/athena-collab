# sweep.sh — Інструкція

Автоматичний параметричний sweep для Athena++.  
Запускає серію симуляцій зі зміненими параметрами інпут-файлу (сорс не чіпається), детектує зависання і генерує Markdown-звіт.

---

## Швидкий старт

```bash
# 1. Зібрати бінарник (якщо ще не зібраний)
bash build.sh

# 2. Визначити тести (масив TESTS у верхній частині скрипта)
#    → відредагуй scripts/practice/run/sweep.sh

# 3. Перевірити без запуску
bash scripts/practice/run/sweep.sh --dry-run

# 4. Запустити вночі у фоні (результати в results/sweeps/sweep-*)
nohup bash scripts/practice/run/sweep.sh > sweep_output.log 2>&1 & echo $!
```

---

## Налаштування (редагувати у верхній частині скрипта)

| Змінна | За замовчуванням | Призначення |
|---|---|---|
| `PROBLEM` | `acc_disk_visc` | Назва задачі (ім'я `.cpp` файлу) |
| `INPUT_TEMPLATE` | `inputs/hydro/athinput.acc_disk_visc` | Шаблон інпут-файлу |
| `HANG_TIMEOUT` | `90` | Секунд без прогресу sim_time → kill |
| `ETA_POLL_INTERVAL` | `5` | Як часто перевіряти прогрес в лозі (сек) |
| `USE_MPI` | `0` | `1` — увімкнути MPI (потрібно зібрати з `-mpi`) |
| `NUM_MPI_PROCS` | `4` | Кількість MPI процесів |

### Масив тестів `TESTS`

Кожен рядок — окремий тест, список `key=value` через пробіл:

```bash
TESTS=(
  "nu_iso=0.1   alpha=0.0"
  "nu_iso=0.5   alpha=0.0"
  "dfloor=1.0e-6   pfloor=1.0e-8   cfl_number=0.3"
  "mesh_nx1=64   mesh_nx2=64   block_nx1=32  block_nx2=32"
)
```

**Спеціальні префікси ключів:**

| Ключ у `TESTS` | Де заміняється |
|---|---|
| `mesh_nx1`, `mesh_nx2` | Блок `<mesh>` |
| `block_nx1`, `block_nx2` | Блок `<meshblock>` |
| Всі інші | Перше входження в будь-якому блоці |

### Таблиця скорочень (для назв папок)

Додай нові скорочення у `ABBREV` за потреби:

```bash
declare -A ABBREV=(
  [nu_iso]="nu"   [alpha]="a"
  [dfloor]="df"   [pfloor]="pf"
  [mesh_nx1]="nx1" [mesh_nx2]="nx2"
  ...
)
```

---

## Структура результатів

```
results/sweeps/sweep-YYYYMMDD-HHMMSS/
  sweep.log                  ← повний лог консолі
  REPORT.md                  ← фінальний Markdown-звіт
  acc_disk_visc.cpp          ← snapshot сорсу на момент запуску
  athinput.acc_disk_visc     ← snapshot шаблону інпуту
  t01_nu0.0_a0.0/
    athinput.in              ← змінений інпут для цього тесту
    params.txt               ← які параметри були змінені
    bin.txt                  ← яка версія бінарника використана
    run.log                  ← вивід Athena++ (прогрес, помилки)
    status                   ← COMPLETED | KILLED_HANG | FAILED | UNKNOWN
    start_time / end_time    ← Unix-таймстемпи
    vis1d.py / vis2d.py / vishst.py  ← скрипти візуалізації (скопійовані)
    data/                    ← *.tab, *.hst та інші файли Athena++
    figs_2d/                 ← PNG-графіки (якщо vis2d.py спрацював)
    figs_hst/                ← графіки від vishst.py (якщо спрацював)
  t02_nu1.0_a0.001/
    ...
```

---

## Моніторинг під час виконання

```bash
# Живий лог (показує поточний прогрес):
tail -f results/sweeps/sweep-*/sweep.log

# Дізнатись PID якщо не записав:
pgrep -af "sweep.sh"

# Переглянути статуси вже виконаних тестів:
grep -h "" results/sweeps/sweep-<timestamp>/t*/status

# Чи запущений якісь тест прямо зараз:
ps aux | grep athena
```

---

## Зупинка

| Команда | Поведінка |
|---|---|
| `kill -SIGTERM <PID>` | Коректна зупинка: чекає завершення поточного тесту, генерує звіт |
| `kill -SIGINT <PID>` | Негайна зупинка: вбиває Athena++, генерує частковий звіт |
| `Ctrl+C` (якщо не у фоні) | Те саме що SIGINT |

> Обидва варіанти завжди генерують `REPORT.md` з результатами вже виконаних тестів.

---

## Вранці: перегляд результатів

```bash
# Список усіх запусків:
ls -dt results/sweeps/sweep-*/ | head -5

# Відкрити звіт найсвіжішого запуску:
cat results/sweeps/sweep-*/REPORT.md | head -100

# Лог конкретного тесту (наприклад t02):
cat results/sweeps/sweep-<timestamp>/t02_*/run.log

# Швидкий огляд усіх статусів:
ls -d results/sweeps/sweep-<timestamp>/t*/ | while read d; do echo "$(basename $d): $(cat $d/status)"; done
```

---

## Детектування зависання (hang detection)

Скрипт моніторить прогрес симуляції двома способами:

1. **`sim_time` progress (PRIMARY)**: якщо `sim_time` в логу не змінився протягом `HANG_TIMEOUT` (90 сек за замовчуванням) → kill.  
   Це ловить справжні зависання (timestep collapse, `dt → 1e-88`).

2. **Output check (FALLBACK)**: якщо `sim_time` ще не з'явився в логу (рання стадія), перевіряєм факт появи нових рядків в логу.  
   Якщо нових рядків нема протягом `HANG_TIMEOUT` → kill.

> **Важливо**: hang-detection оснований на прогресі `sim_time`, не на ETA.  
> Це уникає хибних kill'єрів, коли ETA стоїть на місці але timestep не обвалювався.

---

## Статуси у звіті

| Статус | Значення |
|---|---|
| `✅ COMPLETED` | Симуляція дійшла до `tlim` або `nlim` |
| `⏱️ KILLED (hang)` | `sim_time` не прогресував `HANG_TIMEOUT` секунд → примусово зупинено |
| `❌ FAILED` | Athena++ вивів `FATAL ERROR`, Segfault, `Aborted` або `Error:` |
| `❓ UNKNOWN` | Процес завершився, але ознак успіху/помилки не знайдено |
| `🔍 DRY RUN` | Тест не виконувався (`--dry-run` режим) |

---

## Post-processing (автоматична візуалізація)

На завершення sweep скрипт запускає `vis2d.py` і `vishst.py` для кожного тесту:

- **`vis2d.py`** зчитує `.tab` файли і малює контурні графіки (PNG) → `test_dir/figs_2d/`
- **`vishst.py`** парсить `.hst` (history) файл → таблиці, графіки → `test_dir/figs_hst/`

Якщо симуляція не має вихідних даних (KILLED_HANG, FAILED) — візуалізація пропускається з WARN.

> Post-processing **не блокує** звіт: навіть якщо vis-скрипти впадуть, `REPORT.md` все рівно буде згенерований.
