# sweep.sh — Інструкція

Автоматичний параметричний sweep для Athena++.  
Запускає серію симуляцій зі зміненими параметрами інпут-файлу (сорс не чіпається), детектує зависання і генерує Markdown-звіт.

---

## Швидкий старт

```bash
# 1. Зібрати бінарник (якщо ще не зібраний)
bash build.sh

# 2. Визначити тести (масив TESTS у верхній частині скрипта)
#    → відредагуй scripts/sweep.sh

# 3. Перевірити без запуску
bash scripts/sweep.sh --dry-run

# 4. Запустити вночі у фоні
nohup bash scripts/sweep.sh > /dev/null 2>&1 & echo $!   # запиши PID!
```

---

## Налаштування (редагувати у верхній частині скрипта)

| Змінна | За замовчуванням | Призначення |
|---|---|---|
| `PROBLEM` | `acc_disk_temp_visc` | Назва задачі (ім'я `.cpp` файлу) |
| `INPUT_TEMPLATE` | `inputs/hydro/athinput.acc_disk_temp_visc` | Шаблон інпут-файлу |
| `HANG_TIMEOUT` | `60` | Секунд без зменшення ETA → kill |
| `ETA_POLL_INTERVAL` | `5` | Як часто перевіряти ETA в лозі (сек) |
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
results/SWEEP/sweep-YYYYMMDD-HHMMSS/
  sweep.log                  ← повний лог консолі
  REPORT.md                  ← фінальний звіт
  acc_disk_temp_visc.cpp     ← snapshot сорсу на момент запуску
  athinput.acc_disk_temp_visc ← snapshot шаблону інпуту
  t01_nu0.1_a0.0/
    athinput.in              ← змінений інпут для цього тесту
    params.txt               ← які параметри були змінені
    run.log                  ← вивід Athena++ (ETA, прогрес, помилки)
    status                   ← COMPLETED | KILLED_HANG | FAILED | UNKNOWN
    start_time / end_time    ← Unix-таймстемпи
    vis1d.py / vis2d.py / vishst.py  ← скрипти візуалізації
    data/                    ← *.tab, *.hst та інші файли Athena++
  t02_nu0.5_a0.0/
    ...
```

---

## Моніторинг під час виконання

```bash
# Живий лог (показує поточний прогрес і ETA):
tail -f results/SWEEP/sweep-*/sweep.log

# Дізнатись PID якщо не записав:
pgrep -af sweep.sh

# Переглянути статуси вже виконаних тестів:
grep -h "" results/SWEEP/sweep-<timestamp>/t*/status
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
# Відкрити звіт:
cat results/SWEEP/sweep-<timestamp>/REPORT.md

# Лог конкретного тесту:
cat results/SWEEP/sweep-<timestamp>/t03_nu1.0_a0.0/run.log

# Швидкий огляд усіх статусів:
grep -h "" results/SWEEP/sweep-<timestamp>/t*/status
```

---

## Статуси у звіті

| Статус | Значення |
|---|---|
| `✅ COMPLETED` | Симуляція дійшла до `tlim` або `nlim` |
| `⏱️ KILLED (hang)` | ETA не зменшувалась `HANG_TIMEOUT` секунд → примусово зупинено |
| `❌ FAILED` | Athena++ вивів `FATAL ERROR`, Segfault або `Aborted` |
| `❓ UNKNOWN` | Процес завершився, але ознак успіху/помилки не знайдено |
| `🔍 DRY RUN` | Тест не виконувався (`--dry-run` режим) |
