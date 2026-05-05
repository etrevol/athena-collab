# VL2 Pipeline та виправлення атмосферної нестабільності

## 1. Структура VL2 інтегратора в Athena++

Метод Van Leer 2-го порядку (VL2) виконує **два stage** на кожний часовий крок `dt`.

- **Stage 1** — предиктор: інтегрує з кроком `dt/2`, отримує стан на половині кроку
- **Stage 2** — коректор: використовує стан з stage 1, интегрує повний `dt`

Для кожного stage виконується **однаковий набір задач** у фіксованому порядку:

```
CALC_HYDFLX   → обчислення потоків на гранях комірок (PLM + Riemann solver)
INT_HYD       → оновлення консервативних змінних u = u + dt * F
SRC_TERM      → додавання джерельних доданків (гравітація, в'язкість, ...)
SEND_HYD      → пакування u в буфери та відправка сусіднім MeshBlock-ам
RECV_HYD      → отримання ghost zone даних від сусідів (чекає на SEND)
SETB_HYD      → запис отриманих даних у ghost zone масиви
CONS2PRIM     → перерахунок примітивних змінних w з консервативних u
PHY_BVAL      → застосування фізичних граничних умов до w
USERWORK      → UserWorkInLoop  ← ТІЛЬКИ після stage 2 !
NEW_DT        → обчислення нового dt
```

**Ключовий факт:** `SEND_HYD` відправляє сусідам консервативні змінні `u = {D, M1, M2, M3, E}` у тому стані, в якому вони знаходяться **відразу після SRC_TERM** — до будь-якого виправлення.

---

## 2. Проблема: гравітація на атмосферних комірках

### Що таке атмосферна комірка?

Athena++ не може працювати з `rho = 0` — Riemann solver вимагає `rho > 0`. Тому всі комірки поза диском заповнюються числовою атмосферою:

```
rho   = dfloor   (наприклад, 1e-4)
press = pfloor   (наприклад, 1e-6)
v_r   = 0
v_phi = l_const / r
```

Ця атмосфера — **не фізична речовина**, а числовий placeholder.

### Що відбувається в SRC_TERM (Stage 1)?

`NewtonianGravity` застосовує гравітацію до **всіх** комірок, включаючи атмосферні:

```
g = -β/r²  ≈  -1.07×10⁶  при r = 0.6

ΔM₁ = rho_floor × g × dt  =  1e-4 × (-1.07×10⁶) × 6.4×10⁻⁵  =  -6.85

v_r_new = M₁_new / rho_floor  =  -6.85 / 1e-4  =  -68 500
```

Тепер кінетична енергія:

```
e_k = M₁² / (2ρ)  =  (6.85)² / (2 × 1e-4)  ≈  2.35×10⁵
```

Але повна енергія комірки була:

```
e_total = press_floor/(γ-1) + 0.5 × rho_floor × v_phi²
        = 1e-6/0.3 + 0.5 × 1e-4 × (620/0.6)²
        ≈  3.3×10⁻⁶  +  3.34×10⁻²  ≈  3.34×10⁻²
```

Після гравітації:

```
p_post = (γ-1) × (e_total - e_k)  =  0.3 × (3.34×10⁻² - 2.35×10⁵)  ≈  -7×10⁴
```

**Тиск від'ємний.** Термодинамічно некоректний стан.

### Чому це катастрофа?

```
Stage 1:  SRC_TERM → [некоректний стан] → SEND_HYD → (ghost zone exchange) → CONS2PRIM
                                                 ↑
                              сусідній MeshBlock отримує p < 0 в ghost zone
```

Сусідній MeshBlock у Stage 2 обчислює потоки через границю з ghost zone, де `p < 0`.
HLLC/HLLD Riemann solver з від'ємним тиском видає некоректні потоки.
Новий розрахунок отримує ще більший `|v_r|` → ланцюгова реакція по всій сітці.

`UserWorkInLoop` це **не рятує**, бо вона запускається тільки після Stage 2 — на той момент ghost zone обмін і обчислення потоків Stage 2 вже відбулись.

---

## 3. Виправлення: atmosphere enforcement у NewtonianGravity

Після застосування гравітації — одразу перевіряємо стан і виправляємо атмосферні комірки **до** `SEND_HYD`.

### Покроковий алгоритм

**Крок 1.** Застосувати гравітацію (нічого не змінилось):
```cpp
Real g = -beta_param / (r * r);
cons(IM1,k,j,i) += dt * rho * g;
cons(IEN,k,j,i) += dt * rho * g * v_r;
```

**Крок 2.** Зчитати стан після гравітації:
```cpp
Real rho_post = cons(IDN,k,j,i);   // щільність (гравітація її не змінює)
Real M1_post  = cons(IM1,k,j,i);   // радіальний імпульс — змінився!
Real M2_post  = cons(IM2,k,j,i);   // азимутальний імпульс
Real e_post   = cons(IEN,k,j,i);   // повна енергія — змінилась!
```

**Крок 3.** Обчислити кінетичну та теплову енергію:
```cpp
Real e_k    = 0.5 / max(rho_post, rho_floor) * (M1_post² + M2_post²);
Real p_post = (gamma - 1.0) * (e_post - e_k);
```

`max(rho_post, rho_floor)` захищає від ділення на нуль.

**Крок 4.** Перевірити три умови некоректності:
```cpp
if (rho_post <= rho_floor   // щільність на рівні або нижче атмосфери
 || p_post <= press_floor   // тиск від'ємний або підозріло малий
 || !isfinite(rho_post))    // NaN або Inf
```

**Крок 5.** При спрацюванні — скинути до консистентного атмосферного стану:
```cpp
v_phi_atm = l_const / r    // Keplerian кутова швидкість

cons(IDN) = rho_floor
cons(IM1) = 0.0                          // v_r = 0: атмосфера статична
cons(IM2) = rho_floor * v_phi_atm        // v_phi: Keplerian
cons(IM3) = 0.0
cons(IEN) = press_floor/(γ-1) + 0.5 × rho_floor × v_phi_atm²   // E консистентна
```

Перевірка консистентності: `p = (γ-1)(E - e_k) = press_floor` ✓

---

## 4. Чому фізичні комірки диска не зачіпаються?

Для комірки диска (`rho >> rho_floor`, `press >> press_floor`):

```
P_disk / P_atm  ≈  2700 / 1e-6  =  2.7×10⁹
```

Після гравітації тиск залишається на рівні `~10³ ÷ 10⁴`, що багатократно перевищує `press_floor = 1e-6`.
Умова `p_post <= press_floor` **не спрацьовує**.
Гравітація діє на фізичну матерію повністю некоректно, без будь-якого обмеження.

---

## 5. Порівняння з UserWorkInLoop

| Критерій | `UserWorkInLoop` | Блок в `NewtonianGravity` |
|---|---|---|
| Запускається | тільки stage 2 | кожен stage (1 і 2) |
| Момент у pipeline | після `PHY_BVAL` | **до `SEND_HYD`** |
| Оновлює `w` (primitives) | так | ні — тільки `cons` |
| Оновлює `u` (conservative) | так | так |
| Захищає ghost zone exchange | **ні** | **так** |

Логіка однакова, але timing вирішальний: тільки виправлення **всередині SRC_TERM** гарантує, що `SEND_HYD` відправить сусідам коректний стан.

---

## 6. Схема потоку для одного часового кроку

```
┌─────────────────────────────────────────────────────┐
│                     STAGE 1                         │
│                                                     │
│  CALC_HYDFLX  →  flux з коректних w (stage 0)       │
│  INT_HYD      →  u = u + dt × F                     │
│  SRC_TERM:                                          │
│    NewtonianGravity:                                │
│      1. cons(IM1) += dt × rho × g       ← гравітація│
│      2. cons(IEN) += dt × rho × g × v_r             │
│      3. if (p_post <= floor) → reset    ← FIX       │
│                                         ← SAFE STATE│
│  SEND_HYD     →  відправка u сусідам   ← коректно! │
│  SETB_HYD     →  заповнення ghost zone              │
│  CONS2PRIM    →  w = f(u)                           │
│                                                     │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│                     STAGE 2                         │
│                                                     │
│  CALC_HYDFLX  →  flux з w stage 1                   │
│  INT_HYD      →  u = u_0 + dt × F                   │
│  SRC_TERM     →  (аналогічний reset)    ← FIX       │
│  SEND_HYD     →  відправка                          │
│  SETB_HYD                                           │
│  CONS2PRIM                                          │
│  PHY_BVAL     →  фізичні BC                         │
│  USERWORK     →  UserWorkInLoop         ← stage 2   │
│  NEW_DT                                             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 7. Відповідні файли Athena++

| Файл | Що містить |
|---|---|
| `src/task_list/time_integrator.cpp` | Порядок задач VL2, визначення залежностей між ними |
| `src/eos/adiabatic_hydro.cpp` | `ConservativeToPrimitive`: застосування floor до ghost zone |
| `src/bvals/cc/bvals_cc.cpp` | `LoadBoundaryBufferSameLevel`: пакування `u` в буфери |
| `src/pgen/acc_disk_temp_visc.cpp` | `NewtonianGravity` з виправленням, `UserWorkInLoop` |

---

## 8. Навігація по вихідному коду

### Де визначається порядок задач

[src/task_list/time_integrator.cpp](src/task_list/time_integrator.cpp), рядки 908–1000 — тут **будується граф залежностей** (`AddTask(A, B)` означає "A запускається тільки після B"):

```cpp
// L922
AddTask(CALC_HYDFLX, NONE);            // нема залежностей → перша задача
// L936
AddTask(INT_HYD, CALC_HYDFLX);         // INT після CALC
// L979
AddTask(SRC_TERM, INT_HYD);            // SOURCE після INT
// L994  ← ключовий рядок
AddTask(SEND_HYD, src_aterm);          // SEND після SRC_TERM
// L997
AddTask(RECV_HYD, NONE);               // RECV не чекає нікого (async)
// L998
AddTask(SETB_HYD, (RECV_HYD|SRC_TERM)); // SET після обох
```

Далі (рядки 1139–1194):
```cpp
AddTask(CONS2PRIM, SETB_HYD);          // ConsToPrim після заповнення ghost zone
AddTask(PHY_BVAL, CONS2PRIM);          // BC після ConsToPrim
AddTask(USERWORK, PHY_BVAL);           // USERWORK — остання!
```

---

### Реалізація кожної задачі

#### 1. `CALC_HYDFLX` — [L1694](src/task_list/time_integrator.cpp#L1694)

```cpp
// L1700–1703: stage=1 VL2 використовує xorder=1 (дешевша реконструкція)
// stage=2 використовує повний xorder
phydro->CalculateFluxes(phydro->w, ...);  // читає w (primitives) поточного stage
```

**Вхід: `w` (примітиви).** На stage 1 це `w` з кінця попереднього timestep (ICs для першого кроку).

---

#### 2. `INT_HYD` — [L1797](src/task_list/time_integrator.cpp#L1797)

```cpp
// L1826: додає дивергенцію потоків до u
ph->AddFluxDivergence(wght, ph->u);         // u += dt * div(F)
// L1827: геометричні члени (сферична/циліндрична геометрія)
pmb->pcoord->AddCoordTermsDivergence(...);
```

**Оновлює: `u` (conservative).** `w` ще не змінюється.

---

#### 3. `SRC_TERM` → `AddSourceTerms` — [L1890](src/task_list/time_integrator.cpp#L1890)

```cpp
// L1903: обчислює dt для цього stage (stage 1: dt/2, stage 2: dt)
Real dt = (stage_wghts[(stage-1)].beta) * (pmb->pmy_mesh->dt);

// L1907: викликає NewtonianGravity (наш код у pgen)
ph->hsrc.AddSourceTerms(t_start_stage, dt, ph->flux, ph->w, ps->r, pf->bcc,
                         ph->u, ps->s);
//                                               ↑           ↑
//                                    читає w (stale!)   пише в u (cons)
```

`w` тут — **застарілий** стан з попереднього `CONS2PRIM`. `CONS2PRIM` для поточного stage ще не запускався. Саме тому в `NewtonianGravity` ми пишемо тільки в `cons`, а не в `prim` — будь-який запис у `w` тут буде перезаписаний наступним `CONS2PRIM`.

---

#### 4. `SEND_HYD` — [L1970](src/task_list/time_integrator.cpp#L1970)

```cpp
// L1974: явно свопає pointer на u (cons), не w (prim)
pmb->phydro->hbvar.SwapHydroQuantity(pmb->phydro->u, HydroBoundaryQuantity::cons);
//                                              ↑
//                          пакує саме u — стан після SRC_TERM, ДО CONS2PRIM
pmb->phydro->hbvar.SendBoundaryBuffers();
```

Упакування у [src/bvals/cc/bvals_cc.cpp](src/bvals/cc/bvals_cc.cpp), L287–300:

```cpp
// L299–300
AthenaArray<Real> &var = *var_cc;          // var_cc = u, після SwapHydroQuantity
BufferUtility::PackData(var, buf, ...);    // пакує у точності той u, що var_cc вказує
```

**Якщо `u` містить `p < 0` після гравітації → сусід отримає саме цей стан у ghost zone.**

---

#### 5. `SETB_HYD` / `RECV_HYD` — [src/bvals/cc/bvals_cc.cpp](src/bvals/cc/bvals_cc.cpp), L390–418

Сусідній MeshBlock отримує буфер і записує у свій ghost zone масив `u`. Далі обидва блоки продовжують до `CONS2PRIM` з цим станом.

---

#### 6. `CONS2PRIM` → `ConservedToPrimitive` — [src/eos/adiabatic_hydro.cpp](src/eos/adiabatic_hydro.cpp), L39

```cpp
// L62: floor на щільність (оновлює і cons, і prim)
u_d = (u_d > density_floor_) ? u_d : density_floor_;
w_d = u_d;

// L64–67: рахує швидкість з імпульсу
w_vx = u_m1 / u_d;   // ← якщо u_m1 >> u_d (атмосф. комірка після гравітації без фіксу):
                      //   w_vx = -68500 — і це записується в w, без жодної перевірки!

// L73–75: floor на тиск — виправляє u_e і w_p, але w_vx вже записаний!
u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_/gm1) + e_k);
w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;
```

Без нашого фіксу: тиск виправляється, але `w_vx` (= v_r) залишається `-68500`. Це і є джерело каскадної нестабільності.

---

#### 7. `USERWORK` — [L2305](src/task_list/time_integrator.cpp#L2305)

```cpp
TaskStatus TimeIntegratorTaskList::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success;  // L2306 ← виходить на stage 1!

  pmb->UserWorkInLoop();   // L2308 ← ТІЛЬКИ на останньому stage (stage 2 для VL2)
  return TaskStatus::success;
}
```

Для VL2: `nstages = 2`. На stage 1 рядок L2306 повертає `success` **без** виклику `UserWorkInLoop`. Тому `UserWorkInLoop` не може захистити ghost zone exchange на stage 1.

---

### Підсумкова таблиця

| Задача | Файл | Рядки | Читає | Пише |
|---|---|---|---|---|
| `CALC_HYDFLX` | time_integrator.cpp | L1694 | `w` | flux arrays |
| `INT_HYD` | time_integrator.cpp | L1797 | flux | `u` |
| `SRC_TERM` (гравітація) | time_integrator.cpp | L1890 | `w` (stale), `u` | `u` |
| `SEND_HYD` | time_integrator.cpp | L1970 | `u` | буфери сусідів |
| `SETB_HYD` | bvals_cc.cpp | L390 | буфери | ghost zone в `u` |
| `CONS2PRIM` | adiabatic_hydro.cpp | L39 | `u` | `w` + floor на `u` |
| `PHY_BVAL` | time_integrator.cpp | L2278 | `w` | `w` |
| `USERWORK` | time_integrator.cpp | L2305 | `w`, `u` | `w`, `u` |
