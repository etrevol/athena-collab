#!/usr/bin/env python3
"""
Analyze floor-value cells relative to meshblock boundaries in Athena++ tab output.
128×128 cylindrical 2D disk, 4 MeshBlocks (64×64 each):
  block0: inner-r (global i 0-63),  lower-phi (global j 0-63)
  block1: outer-r (global i 64-127), lower-phi (global j 0-63)
  block2: inner-r (global i 0-63),  upper-phi (global j 64-127)
  block3: outer-r (global i 64-127), upper-phi (global j 64-127)
"""

import numpy as np
import glob
import os

DATA_DIR = "/home/etrevol/athena-collab/results/TEST/sample-20260331-110149-acc_disk/data"
BASE     = "acc_disk_visc"
DFLOOR   = 1.0e-4   # actual rho floor observed in data (problem-level atmosphere, not numeric dfloor)
PFLOOR   = 1.0e-6   # actual press floor observed in data
NI_TOTAL = 128
NJ_TOTAL = 128
NI_BLOCK = 64
NJ_BLOCK = 64

# ------------------------------------------------------------------
# Helper: block index → (i_global_offset, j_global_offset)
#   block0: inner-r, lower-phi  → (0,  0)
#   block1: outer-r, lower-phi  → (64, 0)
#   block2: inner-r, upper-phi  → (0,  64)
#   block3: outer-r, upper-phi  → (64, 64)
# ------------------------------------------------------------------
BLOCK_OFFSETS = {
    0: (0,  0),
    1: (64, 0),
    2: (0,  64),
    3: (64, 64),
}

def read_block(filepath):
    """Return dict with arrays: i_loc, j_loc, r, phi, rho, press, vr, vphi, vz"""
    cols = np.loadtxt(filepath, comments="#")
    # columns: i  r  j  phi  rho  press  vr  vphi  vz
    return {
        "i_loc":  cols[:, 0].astype(int),
        "r":      cols[:, 1],
        "j_loc":  cols[:, 2].astype(int),
        "phi":    cols[:, 3],
        "rho":    cols[:, 4],
        "press":  cols[:, 5],
        "vr":     cols[:, 6],
        "vphi":   cols[:, 7],
        "vz":     cols[:, 8],
    }

def load_all_blocks(step):
    """Load all 4 blocks for given step string (e.g. '00001').
    Returns global 2D arrays: rho[NI,NJ], r[NI], phi[NJ].
    Local indices run 2..65 → global offset by block.
    """
    rho_global   = np.full((NI_TOTAL, NJ_TOTAL), np.nan)
    press_global = np.full((NI_TOTAL, NJ_TOTAL), np.nan)
    r_global     = np.full(NI_TOTAL, np.nan)
    phi_global   = np.full(NJ_TOTAL, np.nan)
    block_map    = np.full((NI_TOTAL, NJ_TOTAL), -1, dtype=int)

    for bid in range(4):
        fpath = os.path.join(DATA_DIR, f"{BASE}.block{bid}.out1.{step}.tab")
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        d = read_block(fpath)
        i_off, j_off = BLOCK_OFFSETS[bid]

        # local indices 2..65 → 0..63 within block → add offset for global
        i_global = d["i_loc"] - 2 + i_off   # shifts local 2→0, then to block offset
        j_global = d["j_loc"] - 2 + j_off

        rho_global[i_global, j_global]   = d["rho"]
        press_global[i_global, j_global] = d["press"]
        block_map[i_global, j_global]    = bid

        # fill coordinate arrays from each block
        # r depends only on i, phi depends only on j
        for ii in range(len(d["i_loc"])):
            ig = i_global[ii]
            jg = j_global[ii]
            r_global[ig]   = d["r"][ii]
            phi_global[jg] = d["phi"][ii]

    return rho_global, press_global, r_global, phi_global, block_map

# ------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Load step 00000 and 00001")
print("=" * 70)

rho0, press0, r_arr, phi_arr, bmap0 = load_all_blocks("00000")
rho1, press1, _,     _,       bmap1 = load_all_blocks("00001")

print(f"Global grid shape: {rho0.shape}  (i=radial, j=phi)")
print(f"r range:   {r_arr[0]:.4f} to {r_arr[-1]:.4f}")
print(f"phi range: {phi_arr[0]:.4f} to {phi_arr[-1]:.4f}")
print(f"Step 00000  rho min={rho0.min():.3e}  max={rho0.max():.3e}   press min={press0.min():.3e}")
print(f"Step 00001  rho min={rho1.min():.3e}  max={rho1.max():.3e}   press min={press1.min():.3e}")
print(f"NOTE: actual floor values observed = rho_floor=1e-4, press_floor=1e-6 (problem-level atmosphere)")
print(f"      (dfloor=1e-8 / pfloor=1e-10 in input are numeric floors, not hit here)")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Floor cells at step 00001 (rho <= 1.01e-8)")
print("=" * 70)

FLOOR_THRESH = 1.01 * DFLOOR  # 1.01e-4  (within 1% of problem-level rho floor)
floor_mask1 = rho1 <= FLOOR_THRESH
n_floor1    = floor_mask1.sum()

i_fl1, j_fl1 = np.where(floor_mask1)
print(f"Number of floor cells (rho <= {FLOOR_THRESH:.2e}): {n_floor1}")

if n_floor1 > 0:
    print(f"  i_global range: {i_fl1.min()} .. {i_fl1.max()}")
    print(f"  j_global range: {j_fl1.min()} .. {j_fl1.max()}")

    # Concentration by i index
    i_counts = np.bincount(i_fl1, minlength=NI_TOTAL)
    j_counts = np.bincount(j_fl1, minlength=NJ_TOTAL)

    print("\n--- Floor cell counts per radial index (i_global) ---")
    nonzero_i = np.where(i_counts > 0)[0]
    for ig in nonzero_i:
        marker = " <-- r-split boundary (i=63|64)" if ig in (63, 64) else ""
        print(f"  i={ig:3d}  count={i_counts[ig]:4d}  r={r_arr[ig]:.4f}{marker}")

    print("\n--- Floor cell counts per phi index (j_global) ---")
    nonzero_j = np.where(j_counts > 0)[0]
    for jg in nonzero_j:
        marker = ""
        if jg == 0:   marker = " <-- phi block edge (j=0)"
        if jg == 63:  marker = " <-- phi split boundary (last j in lower block)"
        if jg == 64:  marker = " <-- phi split boundary (first j in upper block)"
        if jg == 127: marker = " <-- phi block edge (j=127)"
        print(f"  j={jg:3d}  count={j_counts[jg]:4d}  phi={phi_arr[jg]:.4f}{marker}")

    print("\n--- Summary for boundary vs interior j indices ---")
    boundary_j = {0, 63, 64, 127}
    cnt_boundary = sum(j_counts[j] for j in boundary_j)
    cnt_interior = n_floor1 - cnt_boundary
    avg_interior = cnt_interior / (NJ_TOTAL - len(boundary_j)) if (NJ_TOTAL - len(boundary_j)) > 0 else 0
    avg_boundary = cnt_boundary / len(boundary_j)
    print(f"  Floor cells at j={{0,63,64,127}} (boundaries): {cnt_boundary}")
    print(f"  Floor cells at all other j (interior):         {cnt_interior}")
    print(f"  Avg floor cells per boundary j slot: {avg_boundary:.1f}")
    print(f"  Avg floor cells per interior j slot: {avg_interior:.1f}")

    # Specifically at phi-split
    print(f"\n  j=63 (last in lower-phi block): {j_counts[63]}")
    print(f"  j=64 (first in upper-phi block): {j_counts[64]}")
    print(f"  j=0  (phi block start):           {j_counts[0]}")
    print(f"  j=127 (phi block end):            {j_counts[127]}")
    print(f"  r-split: i=63: {i_counts[63]},  i=64: {i_counts[64]}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Radial distribution of floor cells at step 00001")
print("=" * 70)

print("i_global  r_value    n_floor_cells")
for ig in range(NI_TOTAL):
    cnt = int(i_counts[ig]) if n_floor1 > 0 else 0
    if cnt > 0:
        print(f"  {ig:3d}    {r_arr[ig]:.5f}    {cnt}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Near-floor cells at step 00000 (rho < 2e-8)")
print("=" * 70)

NEAR_FLOOR_THRESH = 2.0 * DFLOOR  # 2e-4
near_mask0 = rho0 < NEAR_FLOOR_THRESH
n_near0 = near_mask0.sum()
i_nf0, j_nf0 = np.where(near_mask0)

print(f"Cells with rho < {NEAR_FLOOR_THRESH:.1e} at step 00000: {n_near0}")
if n_near0 > 0:
    i_cnts0 = np.bincount(i_nf0, minlength=NI_TOTAL)
    j_cnts0 = np.bincount(j_nf0, minlength=NJ_TOTAL)

    print("\n--- Near-floor by i (radial) at step 0 ---")
    for ig in np.where(i_cnts0 > 0)[0]:
        marker = " <-- r-split" if ig in (63, 64) else ""
        print(f"  i={ig:3d}  count={i_cnts0[ig]:4d}  r={r_arr[ig]:.4f}{marker}")

    print("\n--- Near-floor by j (phi) at step 0 ---")
    for jg in np.where(j_cnts0 > 0)[0]:
        marker = ""
        if jg in (0, 63, 64, 127):
            marker = " <-- meshblock edge"
        print(f"  j={jg:3d}  count={j_cnts0[jg]:4d}  phi={phi_arr[jg]:.4f}{marker}")

    print(f"\n  At j=63: {j_cnts0[63]},  j=64: {j_cnts0[64]}")
    print(f"  At i=63: {i_cnts0[63]},  i=64: {i_cnts0[64]}")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5b: PRESSURE floor cells at step 00001 (press <= 1.01e-6)")
print("=" * 70)

PFLOOR_THRESH = 1.01 * PFLOOR
pfloor_mask1 = press1 <= PFLOOR_THRESH
n_pfl1 = pfloor_mask1.sum()
i_pfl1, j_pfl1 = np.where(pfloor_mask1)
print(f"Number of press-floor cells (press <= {PFLOOR_THRESH:.2e}): {n_pfl1}")
if n_pfl1 > 0:
    pi_counts = np.bincount(i_pfl1, minlength=NI_TOTAL)
    pj_counts = np.bincount(j_pfl1, minlength=NJ_TOTAL)
    print(f"  i_global range: {i_pfl1.min()} .. {i_pfl1.max()}")
    print(f"  j_global range: {j_pfl1.min()} .. {j_pfl1.max()}")
    print(f"  at j=63: {pj_counts[63]},  j=64: {pj_counts[64]},  j=0: {pj_counts[0]},  j=127: {pj_counts[127]}")
    print(f"  at i=63: {pi_counts[63]},  i=64: {pi_counts[64]},  i=0: {pi_counts[0]},  i=127: {pi_counts[127]}")
    # Do rho-floor and pressure-floor cells overlap?
    overlap = (floor_mask1 & pfloor_mask1).sum()
    print(f"  Cells at BOTH rho and press floor: {overlap}")

print("\n" + "=" * 70)
print("=" * 70)

# Focus on radial rings that actually have floor cells (step 1) + a few others
if n_floor1 > 0:
    rings_to_check = sorted(set(i_fl1.tolist()))
else:
    rings_to_check = list(range(NI_TOTAL))  # check all if no floor cells

print(f"Checking {len(rings_to_check)} radial rings that contain floor cells at step 1")
print(f"(Reporting discontinuities > 10% relative jump near j=63/64 split)\n")

discontinuities = []
for ig in rings_to_check:
    rho_row = rho1[ig, :]
    # jump across phi-split boundary: j=63 → j=64
    val_left  = rho_row[63]
    val_right = rho_row[64]
    if val_left > 0 and val_right > 0:
        rel_jump = abs(val_right - val_left) / max(val_left, val_right)
    else:
        rel_jump = 0.0

    # also check j=0/j=127 wrap-around
    val_wrap_l = rho_row[127]
    val_wrap_r = rho_row[0]
    if val_wrap_l > 0 and val_wrap_r > 0:
        rel_wrap = abs(val_wrap_r - val_wrap_l) / max(val_wrap_l, val_wrap_r)
    else:
        rel_wrap = 0.0

    if rel_jump > 0.05 or val_left <= FLOOR_THRESH or val_right <= FLOOR_THRESH:
        flag = ""
        if val_left <= FLOOR_THRESH or val_right <= FLOOR_THRESH:
            flag = " **FLOOR VALUE**"
        print(f"  i={ig:3d} r={r_arr[ig]:.4f}:  rho[j=63]={val_left:.3e}  rho[j=64]={val_right:.3e}  "
              f"rel_jump={rel_jump*100:.1f}%{flag}")
        discontinuities.append((ig, val_left, val_right, rel_jump))

if not discontinuities:
    print("  No significant discontinuities at phi-split boundary (j=63/64)")

# Show a few representative phi profiles across the split
print("\n--- Sample phi profiles spanning the j=63|64 split ---")
sample_rings = rings_to_check[:min(5, len(rings_to_check))] if rings_to_check else list(range(0, 128, 32))
for ig in sample_rings:
    rho_row = rho1[ig, :]
    print(f"\n  i={ig:3d} r={r_arr[ig]:.4f}")
    print(f"  j=58..69: ", end="")
    for jg in range(58, 70):
        val = rho_row[jg]
        flag = "*" if val <= FLOOR_THRESH else " "
        print(f"j{jg}={val:.2e}{flag}", end="  ")
    print()

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: 2D heatmap / coordinate list of floor cells at step 00001")
print("=" * 70)

if n_floor1 > 0:
    print(f"\nAll (i_global, j_global) floor cells at step 00001:")
    print(f"{'i_global':>9}  {'j_global':>9}  {'r':>10}  {'phi':>10}  {'rho':>12}  block")

    # Sort by i then j for readability
    idx_sort = np.lexsort((j_fl1, i_fl1))
    for k in idx_sort:
        ig = i_fl1[k]
        jg = j_fl1[k]
        b  = bmap1[ig, jg]
        print(f"  {ig:7d}  {jg:9d}  {r_arr[ig]:10.5f}  {phi_arr[jg]:10.5f}  {rho1[ig,jg]:12.4e}  block{b}")

    # ASCII heatmap (compressed): show which (i,j) cells are floor
    print("\n--- ASCII floor map (rows=i, cols=j; '.' = floor, ' ' = ok) ---")
    print("    j: 0" + " " * 28 + "32" + " " * 28 + "64" + " " * 28 + "96" + " "*27 + "127")
    print("    " + "-" * 128)
    for ig in range(NI_TOTAL):
        row_str = ""
        has_floor = False
        for jg in range(NJ_TOTAL):
            if floor_mask1[ig, jg]:
                row_str += "."
                has_floor = True
            else:
                row_str += " "
        if has_floor:
            split_marker = "| " if ig in (63, 64) else "  "
            print(f"i={ig:3d}{split_marker}|{row_str}| r={r_arr[ig]:.4f}")
    print("    " + "-" * 128)
else:
    print("No floor cells found at step 00001.")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

meshblock_j = {63, 64, 0, 127}
meshblock_i = {63, 64, 0, 127}

if n_floor1 > 0:
    fl_at_j_boundary = sum(1 for jg in j_fl1 if jg in meshblock_j)
    fl_at_i_boundary = sum(1 for ig in i_fl1 if ig in meshblock_i)
    fl_at_any_boundary = sum(
        1 for ig, jg in zip(i_fl1, j_fl1)
        if ig in meshblock_i or jg in meshblock_j
    )
    print(f"Total floor cells:                    {n_floor1}")
    print(f"Floor cells at j in {{0,63,64,127}}:    {fl_at_j_boundary}  ({100*fl_at_j_boundary/n_floor1:.1f}%)")
    print(f"Floor cells at i in {{0,63,64,127}}:    {fl_at_i_boundary}  ({100*fl_at_i_boundary/n_floor1:.1f}%)")
    print(f"Floor cells at ANY meshblock boundary: {fl_at_any_boundary}  ({100*fl_at_any_boundary/n_floor1:.1f}%)")

    # Expected if uniform: each boundary col = 4/128 = 3.1% of j slots
    expected_frac = 4 / NJ_TOTAL
    actual_frac_j = fl_at_j_boundary / n_floor1
    print(f"\nExpected fraction at j-boundaries (random): {expected_frac*100:.1f}%")
    print(f"Actual   fraction at j-boundaries:           {actual_frac_j*100:.1f}%")
    if actual_frac_j > 3 * expected_frac:
        print("=> STRONG concentration at phi-meshblock boundaries")
    elif actual_frac_j > 1.5 * expected_frac:
        print("=> MODERATE concentration at phi-meshblock boundaries")
    else:
        print("=> NO preferential concentration at phi-meshblock boundaries")

    print(f"\nPhi-split (j=63/64) specifically:")
    print(f"  j=63 floor count: {j_counts[63]},  j=64 floor count: {j_counts[64]}")
    total_phi_split = j_counts[63] + j_counts[64]
    avg_other = (n_floor1 - total_phi_split) / (NJ_TOTAL - 2) if NJ_TOTAL > 2 else 0
    print(f"  Average floor count at j=63 and j=64 combined: {total_phi_split/2:.1f}")
    print(f"  Average floor count at all other j:             {avg_other:.1f}")

else:
    print("No floor cells at step 00001 - simulation is clean.")
    print(f"Min rho at step 1: {rho1.min():.4e}  (floor is {DFLOOR:.1e})")
    print(f"Min rho at step 0: {rho0.min():.4e}")
    # Show what the actual minimum rho values are near the boundaries
    print("\nRho values near phi-split boundary (j=60..67) for inner radii (i=0..5):")
    print(f"{'i':>4}", end="")
    for jg in range(60, 68):
        print(f"  j={jg:3d}", end="")
    print()
    for ig in range(6):
        print(f"{ig:4d}", end="")
        for jg in range(60, 68):
            print(f"  {rho1[ig,jg]:.2e}", end="")
        print()
