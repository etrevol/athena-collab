#!/usr/bin/env python3
"""
Analyze Athena++ accretion disk output to diagnose "disk disappears at first timestep".
Reads 4 block files for steps 00000 and 00001, merges into 128x128 arrays,
and reports mass conservation, floor hits, and block-boundary discontinuities.
"""

import numpy as np
import os

DATA_DIR = "/home/etrevol/athena-collab/results/TEST/sample-20260331-110149-acc_disk/data/"
DFLOOR = 1e-4
NR_GLOBAL = 128
NPHI_GLOBAL = 128
NR_BLOCK = 64
NPHI_BLOCK = 64

# Block layout (verified from actual data):
#   block0: global i=[0:64],  global j=[0:64]   (r inner, phi lower)
#   block1: global i=[64:128],global j=[0:64]   (r outer, phi lower)
#   block2: global i=[0:64],  global j=[64:128] (r inner, phi upper)
#   block3: global i=[64:128],global j=[64:128] (r outer, phi upper)
BLOCK_INFO = {
    0: (0,  64,  0,  64),   # (i_start, i_end, j_start, j_end)
    1: (64, 128,  0,  64),
    2: (0,  64,  64, 128),
    3: (64, 128, 64, 128),
}

def load_block(step_str, block_id):
    """
    Load one .tab block file. Returns dict with arrays shaped (NR_BLOCK, NPHI_BLOCK).
    Local Athena i/j indices run 2..65 (4096 data lines per block).
    We sort by (i_local, j_local) to ensure correct ordering.
    """
    fname = os.path.join(
        DATA_DIR,
        f"acc_disk_visc.block{block_id}.out1.{step_str}.tab"
    )
    rows = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            i_local = int(parts[0])
            r       = float(parts[1])
            j_local = int(parts[2])
            phi     = float(parts[3])
            rho     = float(parts[4])
            press   = float(parts[5])
            vel1    = float(parts[6])
            vel2    = float(parts[7])
            rows.append((i_local, j_local, r, phi, rho, press, vel1, vel2))

    if len(rows) != NR_BLOCK * NPHI_BLOCK:
        print(f"  WARNING: block{block_id} step {step_str} has {len(rows)} rows (expected {NR_BLOCK*NPHI_BLOCK})")

    # Sort: primary key = i_local (r direction), secondary = j_local (phi direction)
    rows.sort(key=lambda x: (x[0], x[1]))

    r_arr     = np.array([x[2] for x in rows]).reshape(NR_BLOCK, NPHI_BLOCK)
    phi_arr   = np.array([x[3] for x in rows]).reshape(NR_BLOCK, NPHI_BLOCK)
    rho_arr   = np.array([x[4] for x in rows]).reshape(NR_BLOCK, NPHI_BLOCK)
    press_arr = np.array([x[5] for x in rows]).reshape(NR_BLOCK, NPHI_BLOCK)
    vel1_arr  = np.array([x[6] for x in rows]).reshape(NR_BLOCK, NPHI_BLOCK)
    vel2_arr  = np.array([x[7] for x in rows]).reshape(NR_BLOCK, NPHI_BLOCK)

    return dict(r=r_arr, phi=phi_arr, rho=rho_arr, press=press_arr,
                vel1=vel1_arr, vel2=vel2_arr)


def merge_blocks(step_str):
    """Merge all 4 blocks into global 128x128 arrays."""
    r_g     = np.zeros((NR_GLOBAL, NPHI_GLOBAL))
    phi_g   = np.zeros((NR_GLOBAL, NPHI_GLOBAL))
    rho_g   = np.zeros((NR_GLOBAL, NPHI_GLOBAL))
    press_g = np.zeros((NR_GLOBAL, NPHI_GLOBAL))
    vel1_g  = np.zeros((NR_GLOBAL, NPHI_GLOBAL))
    vel2_g  = np.zeros((NR_GLOBAL, NPHI_GLOBAL))

    for bid, (i0, i1, j0, j1) in BLOCK_INFO.items():
        d = load_block(step_str, bid)
        r_g    [i0:i1, j0:j1] = d['r']
        phi_g  [i0:i1, j0:j1] = d['phi']
        rho_g  [i0:i1, j0:j1] = d['rho']
        press_g[i0:i1, j0:j1] = d['press']
        vel1_g [i0:i1, j0:j1] = d['vel1']
        vel2_g [i0:i1, j0:j1] = d['vel2']

    return dict(r=r_g, phi=phi_g, rho=rho_g, press=press_g,
                vel1=vel1_g, vel2=vel2_g)


def compute_cell_areas(r, phi):
    """
    Compute 2D polar cell areas: A_ij = 0.5*(r_out^2 - r_in^2) * dphi
    We approximate cell edges as midpoints between cell centers.
    r shape: (NR, NPHI) — r values are the same for all phi in a row.
    phi shape: (NR, NPHI) — phi values are the same for all r in a column.
    """
    r1d   = r[:, 0]    # (NR,)
    phi1d = phi[0, :]  # (NPHI,)

    # Radial edges (use midpoints; extrapolate outer/inner boundaries)
    r_edges = np.zeros(NR_GLOBAL + 1)
    r_edges[1:-1] = 0.5 * (r1d[:-1] + r1d[1:])
    r_edges[0]  = r1d[0]  - 0.5 * (r1d[1]  - r1d[0])
    r_edges[-1] = r1d[-1] + 0.5 * (r1d[-1] - r1d[-2])
    r_edges = np.clip(r_edges, 0, None)

    # Phi edges
    phi_edges = np.zeros(NPHI_GLOBAL + 1)
    phi_edges[1:-1] = 0.5 * (phi1d[:-1] + phi1d[1:])
    phi_edges[0]  = phi1d[0]  - 0.5 * (phi1d[1]  - phi1d[0])
    phi_edges[-1] = phi1d[-1] + 0.5 * (phi1d[-1] - phi1d[-2])

    dr   = r_edges[1:] - r_edges[:-1]       # (NR,)
    dphi = phi_edges[1:] - phi_edges[:-1]   # (NPHI,)

    # Area = 0.5*(r_out^2 - r_in^2)*dphi  ≈ r * dr * dphi
    area = (r_edges[1:]**2 - r_edges[:-1]**2) / 2.0  # (NR,)
    area = area[:, np.newaxis] * dphi[np.newaxis, :]   # (NR, NPHI)
    return area


def analyze_step(step_str, label):
    print(f"\n{'='*70}")
    print(f"  STEP {step_str}  ({label})")
    print(f"{'='*70}")

    d = merge_blocks(step_str)
    rho   = d['rho']
    press = d['press']
    vel1  = d['vel1']
    vel2  = d['vel2']
    r     = d['r']
    phi   = d['phi']

    # Verify coordinates look sensible
    print(f"\n[Coordinate check]")
    print(f"  r    range: [{r.min():.5f}, {r.max():.5f}]")
    print(f"  phi  range: [{phi.min():.5f}, {phi.max():.5f}]")
    print(f"  r[i=0,  j=0] = {r[0,   0]:.5f}   r[i=63, j=0] = {r[63,  0]:.5f}")
    print(f"  r[i=64, j=0] = {r[64,  0]:.5f}   r[i=127,j=0] = {r[127, 0]:.5f}")
    print(f"  phi[i=0,j=0] = {phi[0,  0]:.5f}  phi[i=0,j=63]= {phi[0, 63]:.5f}")
    print(f"  phi[i=0,j=64]= {phi[0, 64]:.5f}  phi[i=0,j=127]= {phi[0,127]:.5f}")

    cell_area = compute_cell_areas(r, phi)

    # ── 3a/3b: Disk mass ──────────────────────────────────────────────────────
    total_mass = np.sum(rho * cell_area)
    nonvac_mask = rho > 2 * DFLOOR
    nonvac_mass = np.sum(rho[nonvac_mask] * cell_area[nonvac_mask])
    print(f"\n[Disk mass]")
    print(f"  Total mass (all cells):              {total_mass:.6e}")
    print(f"  Non-vacuum mass (rho > 2e-4):        {nonvac_mass:.6e}")
    print(f"  Non-vacuum cells:                    {nonvac_mask.sum()} / {NR_GLOBAL*NPHI_GLOBAL}")

    # ── 3c: Global min/max/mean ───────────────────────────────────────────────
    print(f"\n[Global statistics]")
    for name, arr in [('rho', rho), ('press', press), ('vel1', vel1), ('vel2', vel2)]:
        print(f"  {name:6s}: min={arr.min():.4e}  max={arr.max():.4e}  mean={arr.mean():.4e}")

    # ── 3d: Radial profile (mean rho over phi) ────────────────────────────────
    print(f"\n[Radial profile: mean rho (phi-average)]")
    mean_rho_r = rho.mean(axis=1)
    print(f"  {'i_global':>8}  {'r':>10}  {'mean_rho':>12}  {'min_rho':>12}  {'max_rho':>12}")
    for i in range(0, NR_GLOBAL, 8):
        print(f"  {i:8d}  {r[i,0]:10.5f}  {mean_rho_r[i]:12.5e}  {rho[i,:].min():12.5e}  {rho[i,:].max():12.5e}")

    # ── 3e: Floor hits ────────────────────────────────────────────────────────
    floor_mask = rho <= 1.01 * DFLOOR
    n_floor = floor_mask.sum()
    print(f"\n[Floor hits: rho <= 1.01e-4]")
    print(f"  Count: {n_floor}")
    if n_floor > 0:
        i_floor, j_floor = np.where(floor_mask)
        print(f"  i range: [{i_floor.min()}, {i_floor.max()}]")
        print(f"  j range: [{j_floor.min()}, {j_floor.max()}]")
        # Histogram by radial ring
        floor_by_i = floor_mask.sum(axis=1)
        print(f"  {'i_global':>8}  {'r':>10}  {'floor_cells_in_ring':>20}")
        for i in range(NR_GLOBAL):
            if floor_by_i[i] > 0:
                print(f"  {i:8d}  {r[i,0]:10.5f}  {floor_by_i[i]:20d}")

    return dict(total_mass=total_mass, nonvac_mass=nonvac_mass,
                rho=rho, press=press, vel1=vel1, vel2=vel2, r=r, phi=phi)


def analyze_phi_boundary(d0, d1):
    """
    Item 5: PHI-SPLIT boundary at j_global=63/64.
    For innermost 11 radial rings (i_global=0..10).
    """
    print(f"\n{'='*70}")
    print("  PHI-SPLIT BOUNDARY ANALYSIS  (j=63/64)")
    print(f"{'='*70}")
    print("  (Block 0/1 ends at j=63; Block 2/3 starts at j=64)\n")

    header = f"  {'i':>3}  {'r':>8}  " \
             f"{'rho[j62]':>10}  {'rho[j63]':>10}  | {'rho[j64]':>10}  {'rho[j65]':>10}  " \
             f"| {'v1[j62]':>10}  {'v1[j63]':>10}  | {'v1[j64]':>10}  {'v1[j65]':>10}"

    for step_str, data, label in [('00000', d0, 'step0'), ('00001', d1, 'step1')]:
        rho  = data['rho']
        vel1 = data['vel1']
        r    = data['r']
        print(f"  --- {label} ---")
        print(header)
        for i in range(11):
            row = (f"  {i:3d}  {r[i,0]:8.5f}  "
                   f"{rho[i,62]:10.4e}  {rho[i,63]:10.4e}  | "
                   f"{rho[i,64]:10.4e}  {rho[i,65]:10.4e}  | "
                   f"{vel1[i,62]:10.4e}  {vel1[i,63]:10.4e}  | "
                   f"{vel1[i,64]:10.4e}  {vel1[i,65]:10.4e}")
            # Flag discontinuity: ratio of rho at j63/j64 > 10x or < 0.1x
            ratio = rho[i, 63] / (rho[i, 64] + 1e-30)
            flag = "  <<< DISC" if (ratio > 10 or ratio < 0.1) else ""
            print(row + flag)
        print()


def analyze_r_boundary(d0, d1):
    """
    Item 6: R-SPLIT boundary at i_global=63/64.
    For j=0 to j=5.
    """
    print(f"\n{'='*70}")
    print("  R-SPLIT BOUNDARY ANALYSIS  (i=63/64)")
    print(f"{'='*70}")
    print("  (Block 0/2 ends at i=63; Block 1/3 starts at i=64)\n")

    for step_str, data, label in [('00000', d0, 'step0'), ('00001', d1, 'step1')]:
        rho  = data['rho']
        vel1 = data['vel1']
        r    = data['r']
        print(f"  --- {label} ---")
        print(f"  {'j':>3}  {'phi_j':>8}  "
              f"{'rho[i62]':>10}  {'rho[i63]':>10}  | {'rho[i64]':>10}  {'rho[i65]':>10}  "
              f"| {'v1[i62]':>10}  {'v1[i63]':>10}  | {'v1[i64]':>10}  {'v1[i65]':>10}")
        phi = data['phi']
        for j in range(6):
            row = (f"  {j:3d}  {phi[0,j]:8.5f}  "
                   f"{rho[62,j]:10.4e}  {rho[63,j]:10.4e}  | "
                   f"{rho[64,j]:10.4e}  {rho[65,j]:10.4e}  | "
                   f"{vel1[62,j]:10.4e}  {vel1[63,j]:10.4e}  | "
                   f"{vel1[64,j]:10.4e}  {vel1[65,j]:10.4e}")
            ratio = rho[63, j] / (rho[64, j] + 1e-30)
            flag = "  <<< DISC" if (ratio > 10 or ratio < 0.1) else ""
            print(row + flag)
        print()


def main():
    print("Athena++ Disk Disappearance Diagnostic")
    print("=" * 70)

    d0 = analyze_step('00000', 'Initial condition')
    d1 = analyze_step('00001', 'After first timestep')

    # ── Item 4: Mass ratio ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MASS CONSERVATION CHECK")
    print(f"{'='*70}")
    ratio_total  = d1['total_mass']  / d0['total_mass']
    ratio_nonvac = d1['nonvac_mass'] / d0['nonvac_mass'] if d0['nonvac_mass'] > 0 else float('nan')
    print(f"  Total mass step0:    {d0['total_mass']:.6e}")
    print(f"  Total mass step1:    {d1['total_mass']:.6e}")
    print(f"  Ratio (step1/step0): {ratio_total:.6f}")
    if abs(ratio_total - 1.0) < 0.01:
        print("  -> Total mass CONSERVED to <1%  (good)")
    elif abs(ratio_total - 1.0) < 0.10:
        print("  -> Total mass changed by <10%   (moderate)")
    else:
        print(f"  -> Total mass CHANGED DRAMATICALLY ({(ratio_total-1)*100:.1f}%)")
    print(f"\n  Non-vacuum mass step0: {d0['nonvac_mass']:.6e}")
    print(f"  Non-vacuum mass step1: {d1['nonvac_mass']:.6e}")
    print(f"  Non-vac ratio:         {ratio_nonvac:.6f}")
    if abs(ratio_nonvac - 1.0) < 0.01:
        print("  -> Non-vacuum mass CONSERVED to <1%")
    else:
        print(f"  -> Non-vacuum mass CHANGED by {(ratio_nonvac-1)*100:.1f}%")

    # ── Items 5 & 6: Boundary discontinuities ────────────────────────────────
    analyze_phi_boundary(d0, d1)
    analyze_r_boundary(d0, d1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    rho0 = d0['rho'];  rho1 = d1['rho']
    floor0 = (rho0 <= 1.01*DFLOOR).sum()
    floor1 = (rho1 <= 1.01*DFLOOR).sum()
    print(f"  Floor cells step0: {floor0} / {NR_GLOBAL*NPHI_GLOBAL}  ({100*floor0/(NR_GLOBAL*NPHI_GLOBAL):.1f}%)")
    print(f"  Floor cells step1: {floor1} / {NR_GLOBAL*NPHI_GLOBAL}  ({100*floor1/(NR_GLOBAL*NPHI_GLOBAL):.1f}%)")
    print(f"  Floor cell change: {floor1 - floor0:+d}")
    print(f"\n  max(rho) step0: {rho0.max():.4e}   step1: {rho1.max():.4e}   ratio: {rho1.max()/rho0.max():.4f}")
    print(f"  mean(rho) step0: {rho0.mean():.4e}  step1: {rho1.mean():.4e}  ratio: {rho1.mean()/rho0.mean():.4f}")
    print()


if __name__ == '__main__':
    main()
