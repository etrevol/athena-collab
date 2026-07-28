# Papaloizou-Pringle Disk: Calculation Report

**Generated:** 2026-02-17 14:53:33  
**Black Hole Mass:** M_BH = 1.70e+07 M☉  
**Scaling:** χ = 1000.00, C' = 0.3, γ = 1.666667  
**Viscosity:** ν̃ = 0.0000e+00 (INVISCID)

---

## Fundamental Formulas

### Gravitational Radius
$$r_g = \frac{2GM}{c^2} = 5.0207 \times 10^{12} \text{ cm} = 0.3356 \text{ AU}$$

### Characteristic Scales
$$r_0 = \chi \cdot r_g = 5.0207 \times 10^{15} \text{ cm} = 335.61 \text{ AU}$$

$$v_0 = \frac{c}{\sqrt{2\chi}} = 6.7036 \times 10^{8} \text{ cm/s} = 0.022361 \, c$$

$$t_0 = 2\pi\sqrt{\frac{r_0^3}{GM}} = 4.7058 \times 10^{7} \text{ s} = 544.66 \text{ days}$$

### Polytropic Index
$$n = \frac{1}{\gamma - 1} = 1.5000$$

### Disk Boundaries
Solve: $f(\tilde{r}) = \frac{1}{\tilde{r}} - \frac{1}{2\tilde{r}^2} - C' = 0$

$$\tilde{r}_{\text{inner}} = 0.6126 \quad (r = 205.584 \text{ AU})$$

$$\tilde{r}_{\text{outer}} = 2.7208 \quad (r = 913.105 \text{ AU})$$

### Equilibrium Pressure-Density Relation
$$\frac{\tilde{p}}{\tilde{\rho}} = \frac{1}{n+1}\left(\frac{1}{\tilde{r}} - \frac{1}{2\tilde{r}^2} - C'\right)$$

### Velocity Profile (Sub-Keplerian)
$$\tilde{v}_r = 0 \quad \text{(equilibrium)}$$

$$\tilde{v}_\phi = \frac{1}{\tilde{r}} \quad \text{(sub-Keplerian)}$$

### Viscosity
$$\tilde{\nu} = 0 \quad \text{(INVISCID - no viscous evolution)}$$

---

## Athena++ Input File Format

```
<comment>
problem   = Papaloizou-Pringle accretion disk
reference = theory.txt
configure = --prob=acc_disk --coord=cylindrical

<job>
problem_id = acc_disk      # Problem identifier

<output1>
file_type  = hdf5          # HDF5 output format
variable   = prim          # Primitive variables (rho, v, p)
dt         = 6.283185      # Output cadence [3422.18 days]
dcycle     = -1            # Disable cycle-based output

<time>
cfl_number = 0.3            # CFL stability condition
nlim       = -1            # No cycle limit (use tlim)
tlim       = 628.318531      # End time [342217.8 days]
integrator = vl2           # Van Leer 2nd order integrator
xorder     = 2             # 2nd order spatial reconstruction
ncycle_out = 10            # Console output frequency

<mesh>
nx1        = 256            # Radial resolution
x1min      = 0.3            # Inner boundary (code units)
x1max      = 3.0            # Outer boundary (code units)
x1rat      = 1.0           # Uniform radial spacing
ix1_bc     = outflow       # Inner radial boundary condition
ox1_bc     = outflow       # Outer radial boundary condition

nx2        = 256            # Azimuthal resolution
x2min      = 0.0           # Azimuthal start (radians)
x2max      = 6.283185307179586  # Azimuthal end (2π radians)
ix2_bc     = periodic      # Azimuthal boundary (periodic)
ox2_bc     = periodic      # Azimuthal boundary (periodic)

nx3        = 1              # Vertical resolution (2D disk)
x3min      = -0.5          # Vertical minimum
x3max      = 0.5           # Vertical maximum
ix3_bc     = periodic      # Vertical boundary
ox3_bc     = periodic      # Vertical boundary

<meshblock>
nx1        = 64             # MeshBlock size (radial)
nx2        = 64             # MeshBlock size (azimuthal)
nx3        = 1             # MeshBlock size (vertical)

<hydro>
gamma      = 1.666666666666667  # Adiabatic index
iso_sound_speed = 0.0     # Not used (adiabatic)
dfloor     = 1.00e-08      # Density floor (numerical stability)
pfloor     = 1.00e-10     # Pressure floor (numerical stability)
# nu_iso = 0.0              # INVISCID (viscosity disabled)

<problem>
C_prime    = 0.300000000000000  # Disk width parameter

# chi        = 1000.000000000000000  # Scaling parameter (r₀ = χ·r_g)
# gm        = 2.256179704570000e+33  # GM (for reference, cm³/s²)
# r_g       = 5.020676949513693e+12  # Gravitational radius (for reference, cm)
# r_0       = 5.020676949513693e+15  # Characteristic length (for reference, cm)
# v_0       = 6.703563152297506e+08  # Characteristic velocity (for reference, cm/s)
# t_0       = 4.705832245418316e+07  # Characteristic time (for reference, s)
```

---

## Physical Interpretation

| Quantity | Code Units | Physical Units |
|----------|------------|----------------|
| Domain (radial) | [0.3, 3.0] | [100.68, 1006.82] AU |
| Disk extent | [0.613, 2.721] | [205.58, 913.11] AU |
| Simulation time | 628.32 t₀ | 342217.8 days = 936.9412 years |
| Output interval | 6.28 t₀ | 3422.18 days |

--- 
**Files saved in:** `/home/etrevol/athena-collab/materials/theory-scripts`
