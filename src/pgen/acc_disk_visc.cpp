//========================================================================================
// Papaloizou-Pringle torus with alpha viscosity. Cylindrical (r, phi), 2D.
//
// A constant-angular-momentum torus superposed on a centrifugally balanced ambient
// medium; v_phi follows from exact radial force balance, so the surface is continuous.
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

namespace {
  // Physical Constants (CGS units)
  const Real C_LIGHT = 2.99792458e10;  // Speed of light [cm/s]
  const Real K_B     = 1.380649e-16;   // Boltzmann constant [erg/K]
  const Real M_P     = 1.67262192e-24; // Proton mass [g]
  const Real G_GRAV  = 6.67430e-8;     // Gravitational constant [cm^3/(g·s^2)]
  const Real M_SUN   = 1.98841e33;     // Solar mass [g]
  const Real YR_TO_S = 3.15576e7;      // Year to seconds conversion

  // Input Physical Parameters
  Real T_0;          // Reference temperature [K]
  Real mu_gas;       // Mean molecular weight
  Real chi_param;    // Scaling parameter (r_0 / r_g)
  Real M_bh;         // Black hole mass [M_sun]
  Real rho_0;        // Reference physical density [g/cm^3]

  // Derived Dimensionless Parameters
  Real beta_param;   // Dimensionless gravity (calculated internally)
  Real r_center;     // Dimensionless shifted center of the disk
  Real C_prime;      // Geometric disk thickness parameter
  Real gamma_gas;    // Adiabatic index
  Real n_poly;       // Polytropic index
  Real nu_iso;       // Kinematic viscosity (isotropic, Athena++ core parameter)
  Real alpha_visc;   // Alpha viscosity parameter
  Real rho_floor;    // Density floor (hard numerical safety net)
  Real press_floor;  // Pressure floor (hard numerical safety net)
  Real r_inner;      // Inner disk geometric boundary
  Real r_outer;      // Outer disk geometric boundary

  // Ambient medium ("atmosphere") -- a genuine equilibrium state, NOT the floor
  Real eps_soft;     // Softening of the central mass; 0 reproduces the point mass
  Real pert_amp;     // Per-cell white-noise density seed
  Real pert_mode_amp;  // Coherent seed amplitude per harmonic, m = 1..5
  Real rho_body;     // Density above which a cell counts as torus, for diagnostics
  Real l0_sq;        // l^2 that puts the pressure maximum at r_center

  Real rho_atm;      // Ambient density
  Real cs2_atm;      // Ambient p/rho
  Real p_atm;        // Ambient pressure
  Real visc_rho_cut; // Density below which alpha-viscosity is smoothly switched off

  // Derived torus normalisation
  Real f_center;     // f at the density maximum = 0.5 - C'
  Real p_norm;       // beta / (r_center*(n+1)*f_center^n), so that p_d = p_norm*f^(n+1)

  // Physical Scaling Factors (for history output)
  Real r_g;          // Schwarzschild radius 2GM/c^2 [cm]
  Real L_0;          // Length scale [cm]
  Real V_0;          // Velocity scale (sound speed) [cm/s]
  Real T_scale;      // Time scale [s]
  Real mass_scale;   // Mass scale [M_sun]
  Real mdot_scale;   // Mass accretion rate scale [M_sun/yr]
}

// Function declarations
void NewtonianGravity(MeshBlock *pmb, const Real time, const Real dt,
                 const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                 AthenaArray<Real> &cons_scalar);
void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                   int is, int ie, int js, int je, int ks, int ke);
Real TotalDiskMass(MeshBlock *pmb, int iout);
Real TorusHistory(MeshBlock *pmb, int iout);
Real AccretionRate(MeshBlock *pmb, int iout);
void InnerX1OutflowBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                   FaceField &b, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1OutflowBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                   FaceField &b, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//----------------------------------------------------------------------------------------
//! Geometric shape function f(r) of the Papaloizou-Pringle torus.
//!
//! With a softened central mass, Phi = -beta/sqrt(r^2+eps^2), the pressure maximum stays
//! at r_center only if l^2 = beta r_c^4 / (r_c^2+eps^2)^{3/2}, and the shape function
//! generalises to
//!
//!     f(r) = r_c/sqrt(r^2+eps^2) - r_c^5 / (2 (r_c^2+eps^2)^{3/2} r^2) - C'.
//!
//! At eps = 0 this is exactly r_c/r - r_c^2/(2r^2) - C', the original expression, and
//! the eps = 0 branch is taken literally so that behaviour is unchanged bit for bit.
//! The equilibrium must be built on the same potential the solver applies: balancing a
//! torus against a point mass while integrating it in a softened one starts it out of
//! equilibrium by the difference between them.
Real DiskFunction(Real r) {
  if (eps_soft <= 0.0) {
    Real x = r_center / r;
    return x - 0.5 * x * x - C_prime;
  }
  return r_center / std::sqrt(r * r + eps_soft * eps_soft)
         - 0.5 * l0_sq * r_center / (beta_param * r * r) - C_prime;
}

//! d f / d r
Real DiskFunctionDeriv(Real r) {
  if (eps_soft <= 0.0) {
    return -r_center / (r * r) + r_center * r_center / (r * r * r);
  }
  Real s = r * r + eps_soft * eps_soft;
  return -r_center * r / (s * std::sqrt(s))
         + l0_sq * r_center / (beta_param * r * r * r);
}

//! Torus density (identically zero outside the torus surface f <= 0).
Real DiskDensity(Real r) {
  Real f = DiskFunction(r);
  if (f <= 0.0) return 0.0;
  return std::pow(f / f_center, n_poly);
}

//! Torus pressure (identically zero outside the torus surface).
Real DiskPressure(Real r) {
  Real f = DiskFunction(r);
  if (f <= 0.0) return 0.0;
  return p_norm * std::pow(f, n_poly + 1.0);
}

//! d p_torus / d r  (identically zero outside the torus surface).
Real DiskPressureDeriv(Real r) {
  Real f = DiskFunction(r);
  if (f <= 0.0) return 0.0;
  return p_norm * (n_poly + 1.0) * std::pow(f, n_poly) * DiskFunctionDeriv(r);
}

//----------------------------------------------------------------------------------------
//! v_phi^2 from radial force balance; l = const inside the torus, Keplerian outside.
Real EquilibriumVphi2(Real r) {
  Real rho = DiskDensity(r) + rho_atm;
  Real g = beta_param / r;                       // point-mass limit
  if (eps_soft > 0.0) {
    Real s = r * r + eps_soft * eps_soft;
    g = beta_param * r * r / (s * std::sqrt(s));
  }
  Real v2 = g + (r / rho) * DiskPressureDeriv(r);
  return (v2 > 0.0) ? v2 : 0.0;
}

//----------------------------------------------------------------------------------------
//! Reproducible white noise in [-1, 1], hashed from the cell coordinates, and a coherent
//! seed in the low harmonics. An instability amplifies whatever asymmetry is already
//! present; a grid started from an analytic profile has almost none, so without a seed
//! the mode grows out of round-off and nothing is measured in a reasonable run length.
//! White noise averages down over the cells in the torus and is the weaker of the two;
//! the coherent seed puts a known amplitude into each of m = 1..5 with its own random
//! phase, privileging no harmonic. Both default to zero.
Real CellNoise(Real x, Real y) {
  auto q = [](Real v) { return static_cast<unsigned int>(
      static_cast<long long>(std::floor(v * 1.0e6)) & 0xffffffffLL); };
  unsigned int h = q(x) * 73856093u ^ q(y) * 19349663u;
  h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
  return 2.0 * (static_cast<Real>(h) / 4294967295.0) - 1.0;
}

Real ModeNoise(Real phi) {
  Real s = 0.0;
  for (int m = 1; m <= 5; ++m) {
    unsigned int h = 2654435761u * static_cast<unsigned int>(m) + 1013904223u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    s += std::cos(m * phi - 2.0 * PI * (static_cast<Real>(h) / 4294967295.0));
  }
  return s;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  C_prime    = pin->GetReal("problem", "C_prime");
  r_center   = pin->GetReal("problem", "r_center");
  nu_iso     = pin->GetOrAddReal("problem", "nu_iso", 0.0);
  alpha_visc = pin->GetOrAddReal("problem", "alpha", 0.0);
  gamma_gas  = pin->GetReal("hydro", "gamma");

  eps_soft      = pin->GetOrAddReal("problem", "eps_soft", 0.0);
  pert_amp      = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  pert_mode_amp = pin->GetOrAddReal("problem", "pert_mode_amp", 0.0);

  rho_floor   = pin->GetOrAddReal("hydro", "dfloor", 1.0e-9);
  press_floor = pin->GetOrAddReal("hydro", "pfloor", 1.0e-10);

  T_0       = pin->GetReal("problem", "T_0");
  mu_gas    = pin->GetReal("problem", "mu");
  chi_param = pin->GetReal("problem", "chi");
  M_bh      = pin->GetReal("problem", "M_bh");
  rho_0     = pin->GetReal("problem", "rho_0");

  n_poly = 1.0 / (gamma_gas - 1.0);

  // Adiabatic sound speed at the reference temperature [cm/s]
  Real cs0_sq = (gamma_gas * K_B * T_0) / (mu_gas * M_P);
  Real cs0    = std::sqrt(cs0_sq);

  // Physical scaling factors
  r_g = 2.0 * G_GRAV * M_bh * M_SUN / (C_LIGHT * C_LIGHT);
  L_0 = chi_param * r_g;
  V_0 = cs0;
  T_scale = L_0 / V_0;

  mass_scale = rho_0 * std::pow(L_0, 3.0) / M_SUN;

  mdot_scale = rho_0 * L_0 * L_0 * V_0 * YR_TO_S / M_SUN;

  // Dimensionless gravity parameter
  beta_param = (C_LIGHT * C_LIGHT) / (2.0 * chi_param * cs0_sq);

  // l^2 that keeps the pressure maximum at r_center under the softened potential;
  // reduces to beta*r_center at eps = 0.
  {
    Real sc = r_center * r_center + eps_soft * eps_soft;
    l0_sq = beta_param * std::pow(r_center, 4.0) / (sc * std::sqrt(sc));
  }

  // Exact geometric disk boundaries
  Real discriminant = 1.0 - 2.0 * C_prime;
  if (discriminant < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in acc_disk_visc.cpp" << std::endl
        << "Invalid C_prime = " << C_prime << ". Must satisfy 2*C' < 1." << std::endl;
    ATHENA_ERROR(msg);
  }
  r_inner = r_center * ( (1.0 - std::sqrt(discriminant)) / (2.0 * C_prime) );
  r_outer = r_center * ( (1.0 + std::sqrt(discriminant)) / (2.0 * C_prime) );

  // Torus normalisation:  p_d(r) = p_norm * f(r)^(n+1),  rho_d(r) = (f/f_c)^n
  f_center = 0.5 - C_prime;
  p_norm   = beta_param / (r_center * (n_poly + 1.0) * std::pow(f_center, n_poly));

  // Ambient medium: an equilibrium state kept well above the floors so they never activate.
  rho_atm = pin->GetOrAddReal("problem", "rho_atm", 1.0e-6);
  Real t_atm_frac = pin->GetOrAddReal("problem", "t_atm_frac", 1.0);
  cs2_atm = t_atm_frac * beta_param * f_center / (r_center * (n_poly + 1.0));
  p_atm   = rho_atm * cs2_atm;

  // Viscosity is tapered off in the ambient, removing artificial torque at the boundaries.
  visc_rho_cut = pin->GetOrAddReal("problem", "visc_rho_cut", 10.0 * rho_atm);

  if (rho_atm <= rho_floor || p_atm <= press_floor) {
    std::stringstream msg;
    msg << "### FATAL ERROR in acc_disk_visc.cpp" << std::endl
        << "The ambient medium must sit strictly above the numerical floors." << std::endl
        << "  rho_atm = " << rho_atm << "  dfloor = " << rho_floor << std::endl
        << "  p_atm   = " << p_atm   << "  pfloor = " << press_floor << std::endl;
    ATHENA_ERROR(msg);
  }

  // Athena++ gates ViscousFluxIso on nu_iso, so alpha > 0 with nu_iso = 0 runs inviscid.
  // Mirror trap: nu_iso > 0 with alpha = 0 applies a CONSTANT nu, it is not inviscid.
  if (alpha_visc <= 0.0 && nu_iso > 0.0 && Globals::my_rank == 0) {
    std::cout << std::endl
              << "  *** WARNING: alpha = 0 but nu_iso = " << nu_iso << " > 0." << std::endl
              << "      This is NOT an inviscid run: Athena++ will apply a constant"
              << std::endl
              << "      kinematic viscosity nu = " << nu_iso << " everywhere." << std::endl
              << "      Set nu_iso = 0 for a genuinely inviscid control run." << std::endl
              << std::endl;
  }

  if (alpha_visc > 0.0 && nu_iso <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in acc_disk_visc.cpp" << std::endl
        << "alpha = " << alpha_visc << " > 0 requires <problem>/nu_iso > 0 as well."
        << std::endl
        << "Athena++ gates ViscousFluxIso() on nu_iso > 0; without it the enrolled"
        << std::endl
        << "DiskViscosity() coefficient is computed but never applied." << std::endl
        << "Set nu_iso to any positive placeholder (it is overwritten by DiskViscosity)."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // The torus surface must stay interior; on a boundary the outflow condition is ill-posed.
  if (Globals::my_rank == 0
      && (mesh_size.x1min >= r_inner || mesh_size.x1max <= r_outer)) {
    std::cout << std::endl
              << "  *** WARNING: the radial domain does not enclose the torus." << std::endl
              << "      x1min = " << mesh_size.x1min << " must be < r_inner = "
              << r_inner << std::endl
              << "      x1max = " << mesh_size.x1max << " must be > r_outer = "
              << r_outer << std::endl
              << "      Placing the torus surface on the boundary makes the outflow"
              << std::endl
              << "      condition ill-posed and drains the disk." << std::endl << std::endl;
  }

  // Output model parameters to console
  if (Globals::my_rank == 0) {
    std::cout << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Papaloizou-Pringle Disk with Alpha Viscosity" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  --- Physical Inputs ---" << std::endl;
    std::cout << "  Black Hole Mass (M_bh):         " << M_bh << " M_sun" << std::endl;
    std::cout << "  Reference Density (rho_0):      " << rho_0 << " g/cm^3" << std::endl;
    std::cout << "  Temperature (T_0):              " << T_0 << " K" << std::endl;
    std::cout << "  Mean molecular weight (mu):     " << mu_gas << std::endl;
    std::cout << "  Scaling parameter (chi):        " << chi_param << std::endl;
    std::cout << "  Calculated Sound Speed (cs_0):  " << cs0 << " cm/s" << std::endl;
    std::cout << "  --- Physical Scales ---" << std::endl;
    std::cout << "  Gravitational radius (r_g):     " << r_g << " cm" << std::endl;
    std::cout << "  Length scale (L_0):             " << L_0 << " cm" << std::endl;
    std::cout << "  Time scale (T_0):               " << T_scale << " s" << std::endl;
    std::cout << "  Mass scale:                     " << mass_scale << " M_sun" << std::endl;
    std::cout << "  Mdot scale:                     " << mdot_scale << " M_sun/yr" << std::endl;
    std::cout << "  --- Dimensionless Parameters ---" << std::endl;
    std::cout << "  Calculated Gravity (beta):      " << beta_param << std::endl;
    std::cout << "  Disk Center Shift (r_center):   " << r_center << std::endl;
    std::cout << "  Disk Width Parameter (C'):      " << C_prime << std::endl;
    std::cout << "  Adiabatic Index (gamma):        " << gamma_gas << std::endl;
    std::cout << "  Alpha Viscosity Parameter:      " << alpha_visc << std::endl;
    std::cout << "  Inner Geometric Boundary:       " << r_inner << std::endl;
    std::cout << "  Outer Geometric Boundary:       " << r_outer << std::endl;
    std::cout << "  --- Ambient Medium & Floors ---" << std::endl;
    std::cout << "  Ambient density (rho_atm):      " << rho_atm << std::endl;
    std::cout << "  Ambient pressure (p_atm):       " << p_atm << std::endl;
    std::cout << "  Ambient sound speed:            " << std::sqrt(gamma_gas*cs2_atm)
              << std::endl;
    std::cout << "  Density floor (dfloor):         " << rho_floor << std::endl;
    std::cout << "  Pressure floor (pfloor):        " << press_floor << std::endl;
    std::cout << "  Viscosity cut-off density:      " << visc_rho_cut << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::endl;
  }

  EnrollUserExplicitSourceFunction(NewtonianGravity);
  if (alpha_visc > 0.0) {
    EnrollViscosityCoefficient(DiskViscosity);
  }
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1OutflowBC);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1OutflowBC);

  rho_body = 10.0 * rho_atm;

  // The first two columns keep their names and positions, so anything that already
  // reads this history file is unaffected. What follows is the same set of mode
  // diagnostics the 3D generator writes, under the same names, so that the analysis
  // tools run on 2D and 3D output without knowing which they were given. z-quantities
  // are present but zero: there is no third dimension here.
  AllocateUserHistoryOutput(23);
  EnrollUserHistoryOutput(0, TotalDiskMass, "disk_mass", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(1, AccretionRate, "mdot_in", UserHistoryOperation::sum);
  const char *mnames[21] = {"tor_mass", "tor_mx", "tor_my", "tor_mz",
                            "a1", "b1", "a2", "b2", "a3", "b3",
                            "a4", "b4", "a5", "b5",
                            "E_kin", "E_int", "E_grav_c", "E_grav_self",
                            "n_dfloor", "n_pfloor", "v_max"};
  for (int i = 0; i < 21; ++i)
    EnrollUserHistoryOutput(i + 2, TorusHistory, mnames[i],
                            i == 20 ? UserHistoryOperation::max
                                    : UserHistoryOperation::sum);

  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        // Volume-centroid radius: the same radius Athena++'s geometric source term uses.
        Real r = pcoord->x1v(i);

        Real rho_d = DiskDensity(r);
        if (rho_d > 0.0 && (pert_amp > 0.0 || pert_mode_amp > 0.0)) {
          Real phi = pcoord->x2v(j);
          Real seed = 0.0;
          if (pert_amp > 0.0)
            seed += pert_amp * CellNoise(r * std::cos(phi), r * std::sin(phi));
          if (pert_mode_amp > 0.0) seed += pert_mode_amp * ModeNoise(phi);
          rho_d *= 1.0 + seed;
        }
        Real rho   = rho_d + rho_atm;
        Real press = DiskPressure(r) + p_atm;
        Real v_r   = 0.0;
        Real v_phi = std::sqrt(EquilibriumVphi2(r));

        phydro->w(IDN,k,j,i) = rho;
        phydro->w(IVX,k,j,i) = v_r;
        phydro->w(IVY,k,j,i) = v_phi;
        phydro->w(IVZ,k,j,i) = 0.0;
        phydro->w(IPR,k,j,i) = press;

        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = rho * v_r;
        phydro->u(IM2,k,j,i) = rho * v_phi;
        phydro->u(IM3,k,j,i) = 0.0;

        Real kinetic_energy = 0.5 * rho * (v_r*v_r + v_phi*v_phi);
        Real internal_energy = press / (gamma_gas - 1.0);
        phydro->u(IEN,k,j,i) = internal_energy + kinetic_energy;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Point-mass gravity, using the same geometric factor as Athena++'s cylindrical
//!        source term so that a balanced ambient medium is an exact discrete equilibrium.
//!        Energy follows the numerical mass flux, not the cell-centred rho*v_r.
void NewtonianGravity(MeshBlock *pmb, const Real time, const Real dt,
                 const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                 AthenaArray<Real> &cons_scalar) {
  const Real gm1 = gamma_gas - 1.0;
  AthenaArray<Real> &x1flux = pmb->phydro->flux[X1DIR];

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rm = pmb->pcoord->x1f(i);
        Real rp = pmb->pcoord->x1f(i+1);
        Real rv = pmb->pcoord->x1v(i);
        Real src1 = 2.0 / (rm + rp);   // == Coordinates::coord_src1_i_(i)

        // Radial momentum: -rho * beta * <1/r> / r_v, times r^3/(r^2+eps^2)^{3/2}
        // when the mass is softened, which is 1 at eps = 0.
        Real soft = 1.0;
        if (eps_soft > 0.0) {
          Real ss = rv * rv + eps_soft * eps_soft;
          soft = rv * rv * rv / (ss * std::sqrt(ss));
        }
        cons(IM1,k,j,i) -= dt * prim(IDN,k,j,i) * beta_param * src1 * soft / rv;

        // Total energy: work done by gravity on the numerical mass flux
        cons(IEN,k,j,i) -= dt * 0.5 * beta_param * soft
                           * ( x1flux(IDN,k,j,i)   / (rv * rm)
                             + x1flux(IDN,k,j,i+1) / (rv * rp) );

        // Safety net, before SEND_HYD: pressure violations correct the energy only.
        Real &d = cons(IDN,k,j,i);
        if (!std::isfinite(d) || d < rho_floor) {
          Real v_phi_atm = std::sqrt(beta_param / rv);
          d               = rho_floor;
          cons(IM1,k,j,i) = 0.0;
          cons(IM2,k,j,i) = rho_floor * v_phi_atm;
          cons(IM3,k,j,i) = 0.0;
          cons(IEN,k,j,i) = press_floor / gm1
                            + 0.5 * rho_floor * v_phi_atm * v_phi_atm;
          pmb->ruser_meshblock_data[0](0) += 1.0;
        } else {
          Real e_k = 0.5 * ( SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i))
                           + SQR(cons(IM3,k,j,i)) ) / d;
          Real e_int = cons(IEN,k,j,i) - e_k;
          if (!std::isfinite(e_int) || e_int < press_floor / gm1) {
            cons(IEN,k,j,i) = press_floor / gm1 + e_k;
            pmb->ruser_meshblock_data[0](1) += 1.0;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Alpha viscosity nu = alpha*(gamma/sqrt(beta))*(p/rho)*r^(3/2), tapered to zero
//!        in the ambient medium to avoid an artificial torque at the boundaries.
void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                   int is, int ie, int js, int je, int ks, int ke) {
  Real coeff = alpha_visc * gamma_gas / std::sqrt(beta_param);
  Real cut2  = visc_rho_cut * visc_rho_cut;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real rho = prim(IDN,k,j,i);
        Real press = prim(IPR,k,j,i);

        // Smooth density weight: ~1 in the disk body, ~0 in the ambient medium.
        Real w = (cut2 > 0.0) ? (rho*rho / (rho*rho + cut2)) : 1.0;

        // nu = alpha * (gamma/sqrt(beta)) * (p/rho) * r^(3/2)
        Real nu_func = coeff * (press / rho) * std::pow(r, 1.5) * w;

        phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) = nu_func;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! Mass of the disk body only [M_sun]. The uniform ambient medium is subtracted so that
//! this diagnostic tracks the torus rather than the numerical background.
Real TotalDiskMass(MeshBlock *pmb, int iout) {
  Real mass_sum = 0.0;
  const Real rho_cut = 2.0 * rho_atm;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rho_code = pmb->phydro->w(IDN,k,j,i);
        if (rho_code <= rho_cut) continue;
        Real vol = pmb->pcoord->GetCellVolume(k,j,i);

        mass_sum += (rho_code - rho_atm) * vol;
      }
    }
  }

  return mass_sum * mass_scale;
}

//----------------------------------------------------------------------------------------
//! Mass accretion rate [M_sun/yr] through the inner boundary (positive = inflow).
//! Only the MeshBlock owning that boundary contributes, tested against the mesh geometry.
Real AccretionRate(MeshBlock *pmb, int iout) {
  Real mdot_sum = 0.0;

  const Real x1min_mesh = pmb->pmy_mesh->mesh_size.x1min;
  const Real tol = 1.0e-10 * std::max(1.0, std::fabs(x1min_mesh));
  if (std::fabs(pmb->block_size.x1min - x1min_mesh) > tol) return 0.0;

  int i = pmb->is;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real rho_code = pmb->phydro->w(IDN,k,j,i);
      Real vr_code = pmb->phydro->w(IVX,k,j,i);
      Real area = pmb->pcoord->GetFace1Area(k,j,i);

      mdot_sum += -rho_code * vr_code * area;
    }
  }
  return mdot_sum * mdot_scale;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Two cumulative counters: density- and pressure-floor activations. In a healthy run
  // both stay at zero; any nonzero value has to be explained before the run is used.
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(2);
  ruser_meshblock_data[0](0) = 0.0;
  ruser_meshblock_data[0](1) = 0.0;

  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "f_grav");
  SetUserOutputVariableName(1, "f_centr");
  SetUserOutputVariableName(2, "f_press");
  SetUserOutputVariableName(3, "f_sum");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r     = pcoord->x1v(i);
        Real rho   = phydro->w(IDN,k,j,i);
        Real press = phydro->w(IPR,k,j,i);
        Real v_phi = phydro->w(IVY,k,j,i);

        // Gravitational force per unit mass (radial)
        Real f_grav = -beta_param / (r * r);

        // Centrifugal force per unit mass (cylindrical coords)
        Real f_centr = v_phi * v_phi / r;

        // Pressure gradient force per unit mass: -(1/rho) * dP/dr
        Real dPdr;
        if (i > is && i < ie) {
          Real r_p = pcoord->x1v(i+1);
          Real r_m = pcoord->x1v(i-1);
          Real P_p = phydro->w(IPR,k,j,i+1);
          Real P_m = phydro->w(IPR,k,j,i-1);
          dPdr = (P_p - P_m) / (r_p - r_m);
        } else if (i == is) {
          Real r_p = pcoord->x1v(i+1);
          Real P_p = phydro->w(IPR,k,j,i+1);
          dPdr = (P_p - press) / (r_p - r);
        } else {
          Real r_m = pcoord->x1v(i-1);
          Real P_m = phydro->w(IPR,k,j,i-1);
          dPdr = (press - P_m) / (r - r_m);
        }
        Real f_press = -(1.0 / std::max(rho, rho_floor)) * dPdr;

        // 4. Net radial force
        Real f_sum = f_grav + f_centr + f_press;

        user_out_var(0,k,j,i) = f_grav;
        user_out_var(1,k,j,i) = f_centr;
        user_out_var(2,k,j,i) = f_press;
        user_out_var(3,k,j,i) = f_sum;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Inner radial boundary: zero-gradient rho and p, diode in v_r (outflow only),
//!        v_phi extrapolated as r^(-1/2) so the ghost zones stay centrifugally balanced.
void InnerX1OutflowBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                   FaceField &b, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real scale = std::sqrt(pco->x1v(il) / pco->x1v(il-i));
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        prim(IVX,k,j,il-i) = std::min(prim(IVX,k,j,il), 0.0); // diode: only v_r <= 0
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il) * scale;
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Outer radial boundary: same construction, diode reversed (v_r >= 0).
void OuterX1OutflowBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                   FaceField &b, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real scale = std::sqrt(pco->x1v(iu) / pco->x1v(iu+i));
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        prim(IVX,k,j,iu+i) = std::max(prim(IVX,k,j,iu), 0.0); // diode: only v_r >= 0
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu) * scale;
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \brief Mode diagnostics, in the same columns and under the same names as the 3D
//!        generator writes, so one analysis tool serves both geometries.
//!
//! All twenty-one come from a single sweep, cached per block and per time: called once
//! per column they would otherwise cost twenty-one passes over the mesh at every history
//! dump. Only cells above rho_body contribute, so the ambient cannot masquerade as torus
//! structure. Coefficients are mass-weighted volume sums; normalised by the torus mass
//! they are directly comparable with the 3D runs and with published particle counts.
Real TorusHistory(MeshBlock *pmb, int iout) {
  static thread_local int cached_gid = -1;
  static thread_local Real cached_time = -1.0;
  static thread_local Real v[21];

  Real now = pmb->pmy_mesh->time;
  if (pmb->gid != cached_gid || now != cached_time) {
    for (int n = 0; n < 21; ++n) v[n] = 0.0;
    v[18] = pmb->ruser_meshblock_data[0](0);
    v[19] = pmb->ruser_meshblock_data[0](1);
    const Real gm1 = gamma_gas - 1.0;

    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        Real phi = pmb->pcoord->x2v(j);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real rho = pmb->phydro->w(IDN,k,j,i);
          Real vr = pmb->phydro->w(IVX,k,j,i);
          Real vp = pmb->phydro->w(IVY,k,j,i);
          Real sp = std::sqrt(vr*vr + vp*vp);
          if (sp > v[20]) v[20] = sp;

          if (rho <= rho_body) continue;
          Real r = pmb->pcoord->x1v(i);
          Real dV = pmb->pcoord->GetCellVolume(k,j,i);
          Real dm = rho * dV;

          v[0] += dm;
          v[1] += dm * r * std::cos(phi);
          v[2] += dm * r * std::sin(phi);
          for (int m = 1; m <= 5; ++m) {
            v[2*m + 2] += dm * std::cos(m * phi);
            v[2*m + 3] += dm * std::sin(m * phi);
          }
          v[14] += 0.5 * dm * (vr*vr + vp*vp);
          v[15] += pmb->phydro->w(IPR,k,j,i) * dV / gm1;
          Real pot = -beta_param / r;
          if (eps_soft > 0.0)
            pot = -beta_param / std::sqrt(r*r + eps_soft*eps_soft);
          v[16] += dm * pot;
        }
      }
    }
    cached_gid = pmb->gid;
    cached_time = now;
  }
  // The first two history columns belong to TotalDiskMass and AccretionRate, so these
  // are enrolled from index 2 onwards while the cache is indexed from zero. Returning
  // v[iout] read two past the end and reported uninitialised memory as physics.
  int n = iout - 2;
  return (n >= 0 && n < 21) ? v[n] : 0.0;
}
