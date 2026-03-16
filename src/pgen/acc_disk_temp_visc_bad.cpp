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
  const Real G_GRAV  = 6.67430e-8;     // Gravitational constant [cm³/(g·s²)]
  const Real M_SUN   = 1.98841e33;     // Solar mass [g]
  const Real YR_TO_S = 3.15576e7;      // Year to seconds conversion

  // Input Physical Parameters
  Real T_0;          // Reference temperature [K]
  Real mu_gas;       // Mean molecular weight
  Real chi_param;    // Scaling parameter (r_0 / r_g)
  Real M_bh;         // Black hole mass [M_sun]
  Real rho_0;        // Reference physical density [g/cm³]

  // Derived Dimensionless Parameters
  Real beta_param;   // Dimensionless gravity (calculated internally)
  Real r_center;     // Dimensionless shifted center of the disk
  Real C_prime;      // Geometric disk thickness parameter
  Real gamma_gas;    // Adiabatic index
  Real n_poly;       // Polytropic index
  Real nu_iso;       // Kinematic viscosity (isotropic)
  Real alpha_visc;   // Alpha viscosity parameter
  Real rho_floor;    // Density floor
  Real press_floor;  // Pressure floor
  Real r_inner;      // Inner disk geometric boundary
  Real r_outer;      // Outer disk geometric boundary

  // Physical Scaling Factors (for history output)
  Real r_g;          // Gravitational radius (Schwarzschild) [cm]
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
Real AccretionRate(MeshBlock *pmb, int iout);

// Calculates the geometric shape function f(r)
Real DiskFunction(Real r) {
  Real x = r_center / r;
  return x - 0.5 * x * x - C_prime;
}

// Calculates p/rho
Real PressureOverDensity(Real r) {
  Real f = DiskFunction(r);
  if (f > 0.0) {
    return (beta_param / (r_center * (n_poly + 1.0))) * f;
  }
  return 0.0;
}

// Calculates normalized density profile
Real DiskDensity(Real r) {
  Real f = DiskFunction(r);
  if (f > 0.0) {
    Real f_center = 0.5 - C_prime; // Maximum value of f(r) occurs exactly at r = r_center
    return std::pow(f / f_center, n_poly);
  }
  return rho_floor;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // 1. Read geometric and numerical parameters
  C_prime    = pin->GetReal("problem", "C_prime");
  r_center   = pin->GetReal("problem", "r_center");
  nu_iso     = pin->GetOrAddReal("problem", "nu_iso", 0.0);
  alpha_visc = pin->GetOrAddReal("problem", "alpha", 0.0);
  gamma_gas  = pin->GetReal("hydro", "gamma");
  
  rho_floor   = pin->GetOrAddReal("hydro", "dfloor", 1.0e-8);
  press_floor = pin->GetOrAddReal("hydro", "pfloor", 1.0e-10);
  
  // 2. Read physical parameters
  T_0       = pin->GetReal("problem", "T_0");
  mu_gas    = pin->GetReal("problem", "mu");
  chi_param = pin->GetReal("problem", "chi");
  M_bh      = pin->GetReal("problem", "M_bh");   // Black hole mass [M_sun]
  rho_0     = pin->GetReal("problem", "rho_0");  // Reference density [g/cm³]

  // 3. Calculate internal physics
  n_poly = 1.0 / (gamma_gas - 1.0);

  // Calculate isothermal sound speed squared (cs0^2) in CGS [cm^2/s^2]
  Real cs0_sq = (gamma_gas * K_B * T_0) / (mu_gas * M_P);
  Real cs0    = std::sqrt(cs0_sq);

  // Calculate physical scaling factors
  r_g = 2.0 * G_GRAV * M_bh * M_SUN / (C_LIGHT * C_LIGHT);  // Gravitational radius [cm]
  L_0 = chi_param * r_g;                                    // Length scale [cm]
  V_0 = cs0;                                                // Velocity scale [cm/s]
  T_scale = L_0 / V_0;                                      // Time scale [s]
  
  // Mass scale: converts dimensionless integrated density to solar masses
  // M_physical = rho_0 * L_0³ * M_dimensionless / M_sun
  mass_scale = rho_0 * std::pow(L_0, 3.0) / M_SUN;
  
  // Mass accretion rate scale: converts to M_sun/yr
  // Mdot = rho * v * A, dimensions: [g/cm³] * [cm/s] * [cm²] = [g/s]
  // Convert to M_sun/yr: [g/s] * [yr/s] / [g/M_sun]
  mdot_scale = rho_0 * L_0 * L_0 * V_0 * YR_TO_S / M_SUN;

  // Calculate dimensionless gravity parameter beta
  beta_param = (C_LIGHT * C_LIGHT) / (2.0 * chi_param * cs0_sq);

  // 4. Calculate exact geometric disk boundaries
  Real discriminant = 1.0 - 2.0 * C_prime;
  if (discriminant < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in acc_disk_temp_visc.cpp" << std::endl
        << "Invalid C_prime = " << C_prime << ". Must satisfy 2*C' < 1." << std::endl;
    ATHENA_ERROR(msg);
  }
  r_inner = r_center * ( (1.0 - std::sqrt(discriminant)) / (2.0 * C_prime) );
  r_outer = r_center * ( (1.0 + std::sqrt(discriminant)) / (2.0 * C_prime) );
  
  // 5. Output model parameters to console
  if (Globals::my_rank == 0) {
    std::cout << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Thermally-Scaled Papaloizou-Pringle Disk" << std::endl;
    std::cout << "  with Spatially-Varying Alpha Viscosity" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  --- Physical Inputs ---" << std::endl;
    std::cout << "  Black Hole Mass (M_bh):         " << M_bh << " M_sun" << std::endl;
    std::cout << "  Reference Density (rho_0):      " << rho_0 << " g/cm³" << std::endl;
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
    std::cout << "===========================================================" << std::endl;
    std::cout << std::endl;
  }
  
  EnrollUserExplicitSourceFunction(NewtonianGravity);
  if (alpha_visc > 0.0) {
    EnrollViscosityCoefficient(DiskViscosity);
  }
  
  // Enroll user history output functions
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, TotalDiskMass, "disk_mass", UserHistoryOperation::sum);
  EnrollUserHistoryOutput(1, AccretionRate, "mdot_in", UserHistoryOperation::sum);
  
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real v_phi_constant = std::sqrt(beta_param * r_center);
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r = pcoord->x1v(i);
        
        Real rho, press, v_r, v_phi;
        Real f = DiskFunction(r);
        
        if (f > 0.0) {
          // Inside disk
          rho = DiskDensity(r);
          if (rho < rho_floor) rho = rho_floor;
          
          press = rho * PressureOverDensity(r);
          if (press < press_floor) press = press_floor;
          
          v_r = 0.0;
          v_phi = v_phi_constant / r; // l = const profile
        } else {
          rho = rho_floor;
          press = press_floor;
          v_r = 0.0;
          v_phi = v_phi_constant / r;
        }

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

// Numerical stabilizer executed after every integration step
void MeshBlock::UserWorkInLoop() {
  Real v_phi_constant = std::sqrt(beta_param * r_center);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r = pcoord->x1v(i);
        
        Real rho = phydro->u(IDN,k,j,i);
        Real v_r = phydro->w(IVX,k,j,i);
        Real v_phi = phydro->w(IVY,k,j,i);
        
        bool need_reset = false;
        
        // Check violations
        if (rho < rho_floor || !std::isfinite(rho)) need_reset = true;
        
        Real kinetic = 0.5 * rho * (v_r*v_r + v_phi*v_phi);
        Real total_energy = phydro->u(IEN,k,j,i);
        Real internal = total_energy - kinetic;
        Real press = internal * (gamma_gas - 1.0);
        
        if (press < press_floor || !std::isfinite(press) || internal < 0.0) {
          need_reset = true;
        }
        
        if (need_reset) {
          Real f = DiskFunction(r);
          Real rho_new, press_new;
          
          if (f > 0.0) {
            rho_new = DiskDensity(r);
            if (rho_new < rho_floor) rho_new = rho_floor;
            press_new = rho_new * PressureOverDensity(r);
            if (press_new < press_floor) press_new = press_floor;
          } else {
            rho_new = rho_floor;
            press_new = press_floor;
          }
          
          Real v_phi_new = v_phi_constant / r;
          
          // Apply corrected values
          phydro->u(IDN,k,j,i) = rho_new;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = rho_new * v_phi_new;
          phydro->u(IM3,k,j,i) = 0.0;
          
          Real kin_new = 0.5 * rho_new * v_phi_new * v_phi_new;
          Real int_new = press_new / (gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) = int_new + kin_new;
          
          phydro->w(IDN,k,j,i) = rho_new;
          phydro->w(IVX,k,j,i) = 0.0;
          phydro->w(IVY,k,j,i) = v_phi_new;
          phydro->w(IVZ,k,j,i) = 0.0;
          phydro->w(IPR,k,j,i) = press_new;
        }
      }
    }
  }
  return;
}

void NewtonianGravity(MeshBlock *pmb, const Real time, const Real dt,
                 const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                 AthenaArray<Real> &cons_scalar) {
  
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real rho = prim(IDN,k,j,i);
        Real v_r = prim(IVX,k,j,i);
        
        Real g = -beta_param / (r * r);
        
        cons(IM1,k,j,i) += dt * rho * g;
        cons(IEN,k,j,i) += dt * rho * g * v_r;
      }
    }
  }
  return;
}

// Spatially-varying alpha viscosity: nu = alpha * (gamma/sqrt(beta)) * (p/rho) * r^(3/2)
void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                   int is, int ie, int js, int je, int ks, int ke) {
  Real coeff = alpha_visc * gamma_gas / std::sqrt(beta_param);
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real rho = prim(IDN,k,j,i);
        Real press = prim(IPR,k,j,i);
        
        // nu = alpha * (gamma/sqrt(beta)) * (p/rho) * r^(3/2)
        Real nu_func = coeff * (press / rho) * std::pow(r, 1.5);
        
        phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) = nu_func;
      }
    }
  }
  return;
}

// Total disk mass [M_sun]
Real TotalDiskMass(MeshBlock *pmb, int iout) {
  Real mass_sum = 0.0;
  
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rho_code = pmb->phydro->w(IDN,k,j,i);
        Real vol = pmb->pcoord->GetCellVolume(k,j,i);
        
        mass_sum += rho_code * vol;
      }
    }
  }
  
  return mass_sum * mass_scale;
}

// Mass accretion rate [M_sun/yr] through inner boundary
Real AccretionRate(MeshBlock *pmb, int iout) {
  Real mdot_sum = 0.0;
  int i = pmb->is;
  
  if (pmb->pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::outflow) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        Real rho_code = pmb->phydro->w(IDN,k,j,i);
        Real vr_code = pmb->phydro->w(IVX,k,j,i);
        Real area = pmb->pcoord->GetFace1Area(k,j,i);
      
        mdot_sum += -rho_code * vr_code * area;
      }
    }
  }
  return mdot_sum * mdot_scale;
}
