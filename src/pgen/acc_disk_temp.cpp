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

  // Input Physical Parameters
  Real T_0;          // Reference temperature [K]
  Real mu_gas;       // Mean molecular weight
  Real chi_param;    // Scaling parameter (r_0 / r_g)

  // Derived Dimensionless Parameters
  Real beta_param;   // Dimensionless gravity (calculated internally)
  Real r_center;     // Dimensionless shifted center of the disk
  Real C_prime;      // Geometric disk thickness parameter
  Real gamma_gas;    // Adiabatic index
  Real n_poly;       // Polytropic index
  Real nu_iso;       // Kinematic viscosity
  Real rho_floor;    // Density floor
  Real press_floor;  // Pressure floor
  Real r_inner;      // Inner disk geometric boundary
  Real r_outer;      // Outer disk geometric boundary
}

// Function declarations
void NewtonianGravity(MeshBlock *pmb, const Real time, const Real dt,
                 const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                 AthenaArray<Real> &cons_scalar);
void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                   int is, int ie, int js, int je, int ks, int ke);

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
  gamma_gas  = pin->GetReal("hydro", "gamma");
  
  rho_floor   = pin->GetOrAddReal("hydro", "dfloor", 1.0e-8);
  press_floor = pin->GetOrAddReal("hydro", "pfloor", 1.0e-10);
  
  // 2. Read physical parameters
  T_0       = pin->GetReal("problem", "T_0");
  mu_gas    = pin->GetReal("problem", "mu");
  chi_param = pin->GetReal("problem", "chi");

  // 3. Calculate internal physics
  n_poly = 1.0 / (gamma_gas - 1.0);

  // Calculate isothermal sound speed squared (cs0^2) in CGS [cm^2/s^2]
  Real cs0_sq = (gamma_gas * K_B * T_0) / (mu_gas * M_P);
  Real cs0    = std::sqrt(cs0_sq);

  // Calculate dimensionless gravity parameter beta
  beta_param = (C_LIGHT * C_LIGHT) / (2.0 * chi_param * cs0_sq);

  // 4. Calculate exact geometric disk boundaries
  Real discriminant = 1.0 - 2.0 * C_prime;
  if (discriminant < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in acc_disk.cpp" << std::endl
        << "Invalid C_prime = " << C_prime << ". Must satisfy 2*C' < 1." << std::endl;
    ATHENA_ERROR(msg);
  }
  r_inner = r_center * ( (1.0 - std::sqrt(discriminant)) / (2.0 * C_prime) );
  r_outer = r_center * ( (1.0 + std::sqrt(discriminant)) / (2.0 * C_prime) );
  
  // 5. Output model parameters to console
  if (Globals::my_rank == 0) {
    std::cout << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Papaloizou-Pringle Disk" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  --- Physical Inputs ---" << std::endl;
    std::cout << "  Temperature (T_0):              " << T_0 << " K" << std::endl;
    std::cout << "  Mean molecular weight (mu):     " << mu_gas << std::endl;
    std::cout << "  Scaling parameter (chi):        " << chi_param << std::endl;
    std::cout << "  Calculated Sound Speed (cs_0):  " << cs0 << " cm/s" << std::endl;
    std::cout << "  --- Dimensionless Parameters ---" << std::endl;
    std::cout << "  Calculated Gravity (beta):      " << beta_param << std::endl;
    std::cout << "  Disk Center Shift (r_center):   " << r_center << std::endl;
    std::cout << "  Disk Width Parameter (C'):      " << C_prime << std::endl;
    std::cout << "  Adiabatic Index (gamma):        " << gamma_gas << std::endl;
    std::cout << "  Inner Geometric Boundary:       " << r_inner << std::endl;
    std::cout << "  Outer Geometric Boundary:       " << r_outer << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::endl;
  }
  
  EnrollUserExplicitSourceFunction(NewtonianGravity);
  if (nu_iso > 0.0) {
    EnrollViscosityCoefficient(DiskViscosity);
  }
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
          v_phi = v_phi_constant / r;
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

void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                   int is, int ie, int js, int je, int ks, int ke) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) = nu_iso;
      }
    }
  }
  return;
}