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

// Global variables for disk parameters
namespace {
  Real chi;          // Scaling parameter (r_0 = chi * r_g)
  Real C;            // Disk width parameter
  Real gamma_gas;    // Adiabatic index
  Real n;            // Polytropic index n = 1/(gamma-1)
  Real gm;           // Dimensionless GM coefficient: c^2/(2*chi*v_0^2) = 1
  Real nu_iso;       // Kinematic viscosity
  Real rho_floor;    // Density floor
  Real press_floor;  // Pressure floor
  Real r_inner;      // Inner disk boundary
  Real r_outer;      // Outer disk boundary
}

// Function declarations
void NewtonianGravity(MeshBlock *pmb, const Real time, const Real dt,
                 const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                 AthenaArray<Real> &cons_scalar);

void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                   int is, int ie, int js, int je, int ks, int ke);

Real DiskFunction(Real r) {
  return 1.0/r - 1.0/(2.0 * r * r) - C;
}

Real PressureOverDensity(Real r) {
  Real f = DiskFunction(r);
  if (f > 0.0) {
    return f / (n + 1.0);
  }
  return 0.0;
}

Real DiskDensity(Real r, Real p_over_rho_center) {
  Real p_over_rho = PressureOverDensity(r);
  if (p_over_rho > 0.0) {
    // Polytropic relation: rho = [p/rho / (p/rho)_center]^(1/(gamma-1))
    return std::pow(p_over_rho / p_over_rho_center, n);
  }
  return rho_floor;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters
  C = pin->GetReal("problem", "C");
  gamma_gas = pin->GetReal("hydro", "gamma");
  nu_iso = pin->GetOrAddReal("hydro", "nu_iso", 0.0);
  
  // Read floors
  rho_floor = pin->GetOrAddReal("hydro", "dfloor", 1.0e-8);
  press_floor = pin->GetOrAddReal("hydro", "pfloor", 1.0e-10);
  
  // Calculate derived quantities
  n = 1.0 / (gamma_gas - 1.0);
  gm = 1.0;  // Dimensionless gravity coefficient (by construction c^2/(2*chi*v_0^2) = 1)
  
  // Calculate disk boundaries (solve f(r) = 0)
  Real discriminant = 1.0 - 2.0*C;
  if (discriminant < 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in acc_disk.cpp" << std::endl
        << "Invalid C = " << C << ", must satisfy 2*C < 1" << std::endl;
    ATHENA_ERROR(msg);
  }
  r_inner = (1.0 - std::sqrt(discriminant)) / (2.0 * C);
  r_outer = (1.0 + std::sqrt(discriminant)) / (2.0 * C);
  
  // Output disk parameters
  if (Globals::my_rank == 0) {
    std::cout << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Papaloizou-Pringle Accretion Disk" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << "  Disk width parameter: C      = " << C << std::endl;
    std::cout << "  Adiabatic index:      gamma       = " << gamma_gas << std::endl;
    std::cout << "  Polytropic index:     n       = " << n << std::endl;
    std::cout << "  Inner boundary:       r_in   = " << r_inner << std::endl;
    std::cout << "  Outer boundary:       r_out  = " << r_outer << std::endl;
    std::cout << "  Kinematic viscosity:  nu      = " << nu_iso << std::endl;
    std::cout << "  Density floor:        rho_floor = " << rho_floor << std::endl;
    std::cout << "  Pressure floor:       p_floor = " << press_floor << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::endl;
  }
  
  // Enroll user-defined functions
  EnrollUserExplicitSourceFunction(NewtonianGravity);
  
  // Enroll viscosity if nu_iso > 0
  if (nu_iso > 0.0) {
    EnrollViscosityCoefficient(DiskViscosity);
  }
  
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Calculate normalization at disk center
  Real p_over_rho_center = PressureOverDensity(1.0);
  
  // Initialize disk structure
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r = pcoord->x1v(i);  // Radial coordinate (dimensionless)
        
        // Check if inside disk
        Real f = DiskFunction(r);
        
        Real rho, press, v_r, v_phi;
        
        if (f > 0.0) {
          // Inside disk: use equilibrium solution
          
          // Density from polytropic relation
          rho = DiskDensity(r, p_over_rho_center);
          if (rho < rho_floor) rho = rho_floor;
          
          // Pressure from equilibrium
          Real p_over_rho = PressureOverDensity(r);
          press = rho * p_over_rho;
          if (press < press_floor) press = press_floor;
          
          // Velocities: equilibrium conditions
          v_r = 0.0;  // No radial motion in equilibrium
          v_phi = 1.0 / r;  // Constant angular momentum v_φ = 1/r
          
        } else {
          // Outside disk: set to floor values
          rho = rho_floor;
          press = press_floor;
          v_r = 0.0;
          v_phi = 1.0 / r;  // Constant angular momentum
        }
        
        // Set conservative variables
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = rho * v_r;      // Radial momentum
        phydro->u(IM2,k,j,i) = rho * v_phi;    // Azimuthal momentum
        phydro->u(IM3,k,j,i) = 0.0;            // Vertical momentum
        
        // Total energy: E = p/(gamma-1) + 0.5*rho*v^2
        Real kinetic_energy = 0.5 * rho * (v_r*v_r + v_phi*v_phi);
        Real internal_energy = press / (gamma_gas - 1.0);
        phydro->u(IEN,k,j,i) = internal_energy + kinetic_energy;
      }
    }
  }
  
  return;
}

void MeshBlock::UserWorkInLoop() {
  // Get pressure/density at center for normalization
  Real p_over_rho_center = PressureOverDensity(1.0);
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r = pcoord->x1v(i);
        
        // Check for invalid values
        Real rho = phydro->u(IDN,k,j,i);
        Real v_r = phydro->w(IVX,k,j,i);
        Real v_phi = phydro->w(IVY,k,j,i);
        
        bool need_reset = false;
        
        // Check density floor
        if (rho < rho_floor || !std::isfinite(rho)) {
          need_reset = true;
        }
        
        // Check energy validity
        Real kinetic = 0.5 * rho * (v_r*v_r + v_phi*v_phi);
        Real total_energy = phydro->u(IEN,k,j,i);
        Real internal = total_energy - kinetic;
        Real press = internal * (gamma_gas - 1.0);
        
        if (press < press_floor || !std::isfinite(press) || internal < 0.0) {
          need_reset = true;
        }
        
        // Reset to equilibrium if needed
        if (need_reset) {
          Real f = DiskFunction(r);
          Real rho_new, press_new;
          
          if (f > 0.0) {
            rho_new = DiskDensity(r, p_over_rho_center);
            if (rho_new < rho_floor) rho_new = rho_floor;
            Real p_over_rho = PressureOverDensity(r);
            press_new = rho_new * p_over_rho;
            if (press_new < press_floor) press_new = press_floor;
          } else {
            rho_new = rho_floor;
            press_new = press_floor;
          }
          
          Real v_phi_new = 1.0 / r;  // Constant angular momentum
          
          // Reset conservative variables
          phydro->u(IDN,k,j,i) = rho_new;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = rho_new * v_phi_new;
          phydro->u(IM3,k,j,i) = 0.0;
          
          Real kin_new = 0.5 * rho_new * v_phi_new * v_phi_new;
          Real int_new = press_new / (gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) = int_new + kin_new;
          
          // Reset primitive variables
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
        
        // Gravitational acceleration: g = -gm/r^2
        Real g = -gm / (r * r);
        
        // Add to radial momentum: dM_r/dt = rho*g
        cons(IM1,k,j,i) += dt * rho * g;
        
        // Add to energy: dE/dt = rho*g*v_r
        cons(IEN,k,j,i) += dt * rho * g * v_r;
      }
    }
  }
  
  return;
}

void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                   int is, int ie, int js, int je, int ks, int ke) {
  
  // Set constant viscosity throughout domain
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
