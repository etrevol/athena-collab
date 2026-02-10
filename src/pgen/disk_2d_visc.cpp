// C++ headers
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../parameter_input.hpp"

// Disk parameters
struct Disk {Real Const; Real xm; Real k; Real Kgas; Real rho_m; Real gm; Real nu;};
Disk disk;

Real T, l_0, Crat, rho_floor, gam, gam_rat;

void SourceTerms(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar);

void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                   const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
                   int ks, int ke);

// Right-hand bracket of the disk equation
Real DiskEquation(Real x, Real xm, Real disk_Const){
  return (xm/x - pow(xm,2)/2.0/pow(x,2) - disk_Const);
}

// Density of the disk
Real DiskDensity(Real x, Disk disk){
  Real eq =  DiskEquation(x, disk.xm, disk.Const)/DiskEquation(disk.xm, disk.xm, disk.Const);
  Real rho = pow(eq, 1.0/(gam-1.0));
  return rho;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  Real vs;

  // Constants
  const Real c   = 2.99792458e10;  // Speed of light (cm/s)
  const Real K_B = 1.380649e-16;   // Boltzmann constant (erg/K)
  const Real MP  = 1.6726219e-24;  // Proton mass (g)
  const Real mu  = 1.0;            // Mean molecular weight (g/mol)

  // Read parameters from the input file
  disk.k = pin->GetOrAddReal("problem", "k", 1.0);
  T = pin->GetOrAddReal("problem", "T", 1.0);
  gam = pin->GetOrAddReal("hydro", "gamma", 1.6666666666666667);
  rho_floor = pin->GetOrAddReal("problem", "rho_floor", 0.0000001);
  disk.xm = pin->GetOrAddReal("problem", "disk_xm", 1.0);
  disk.Const = pin->GetOrAddReal("problem", "disk_Const", 1.0);
  disk.nu = pin->GetOrAddReal("problem", "nu_iso", 0.0);

  // Calculate the sound speed
  //vs = sqrt(K_B * T / (mu * MP));
  vs = 100000;
  // Calculate the ratio of the speed of light to the sound speed
  Crat = c / vs;

  // Calculate the polytropic constant "Kgas"
  disk.gm = pow(Crat, 2)/2.0/disk.k;
  gam_rat = gam/(gam-1.0);
  disk.Kgas = disk.gm / disk.xm / gam_rat * DiskEquation(disk.xm, disk.xm, disk.Const);

  EnrollUserExplicitSourceFunction(SourceTerms);
  
  // Enroll viscosity function if nu > 0
  if (disk.nu > 0.0) {
    EnrollViscosityCoefficient(DiskViscosity);
  }
}

// Initialization of the disk
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real n = 1.0/(gam - 1.0);
  Real f0 = DiskEquation(disk.xm, disk.xm, disk.Const);
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x = pcoord->x1v(i);
        Real f = DiskEquation(x, disk.xm, disk.Const);
        Real v_phi = sqrt(disk.gm/disk.xm) * disk.xm / x;
        
        Real rho, press;
        if (f > 0.0) {
          rho = (DiskDensity(x, disk) > rho_floor) ? DiskDensity(x, disk) : rho_floor;
          press = disk.gm / (n + 1) / disk.xm * pow(f, n+1) / pow(f0, n);
        } else {
          rho = rho_floor;
          press = disk.Kgas * pow(rho_floor, gam);
        }
        
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = v_phi * rho;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = press/(gam - 1.0) + 0.5 * rho * v_phi * v_phi;
      }
    }
  }
}

// Control every cell on each time step
void MeshBlock::UserWorkInLoop() {
  auto &u = phydro->u;
  auto &w = phydro->w;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {

        // Density control
        Real rho = u(IDN,k,j,i);
        Real x = pcoord->x1v(i);

        if (rho < rho_floor || !std::isfinite(phydro->u(IM1,k,j,i))) {
          rho = rho_floor;
          Real press = disk.Kgas * pow(rho, gam);
          Real v_phi = sqrt(disk.gm/disk.xm)*disk.xm/x;
          Real ener = press / (gam - 1.0);
          phydro->u(IDN,k,j,i) = rho;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = sqrt(disk.gm/disk.xm)*disk.xm/x*rho_floor;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(IEN,k,j,i) = ener+ 0.5 * rho_floor * v_phi * v_phi;

          phydro->w(IDN,k,j,i) = rho;
          phydro->w(IM1,k,j,i) = 0.0;
          phydro->w(IM2,k,j,i) = sqrt(disk.gm/disk.xm)*disk.xm/x*rho_floor;
          phydro->w(IM3,k,j,i) = 0.0;
          phydro->w(IPR,k,j,i) = press;
        }

        Real v_r = w(IVX,k,j,i);
        Real v_phi = w(IVY,k,j,i);

        // Pressure control
        Real kin_energy = 0.5 * rho * (v_r*v_r + v_phi*v_phi);
        Real total_energy = u(IEN,k,j,i);
        Real press = (total_energy - kin_energy) * (gam - 1.0);
        if (press < 0.0 || total_energy < kin_energy) {
          u(IEN,k,j,i) = kin_energy;
        }

        // Energy control
        if (u(IEN,k,j,i) < 0.0) {
          u(IEN,k,j,i) = 0.0;
        }
      }
    }
  }
}

// Newtonian Gravity (Viscosity handled by HydroDiffusion)
void SourceTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        // Newtonian Gravity
        Real rho = prim(IDN,k,j,i);
        Real x = pmb->pcoord->x1v(i);
        Real g = disk.gm/(x*x);
        Real vr = prim(IVX,k,j,i);

        // Radial momentum
        cons(IM1,k,j,i) += -dt * rho * g;

        // Energy
        cons(IEN,k,j,i) += -dt * rho * g * vr;
      }
    }
  }
}

void DiskViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                   const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
                   int ks, int ke) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) = disk.nu;
      }
    }
  }
  return;
}
