// C++ headers
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../parameter_input.hpp"

// Disk parameters
struct Disk {Real Const; Real xm; Real k; Real Kgas; Real rho_m; Real gm;};
Disk disk;

Real T, l_0, Crat, Prat, rhoSc, rho_floor, gam, gam_rat;

Real DiskDensity(Real, Disk, Real);

void NewtonianGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar);

// Right-hand bracket of the disk equation
Real DiskEquation(Real x, Real xm, Real disk_Const){
  return (1.0/x - xm/2.0/pow(x,2) - disk_Const);
}

// Density of the disk
Real DiskDensity(Real x, Disk disk){
  Real eq = disk.gm / gam_rat / disk.Kgas * DiskEquation(x, disk.xm, disk.Const);
  Real rho = pow(eq, 1.0/(gam-1.0));
  return rho;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  Real MBH, vs, rg;

  // Constants
  const Real c   = 2.99792458e10;  // Speed of light (cm/s)
  const Real MSOLAR = 1.989e33;    // Solar mass (g)
  const Real K_B = 1.380649e-16;   // Boltzmann constant (erg/K)
  const Real MP  = 1.6726219e-24;  // Proton mass (g)
  const Real G   = 6.67430e-8;     // Gravitational constant (cm^3*g^-1*s^-2)
  const Real mu  = 1.0;            // Mean molecular weight (g/mol)

  // Read parameters from the input file
  MBH = MSOLAR * pin->GetOrAddReal("problem", "MBH", 1.0);
  l_0 = pin->GetOrAddReal("problem", "l_0", 1.0);
  T = pin->GetOrAddReal("problem", "T", 1.0);
  rhoSc = pin->GetOrAddReal("problem", "rhoSc", 1.0);
  gam = pin->GetOrAddReal("hydro", "gamma", 1.6666666666666667);
  rho_floor = rhoSc*pin->GetOrAddReal("problem", "rho_floor", 0.0000001);
  disk.rho_m = pin->GetOrAddReal("problem", "disk_rho_m", 1.0);
  disk.xm = pin->GetOrAddReal("problem", "disk_xm", 1.0);
  disk.Const = pin->GetOrAddReal("problem", "disk_Const", 1.0);

  // Calculate the sound speed
  vs = sqrt(K_B * T / (mu * MP));

  // Calculate the ratio of the speed of light to the sound speed
  Crat = c / vs;

  // Calculate the gravitational radius (Schwarzschild)
  rg = 2 * G * MBH / pow(c, 2.0);

  // Calculate the parameter "k" (ratio of the length scale to gravitational radius)
  disk.k = l_0 / rg;

  // Calculate the polytropic constant "Kgas"
  disk.gm = pow(Crat, 2)/2.0/disk.k;
  gam_rat = gam/(gam-1.0);
  disk.Const = disk.Const / disk.xm;
  disk.Kgas = pow(disk.rho_m, 1.0/gam) * disk.gm / gam_rat * DiskEquation(disk.xm, disk.xm, disk.Const);

  EnrollUserExplicitSourceFunction(NewtonianGravity);

}

// Initialization of the disk
void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        gam = peos->GetGamma();
        Real x = pcoord->x1v(i);
        Real f = DiskEquation(x, disk.xm, disk.Const);

        if (f > 0.0) {
          Real v_phi = sqrt(disk.gm/disk.xm)*disk.xm/x;
          Real rho = DiskDensity(x, disk);
          phydro->u(IDN,k,j,i) = rho;
          phydro->u(IM2,k,j,i) = v_phi * rho;

          Real press = disk.Kgas * pow(rho, gam);
          Real ener = press/(gam-1.0);
          phydro->u(IEN,k,j,i) = ener + 0.5 * rho * v_phi * v_phi;
        } else {
          phydro->u(IDN,k,j,i) = rho_floor;
          phydro->u(IM2,k,j,i) = 0.0;

          Real press = disk.Kgas * pow(rho_floor, gam);
          Real ener = press/(gam-1.0);
          phydro->u(IEN,k,j,i) = ener;
        }

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

      }
    }
  }
}

// Control every sell on each time step
void MeshBlock::UserWorkInLoop() {
  auto &u = phydro->u;
  auto &w = phydro->w;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {

        // Density control
        Real rho = u(IDN,k,j,i);
        if (rho < 0.0) {
          rho = 0.0;
          u(IDN,k,j,i) = 0.0;
        }

        // Radial and azimuthal velocity control
        // w(IVX,k,j,i) = 0.0;  // Radial
        // Real x = pcoord->x1v(i);
        // Real f = 1.0/x - disk.xm/(2.0 * x * x) - disk.Const;
        // if (f <= 0.0) w(IVY,k,j,i) = 0.0;  // Azimuthal

        // Non-negative velocity
        Real v_r = w(IVX,k,j,i);
        Real v_phi = w(IVY,k,j,i);

        // if (v_r < 0.0)   u(IM1,k,j,i) = 0.0;
        // if (v_phi < 0.0) u(IM2,k,j,i) = 0.0;

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

// Newtonian Gravity source term
void NewtonianGravity(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real rho = prim(IDN,k,j,i);
        Real x = pmb->pcoord->x1v(i);
        Real g = disk.gm/(x*x);
        Real vr = prim(IM1)/rho;

        // Momentum equation
        cons(IM1,k,j,i) += -dt * rho * g;

        // Energy equation
        cons(IEN,k,j,i) += -dt * rho * g * vr;
      }
    }
  }
}