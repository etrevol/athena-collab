//========================================================================================
//! \file visc_ring.cpp
//! \brief Viscously spreading ring (Lynden-Bell & Pringle 1974). Cylindrical (r, phi), 2D.
//!
//! A narrow ring of gas on Keplerian orbits with a constant kinematic viscosity spreads
//! according to an exact analytic solution, which makes it the reference test for the
//! viscous operator. Run with nu_iso = 0 it instead measures the scheme's own numerical
//! viscosity, since any spreading that survives is the scheme's.
//!
//! Gravity and viscosity are Athena++'s own: <problem>/GM drives
//! HydroSourceTerms::PointMass, and <problem>/nu_iso with no enrolled coefficient gives
//! ConstViscosity. Nothing here reimplements either.
//!
//! <hydro>/gamma is taken near 1 on purpose. The solution assumes a pressureless disk,
//! and viscous heating would otherwise thicken the disk over one spreading time.
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

namespace {
  Real gm_ring;       // GM, also read by Athena++'s own point-mass source term
  Real r_ring;        // R0, the initial ring radius
  Real tau_init;      // dimensionless time the run starts from
  Real sigma_peak;    // surface density at the ring centre at tau_init
  Real sigma_bg;      // uniform background, so the domain is never empty
  Real aspect;        // h = c_s / v_K
  Real nu_visc;       // constant kinematic viscosity (0 = measure the scheme's own)
  Real gamma_gas;
  Real mass_norm;     // M/(pi R0^2), fixed by sigma_peak
}

void InnerX1KeplerBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1KeplerBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//----------------------------------------------------------------------------------------
//! \brief Exponentially scaled modified Bessel function exp(-z) I_{1/4}(z), z > 0.
//!        Ascending series below z = 15, asymptotic expansion above; the two agree to
//!        ~1e-10 at the join. std::cyl_bessel_i is C++17, this code is built as C++11.
Real BesselI14Scaled(Real z) {
  const Real nu = 0.25;
  if (z <= 0.0) return 0.0;
  if (z < 15.0) {
    Real half = 0.5 * z;
    Real term = std::pow(half, nu) / std::tgamma(nu + 1.0);
    Real sum = term;
    for (int k = 1; k < 60; ++k) {
      term *= half * half / (k * (k + nu));
      sum += term;
      if (term < 1.0e-18 * sum) break;
    }
    return sum * std::exp(-z);
  }
  // Asymptotic expansion, Abramowitz & Stegun 9.7.1
  const Real mu = 4.0 * nu * nu;
  Real t = 1.0, sum = 1.0;
  for (int k = 1; k <= 6; ++k) {
    Real f = (2.0 * k - 1.0) * (2.0 * k - 1.0);
    t *= (mu - f) / (k * 8.0 * z);
    sum += ((k % 2) ? -t : t);
  }
  return sum / std::sqrt(2.0 * PI * z);
}

//----------------------------------------------------------------------------------------
//! \brief Analytic surface density of the spreading ring. The scaled Bessel function and
//!        the combined exponent keep this finite at the small tau the run starts from.
Real RingSigma(Real r, Real tau) {
  Real x = r / r_ring;
  if (x <= 0.0) return 0.0;
  Real arg = (1.0 - x) * (1.0 - x) / tau;
  if (arg > 700.0) return 0.0;              // exp underflows; the ring is not out here
  return mass_norm / (tau * std::pow(x, 0.25))
         * BesselI14Scaled(2.0 * x / tau) * std::exp(-arg);
}

//----------------------------------------------------------------------------------------
//! \brief Radial drift of the analytic solution. Zero when nu is zero, which is what the
//!        numerical-viscosity run needs: no physical drift is imposed there.
Real RingVr(Real r, Real tau) {
  if (nu_visc <= 0.0) return 0.0;
  Real sig = RingSigma(r, tau) + sigma_bg;
  if (sig <= 0.0) return 0.0;
  Real d = 1.0e-4 * r_ring;
  Real fp = (RingSigma(r + d, tau) + sigma_bg) * std::sqrt(r + d);
  Real fm = (RingSigma(r - d, tau) + sigma_bg) * std::sqrt(r - d);
  return -3.0 * nu_visc * (fp - fm) / (2.0 * d) / (sig * std::sqrt(r));
}

//----------------------------------------------------------------------------------------
//! \brief Rotation from exact radial balance, so the run starts in equilibrium apart
//!        from the viscous drift. The analytic solution ignores pressure; letting v_phi
//!        absorb it removes a startup transient of the same size as the spreading.
Real RingVphi(Real r, Real tau) {
  Real d = 1.0e-4 * r_ring;
  auto press = [tau](Real rr) {
    Real sig = RingSigma(rr, tau) + sigma_bg;
    Real cs2 = aspect * aspect * gm_ring / rr;       // c_s = h v_K
    return sig * cs2 / gamma_gas;
  };
  Real rho = RingSigma(r, tau) + sigma_bg;
  Real dpdr = (press(r + d) - press(r - d)) / (2.0 * d);
  Real v2 = gm_ring / r + r * dpdr / rho;
  return (v2 > 0.0) ? std::sqrt(v2) : 0.0;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  gm_ring   = pin->GetOrAddReal("problem", "GM", 1.0);
  r_ring    = pin->GetOrAddReal("problem", "r_ring", 1.0);
  tau_init  = pin->GetOrAddReal("problem", "tau0", 0.018);
  sigma_peak = pin->GetOrAddReal("problem", "sigma_peak", 1.0);
  sigma_bg  = pin->GetOrAddReal("problem", "sigma_bg", 1.0e-6);
  aspect    = pin->GetOrAddReal("problem", "aspect", 0.05);
  nu_visc   = pin->GetOrAddReal("problem", "nu_iso", 0.0);
  gamma_gas = pin->GetReal("hydro", "gamma");

  // Fix M so the analytic peak at x = 1 equals sigma_peak. Taken from the analytic
  // value rather than from the grid, so the normalisation is resolution independent.
  mass_norm = 1.0;
  Real peak = RingSigma(r_ring, tau_init);
  if (peak <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in visc_ring.cpp" << std::endl
        << "The ring has zero amplitude at tau0 = " << tau_init << "." << std::endl;
    ATHENA_ERROR(msg);
  }
  mass_norm = sigma_peak / peak;

  if (Globals::my_rank == 0) {
    std::streamsize saved = std::cout.precision();
    std::cout.precision(12);
    Real width = std::sqrt(tau_init) * r_ring;       // e-folding half-width of the ring
    std::cout << std::endl
              << "=========================================================" << std::endl
              << "  Lynden-Bell & Pringle viscously spreading ring" << std::endl
              << "=========================================================" << std::endl
              << "  GM                 " << gm_ring << std::endl
              << "  R0                 " << r_ring << std::endl
              << "  tau0               " << tau_init << std::endl
              << "  nu_iso             " << nu_visc
              << (nu_visc > 0.0 ? "" : "   (inviscid: any spreading is numerical)")
              << std::endl
              << "  sigma_peak         " << sigma_peak << std::endl
              << "  sigma_bg           " << sigma_bg << std::endl
              << "  aspect h           " << aspect << std::endl
              << "  gamma              " << gamma_gas << std::endl
              << "  mass_norm          " << mass_norm << std::endl
              << "  ring half-width    " << width << "  (needs several cells)"
              << std::endl
              << "  orbital period     " << 2.0 * PI * std::sqrt(r_ring * r_ring * r_ring
                                                                 / gm_ring) << std::endl;
    if (nu_visc > 0.0) {
      std::cout << "  t per unit tau     " << r_ring * r_ring / (12.0 * nu_visc)
                << std::endl;
    }
    std::cout << "=========================================================" << std::endl
              << std::endl;
    std::cout.precision(saved);
  }

  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1KeplerBC);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1KeplerBC);
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real r = pcoord->x1v(i);
        Real rho = RingSigma(r, tau_init) + sigma_bg;
        Real cs2 = aspect * aspect * gm_ring / r;
        Real press = rho * cs2 / gamma_gas;
        Real v_r = RingVr(r, tau_init);
        Real v_phi = RingVphi(r, tau_init);

        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = rho * v_r;
        phydro->u(IM2,k,j,i) = rho * v_phi;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = press / (gamma_gas - 1.0)
                               + 0.5 * rho * (v_r * v_r + v_phi * v_phi);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Radial boundaries: zero-gradient in rho and p, diode in v_r, and v_phi scaled
//!        as r^(-1/2) so the ghost zones stay Keplerian. Copying v_phi would leave them
//!        under-rotating and drive a spurious inflow.
void InnerX1KeplerBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real scale = std::sqrt(pco->x1v(il) / pco->x1v(il-i));
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        prim(IVX,k,j,il-i) = std::min(prim(IVX,k,j,il), 0.0);
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il) * scale;
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il);
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il);
      }
    }
  }
  return;
}

void OuterX1KeplerBC(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real scale = std::sqrt(pco->x1v(iu) / pco->x1v(iu+i));
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        prim(IVX,k,j,iu+i) = std::max(prim(IVX,k,j,iu), 0.0);
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu) * scale;
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
      }
    }
  }
  return;
}
