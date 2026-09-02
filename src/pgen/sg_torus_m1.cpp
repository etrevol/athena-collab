//========================================================================================
// Self-gravitating Papaloizou-Pringle torus around a central point mass. 3D Cartesian.
//
// The hydrodynamic counterpart of the collisionless N-body torus of Bannikova et al.
// (2026, A&A; arXiv:2604.11528), which finds that a global m = 1 slow mode grows
// spontaneously out of an axisymmetric state and displaces the central mass.
//
// Units follow that paper: G = 1, M_c = 1, R_tor = 1, so T_orb = 2 pi at R = 1.
//
// Cartesian rather than cylindrical because Athena++ solves the Poisson equation on
// uniform Cartesian meshes only - there is no cylindrical self-gravity solver, and the
// paper's mode does not exist without self-gravity.
//
// Three pieces of gravity, kept separate on purpose:
//   1. the softened central point mass, an analytic potential;
//   2. the gas self-gravity, the Multigrid solve (SELF_GRAVITY_ENABLED);
//   3. the indirect term - the pull of the gas *on the central mass*. The grid keeps
//      the point mass pinned at the origin, so its acceleration is instead applied to
//      the gas with the opposite sign. Without this the central mass would be nailed
//      to an inertial frame and the paper's headline result, its displacement, could
//      not appear at all.
//
// Build (two binaries, because self-gravity is a configure-time switch):
//   python3 configure.py --prob sg_torus_m1 --grav mg -mpi   # the physical runs
//   python3 configure.py --prob sg_torus_m1 -mpi             # Appendix C control
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {
  // Torus and central mass
  Real C_prime;      // torus thickness parameter
  Real q_rot;        // rotation-law exponent, l = l_0 (R/R_tor)^q; 0 = constant l
  Real M_tor;        // torus mass in units of the central mass
  Real rho_c;        // density at the torus centre, fixed by M_tor
  Real GM_c;         // central point mass, = 1 in these units
  Real eps_soft;     // softening length of the central point mass
  Real gamma_gas;    // adiabatic index
  Real n_poly;       // polytropic index 1/(gamma-1)
  Real Psi_c;        // shape function at the density maximum, = 0.5 - C'
  Real p_c;          // pressure at the torus centre
  Real r_in, r_out;  // mid-plane torus edges

  // Ambient medium, floors, viscosity
  Real rho_atm;
  Real p_atm;
  Real rho_floor;
  Real press_floor;
  Real alpha_visc;
  Real nu_iso;
  Real visc_rho_cut;

  // Switches
  bool use_indirect;
  Real pert_amp;       // per-cell white-noise density seed
  Real pert_mode_amp;  // coherent seed amplitude per harmonic, m = 1..5

  // The gas pull on the central mass, refreshed once per cycle by Mesh::UserWorkInLoop.
  // Lagged by one cycle: an O(dt) error, the same class of lag the alpha-viscosity
  // coefficient already carries, and it starts at zero because t = 0 is axisymmetric.
  Real a_indirect[3] = {0.0, 0.0, 0.0};

  // Density above which a cell counts as torus rather than ambient, for the diagnostics.
  Real rho_body;

  // ---- Self-consistent equilibrium (SCF) -----------------------------------------
  // The analytic shape function balances the gas against the central mass alone. At
  // M_tor/M_c = 0.3 that omits a third of the gravity, and the torus spends its first
  // ~30 orbits readjusting rather than evolving. These hold the iterated solution in
  // which the enthalpy is balanced against the gas potential as well.
  //
  // The gas potential is computed here rather than taken from Multigrid because the
  // solver is only called after ProblemGenerator has run, and because the control run
  // has no solver at all yet must be able to start from the same state. At t = 0 the
  // torus is axisymmetric, so the azimuthal integral of the Green's function is a
  // complete elliptic integral and the whole problem collapses to a 2D quadrature.
  int  scf_iter = 0;         // 0 = off, reproducing every run before this one
  int  scf_nr = 200;         // table resolution in R (~3.5x the mesh)
  int  scf_nz = 100;         // and in z, on the z >= 0 half plane
  Real scf_relax = 0.7;      // under-relaxation between iterations
  Real scf_tol = 1.0e-7;     // convergence on max|dPhi| / max|Phi|
  Real scf_Rmax = 0.0, scf_Zmax = 0.0, scf_dR = 1.0, scf_dZ = 1.0;
  std::vector<Real> scf_phi; // Phi_gas on the table, index k*(nr+1) + i
  bool scf_on = false;
  Real l0sq_eff = 0.0;       // rotation-law normalisation l_0^2, reset by the SCF
  Real C_eff = 0.0;          // Bernoulli constant, likewise
  int  scf_used = 0;         // iterations actually taken
  Real scf_rin_an = 0.0, scf_rout_an = 0.0, scf_zt = 0.0;  // what it started from
  Real scf_resid = 0.0;      // final relative change
}

void TorusGravity(MeshBlock *pmb, const Real time, const Real dt,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                  AthenaArray<Real> &cons_scalar);
void TorusViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                    int is, int ie, int js, int je, int ks, int ke);
Real TorusHistory(MeshBlock *pmb, int iout);
void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//----------------------------------------------------------------------------------------
//! Softened central potential. Plummer, so the force is finite in the empty inner hole.
Real CentralPotential(Real x, Real y, Real z) {
  return -GM_c / std::sqrt(x*x + y*y + z*z + eps_soft*eps_soft);
}

//----------------------------------------------------------------------------------------
//! Torus shape function. Positive inside the torus, zero on its surface.
//!
//! For a power-law rotation law l = l_0 (R/R_tor)^q with l_0^2 = GM R_tor = 1,
//!
//!     Psi = 1/sqrt(R^2+z^2) - R^(2q-2)/(2-2q) - C',
//!
//! whose radial derivative vanishes at R = R_tor = 1 for every q, so the pressure
//! maximum stays put as q is varied. At q = 0 this is 1/sqrt(R^2+z^2) - 1/(2R^2) - C',
//! and in the mid-plane with x = 1/R it becomes x - x^2/2 - C' - the same function the
//! 2D model uses. C' therefore carries over from acc_disk_visc unchanged.
//!
//! q exists because a gas torus with l = const is Papaloizou-Pringle unstable, with
//! m = 1 as its fastest growing mode - the same symmetry as the mode being looked for,
//! and with no counterpart in the collisionless problem. The instability weakens as the
//! rotation law approaches Keplerian (q -> 1/2), which is also what the paper's
//! particle orbits are. The catch is that Psi_c = 1 - 1/(2-2q) - C' must stay positive,
//! so q < 1 - 1/(2(1-C')): a pressure-supported torus cannot be both thick and
//! Keplerian, because for a fluid the thickness IS the departure from Keplerian.
//!
//! The equilibrium is built on the SAME softened potential the solver applies. Using
//! the point-mass one instead looks harmless - the torus never enters the softening
//! region - but it is not: at eps = 0.25 and r_in = 0.56 the softened force is only 76%
//! of the point-mass force at the inner edge, so a torus balanced against the latter
//! starts 24% out of equilibrium there. Consistency costs nothing, and it fixes the
//! angular momentum normalisation too: l_0^2 = (1 + eps^2)^(-3/2) keeps the pressure
//! maximum at R = R_tor = 1 exactly. Both reduce to the unsoftened forms at eps -> 0.

//----------------------------------------------------------------------------------------
//! Complete elliptic integral K(m), m = k^2, by the arithmetic-geometric mean. Quadratic
//! convergence, so twenty steps is far more than the double precision can hold.
Real EllipK(Real m) {
  Real a = 1.0, b = std::sqrt(std::max(1.0 - m, 1.0e-18));
  for (int i = 0; i < 20; ++i) {
    Real an = 0.5 * (a + b);
    Real bn = std::sqrt(a * b);
    if (std::fabs(an - bn) <= 1.0e-15 * an) { a = an; b = bn; break; }
    a = an; b = bn;
  }
  return PI / (a + b);
}

//! Potential at (R, z) of a unit-mass uniform ring of radius Rp in the plane z = zp.
//! The azimuthal integral of 1/|r - r'| over the ring is the elliptic integral above.
Real RingPotential(Real R, Real z, Real Rp, Real zp) {
  Real d2 = (R + Rp) * (R + Rp) + (z - zp) * (z - zp);
  if (d2 <= 0.0) return 0.0;
  Real m = 4.0 * R * Rp / d2;
  if (m > 1.0 - 1.0e-12) m = 1.0 - 1.0e-12;   // the log singularity is integrable
  return -(2.0 / PI) * EllipK(m) / std::sqrt(d2);
}

//! The same for a finite cell, with the near field subdivided. K diverges logarithmically
//! as the source approaches the target, so the midpoint rule is poor within a cell or two;
//! everywhere else it is already converged and the subdivision would be wasted.
Real CellPotential(Real R, Real z, Real Rp, Real zp, bool near) {
  if (!near) return RingPotential(R, z, Rp, zp);
  const int ns = 5;
  Real s = 0.0, w = 0.0;
  for (int a = 0; a < ns; ++a) {
    Real Rs = Rp + scf_dR * ((a + 0.5) / ns - 0.5);
    if (Rs <= 0.0) continue;
    for (int b = 0; b < ns; ++b) {
      Real zs = zp + scf_dZ * ((b + 0.5) / ns - 0.5);
      s += Rs * RingPotential(R, z, Rs, zs);   // ring mass goes as R
      w += Rs;
    }
  }
  return (w > 0.0) ? s / w : RingPotential(R, z, Rp, zp);
}

//! Bilinear interpolation of the tabulated gas potential; a monopole outside the table,
//! which the torus never reaches but the ambient does.
Real PhiGas(Real R, Real z) {
  if (!scf_on) return 0.0;
  Real az = std::fabs(z);
  if (R >= scf_Rmax || az >= scf_Zmax) {
    Real r = std::sqrt(R * R + z * z);
    return -M_tor / std::max(r, 1.0e-3);
  }
  Real fi = R / scf_dR, fk = az / scf_dZ;
  int i = static_cast<int>(fi), k = static_cast<int>(fk);
  if (i > scf_nr - 1) i = scf_nr - 1;
  if (k > scf_nz - 1) k = scf_nz - 1;
  Real ti = fi - i, tk = fk - k;
  const int nrp = scf_nr + 1;
  Real p00 = scf_phi[k*nrp + i],       p10 = scf_phi[k*nrp + i + 1];
  Real p01 = scf_phi[(k+1)*nrp + i],   p11 = scf_phi[(k+1)*nrp + i + 1];
  return (1.0 - tk) * ((1.0 - ti)*p00 + ti*p10)
       +        tk  * ((1.0 - ti)*p01 + ti*p11);
}

Real ShapeFunction(Real R, Real z) {
  if (R <= 0.0) return -1.0;
  Real l0sq = scf_on ? l0sq_eff : std::pow(1.0 + eps_soft*eps_soft, -1.5);
  Real Cst  = scf_on ? C_eff    : C_prime;
  return 1.0 / std::sqrt(R*R + z*z + eps_soft*eps_soft)
         - PhiGas(R, z)
         - l0sq * std::pow(R, 2.0*q_rot - 2.0) / (2.0 - 2.0*q_rot) - Cst;
}

Real TorusDensity(Real R, Real z) {
  if (scf_on && (R < r_in || R > r_out)) return 0.0;
  Real Psi = ShapeFunction(R, z);
  if (Psi <= 0.0) return 0.0;
  return rho_c * std::pow(Psi / Psi_c, n_poly);
}

//----------------------------------------------------------------------------------------
//! Reproducible white noise in [-1, 1], hashed from the global cell coordinates so that
//! the seed field does not depend on how the mesh is divided into blocks.
//!
//! Why seed at all, when the paper stresses that its mode appears with no imposed
//! perturbation. An N-body torus is not smooth: with N particles it carries Poisson
//! noise, and in each azimuthal harmonic that amounts to A_m/Sigma_0 ~ 1/sqrt(2N),
//! about 2e-3 for their N = 128k. That is what their instability grows from. A grid
//! initialised from an analytic profile has none of it: measured here, the m = 1
//! coefficient starts at ~2e-16.
//!
//! Per-cell white noise is a poor way to supply it, and the arithmetic says so. Noise of
//! amplitude eps spread over N_cell ~ 3e5 cells inside the torus averages down to
//! A_m/M_tor ~ eps/sqrt(2 N_cell), so eps = 3e-3 buys only 4e-6 - some 500 times quieter
//! than the N-body, and matching it would take eps = 1.5, which is not a perturbation at
//! all. Most of that power sits at high k, where it simply dissipates.
//!
//! ModeNoise below therefore seeds the low harmonics COHERENTLY, each with its own
//! random phase, which is what Poisson noise actually looks like at low m. It is still
//! broadband over m = 1..5 and never an m = 1 pattern on its own: which harmonic wins
//! remains the result, and pert_mode_amp = 0 is the control. The gain is not cosmetic -
//! from 4e-6 a mode needs 10 e-foldings to reach saturation, from 2e-3 only 4, which for
//! a secular mode growing at ~0.05 per orbit is the difference between 200 orbits and 80.
Real CellNoise(Real x, Real y, Real z) {
  // A cheap integer hash of the quantised position; deterministic across runs and ranks.
  auto q = [](Real v) { return static_cast<unsigned int>(
      static_cast<long long>(std::floor(v * 1.0e6)) & 0xffffffffLL); };
  unsigned int h = q(x) * 73856093u ^ q(y) * 19349663u ^ q(z) * 83492791u;
  h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
  return 2.0 * (static_cast<Real>(h) / 4294967295.0) - 1.0;
}

//! Coherent seed in the low azimuthal harmonics: sum over m = 1..5 of one random phase
//! each, all at the same amplitude, emulating the low-m content of Poisson noise.
Real ModeNoise(Real x, Real y) {
  const int m_seed = 5;
  Real phi = std::atan2(y, x);
  Real s = 0.0;
  for (int m = 1; m <= m_seed; ++m) {
    // fixed, reproducible phase per harmonic; nothing here singles out m = 1
    unsigned int h = 2654435761u * static_cast<unsigned int>(m) + 1013904223u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    Real phase = 2.0 * PI * (static_cast<Real>(h) / 4294967295.0);
    s += std::cos(m * phi - phase);
  }
  return s;
}

Real TorusPressure(Real R, Real z) {
  if (scf_on && (R < r_in || R > r_out)) return 0.0;
  Real Psi = ShapeFunction(R, z);
  if (Psi <= 0.0) return 0.0;
  return p_c * std::pow(Psi / Psi_c, n_poly + 1.0);
}

//----------------------------------------------------------------------------------------
//! \brief Iterate the equilibrium against its own gravity.
//!
//! The torus is a polytrope, so the Bernoulli condition is
//!
//!     H(R,z) = -Phi_c - Phi_gas - l_0^2 R^(2q-2)/(2-2q) - C = 0  on the surface,
//!     rho = rho_c (H/H_max)^n,   p = rho H / (n+1),
//!
//! and the analytic model is the special case Phi_gas = 0. Iterating it needs the gas
//! potential, which is a two-dimensional quadrature while the torus is still
//! axisymmetric: the azimuthal integral of the Green's function over a ring is the
//! complete elliptic integral K.
//!
//! Two constants have to be redetermined at every step, and the choice of what to hold
//! fixed is what the solution means. There are three shape descriptors - the two
//! mid-plane edges and the half-thickness - and only two constants, so one of them has
//! to be allowed to move. The two that are pinned here are the ones the comparison with
//! the paper is actually made through:
//!
//!   * the density maximum stays at R = R_tor = 1. This is the unit of length and the
//!     orbit that T_orb = 2 pi refers to, so every pattern speed quoted as a fraction of
//!     Omega is measured against it. dH/dR = 0 there gives
//!     l_0^2 = (1 + eps^2)^(-3/2) + dPhi_gas/dR, which reduces to the analytic value.
//!
//!   * the half-thickness z_max is unchanged. It is the analogue of their i_max, and the
//!     campaign varies M_tor at fixed thickness on purpose - a thickness that drifted
//!     with the mass would confound the two. The surface is tangent to z = z_max, so
//!     C = max_R [ -Phi_c - Phi_gas - l_0^2 R^(2q-2)/(2-2q) ] evaluated at z = z_max,
//!     which is again an explicit formula rather than a root find.
//!
//! Pinning the edges instead was tried and rejected: at M_tor/M_c = 0.3 it produced a
//! torus 13% thicker with its density maximum at R = 1.13, i.e. it moved both of the
//! quantities being compared. What moves here instead is r_in and r_out, which are
//! recomputed and reported, and which the box has margin for.
//!
//! rho_c is then fixed by the mass integral, as before.
//!
//! Convergence is on max|dPhi|/max|Phi|; the returned H_max and rho_c replace the
//! analytic Psi_c and rho_c in the caller.
void SolveSelfConsistentEquilibrium(Real *Hmax_out, Real *rhoc_out) {
  const int nr = scf_nr, nz = scf_nz, nrp = nr + 1, nzp = nz + 1;

  // Vertical extent of the analytic torus, so the table covers it with room to spare.
  Real z_est = 0.0;
  for (int i = 1; i < 1200; ++i) {
    Real R = r_in + (r_out - r_in) * i / 1200.0;
    for (int k = 0; k < 1200; ++k) {
      Real z = r_out * k / 1200.0;
      if (ShapeFunction(R, z) > 0.0) z_est = std::max(z_est, z);
    }
  }
  if (z_est <= 0.0) z_est = 0.5 * (r_out - r_in);
  const Real z_target = z_est;      // the analytic half-thickness, held fixed
  scf_rin_an = r_in; scf_rout_an = r_out; scf_zt = z_target;

  scf_Rmax = 1.5 * r_out;
  scf_Zmax = 2.0 * z_est;
  scf_dR = scf_Rmax / nr;
  scf_dZ = scf_Zmax / nz;
  scf_phi.assign(nrp * nzp, 0.0);
  scf_on = true;                   // PhiGas now reads the table, which is still zero

  std::vector<Real> phi_new(nrp * nzp, 0.0), rho_tab(nrp * nzp, 0.0);
  std::vector<Real> Rt(nrp), Zt(nzp), wr(nrp, 1.0), wz(nzp, 1.0);
  for (int i = 0; i <= nr; ++i) Rt[i] = i * scf_dR;
  for (int k = 0; k <= nz; ++k) Zt[k] = k * scf_dZ;
  wr[0] = wr[nr] = 0.5;            // trapezoid on the half plane
  wz[0] = wz[nz] = 0.5;

  const Real gexp = 2.0 * q_rot - 2.0, gden = 2.0 - 2.0 * q_rot;
  Real Hmax = 0.0, rho_c_scf = 0.0;

  for (scf_used = 1; scf_used <= scf_iter; ++scf_used) {
    // (a) rotation law: dH/dR = 0 at the density maximum, held at R = 1
    Real hR = scf_dR;
    Real dphidR = (PhiGas(1.0 + hR, 0.0) - PhiGas(1.0 - hR, 0.0)) / (2.0 * hR);
    l0sq_eff = std::pow(1.0 + eps_soft*eps_soft, -1.5) + dphidR;

    // (b) Bernoulli constant: the surface is tangent to z = z_target, so C is the
    //     largest the rest of the enthalpy reaches on that plane.
    C_eff = -1.0e30;
    for (int i = 1; i <= 4000; ++i) {
      Real R = scf_Rmax * i / 4000.0;
      Real a = 1.0 / std::sqrt(R*R + z_target*z_target + eps_soft*eps_soft)
               - PhiGas(R, z_target) - l0sq_eff * std::pow(R, gexp) / gden;
      C_eff = std::max(C_eff, a);
    }

    // (b) the enthalpy, and the shape of the density
    Hmax = 0.0;
    for (int k = 0; k <= nz; ++k) {
      for (int i = 0; i <= nr; ++i) {
        Real R = Rt[i], z = Zt[k];
        Real H = -1.0;
        if (R > 0.0) {
          H = 1.0 / std::sqrt(R*R + z*z + eps_soft*eps_soft) - PhiGas(R, z)
              - l0sq_eff * std::pow(R, gexp) / gden - C_eff;
        }
        rho_tab[k*nrp + i] = (H > 0.0) ? H : 0.0;
        Hmax = std::max(Hmax, H);
      }
    }
    if (Hmax <= 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in sg_torus_m1.cpp" << std::endl
          << "the self-consistent iteration produced no torus at all (H_max <= 0) on"
          << std::endl << "step " << scf_used << ". M_tor = " << M_tor
          << " is too large for this geometry." << std::endl;
      ATHENA_ERROR(msg);
    }

    // (c) normalise to the requested mass
    Real mhat = 0.0;
    for (int k = 0; k <= nz; ++k) {
      for (int i = 0; i <= nr; ++i) {
        Real h = rho_tab[k*nrp + i];
        if (h <= 0.0) continue;
        rho_tab[k*nrp + i] = std::pow(h / Hmax, n_poly);
        mhat += 2.0 * wr[i] * wz[k] * rho_tab[k*nrp + i]
                * 2.0 * PI * Rt[i] * scf_dR * scf_dZ;   // factor 2: the z < 0 half
      }
    }
    rho_c_scf = M_tor / mhat;
    for (int m = 0; m < nrp*nzp; ++m) rho_tab[m] *= rho_c_scf;

    // (d) the potential of that density. Only the occupied cells are sources.
    std::vector<int> si, sk;
    std::vector<Real> sdm;
    for (int k = 0; k <= nz; ++k) {
      for (int i = 0; i <= nr; ++i) {
        Real rr = rho_tab[k*nrp + i];
        if (rr <= 0.0 || Rt[i] <= 0.0) continue;
        si.push_back(i); sk.push_back(k);
        sdm.push_back(rr * wr[i] * wz[k] * 2.0 * PI * Rt[i] * scf_dR * scf_dZ);
      }
    }
    const int ns = static_cast<int>(si.size());
#pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k <= nz; ++k) {
      for (int i = 0; i <= nr; ++i) {
        Real R = Rt[i], z = Zt[k], acc = 0.0;
        for (int m = 0; m < ns; ++m) {
          int ip = si[m], kp = sk[m];
          Real Rp = Rt[ip], zp = Zt[kp];
          bool nr_up = (std::abs(i - ip) <= 1 && std::abs(k - kp) <= 1);
          bool nr_dn = (std::abs(i - ip) <= 1 && (k + kp) <= 1);
          acc += sdm[m] * (CellPotential(R, z, Rp,  zp, nr_up)
                         + CellPotential(R, z, Rp, -zp, nr_dn));
        }
        phi_new[k*nrp + i] = acc;
      }
    }

    // (e) under-relaxed update
    Real dmax = 0.0, pmax = 0.0;
    for (int m = 0; m < nrp*nzp; ++m) {
      Real pn = phi_new[m];
      dmax = std::max(dmax, std::fabs(pn - scf_phi[m]));
      pmax = std::max(pmax, std::fabs(pn));
      scf_phi[m] = (1.0 - scf_relax) * scf_phi[m] + scf_relax * pn;
    }
    scf_resid = (pmax > 0.0) ? dmax / pmax : 0.0;
    if (scf_resid < scf_tol) break;
  }
  if (scf_used > scf_iter) scf_used = scf_iter;

  // The mid-plane edges are an output now, not an input. Recompute them the same way
  // the analytic ones were, so the guard in TorusDensity and the report agree with the
  // solution rather than with the model it started from.
  {
    const int ns2 = 200000;
    const Real lo = 1.0e-3, hi = 1.0e3;
    bool found = false;
    Real a = 1.0, b = 1.0;
    for (int t = 0; t <= ns2; ++t) {
      Real R = lo * std::pow(hi / lo, static_cast<Real>(t) / ns2);
      Real H = 1.0 / std::sqrt(R*R + eps_soft*eps_soft) - PhiGas(R, 0.0)
               - l0sq_eff * std::pow(R, gexp) / gden - C_eff;
      if (H <= 0.0) continue;
      if (!found) { a = R; found = true; }
      b = R;
    }
    if (found) { r_in = a; r_out = b; }
  }
  *Hmax_out = Hmax;
  *rhoc_out = rho_c_scf;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  C_prime   = pin->GetReal("problem", "C_prime");
  q_rot     = pin->GetOrAddReal("problem", "q_rot", 0.0);
  M_tor     = pin->GetReal("problem", "M_tor");
  rho_c     = pin->GetReal("problem", "rho_c");
  GM_c      = pin->GetOrAddReal("problem", "GM_c", 1.0);
  eps_soft  = pin->GetOrAddReal("problem", "eps_soft", 0.25);
  gamma_gas = pin->GetReal("hydro", "gamma");

  rho_floor   = pin->GetOrAddReal("hydro", "dfloor", 1.0e-12);
  press_floor = pin->GetOrAddReal("hydro", "pfloor", 1.0e-14);

  alpha_visc   = pin->GetOrAddReal("problem", "alpha", 0.0);
  nu_iso       = pin->GetOrAddReal("problem", "nu_iso", 0.0);
  rho_atm      = pin->GetOrAddReal("problem", "rho_atm", 1.0e-9);
  Real t_atm   = pin->GetOrAddReal("problem", "t_atm_frac", 1.0);
  visc_rho_cut = pin->GetOrAddReal("problem", "visc_rho_cut", 10.0 * rho_atm);
  use_indirect = pin->GetOrAddInteger("problem", "indirect", 1) != 0;
  pert_amp     = pin->GetOrAddReal("problem", "pert_amp", 1.0e-3);
  pert_mode_amp = pin->GetOrAddReal("problem", "pert_mode_amp", 0.0);
  scf_iter      = pin->GetOrAddInteger("problem", "scf_iter", 0);
  scf_nr        = pin->GetOrAddInteger("problem", "scf_nr", scf_nr);
  scf_nz        = pin->GetOrAddInteger("problem", "scf_nz", scf_nz);

  bool want_self_grav = pin->GetOrAddInteger("problem", "self_grav", 1) != 0;

  if (C_prime <= 0.0 || C_prime >= 0.5) {
    std::stringstream msg;
    msg << "### FATAL ERROR in sg_torus_m1.cpp" << std::endl
        << "C_prime = " << C_prime << " must satisfy 0 < C' < 0.5 for a closed torus."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  n_poly = 1.0 / (gamma_gas - 1.0);
  Psi_c  = ShapeFunction(1.0, 0.0);

  if (Psi_c <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in sg_torus_m1.cpp" << std::endl
        << "No closed torus exists for C' = " << C_prime << " and q_rot = " << q_rot
        << ":" << std::endl
        << "Psi_c = " << Psi_c << " must be positive; at eps_soft -> 0 that needs"
        << std::endl
        << "q < 1 - 1/(2(1-C')) = " << 1.0 - 1.0 / (2.0 * (1.0 - C_prime)) << "."
        << std::endl
        << "A pressure-supported torus cannot be both thick and Keplerian." << std::endl;
    ATHENA_ERROR(msg);
  }
  p_c = rho_c * Psi_c / (n_poly + 1.0);

  // Mid-plane edges. Closed form only at q = 0, so they are bracketed numerically and
  // used for reporting and for the domain check.
  {
    const int ns = 200000;
    const Real lo = 1.0e-3, hi = 1.0e3;
    bool found = false;
    for (int s = 0; s <= ns; ++s) {
      Real R = lo * std::pow(hi / lo, static_cast<Real>(s) / ns);
      if (ShapeFunction(R, 0.0) <= 0.0) continue;
      if (!found) { r_in = R; found = true; }
      r_out = R;
    }
    if (!found) {   // Psi_c > 0 guarantees R = 1 is inside, so this cannot happen
      r_in = r_out = 1.0;
    }
  }

  // The self-consistent equilibrium, if asked for. It replaces the normalisation of the
  // shape function, the rotation law and the central density; the geometry it was handed
  // (r_in, r_out, and everything the generator derived from them) is held fixed.
  Real Psi_c_an = Psi_c, rho_c_an = rho_c;
  Real l0sq_an = std::pow(1.0 + eps_soft*eps_soft, -1.5);
  if (scf_iter > 0) {
    Real Hmax = 0.0, rhoc_new = 0.0;
    SolveSelfConsistentEquilibrium(&Hmax, &rhoc_new);
    Psi_c = Hmax;
    rho_c = rhoc_new;
    p_c   = rho_c * Psi_c / (n_poly + 1.0);
  }

  p_atm    = rho_atm * t_atm * Psi_c / (n_poly + 1.0);
  rho_body = 10.0 * rho_atm;

  // Self-gravity is a configure-time switch, so a mismatch between the binary and the
  // input file would silently run the wrong experiment - the exact confusion the
  // no-self-gravity control exists to avoid.
  if (want_self_grav && !SELF_GRAVITY_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in sg_torus_m1.cpp" << std::endl
        << "<problem>/self_grav = 1 but this binary was configured without a gravity"
        << std::endl
        << "solver. Reconfigure with --grav mg, or set self_grav = 0 to run the"
        << std::endl
        << "no-self-gravity control of Bannikova et al. (2026), Appendix C." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (!want_self_grav && SELF_GRAVITY_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in sg_torus_m1.cpp" << std::endl
        << "<problem>/self_grav = 0 but this binary has a gravity solver compiled in,"
        << std::endl
        << "and Athena++ cannot switch it off at run time. Use a binary configured"
        << std::endl
        << "without --grav for the control run." << std::endl;
    ATHENA_ERROR(msg);
  }

  // Athena++ gates ViscousFluxIso on nu_iso, so alpha alone would run inviscid.
  if (alpha_visc > 0.0 && nu_iso <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in sg_torus_m1.cpp" << std::endl
        << "alpha = " << alpha_visc << " > 0 requires <problem>/nu_iso > 0 as well;"
        << std::endl
        << "the enrolled coefficient is computed but never applied otherwise."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  if (alpha_visc <= 0.0 && nu_iso > 0.0 && Globals::my_rank == 0) {
    std::cout << std::endl
              << "  *** WARNING: alpha = 0 but nu_iso = " << nu_iso << " > 0."
              << std::endl
              << "      This is NOT the collisionless analogue: a constant kinematic"
              << std::endl
              << "      viscosity will be applied everywhere." << std::endl << std::endl;
  }

  // Isolated gravity boundaries need the mass well inside the box.
  if (Globals::my_rank == 0
      && (mesh_size.x1max <= r_out || mesh_size.x2max <= r_out
          || mesh_size.x3max <= r_out)) {
    std::cout << std::endl
              << "  *** WARNING: the box does not enclose the torus (r_out = "
              << r_out << ")." << std::endl
              << "      The multipole boundary values assume the mass is interior."
              << std::endl << std::endl;
  }

#if SELF_GRAVITY_ENABLED
  SetFourPiG(4.0 * PI);   // G = 1, matching the paper's N-body units
#endif

  EnrollUserExplicitSourceFunction(TorusGravity);
  if (alpha_visc > 0.0) EnrollViscosityCoefficient(TorusViscosity);

  // All six faces are diodes. Stock outflow injects mass in a gravitating problem;
  // see the boundary functions at the end of this file for what that cost once.
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiodeInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiodeOuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiodeInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiodeOuterX2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiodeInnerX3);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiodeOuterX3);

  // The mode coefficients are volume sums, so the history file records them per cycle
  // and no snapshot post-processing is needed for the analogues of their Figs. 7 and 8.
  //   0      torus mass
  //   1-3    mass-weighted position (divide by the mass for the barycentre r_tb)
  //   4-13   a_m, b_m for m = 1..5
  //   14-16  kinetic, internal, central-potential energy
  //   17     self-gravitational energy (zero without a gravity solver)
  //   18-19  cumulative floor activations; in a healthy run these stay at zero
  //   20     the largest |v| anywhere, the earliest warning of a vacuum cell
  AllocateUserHistoryOutput(21);
  const char *names[21] = {"tor_mass", "tor_mx", "tor_my", "tor_mz",
                           "a1", "b1", "a2", "b2", "a3", "b3",
                           "a4", "b4", "a5", "b5",
                           "E_kin", "E_int", "E_grav_c", "E_grav_self",
                           "n_dfloor", "n_pfloor", "v_max"};
  for (int i = 0; i < 21; ++i)
    EnrollUserHistoryOutput(i, TorusHistory, names[i],
                            i == 20 ? UserHistoryOperation::max
                                    : UserHistoryOperation::sum);

  if (Globals::my_rank == 0) {
    std::cout << std::endl
      << "===========================================================" << std::endl
      << "  Self-gravitating Papaloizou-Pringle torus (G = M_c = R_tor = 1)" << std::endl
      << "===========================================================" << std::endl
      << "  Thickness parameter (C'):       " << C_prime << std::endl
      << "  Torus mass (M_tor / M_c):       " << M_tor << std::endl
      << "  Central density (rho_c):        " << rho_c << std::endl
      << "  Adiabatic index (gamma):        " << gamma_gas << std::endl
      << "  Mid-plane edges r_in / r_out:   " << r_in << " / " << r_out << std::endl
      << "  Softening (eps_soft):           " << eps_soft << std::endl
      << "  Self-gravity:                   "
      << (SELF_GRAVITY_ENABLED ? "ON (Multigrid)" : "OFF (control run)") << std::endl
      << "  Indirect term:                  " << (use_indirect ? "ON" : "OFF")
      << std::endl
      << "  Self-consistent IC (scf_iter):  "
      << (scf_iter > 0 ? "ON" : "OFF (central mass only)") << std::endl
      << "  White-noise seed (pert_amp):    " << pert_amp << std::endl
      << "  Coherent low-m seed:            " << pert_mode_amp << std::endl
      << "  Alpha viscosity:                " << alpha_visc << std::endl
      << "  Ambient density / pressure:     " << rho_atm << " / " << p_atm << std::endl
      << "  Orbital period at R_tor:        " << 2.0 * PI << std::endl
      << "===========================================================" << std::endl
      << std::endl;
    if (scf_iter > 0) {
      std::cout
        << "  --- self-consistent equilibrium ---------------------------" << std::endl
        << "  iterations / residual:          " << scf_used << " / " << scf_resid
        << "   (limit " << scf_iter << ", tol " << scf_tol << ")" << std::endl
        << "  table (nR x nz), R_max, z_max:  " << scf_nr << " x " << scf_nz
        << ", " << scf_Rmax << ", " << scf_Zmax << std::endl
        << "  l_0^2   analytic -> SCF:        " << l0sq_an << " -> " << l0sq_eff
        << "   (" << 100.0*(l0sq_eff/l0sq_an - 1.0) << " %)" << std::endl
        << "  C       analytic -> SCF:        " << C_prime << " -> " << C_eff
        << std::endl
        << "  Psi_max analytic -> SCF:        " << Psi_c_an << " -> " << Psi_c
        << "   (" << 100.0*(Psi_c/Psi_c_an - 1.0) << " %)" << std::endl
        << "  rho_c   analytic -> SCF:        " << rho_c_an << " -> " << rho_c
        << "   (" << 100.0*(rho_c/rho_c_an - 1.0) << " %)" << std::endl
        << "  Phi_gas at (R_tor, 0):          " << PhiGas(1.0, 0.0)
        << "   against Phi_c = "
        << -GM_c/std::sqrt(1.0 + eps_soft*eps_soft) << std::endl
        << "  rho_atm / rho_c:                " << rho_atm/rho_c << std::endl;
      // What the iteration did to the shape. r_in and r_out are pinned by construction;
      // the half-thickness and the position of the density maximum are not, and the
      // comparison with the paper is made through the thickness (their i_max), so both
      // are reported rather than assumed.
      Real zmax_scf = 0.0, Rpk = 1.0, Hpk = -1.0;
      for (int i = 1; i < 2000; ++i) {
        Real R = r_in + (r_out - r_in) * i / 2000.0;
        Real H0 = ShapeFunction(R, 0.0);
        if (H0 > Hpk) { Hpk = H0; Rpk = R; }
        for (int k = 0; k < 2000; ++k) {
          Real z = r_out * k / 2000.0;
          if (ShapeFunction(R, z) > 0.0) zmax_scf = std::max(zmax_scf, z);
        }
      }
      std::cout
        << "  half-thickness  target -> SCF:  " << scf_zt << " -> " << zmax_scf
        << "   (" << 100.0*(zmax_scf/scf_zt - 1.0) << " %)" << std::endl
        << "  density maximum at R:           " << Rpk
        << "   (held at 1 by construction)" << std::endl
        << "  r_in / r_out  analytic -> SCF:  " << scf_rin_an << " / " << scf_rout_an
        << "  ->  " << r_in << " / " << r_out << std::endl
        << "===========================================================" << std::endl
        << std::endl;
    }
  }
  return;
}

//! Two counters per block: density-floor and pressure-floor activations. They are
//! cumulative, so the history column is a running total and any nonzero value at all
//! means the floors are doing physics rather than standing by.
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(2);
  ruser_meshblock_data[0](0) = 0.0;
  ruser_meshblock_data[0](1) = 0.0;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; ++k) {
    Real z = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x = pcoord->x1v(i);
        Real R = std::sqrt(x*x + y*y);

        Real rho_t = TorusDensity(R, z);
        Real p_t   = TorusPressure(R, z);
        // The seed rides on the torus only; perturbing the ambient would do nothing
        // but add noise to the vacuum. Pressure is left alone, so the perturbation is
        // an entropy one and does not disturb the pressure support.
        if (rho_t > 0.0) {
          Real seed = 0.0;
          if (pert_amp > 0.0)      seed += pert_amp * CellNoise(x, y, z);
          if (pert_mode_amp > 0.0) seed += pert_mode_amp * ModeNoise(x, y);
          rho_t *= 1.0 + seed;
        }
        Real rho   = rho_t + rho_atm;
        Real press = p_t + p_atm;

        // l = const inside the torus; the ambient is left at rest, and what little of
        // it drains into the softened centre is far below the central mass.
        Real vx = 0.0, vy = 0.0;
        if (rho_t > 0.0) {
          // l = l_0 R^q, with l_0 set by the softened potential so that the pressure
          // maximum sits at R = R_tor = 1.
          Real l0 = scf_on ? std::sqrt(l0sq_eff)
                           : std::pow(1.0 + eps_soft*eps_soft, -0.75);
          Real v_phi = l0 * std::pow(R, q_rot - 1.0);
          vx = -v_phi * y / R;
          vy =  v_phi * x / R;
        }

        phydro->w(IDN,k,j,i) = rho;
        phydro->w(IVX,k,j,i) = vx;
        phydro->w(IVY,k,j,i) = vy;
        phydro->w(IVZ,k,j,i) = 0.0;
        phydro->w(IPR,k,j,i) = press;

        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = rho * vx;
        phydro->u(IM2,k,j,i) = rho * vy;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = press / (gamma_gas - 1.0)
                               + 0.5 * rho * (vx*vx + vy*vy);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief The central point mass and the indirect term.
//!
//! The point-mass part uses the same conservative discretisation as Athena++'s own
//! self-gravity source term - the momentum from the centred potential difference, the
//! energy from the numerical mass flux - so that the two gravities are treated alike
//! and their sum does not carry a spurious relative discretisation error.
void TorusGravity(MeshBlock *pmb, const Real time, const Real dt,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                  AthenaArray<Real> &cons_scalar) {
  AthenaArray<Real> &x1flux = pmb->phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux = pmb->phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux = pmb->phydro->flux[X3DIR];

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real z = pmb->pcoord->x3v(k);
    Real dx3 = pmb->pcoord->dx3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      Real dx2 = pmb->pcoord->dx2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real dx1 = pmb->pcoord->dx1v(i);
        Real rho = prim(IDN,k,j,i);

        Real phi = CentralPotential(x, y, z);

        // x-direction
        Real hdtodx1 = 0.5 * dt / dx1;
        Real dpl = -(phi - CentralPotential(x - dx1, y, z));
        Real dpr = -(CentralPotential(x + dx1, y, z) - phi);
        cons(IM1,k,j,i) += hdtodx1 * rho * (dpl + dpr);
        cons(IEN,k,j,i) += hdtodx1 * (x1flux(IDN,k,j,i)   * dpl
                                    + x1flux(IDN,k,j,i+1) * dpr);

        // y-direction
        Real hdtodx2 = 0.5 * dt / dx2;
        dpl = -(phi - CentralPotential(x, y - dx2, z));
        dpr = -(CentralPotential(x, y + dx2, z) - phi);
        cons(IM2,k,j,i) += hdtodx2 * rho * (dpl + dpr);
        cons(IEN,k,j,i) += hdtodx2 * (x2flux(IDN,k,j,i)   * dpl
                                    + x2flux(IDN,k,j+1,i) * dpr);

        // z-direction
        Real hdtodx3 = 0.5 * dt / dx3;
        dpl = -(phi - CentralPotential(x, y, z - dx3));
        dpr = -(CentralPotential(x, y, z + dx3) - phi);
        cons(IM3,k,j,i) += hdtodx3 * rho * (dpl + dpr);
        cons(IEN,k,j,i) += hdtodx3 * (x3flux(IDN,k,j,i)   * dpl
                                    + x3flux(IDN,k+1,j,i) * dpr);

        // Indirect term: a uniform acceleration, so the work follows the cell momentum.
        if (use_indirect) {
          Real m1 = cons(IM1,k,j,i), m2 = cons(IM2,k,j,i), m3 = cons(IM3,k,j,i);
          cons(IEN,k,j,i) -= dt * (m1 * a_indirect[0] + m2 * a_indirect[1]
                                 + m3 * a_indirect[2]);
          cons(IM1,k,j,i) -= dt * rho * a_indirect[0];
          cons(IM2,k,j,i) -= dt * rho * a_indirect[1];
          cons(IM3,k,j,i) -= dt * rho * a_indirect[2];
        }

        // Vacuum guard. A cell at the torus surface can be swept down to the density
        // floor while keeping its momentum; the EOS then floors rho but not rho*v, and
        // the velocity it reports is m/rho_floor. Measured on the first attempt at this
        // problem: rho hit dfloor exactly and |v| reached 1.9e5, collapsing the timestep
        // by six orders of magnitude in a single step. Flooring the three quantities
        // together is the fix; counting the activations is what keeps it a safety net
        // rather than a silent background state.
        Real &d = cons(IDN,k,j,i);
        if (!std::isfinite(d) || d < rho_floor) {
          d               = rho_floor;
          cons(IM1,k,j,i) = 0.0;
          cons(IM2,k,j,i) = 0.0;
          cons(IM3,k,j,i) = 0.0;
          cons(IEN,k,j,i) = press_floor / (gamma_gas - 1.0);
          pmb->ruser_meshblock_data[0](0) += 1.0;
        } else {
          Real e_k = 0.5 * (SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i))
                          + SQR(cons(IM3,k,j,i))) / d;
          Real e_int = cons(IEN,k,j,i) - e_k;
          if (!std::isfinite(e_int) || e_int < press_floor / (gamma_gas - 1.0)) {
            cons(IEN,k,j,i) = press_floor / (gamma_gas - 1.0) + e_k;
            pmb->ruser_meshblock_data[0](1) += 1.0;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief The gas pull on the central mass, by direct summation over the gas.
//!
//! Direct summation rather than a gradient of the Multigrid potential at the origin:
//! the origin sits on a cell corner of a symmetric even-sized mesh, so any sampled
//! gradient there would be an interpolation, and the softened kernel used here is
//! exactly the one the gas feels from the point mass, keeping the pair reciprocal.
void Mesh::UserWorkInLoop() {
  if (!use_indirect) return;

  Real acc[3] = {0.0, 0.0, 0.0};
  for (int b = 0; b < nblocal; ++b) {
    MeshBlock *pmb = my_blocks(b);
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real z = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; ++j) {
        Real y = pmb->pcoord->x2v(j);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real x = pmb->pcoord->x1v(i);
          Real rho = pmb->phydro->w(IDN,k,j,i);
          if (rho <= rho_body) continue;   // the ambient must not steer the centre
          Real dm = rho * pmb->pcoord->GetCellVolume(k,j,i);
          Real r2 = x*x + y*y + z*z + eps_soft*eps_soft;
          Real inv = 1.0 / (r2 * std::sqrt(r2));
          acc[0] += dm * x * inv;
          acc[1] += dm * y * inv;
          acc[2] += dm * z * inv;
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, acc, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int d = 0; d < 3; ++d) a_indirect[d] = acc[d];
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Alpha viscosity nu = alpha c_s^2 / Omega_K, tapered off in the ambient.
//!        Off in the runs that stand in for the collisionless system; on in the one run
//!        that deliberately departs from it.
void TorusViscosity(HydroDiffusion *phdif, MeshBlock *pmb,
                    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                    int is, int ie, int js, int je, int ks, int ke) {
  Real cut2 = visc_rho_cut * visc_rho_cut;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real R = std::max(std::sqrt(x*x + y*y), eps_soft);
        Real rho = prim(IDN,k,j,i);
        Real press = prim(IPR,k,j,i);

        Real w = (cut2 > 0.0) ? (rho*rho / (rho*rho + cut2)) : 1.0;
        // Omega_K = sqrt(GM/R^3), so nu = alpha gamma (p/rho) R^{3/2} / sqrt(GM)
        Real nu = alpha_visc * gamma_gas * (press / rho)
                  * std::pow(R, 1.5) / std::sqrt(GM_c) * w;
        phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) = nu;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief The torus diagnostics: mass, barycentre, azimuthal Fourier coefficients and
//!        the energies that make up the virial quantity 2 E_kin + E_pot.
//!
//! Only cells above rho_body contribute, so the ambient and whatever drains into the
//! centre cannot masquerade as torus structure. The Fourier coefficients are
//! mass-weighted volume sums, a_m = sum rho dV cos(m phi); normalised by the mass they
//! are directly comparable with the paper's A_m ratios, k = A_2/A_1 and Q_3, which are
//! themselves normalisation-independent.
//! All eighteen come from one sweep, cached per block and per time: called once per
//! column they would otherwise cost eighteen passes over the mesh at every history
//! dump, which at this cadence is more arithmetic than the hydrodynamics itself.
Real TorusHistory(MeshBlock *pmb, int iout) {
  static thread_local int cached_gid = -1;
  static thread_local Real cached_time = -1.0;
  static thread_local Real v[21];

  Real now = pmb->pmy_mesh->time;
  if (pmb->gid != cached_gid || now != cached_time) {
    for (int n = 0; n < 21; ++n) v[n] = 0.0;
    v[18] = pmb->ruser_meshblock_data[0](0);
    v[19] = pmb->ruser_meshblock_data[0](1);

    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real z = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; ++j) {
        Real y = pmb->pcoord->x2v(j);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real rho = pmb->phydro->w(IDN,k,j,i);

          // v_max is deliberately taken over every cell, not just the torus body:
          // the vacuum cells outside it are exactly where a runaway shows up first.
          Real sp = std::sqrt(SQR(pmb->phydro->w(IVX,k,j,i))
                            + SQR(pmb->phydro->w(IVY,k,j,i))
                            + SQR(pmb->phydro->w(IVZ,k,j,i)));
          if (sp > v[20]) v[20] = sp;

          if (rho <= rho_body) continue;
          Real x = pmb->pcoord->x1v(i);

          Real dV = pmb->pcoord->GetCellVolume(k,j,i);
          Real dm = rho * dV;
          Real phi = std::atan2(y, x);

          v[0] += dm;
          v[1] += dm * x;
          v[2] += dm * y;
          v[3] += dm * z;
          for (int m = 1; m <= 5; ++m) {
            v[2*m + 2] += dm * std::cos(m * phi);
            v[2*m + 3] += dm * std::sin(m * phi);
          }
          Real vx = pmb->phydro->w(IVX,k,j,i);
          Real vy = pmb->phydro->w(IVY,k,j,i);
          Real vz = pmb->phydro->w(IVZ,k,j,i);
          v[14] += 0.5 * dm * (vx*vx + vy*vy + vz*vz);
          v[15] += pmb->phydro->w(IPR,k,j,i) * dV / (gamma_gas - 1.0);
          v[16] += dm * CentralPotential(x, y, z);
#if SELF_GRAVITY_ENABLED
          v[17] += 0.5 * dm * pmb->pgrav->phi(k,j,i);
#endif
        }
      }
    }
    cached_gid = pmb->gid;
    cached_time = now;
  }
  return v[iout];
}

//----------------------------------------------------------------------------------------
//! \brief Diode outflow on the six Cartesian faces: gas may leave, nothing may enter.
//!
//! Stock "outflow" is zero-gradient in every variable, the normal velocity included,
//! which in a gravitating problem is not an outflow condition at all. Gravity pulls
//! inward at the boundary, the ghost zones copy the interior state along with that
//! inward velocity, and the face becomes an infinite reservoir: mass enters, the
//! density near the face rises, and the inflow feeds itself.
//!
//! This is not hypothetical. The first canonical run used stock outflow and the total
//! gas mass grew from 0.1005 to 5.63 - a factor of 56 - over 65 orbits, with the
//! ambient rising four orders of magnitude until the entire box sat at the torus's
//! original central density. A cube has four faces around the rotation axis, so the
//! injected material also stamped a spurious m = 4 pattern on the disc that grew to
//! dominate every physical harmonic. The 2D pgen has guarded against this from the
//! start; the guard had simply never been carried over to Cartesian geometry.
//!
//! Clamping the normal velocity is what makes it a diode. Density and pressure keep the
//! zero-gradient copy, so genuine outflow still leaves smoothly.

void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,il-i) = prim(n,k,j,il);
        }
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IVX,k,j,il-i) = std::min(prim(IVX,k,j,il), static_cast<Real>(0.0));
      }
    }
  }
  return;
}

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,iu+i) = prim(n,k,j,iu);
        }
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IVX,k,j,iu+i) = std::max(prim(IVX,k,j,iu), static_cast<Real>(0.0));
      }
    }
  }
  return;
}

void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          prim(n,k,jl-j,i) = prim(n,k,jl,i);
        }
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IVY,k,jl-j,i) = std::min(prim(IVY,k,jl,i), static_cast<Real>(0.0));
      }
    }
  }
  return;
}

void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          prim(n,k,ju+j,i) = prim(n,k,ju,i);
        }
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IVY,k,ju+j,i) = std::max(prim(IVY,k,ju,i), static_cast<Real>(0.0));
      }
    }
  }
  return;
}

void DiodeInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          prim(n,kl-k,j,i) = prim(n,kl,j,i);
        }
      }
    }
  }
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IVZ,kl-k,j,i) = std::min(prim(IVZ,kl,j,i), static_cast<Real>(0.0));
      }
    }
  }
  return;
}

void DiodeOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          prim(n,ku+k,j,i) = prim(n,ku,j,i);
        }
      }
    }
  }
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        prim(IVZ,ku+k,j,i) = std::max(prim(IVZ,ku,j,i), static_cast<Real>(0.0));
      }
    }
  }
  return;
}
