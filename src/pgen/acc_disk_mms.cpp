//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file acc_disk_mms.cpp
//! \brief Method of Manufactured Solutions for the 2D cylindrical Euler equations.
//!
//! Every other verification problem in this repository compares against a solution the
//! physics happens to admit, so it can only exercise the states that solution reaches.
//! Here the solution is chosen first -- deliberately awkward, varying in both r and phi,
//! with all four conserved variables non-trivial -- and a source term is derived that
//! makes it exact anyway. That is the only way to reach code paths a physical solution
//! never visits.
//!
//! The manufactured solution is STEADY, so u_m is an exact steady state of the modified
//! equations: start there and the run should stay there, and whatever separates it from
//! u_m afterwards is the discretisation error and nothing else. It also means the
//! boundary values do not depend on time.
//!
//! The source term in mms_source_generated.hpp is emitted by
//!     vv mms --config-path vvkit/m5_mms/vvcase_mms_euler.yaml --language cpp
//! from the same operators and the same manufactured solution the study measures
//! against. Regenerate it whenever either changes; nothing here re-derives it, and a
//! stale header would be measuring one problem against another's answer.
//!
//! A mismatch between the operators declared there and what Athena++ actually advances
//! does not hide: u_m stops being an exact solution of the modified system, the run
//! converges to something else, and the observed order collapses towards zero.

// C headers

// C++ headers
#include <cmath>      // sin(), cos()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

// The generated source term. Emitted by `vv mms`; do not edit by hand.
#include "mms_source_generated.hpp"

namespace {
Real gamma_gas;

// The manufactured solution. These MUST match mms.solution in the vvcase file exactly:
// the source term was derived from those expressions, and the study measures against
// them. Any divergence here silently turns the study into a comparison between two
// different problems.
Real MMSDensity(Real x, Real y)  { return 2.0 + 0.5 * std::sin(x) * std::cos(y); }
Real MMSPressure(Real x, Real y) { return 3.0 + 0.4 * std::cos(x) * std::cos(y); }
Real MMSVel1(Real x, Real y)     { return 0.2 * std::cos(x) * std::sin(y); }
Real MMSVel2(Real x, Real y)     { return 1.0 + 0.3 * std::sin(x) * std::sin(y); }

void MMSPrimitive(Real x, Real y, Real *rho, Real *press, Real *v1, Real *v2) {
  *rho   = MMSDensity(x, y);
  *press = MMSPressure(x, y);
  *v1    = MMSVel1(x, y);
  *v2    = MMSVel2(x, y);
}
}  // namespace

void MMSSourceTerm(MeshBlock *pmb, const Real time, const Real dt,
                   const AthenaArray<Real> &prim,
                   const AthenaArray<Real> &prim_scalar,
                   const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                   AthenaArray<Real> &cons_scalar);

void MMSInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void MMSOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//----------------------------------------------------------------------------------------
void Mesh::InitUserMeshData(ParameterInput *pin) {
  gamma_gas = pin->GetReal("hydro", "gamma");

  // The source term was derived with a specific gamma baked into it. Running with a
  // different one leaves u_m no longer a solution, and the study would report a
  // convergence failure whose cause is in the input file rather than in the code.
  const Real gamma_expected = 5.0 / 3.0;
  if (std::abs(gamma_gas - gamma_expected) > 1.0e-12) {
    std::stringstream msg;
    msg << "### FATAL ERROR in acc_disk_mms.cpp" << std::endl
        << "gamma = " << gamma_gas << ", but the generated source term was derived for "
        << gamma_expected << "." << std::endl
        << "Regenerate mms_source_generated.hpp with a matching mms.symbols.g, or set "
        << "hydro/gamma to " << gamma_expected << "." << std::endl;
    ATHENA_ERROR(msg);
  }

  EnrollUserExplicitSourceFunction(MMSSourceTerm);

  // Dirichlet boundaries holding the exact solution. An MMS measures the interior
  // scheme, so the boundary must not contribute an error of its own: extrapolating
  // there would put a first-order boundary term into a second-order measurement.
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, MMSInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, MMSOuterX1);
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Start the run exactly on the manufactured solution.
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x = pcoord->x1v(i);
        Real rho, press, v1, v2;
        MMSPrimitive(x, y, &rho, &press, &v1, &v2);

        phydro->w(IDN,k,j,i) = rho;
        phydro->w(IVX,k,j,i) = v1;
        phydro->w(IVY,k,j,i) = v2;
        phydro->w(IVZ,k,j,i) = 0.0;
        phydro->w(IPR,k,j,i) = press;

        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IM1,k,j,i) = rho * v1;
        phydro->u(IM2,k,j,i) = rho * v2;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = press / (gamma_gas - 1.0)
                               + 0.5 * rho * (v1 * v1 + v2 * v2);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Add dt * S to the conserved variables, S being the divergence of the exact
//!        flux evaluated on the manufactured solution.
//!
//! The r-momentum entry already carries -(rho v_phi^2 + p)/r. Athena++ applies the
//! matching +(rho v_phi^2 + p)/r geometric source internally, and the two are meant to
//! cancel at u_m; dropping either one leaves a term of the same size as the solution.
void MMSSourceTerm(MeshBlock *pmb, const Real time, const Real dt,
                   const AthenaArray<Real> &prim,
                   const AthenaArray<Real> &prim_scalar,
                   const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                   AthenaArray<Real> &cons_scalar) {
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        cons(IDN,k,j,i) += dt * mms_source_dens(x, y);
        cons(IM1,k,j,i) += dt * mms_source_mom1(x, y);
        cons(IM2,k,j,i) += dt * mms_source_mom2(x, y);
        cons(IEN,k,j,i) += dt * mms_source_etot(x, y);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Ghost zones held at the exact solution.
namespace {
void FillGhostFromExact(Coordinates *pco, AthenaArray<Real> &prim,
                        int i_first, int i_last, int jl, int ju, int kl, int ku) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      Real y = pco->x2v(j);
      for (int i=i_first; i<=i_last; ++i) {
        Real x = pco->x1v(i);
        Real rho, press, v1, v2;
        MMSPrimitive(x, y, &rho, &press, &v1, &v2);
        prim(IDN,k,j,i) = rho;
        prim(IVX,k,j,i) = v1;
        prim(IVY,k,j,i) = v2;
        prim(IVZ,k,j,i) = 0.0;
        prim(IPR,k,j,i) = press;
      }
    }
  }
}
}  // namespace

void MMSInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  FillGhostFromExact(pco, prim, il-ngh, il-1, jl, ju, kl, ku);
  return;
}

void MMSOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  FillGhostFromExact(pco, prim, iu+1, iu+ngh, jl, ju, kl, ku);
  return;
}
