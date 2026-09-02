#!/usr/bin/env bash
# The self-gravitating torus campaign: four runs, one after another.
#
#   CAMPAIGN=sg_torus_m1 ./sg_campaign.sh [run ...]     # default: canonical nosg
#
# Sequential on purpose. Multigrid is ~90% of the runtime and the machine is memory
# bound, so two concurrent runs finish later than two consecutive ones.
#
# Each run reproduces one claim of Bannikova et al. (2026):
#   canonical  the spontaneous m = 1 mode and its geometric harmonic spectrum
#   thin       larger C', testing their thickness requirement (Sect. 5)
#   nosg       self-gravity off, their Appendix C control - note that a gas torus
#              is Papaloizou-Pringle unstable on its own, so unlike the
#              collisionless control this one is EXPECTED to grow an m = 1 mode;
#              the pattern speed is what separates the two mechanisms
#   viscous    alpha > 0, which has no collisionless counterpart at all
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# CAMPAIGN selects the results subdirectory: ppi for the phase-1 constant-l
# runs, sg_torus_m1 for the near-Keplerian reproduction of the paper.
results="${repo}/results/${CAMPAIGN:-sg_torus_m1}"
threads="${OMP_NUM_THREADS:-15}"

runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(canonical nosg)

for run in "${runs[@]}"; do
  dir="${results}/${run}"
  # nosg needs the binary built WITHOUT --grav: Athena++ cannot switch the solver
  # off at run time, and the pgen refuses the mismatch rather than run the wrong
  # experiment silently.
  bin="${repo}/bin/athena_sg"
  # any run whose name carries "nosg" is a control and needs the solverless binary
  case "${run}" in *nosg*) bin="${repo}/bin/athena_nosg" ;; esac

  if [ ! -f "${dir}/athinput" ]; then
    echo "no input for '${run}'; generate it with scripts/torus/theory/torus_sg_model.py" >&2
    exit 1
  fi
  if [ -f "${dir}/done" ]; then
    echo "=== ${run}: already finished, skipping"
    continue
  fi

  echo "=== ${run}: starting $(date '+%F %T')"
  python3 "${repo}/scripts/torus/run/describe_run.py" "${dir}" >/dev/null 2>&1 || true
  # Raw output goes to <run>/data/ so that the run directory itself stays readable:
  # input, description and figures, not several hundred snapshots.
  mkdir -p "${dir}/data"
  # A failed run must not cost the runs queued behind it, for the same reason a failed
  # figure must not: the queue is the night's whole output and nobody is awake to restart
  # it. The failure is reported and the directory is left without its "done" marker.
  if ( cd "${dir}/data" && OMP_NUM_THREADS="${threads}" "${bin}" -i ../athinput \
        >> ../run.log 2>&1 ); then
    touch "${dir}/done"
  else
    echo "=== ${run}: FAILED $(date '+%F %T'), see ${dir}/run.log; queue continues" >&2
    continue
  fi
  echo "=== ${run}: finished $(date '+%F %T')"

  # Post-processing runs here rather than by hand afterwards. Every finished run then
  # carries its own numbers, figures and animation, and nothing has to be remembered
  # or re-invoked later. Failures are reported but never abort the campaign: a broken
  # plot must not cost the queued runs behind it.
  echo "--- ${run}: post-processing"
  ( cd "${repo}" && python3 scripts/torus/analysis/torus_modes.py "${dir}" --last 0.3 \
      >> "${dir}/analysis.log" 2>&1 ) || echo "    analysis failed, see analysis.log" >&2
  ( cd "${repo}" && python3 scripts/torus/analysis/torus_figures.py "${dir}" \
      -o "${dir}/figs" --multiview >> "${dir}/analysis.log" 2>&1 ) \
      || echo "    figures failed, see analysis.log" >&2
  python3 "${repo}/scripts/torus/run/describe_run.py" "${dir}" >/dev/null \
      || echo "    description failed" >&2
  echo "--- ${run}: done, see ${dir}/README.md"
done
