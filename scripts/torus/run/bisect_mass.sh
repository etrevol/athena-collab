#!/usr/bin/env bash
# Bracket the torus mass at which the self-consistent equilibrium stops surviving its own
# m = 1 mode. mu = 0.10 keeps 92% of its mass, mu = 0.30 keeps 33%; the transition is
# between, and the cheapest way to find it is bisection.
#
# The decision after each run is made HERE, by the machine, from the run's own history
# file - not by a person reading a notification. That rule is in docs/WORKFLOW.md and it
# was written after a notification that never arrived cost this project a whole night.
set -uo pipefail
cd /home/etrevol/athena-torus
export CAMPAIGN=scfeq
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-14}"
LATEST_START="${LATEST_START:-05:30}"   # nothing new begins after this

# Mass fraction remaining at the end of a run. Above SURVIVE it held together.
SURVIVE=0.70
frac() {
  python3 - "$1" <<'PY'
import sys, pathlib
import numpy as np
sys.path.insert(0, "scripts/torus/analysis")
from torus_modes import read_hst
h = read_hst(str(next(pathlib.Path(sys.argv[1]).glob("data/*.hst"))))
m = np.asarray(h["mass"])
print(f"{m[-1]/m[0]:.4f}")
PY
}

run() {
  local name="$1"
  if [ ! -f "results/scfeq/${name}/athinput" ]; then
    echo "$(date '+%F %T') no input for ${name}"; return 1
  fi
  if [ "$(date +%s)" -ge "$(date -d "${LATEST_START}" +%s)" ]; then
    echo "$(date '+%F %T') past ${LATEST_START}, not starting ${name}"; return 1
  fi
  echo "$(date '+%F %T') starting ${name}"
  ./scripts/torus/run/sg_campaign.sh "${name}" || true
  [ -f "results/scfeq/${name}/done" ] || { echo "${name} did not finish"; return 1; }
  return 0
}

lo=0.10; hi=0.30            # known bracket: 0.10 survives, 0.30 does not
for step in 1 2 3; do
  case "${step}" in
    1) name=mu020 ; mu=0.20 ;;
    2) name="${NEXT_NAME:-}" ; mu="${NEXT_MU:-}" ;;
    3) name="${NEXT_NAME:-}" ; mu="${NEXT_MU:-}" ;;
  esac
  [ -z "${name}" ] && { echo "bracket is [${lo}, ${hi}]; nothing further to run"; break; }
  run "${name}" || break
  f="$(frac "results/scfeq/${name}")"
  if awk "BEGIN{exit !(${f} > ${SURVIVE})}"; then
    echo "$(date '+%F %T') ${name}: mass ${f} -> SURVIVED; bracket becomes [${mu}, ${hi}]"
    lo="${mu}"
  else
    echo "$(date '+%F %T') ${name}: mass ${f} -> DESTROYED; bracket becomes [${lo}, ${mu}]"
    hi="${mu}"
  fi
  # next midpoint, on the 0.05 grid the inputs were generated on
  nxt="$(awk "BEGIN{printf \"%.2f\", (${lo}+${hi})/2}")"
  case "${nxt}" in
    0.15) NEXT_NAME=mu015; NEXT_MU=0.15 ;;
    0.20) NEXT_NAME=mu020; NEXT_MU=0.20 ;;
    0.25) NEXT_NAME=mu025; NEXT_MU=0.25 ;;
    *)    NEXT_NAME=""   ; NEXT_MU=""   ;;
  esac
  [ -f "results/scfeq/${NEXT_NAME}/done" ] && { NEXT_NAME=""; NEXT_MU=""; }
done
echo "$(date '+%F %T') bisection finished; bracket [${lo}, ${hi}]"
