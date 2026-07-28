#!/usr/bin/env bash
# ==============================================================================
#  sweep.sh – Automated parameter sweep for Athena++ simulations
# ==============================================================================
#
#  USAGE (from repo root):
#    bash scripts/sweep.sh [--dry-run]
#
#  DESCRIPTION:
#    Runs a series of Athena++ simulations, each with different parameter
#    values overridden in a copy of the template input file.  The source code
#    is never touched.  A background monitor tracks the progress-bar ETA; if
#    the ETA stops decreasing for longer than HANG_TIMEOUT seconds the
#    simulation is killed and the sweep continues with the next test.
#    A detailed Markdown report is generated at the end (and also on Ctrl+C).
#
#  OUTPUT STRUCTURE:
#    results/SWEEP/sweep-YYYYMMDD-HHMMSS/
#      sweep.log                – full console transcript
#      REPORT.md                – final summary report
#      <test-id>/
#        athinput.in            – modified input file for this test
#        params.txt             – parameter overrides
#        run.log                – Athena++ stdout/stderr (carriage-returns
#                                 converted to newlines for easy grepping)
#        status                 – COMPLETED | KILLED_HANG | FAILED | UNKNOWN
#        start_time / end_time  – Unix timestamps
#        data/                  – Athena++ output files (*.tab, *.hst, …)
#
# ==============================================================================

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

PROBLEM="acc_disk_visc"
INPUT_TEMPLATE="inputs/hydro/athinput.acc_disk_visc"
ATHENA_BIN="bin/athena"
RESULTS_BASE="results/SWEEP"

# Hang detection
HANG_TIMEOUT=90          # seconds: kill if ETA has not decreased
ETA_POLL_INTERVAL=5      # seconds: how often to check ETA in log

# MPI (set USE_MPI=1 and rebuild with -mpi if needed)
USE_MPI=0
NUM_MPI_PROCS=4

# ── TEST DEFINITIONS ──────────────────────────────────────────────────────────
#
#  Each entry is a space-separated list of   key=value   overrides.
#
#  Special key prefixes:
#    mesh_nx1 / mesh_nx2    → nx1/nx2 inside the  <mesh>      config block
#    block_nx1 / block_nx2  → nx1/nx2 inside the  <meshblock> config block
#
#  All other keys are matched by name and replaced at their first occurrence
#  in the file (works for <problem>, <hydro>, <time>, etc.).
#
# ── Edit the array below to define your test matrix ──────────────────────────
TESTS=(
  # Base (hllc, nghost=2, vl2, xorder=2)
  "block_nx1=128 block_nx2=128"
  "block_nx1=64 block_nx2=64"
  "block_nx1=32 block_nx2=32"

  # Riemann Solvers
  "b_flux=hlle"
  "b_flux=roe"
  "b_flux=llf"

  # Time Integrators
  "integrator=rk1"
  "integrator=rk2"
  "integrator=rk3"
  "integrator=rk4"
  "integrator=ssprk5_4"

  # Ghost Cells Pairs
  "b_nghost=1"
  "b_nghost=3"
  "b_nghost=4"

  # Spatial Reconstruction + Ghost Cells Pairs
  "xorder=1 b_nghost=1"
  "xorder=3 b_nghost=3"
  "xorder=4 b_nghost=4"

  # Combined High-Order Test
  "xorder=4 integrator=rk4 b_nghost=4"
)

# Abbreviation map: long key → short label used in directory names
declare -A ABBREV=(
  [nu_iso]="nu"        [alpha]="a"
  [dfloor]="df"        [pfloor]="pf"
  [mesh_nx1]="nx1"     [mesh_nx2]="nx2"
  [mesh_x1min]="x1mn"  [mesh_x1max]="x1mx"
  [mesh_x2min]="x2mn"  [mesh_x2max]="x2mx"
  [block_nx1]="bx1"    [block_nx2]="bx2"
  [gamma]="g"          [cfl_number]="cfl"
  [tlim]="tlim"        [r_center]="rc"
  [C_prime]="cp"       [T_0]="T0"
  [M_bh]="Mbh"         [chi]="chi"
  [b_flux]="flu"       [b_nghost]="ngh"
  [integrator]="int"       [xorder]="xo"
)

# ── SCRIPT INTERNALS ────────────────────────────

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATHENA_BIN="${REPO_DIR}/${ATHENA_BIN}"
INPUT_TEMPLATE="${REPO_DIR}/${INPUT_TEMPLATE}"
RESULTS_BASE="${REPO_DIR}/${RESULTS_BASE}"

SWEEP_LABEL="sweep-$(date +%Y%m%d-%H%M%S)"
SWEEP_DIR="${RESULTS_BASE}/${SWEEP_LABEL}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Global PID of currently running Athena++ (for Ctrl+C cleanup)
CURRENT_ATHENA_PID=""
ALL_TEST_DIRS=()   # populated as tests are launched

# ── UTILITIES ─────────────────────────────────────────────────────────────────

ts()   { date "+%H:%M:%S"; }
info() { echo "[$(ts)] INFO  $*"; }
warn() { echo "[$(ts)] WARN  $*" >&2; }
err()  { echo "[$(ts)] ERROR $*" >&2; }
sep()  { echo "[$(ts)] ──────────────────────────────────────────────────────"; }

# format_duration SECONDS → "HH:MM:SS"
format_duration() {
  local s="${1:-0}"
  printf '%02d:%02d:%02d' "$(( s/3600 ))" "$(( (s%3600)/60 ))" "$(( s%60 ))"
}

# eta_to_secs "HH:MM:SS" → integer seconds  (returns -1 on failure)
eta_to_secs() {
  local str="$1"
  if [[ "$str" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
    echo $(( 10#${BASH_REMATCH[1]} * 3600 \
           + 10#${BASH_REMATCH[2]} * 60  \
           + 10#${BASH_REMATCH[3]} ))
  else
    echo -1
  fi
}

# make_dir_name "key1=val1 key2=val2" → "short1val1_short2val2"
make_dir_name() {
  local -a parts=()
  for kv in $1; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    local short="${ABBREV[$key]:-$key}"
    parts+=("${short}${val}")
  done
  # replace '+' and spaces with underscores; keep '-' only inside a value
  local name
  name="$(IFS=_; echo "${parts[*]}")"
  echo "${name//+/p}"          # e.g. 1.0e+05 → 1.0ep05
}

# ── PARAMETER APPLICATION (Python) ────────────────────────────────────────────
#
# Section-aware replacement: mesh_nx1/mesh_nx2 → <mesh> block,
# block_nx1/block_nx2 → <meshblock> block, all others → first match.

apply_params() {
  local target="$1"
  local params="$2"

  python3 - "$target" "$params" <<'PYEOF'
import sys, re

filepath   = sys.argv[1]
params_str = sys.argv[2]

with open(filepath) as f:
    lines = f.readlines()

# ── parse overrides ────────────────────────────────────────────────────────
global_ov  = {}                 # key → value  (matched anywhere)
section_ov = {}                 # (section, key) → value  (matched in section)

for kv in params_str.split():
    if '=' not in kv:
        continue
    if kv.startswith('b_flux=') or kv.startswith('b_nghost='):
        continue
    key, _, val = kv.partition('=')
    if key.startswith('mesh_'):
        section_ov[('mesh', key[5:])] = val
    elif key.startswith('block_'):
        section_ov[('meshblock', key[6:])] = val
    else:
        global_ov[key] = val

# ── helper: replace value, preserve inline comment ────────────────────────
def replace_value(line, new_val):
    m = re.match(r'^(\s*\S+\s*=\s*)', line)
    if not m:
        return line
    rest   = line[m.end():]
    hi     = rest.find('#')
    comment = ('  ' + rest[hi:].rstrip()) if hi >= 0 else ''
    return m.group(1) + new_val + comment + '\n'

# ── single-pass rewrite ────────────────────────────────────────────────────
current_section = ''
replaced_global  = set()
replaced_section = set()
result = []

for line in lines:
    stripped = line.strip()
    # Detect block headers: <name>
    if re.fullmatch(r'<[^>]+>', stripped):
        current_section = stripped[1:-1].lower()

    done = False

    # Section-specific overrides (only the first match per (section,key))
    for (sec, key), val in section_ov.items():
        sk = (sec, key)
        if sk not in replaced_section and current_section == sec:
            if re.match(r'^\s*' + re.escape(key) + r'\s*=', line):
                result.append(replace_value(line, val))
                replaced_section.add(sk)
                done = True
                break

    if not done:
        # Global overrides (only the first match per key)
        for key, val in global_ov.items():
            if key not in replaced_global:
                if re.match(r'^\s*' + re.escape(key) + r'\s*=', line):
                    result.append(replace_value(line, val))
                    replaced_global.add(key)
                    done = True
                    break

    if not done:
        result.append(line)

# ── warn about anything not found ─────────────────────────────────────────
for key in global_ov:
    if key not in replaced_global:
        print(f"WARNING: parameter '{key}' not found in {filepath}",
              file=sys.stderr)
for (sec, key) in section_ov:
    if (sec, key) not in replaced_section:
        print(f"WARNING: parameter '{key}' (section <{sec}>) not found in "
              f"{filepath}", file=sys.stderr)

with open(filepath, 'w') as f:
    f.writelines(result)
PYEOF
}

# ── SIMULATION RUNNER WITH HANG DETECTION ─────────────────────────────────────
#
# Runs Athena++ with the -p (progress-bar) flag.
# The \r-separated progress bar output is converted to \n-per-update lines,
# so the log file can be grep-parsed normally.
#
# ETA monitoring (background loop):
#   • extracts  "ETA: HH:MM:SS"  from the latest log line
#   • if ETA has not decreased for HANG_TIMEOUT seconds → kill
#   • if "ETA: calculating..." persists (early stage), falls back to
#     checking that at least one new line appears per HANG_TIMEOUT seconds


ensure_binary() {
  local params="$1"
  local flux=""
  local nghost=""
  
  for kv in $params; do
    if [[ "$kv" == b_flux=* ]]; then
      flux="${kv#*=}""
    elif [[ "$kv" == b_nghost=* ]]; then
      nghost="${kv#*=}""
    fi
  done
  
  local current_flux="${flux:-hllc}"
  local current_nghost="${nghost:-2}"
  local bin_name="bin/athena_${current_flux}_${current_nghost}"
  
  if [[ ! -x "${REPO_DIR}/${bin_name}" && "$DRY_RUN" -eq 0 ]]; then
      info "Binary ${bin_name} not found. Compiling..." >&2
      pushd "${REPO_DIR}" > /dev/null
      python3 configure.py --prob="${PROBLEM}" --coord=cylindrical -debug --flux="${current_flux}" --nghost="${current_nghost}" >> "${SWEEP_DIR}/compile.log" 2>&1
      make clean >> "${SWEEP_DIR}/compile.log" 2>&1
      if make -j"$(nproc)" >> "${SWEEP_DIR}/compile.log" 2>&1; then
        mv bin/athena "${bin_name}"
      else
        err "Compilation failed for ${bin_name}. See ${SWEEP_DIR}/compile.log" >&2
        popd > /dev/null
        return 1
      fi
      popd > /dev/null
  fi
  echo "${REPO_DIR}/${bin_name}"
}

run_test() {
  local test_idx="$1"
  local test_dir="$2"
  local input_file="$3"
  local bin_to_run="$4"
  local data_dir="${test_dir}/data"
  local log_file="${test_dir}/run.log"
  local status_file="${test_dir}/status"

  # ── dry run ──────────────────────────────────────────────────────────────
  if [[ "$DRY_RUN" -eq 1 ]]; then
    info "  [DRY RUN] Would execute: ${bin_to_run} -p -i ${input_file}"
    echo "DRY_RUN" > "$status_file"
    echo "$(date +%s)" > "${test_dir}/start_time"
    echo "$(date +%s)" > "${test_dir}/end_time"
    return 0
  fi

  local start_wall
  start_wall=$(date +%s)
  echo "$start_wall" > "${test_dir}/start_time"

  # ── launch Athena++ ───────────────────────────────────────────────────────
  # The subshell writes its own PID (= the PID that exec will inherit) to a
  # file, then replaces itself with Athena via exec.  This lets us kill the
  # correct process later.
  local pid_file="${test_dir}/athena.pid"

  # Determine the launcher command
  local run_cmd
  if command -v stdbuf &>/dev/null; then
    run_cmd="stdbuf -oL"
  else
    run_cmd=""
  fi

  if [[ "$USE_MPI" -eq 1 ]]; then
    {
      echo $BASHPID > "$pid_file"
      exec mpirun -np "$NUM_MPI_PROCS" \
           ${run_cmd} "${ATHENA_BIN}" -p -i "$input_file" 2>&1
    } | tr '\r' '\n' > "$log_file" &
  else
    {
      echo $BASHPID > "$pid_file"
      exec ${run_cmd} "${ATHENA_BIN}" -p -i "$input_file" 2>&1
    } | tr '\r' '\n' > "$log_file" &
  fi

  local pipe_pid=$!

  # Give the subshell time to write its PID before exec replaces it
  sleep 0.5
  local athena_pid
  athena_pid=$(cat "$pid_file" 2>/dev/null || echo "$pipe_pid")
  CURRENT_ATHENA_PID="$athena_pid"
  info "  PID: $athena_pid  log: $(basename "$log_file")"

  # ── ETA monitor ───────────────────────────────────────────────────────────
  local last_eta_secs=-1
  local last_decrease_time
  last_decrease_time=$(date +%s)
  local last_log_linecount=0
  local last_sim_time_str=""
  local last_simtime_change_time
  last_simtime_change_time=$(date +%s)
  local hung=0

  while kill -0 "$athena_pid" 2>/dev/null || kill -0 "$pipe_pid" 2>/dev/null; do
    sleep "$ETA_POLL_INTERVAL"

    # ── 1. ETA-based check ────────────────────────────────────────────────
    local eta_str
    eta_str=$(grep -oE 'ETA: [0-9]{2}:[0-9]{2}:[0-9]{2}' "$log_file" \
              2>/dev/null | tail -1 | awk '{print $2}')

    if [[ -n "$eta_str" ]]; then
      local cur_eta
      cur_eta=$(eta_to_secs "$eta_str")
      if (( cur_eta >= 0 )); then
        if (( last_eta_secs < 0 || cur_eta < last_eta_secs )); then
          # ETA is progressing normally
          last_eta_secs=$cur_eta
          last_decrease_time=$(date +%s)
        else
          # ETA same or increased – check timeout
          local stall=$(( $(date +%s) - last_decrease_time ))
          if (( stall >= HANG_TIMEOUT )); then
            warn "  HANG: ETA (${eta_str}) has not decreased for ${stall}s" \
                  "(limit: ${HANG_TIMEOUT}s) – killing PID ${athena_pid}"
            hung=1
          fi
        fi
      fi
    else
      # ── 2. Fallback: ETA still "calculating..." ────────────────────────
      # Primary: track sim_time progress (catches frozen dt → 0)
      local cur_sim_time_str
      cur_sim_time_str=$(grep -oE 'sim_time=[0-9.e+\-]+' "$log_file" \
                         2>/dev/null | tail -1 | sed 's/sim_time=//')

      if [[ -n "$cur_sim_time_str" ]]; then
        if [[ "$cur_sim_time_str" != "$last_sim_time_str" ]]; then
          last_sim_time_str="$cur_sim_time_str"
          last_simtime_change_time=$(date +%s)
        else
          local stall=$(( $(date +%s) - last_simtime_change_time ))
          if (( stall >= HANG_TIMEOUT )); then
            warn "  HANG: sim_time (${cur_sim_time_str}) has not advanced for ${stall}s" \
                  "(limit: ${HANG_TIMEOUT}s) – killing PID ${athena_pid}"
            hung=1
          fi
        fi
      else
        # Secondary: no sim_time in log yet – check for any new output
        local cur_lines
        cur_lines=$(wc -l < "$log_file" 2>/dev/null || echo 0)
        if (( cur_lines > last_log_linecount )); then
          last_log_linecount=$cur_lines
          last_decrease_time=$(date +%s)
        else
          local stall=$(( $(date +%s) - last_decrease_time ))
          if (( stall >= HANG_TIMEOUT )); then
            warn "  HANG: no output for ${stall}s (limit: ${HANG_TIMEOUT}s)" \
                  "– killing PID ${athena_pid}"
            hung=1
          fi
        fi
      fi
    fi

    if (( hung )); then
      kill "$athena_pid"   2>/dev/null || true
      sleep 3
      kill -9 "$athena_pid" 2>/dev/null || true
      wait "$pipe_pid"     2>/dev/null || true
      break
    fi
  done

  # Wait for the pipe (tr) to finish flushing
  wait "$pipe_pid" 2>/dev/null || true
  CURRENT_ATHENA_PID=""

  local end_wall
  end_wall=$(date +%s)
  echo "$end_wall" > "${test_dir}/end_time"
  local dur
  dur=$(format_duration $(( end_wall - start_wall )))

  # ── determine exit status ─────────────────────────────────────────────────
  if (( hung )); then
    echo "KILLED_HANG" > "$status_file"
    info "  Status: KILLED_HANG  Duration: $dur"
    return 1
  elif grep -qE "Terminating on (time|cycle) limit" "$log_file" 2>/dev/null; then
    echo "COMPLETED" > "$status_file"
    info "  Status: COMPLETED    Duration: $dur"
    return 0
  elif grep -qiE "FATAL ERROR|Segmentation fault|Aborted|Error:" \
            "$log_file" 2>/dev/null; then
    echo "FAILED" > "$status_file"
    warn "  Status: FAILED       Duration: $dur"
    return 2
  else
    echo "UNKNOWN" > "$status_file"
    warn "  Status: UNKNOWN      Duration: $dur"
    return 0
  fi
}

# ── LOG PARSERS ───────────────────────────────────────────────────────────────

# Last cycle number written to log (works for both -p and default output)
last_cycle() {
  grep -oE 'cycle=[0-9]+' "$1" 2>/dev/null \
    | grep -v '/' | tail -1 | grep -oE '[0-9]+' || echo "N/A"
}

# Last sim_time written to progress bar line
last_sim_time() {
  # progress bar:  sim_time=X.XXXe+XX/Y.YYYe+XX
  local t
  t=$(grep -oE 'sim_time=[0-9.e+\-]+/[0-9.e+\-]+' "$1" 2>/dev/null \
      | tail -1 | grep -oE '^[^/]+' | grep -oE '[0-9.e+\-]+')
  # fallback: final termination line "time=X.XXX"
  [[ -z "$t" ]] && t=$(grep -E '^time=' "$1" 2>/dev/null \
                        | tail -1 | grep -oE '[0-9.e+\-]+')
  echo "${t:-N/A}"
}

# ── MARKDOWN REPORT ──────────────────────────────────────────────────────────

generate_report() {
  local sweep_dir="$1"; shift
  local -a test_dirs=("$@")

  local report_file="${sweep_dir}/REPORT.md"
  local ts_now
  ts_now=$(date "+%Y-%m-%d %H:%M:%S")
  local host
  host=$(hostname 2>/dev/null || echo "unknown")

  # ── collect per-test metadata ─────────────────────────────────────────────
  local -a t_names t_params t_statuses t_durations t_cycles t_simtimes

  local n_comp=0 n_kill=0 n_fail=0 n_unk=0

  for tdir in "${test_dirs[@]}"; do
    t_names+=("$(basename "$tdir")")

    local p="N/A"
    [[ -f "${tdir}/params.txt" ]] && p=$(cat "${tdir}/params.txt")
    t_params+=("$p")

    local st="N/A"
    [[ -f "${tdir}/status" ]] && st=$(cat "${tdir}/status")
    t_statuses+=("$st")

    local dur="N/A"
    if [[ -f "${tdir}/start_time" && -f "${tdir}/end_time" ]]; then
      dur=$(format_duration $(( $(cat "${tdir}/end_time") \
                               - $(cat "${tdir}/start_time") )))
    fi
    t_durations+=("$dur")

    local cyc="N/A"
    [[ -f "${tdir}/run.log" ]] && cyc=$(last_cycle "${tdir}/run.log")
    t_cycles+=("$cyc")

    local stime="N/A"
    [[ -f "${tdir}/run.log" ]] && stime=$(last_sim_time "${tdir}/run.log")
    t_simtimes+=("$stime")

    case "$st" in
      COMPLETED)   n_comp=$(( n_comp + 1 )) ;;
      KILLED_HANG) n_kill=$(( n_kill + 1 )) ;;
      FAILED)      n_fail=$(( n_fail + 1 )) ;;
      *)           n_unk=$(( n_unk + 1 ))   ;;
    esac
  done

  # ── helper: status → icon ─────────────────────────────────────────────────
  status_icon() {
    case "$1" in
      COMPLETED)   echo "✅ COMPLETED"              ;;
      KILLED_HANG) echo "⏱️ KILLED (hang)"           ;;
      FAILED)      echo "❌ FAILED"                 ;;
      DRY_RUN)     echo "🔍 DRY RUN"                ;;
      *)           echo "❓ ${1}"                   ;;
    esac
  }

  # ── write report ──────────────────────────────────────────────────────────
  {
    echo "# Athena++ Parameter Sweep Report"
    echo ""
    echo "## Overview"
    echo ""
    echo "| | |"
    echo "|---|---|"
    echo "| **Generated** | ${ts_now} |"
    echo "| **Host** | \`${host}\` |"
    echo "| **Problem** | \`${PROBLEM}\` |"
    echo "| **Template input** | \`${INPUT_TEMPLATE}\` |"
    echo "| **Binary** | \`${ATHENA_BIN}\` |"
    echo "| **Sweep directory** | \`${sweep_dir}\` |"
    echo "| **Hang timeout** | ${HANG_TIMEOUT} s |"
    echo "| **Total tests** | ${#test_dirs[@]} |"
    echo "| **Completed** | ✅ ${n_comp} |"
    echo "| **Killed (hang)** | ⏱️ ${n_kill} |"
    echo "| **Failed (error)** | ❌ ${n_fail} |"
    echo "| **Unknown / dry-run** | ❓ ${n_unk} |"
    echo ""
    echo "---"
    echo ""

    # ── summary table ───────────────────────────────────────────────────────
    echo "## Summary Table"
    echo ""
    echo "| \# | Test directory | Parameters | Status | Duration | Cycles | Sim time |"
    echo "|---|---|---|---|---|---|---|"

    for i in "${!test_dirs[@]}"; do
      # Format parameters: "key=val key=val" → "`key=val`, `key=val`"
      local pfmt
      pfmt=$(echo "${t_params[$i]}" | sed 's/  */ /g; s/ /, /g')
      printf "| %d | \`%s\` | %s | %s | %s | %s | %s |\n" \
        "$(( i+1 ))" \
        "${t_names[$i]}" \
        "$pfmt" \
        "$(status_icon "${t_statuses[$i]}")" \
        "${t_durations[$i]}" \
        "${t_cycles[$i]}" \
        "${t_simtimes[$i]}"
    done
    echo ""
    echo "---"
    echo ""

    # ── per-test detail sections ─────────────────────────────────────────────
    echo "## Test Details"
    echo ""

    for i in "${!test_dirs[@]}"; do
      local tdir="${test_dirs[$i]}"
      local tname="${t_names[$i]}"
      local num=$(( i+1 ))

      echo "### Test ${num}: \`${tname}\`"
      echo ""
      echo "| Field | Value |"
      echo "|---|---|"
      echo "| **Status** | $(status_icon "${t_statuses[$i]}") |"
      echo "| **Duration** | ${t_durations[$i]} |"
      echo "| **Cycles completed** | ${t_cycles[$i]} |"
      echo "| **Final sim time** | ${t_simtimes[$i]} |"
      echo "| **Input file** | \`${tdir}/athinput.in\` |"
      echo ""

      # Parameters
      echo "#### Parameter overrides"
      echo ""
      echo "\`\`\`"
      for kv in ${t_params[$i]}; do
        printf "  %-20s\n" "$kv"
      done
      echo "\`\`\`"
      echo ""

      # Last 20 log lines
      if [[ -f "${tdir}/run.log" ]]; then
        echo "#### Last output"
        echo ""
        echo "\`\`\`"
        tail -20 "${tdir}/run.log" 2>/dev/null || echo "(log file empty)"
        echo "\`\`\`"
        echo ""
      fi

      # History file snippet
      local hst
      hst=$(find "${tdir}/data" -maxdepth 1 -name "*.hst" 2>/dev/null | head -1)
      if [[ -n "$hst" && -f "$hst" ]]; then
        local nlines
        nlines=$(wc -l < "$hst")
        echo "#### History file: \`$(basename "$hst")\`  (${nlines} lines)"
        echo ""
        echo "\`\`\`"
        head -5  "$hst"
        echo "  ..."
        tail -3  "$hst"
        echo "\`\`\`"
        echo ""
      fi

      echo "---"
      echo ""
    done

    echo "*Report generated by \`scripts/sweep.sh\` — problem \`${PROBLEM}\`*"
  } > "$report_file"

  info "Report → ${report_file}"
}

# ── SIGNAL HANDLER ────────────────────────────────────────────────────────────

_on_interrupt() {
  echo ""
  warn "Interrupted!  Cleaning up..."
  if [[ -n "$CURRENT_ATHENA_PID" ]]; then
    warn "Killing Athena++ PID ${CURRENT_ATHENA_PID}"
    kill "$CURRENT_ATHENA_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$CURRENT_ATHENA_PID" 2>/dev/null || true
  fi
  if [[ ${#ALL_TEST_DIRS[@]} -gt 0 ]]; then
    info "Generating partial report for ${#ALL_TEST_DIRS[@]} test(s)..."
    generate_report "$SWEEP_DIR" "${ALL_TEST_DIRS[@]}"
  fi
  exit 130
}

trap '_on_interrupt' SIGINT SIGTERM

fi

if [[ ! -f "${INPUT_TEMPLATE}" ]]; then
  err "Input template '${INPUT_TEMPLATE}' not found."
  exit 1
fi

# ── SETUP ─────────────────────────────────────────────────────────────────────

mkdir -p "${SWEEP_DIR}"

# Tee all output to sweep.log from this point on
exec > >(tee -a "${SWEEP_DIR}/sweep.log") 2>&1

info "════════════════════════════════════════════════════════"
info "  Athena++ Parameter Sweep"
info "  Problem      : ${PROBLEM}"
info "  Tests        : ${#TESTS[@]}"
info "  Hang timeout : ${HANG_TIMEOUT} s"
info "  Sweep dir    : ${SWEEP_DIR}"
[[ "$DRY_RUN" -eq 1 ]] && info "  Mode         : DRY RUN (no simulations run)"
info "════════════════════════════════════════════════════════"

# Copy source file and input template into the sweep root for provenance
_src="${REPO_DIR}/src/pgen/${PROBLEM}.cpp"
if [[ -f "$_src" ]]; then
  cp "$_src" "${SWEEP_DIR}/${PROBLEM}.cpp"
  info "Source snapshot → ${SWEEP_DIR}/${PROBLEM}.cpp"
fi
cp "${INPUT_TEMPLATE}" "${SWEEP_DIR}/$(basename "${INPUT_TEMPLATE}")"
info "Input template  → ${SWEEP_DIR}/$(basename "${INPUT_TEMPLATE}")"
echo ""

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

total_tests=${#TESTS[@]}
pad_width=${#total_tests}   # digits needed for zero-padding

for i in "${!TESTS[@]}"; do
  params="${TESTS[$i]}"
  test_num=$(( i + 1 ))
  printf -v num_prefix "t%0${pad_width}d" "$test_num"
  dir_name="${num_prefix}_$(make_dir_name "$params")"
  test_dir="${SWEEP_DIR}/${dir_name}"
  data_dir="${test_dir}/data"

  mkdir -p "$data_dir"
  ALL_TEST_DIRS+=("$test_dir")
  echo "$params" > "${test_dir}/params.txt"

  # Copy visualization scripts alongside the input file
  for _vis in vis1d.py vis2d.py vishst.py; do
    [[ -f "${REPO_DIR}/scripts/${_vis}" ]] && \
      cp "${REPO_DIR}/scripts/${_vis}" "${test_dir}/"
  done

  # Prepare modified input file
  local_input="${test_dir}/athinput.in"
  cp "${INPUT_TEMPLATE}" "$local_input"
  apply_params "$local_input" "$params"

  sep
  info "Test ${test_num}/${#TESTS[@]}: ${dir_name}"
  info "  Params: ${params}"

  # Change to data dir so Athena writes output there; use absolute paths
  pushd "$data_dir" > /dev/null
  custom_bin=$(ensure_binary "$params") || exit 1
  run_test "$test_num" "$test_dir" "$local_input" "$custom_bin"
  popd > /dev/null

  echo ""
done

# ── FINAL REPORT ─────────────────────────────────────────────────────────────

sep
info "All ${#TESTS[@]} tests finished.  Generating report..."
generate_report "$SWEEP_DIR" "${ALL_TEST_DIRS[@]}"

# ── POST-PROCESSING: run vis2d.py + vishst.py for each test (sequentially) ──────

sep
info "Post-processing: running vis2d.py + vishst.py for each test..."
echo ""

for tdir in "${ALL_TEST_DIRS[@]}"; do
  local_vis="${tdir}/vis2d.py"
  local_vishst="${tdir}/vishst.py"
  local_data="${tdir}/data"
  local_figs="${tdir}/figs_2d"
  local_figs_hst="${tdir}/figs_hst"
  tname="$(basename "$tdir")"

  # ── vis2d ──────────────────────────────────────────────────────────────
  if [[ ! -f "$local_vis" ]]; then
    warn "  [${tname}] vis2d.py not found – skipping"
  elif [[ ! -d "$local_data" ]] || [[ -z "$(ls "${local_data}"/*.tab 2>/dev/null)" ]]; then
    warn "  [${tname}] no .tab files in data/ – skipping vis2d"
  else
    info "  [${tname}] running vis2d.py --subsample 1 ..."
    mkdir -p "$local_figs"
    if python3 "$local_vis" \
         --subsample 1 \
         --data_dir "$local_data" \
         --output_dir "$local_figs" \
         >> "${tdir}/vis2d.log" 2>&1; then
      info "  [${tname}] vis2d done → ${local_figs}"
    else
      warn "  [${tname}] vis2d.py exited with errors – check ${tdir}/vis2d.log"
    fi
  fi

  # ── vishst ─────────────────────────────────────────────────────────────
  if [[ ! -f "$local_vishst" ]]; then
    warn "  [${tname}] vishst.py not found – skipping"
  else
    hst_file=$(find "$local_data" -maxdepth 1 -name "*.hst" 2>/dev/null | head -1)
    if [[ -z "$hst_file" ]]; then
      warn "  [${tname}] no .hst file in data/ – skipping vishst"
    else
      info "  [${tname}] running vishst.py ..."
      mkdir -p "$local_figs_hst"
      if python3 "$local_vishst" \
           --hst_file "$hst_file" \
           --output_dir "$local_figs_hst" \
           >> "${tdir}/vishst.log" 2>&1; then
        info "  [${tname}] vishst done → ${local_figs_hst}"
      else
        warn "  [${tname}] vishst.py exited with errors – check ${tdir}/vishst.log"
      fi
    fi
  fi

  echo ""
done

sep
info "Sweep complete."
info "  Results : ${SWEEP_DIR}"
info "  Report  : ${SWEEP_DIR}/REPORT.md"
