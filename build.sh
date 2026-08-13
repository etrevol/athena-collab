#!/bin/bash
#
# Configure, build and run one Athena++ simulation.
#
#   ./build.sh
#
# Rebuilds only when the problem generator or a configure flag changed, then writes
# results/<project_directory>/sample-<timestamp><specification>/{data,materials}.
# Edit the variables below to choose what is built and where it lands.
#
# Generate the input file first if the physics changed:
#   python3 scripts/theory/disk_model.py
#
# To continue a run that was interrupted, set resume="yes" below and run again.
# That needs the run's input file to have had an `rst` output block; set
# RUN_SETUP["restart_every_orbits"] in scripts/theory/disk_model.py before the
# first launch, or declare <output4> file_type = rst by hand. Without one there
# is nothing to continue from, and the script says so instead of silently
# starting over.
#
# Known consequence of resuming, from Athena++ rather than from this script:
# the .hst file is appended to, not truncated, so it keeps the rows written
# between the restart dump and the interruption. Its time column therefore steps
# backwards once at the join, and a time-series plot will loop back on itself
# there. The .tab/.athdf frames do not have this problem - they are numbered by
# output index and simply overwrite.
#
set -e  # abort on the first failing command

problem="acc_disk_visc"        # problem generator: src/pgen/<problem>.cpp
input="acc_disk_visc"          # input file: inputs/hydro/athinput.<input>
specification="-acc_disk_visc" # suffix appended to the run directory name

use_mpi=0              # 1 = build and run with MPI, 0 = serial
num_mpi_procs=4        # MPI ranks, used only when use_mpi=1

show_progress="yes"    # pass -p to athena for the progress bar with ETA

# Continue an interrupted run instead of starting a new one. Requires that the
# athinput used for that run had an `rst` output block (scripts/theory/disk_model.py
# emits one when RUN_SETUP["restart_every_orbits"] is set; a hand-written input can
# declare its own). This block is deliberately problem-agnostic: it looks only for
# .rst files on disk and hands the newest to `athena -r`, so it works for any
# problem generator and any input file, not just this one.
resume="no"            # "yes" = continue the newest run under results/<project>/
resume_directory=""    # optional: resume this run directory instead of the newest

project_directory="runs"   # subdirectory of results/ that receives the run

repo_directory=$(pwd)

results_directory="${repo_directory}/results"
timestamp=$(date +"%Y%m%d-%H%M%S")
sample_directory="sample-${timestamp}${specification}"
data_directory="data"
materials_directory="materials"

# When resuming, the run directory is an existing one rather than a fresh
# timestamp. Resolved here so every path built below points at the right place.
restart_file=""
if [ "$resume" = "yes" ]; then
    if [ -n "$resume_directory" ]; then
        resume_path="$resume_directory"
    else
        # newest run directory under results/<project_directory>/
        resume_path=$(ls -1dt "${results_directory}/${project_directory}"/*/ 2>/dev/null | head -n1)
    fi
    resume_path="${resume_path%/}"
    if [ -z "$resume_path" ] || [ ! -d "$resume_path" ]; then
        echo "Error: resume=\"yes\" but no run directory found under" \
             "${results_directory}/${project_directory}/"
        exit 1
    fi
    # Newest .rst anywhere in that run. *.final.rst (written when athena hits its
    # -t wall limit) sorts alongside the periodic dumps, and -t picks whichever
    # was actually written last, which is what we want either way.
    restart_file=$(ls -1t "${resume_path}/${data_directory}"/*.rst 2>/dev/null | head -n1)
    if [ -z "$restart_file" ]; then
        echo "Error: no .rst file in ${resume_path}/${data_directory}/"
        echo "       The run cannot be continued: its input file had no rst output block."
        echo "       Add one (RUN_SETUP[\"restart_every_orbits\"] in scripts/theory/disk_model.py,"
        echo "       or a <output4> file_type = rst block by hand) and rerun from the start."
        exit 1
    fi
    sample_directory=$(basename "$resume_path")
    echo "Resuming: ${sample_directory}"
    echo "  restart file: $(basename "$restart_file")"
fi

# Rebuild only when the generator source or a configure flag changed
build_state_file=".build_state"
need_rebuild=false

# Hash of the generator source, stored in .build_state
source_file="src/pgen/${problem}.cpp"
if [ -f "${source_file}" ]; then
    current_hash=$(md5sum "${source_file}" | cut -d' ' -f1)
else
    echo "Error: Source file ${source_file} not found!"
    exit 1
fi

# Check if build state exists and compare
if [ -f "${build_state_file}" ]; then
    saved_problem=$(grep "^PROBLEM=" "${build_state_file}" | cut -d'=' -f2)
    saved_hash=$(grep "^HASH=" "${build_state_file}" | cut -d'=' -f2)
    
    if [ "${saved_problem}" != "${problem}" ] || [ "${saved_hash}" != "${current_hash}" ]; then
        need_rebuild=true
        echo "Changes detected (problem or source file changed). Rebuilding..."
    else
        echo "No changes detected. Skipping compilation..."
    fi
else
    need_rebuild=true
    echo "First build or state file missing. Building..."
fi

# 1. Configure and build
if [ "$need_rebuild" = true ]; then
    echo "Configuring with problem: ${problem}"
    if [ "$use_mpi" -eq 1 ]; then
        echo "MPI parallelization: ENABLED"
        python3 configure.py --prob "${problem}" --coord=cylindrical -mpi
    else
        echo "MPI parallelization: DISABLED (single block mode)"
        python3 configure.py --prob "${problem}" --coord=cylindrical
    fi

    echo "Building..."
    make clean
    make -j$(($(nproc) - 1))
    
    # Save current state
    echo "PROBLEM=${problem}" > "${build_state_file}"
    echo "HASH=${current_hash}" >> "${build_state_file}"
    echo "Build completed successfully!"
else
    echo "Using existing build."
fi

# 2. Run directory: data/ for output, materials/ for the exact inputs used
mkdir -p "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"
mkdir -p "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}"

# 3. Record what this run was made with. On resume these are left untouched: the
#    continued run must keep the input and generator it actually started with,
#    not whatever happens to be in the repository now.
if [ "$resume" != "yes" ]; then
    cp "${repo_directory}/inputs/hydro/athinput.${input}" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
    cp "${repo_directory}/src/pgen/${problem}.cpp" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
fi

# The visualization scripts live outside the repository (scripts/ is gitignored), so a
# fresh clone will not have them. They are a convenience copy, not something the run
# depends on - skip whatever is missing instead of aborting under `set -e`.
vis_scripts_dir="${repo_directory}/scripts/practice/vis"
for vis_script in athena_data.py vis.py vis1d.py vis2d.py vishst.py visforces.py athvis.py; do
    if [ -f "${vis_scripts_dir}/${vis_script}" ]; then
        cp "${vis_scripts_dir}/${vis_script}" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
    else
        echo "Note: ${vis_script} not found in ${vis_scripts_dir} - skipping."
    fi
done

# 4. Athena++ writes into the current directory
cd "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"

# 5. Drop any output left from an earlier run in this directory. Skipped entirely
#    on resume - that output IS the run being continued. Athena++ carries on the
#    frame numbering from the restart file, so old and new frames form one series.
if [ "$resume" != "yes" ]; then
    rm -f *.tab *.athdf *.athdf.xdmf *.hst
fi

# 6. Run.  `-r <file>` continues from a restart dump; `-i <file>` starts fresh.
#    Athena++ takes everything it needs from the .rst, so no -i is passed on
#    resume - that is also what keeps this block problem-agnostic.
if [ "$resume" = "yes" ]; then
    athena_args=(-r "$restart_file")
else
    athena_args=(-i "../${materials_directory}/athinput.${input}")
fi
if [ "$show_progress" = "yes" ]; then
    athena_args=(-p "${athena_args[@]}")
fi

if [ "$use_mpi" -eq 1 ]; then
    echo "Running with ${num_mpi_procs} MPI processes..."
    mpirun -np ${num_mpi_procs} "${repo_directory}/bin/athena" "${athena_args[@]}"
else
    echo "Running in single block mode (no MPI)..."
    "${repo_directory}/bin/athena" "${athena_args[@]}"
fi

# 7. Report where everything went
# ls

echo ""
echo "Results saved to: ${results_directory}/${project_directory}/${sample_directory}/${data_directory}"
echo "Materials saved to: ${results_directory}/${project_directory}/${sample_directory}/${materials_directory}"