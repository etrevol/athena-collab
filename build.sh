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
#   python3 scripts/theory/make_athinput.py -o inputs/hydro/athinput.acc_disk_visc_vv
#   python3 scripts/theory/verify_athinput.py inputs/hydro/athinput.acc_disk_visc_vv
#
set -e  # abort on the first failing command

problem="acc_disk_visc_vv"        # problem generator: src/pgen/<problem>.cpp
input="acc_disk_visc_vv"          # input file: inputs/hydro/athinput.<input>
specification="-acc_disk_visc_vv" # suffix appended to the run directory name

use_mpi=0              # 1 = build and run with MPI, 0 = serial
num_mpi_procs=4        # MPI ranks, used only when use_mpi=1

show_progress="yes"    # pass -p to athena for the progress bar with ETA

project_directory="runs"   # subdirectory of results/ that receives the run

repo_directory=$(pwd)

results_directory="${repo_directory}/results"
timestamp=$(date +"%Y%m%d-%H%M%S")
sample_directory="sample-${timestamp}${specification}"
data_directory="data"
materials_directory="materials"

# Rebuild only when the generator source or a configure flag changed
build_state_file=".build_state"
need_rebuild=false

# The full configure line, so a change of ANY flag forces a rebuild. Storing only the
# problem name meant `./build.sh` with use_mpi flipped reported "no changes" and silently
# ran the previous binary -- which would quietly invalidate a convergence study.
configure_args="--prob ${problem} --coord=cylindrical"
if [ "$use_mpi" -eq 1 ]; then
    configure_args="${configure_args} -mpi"
fi

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
    saved_configure=$(sed -n 's/^CONFIGURE=//p' "${build_state_file}")

    if [ "${saved_problem}" != "${problem}" ] || [ "${saved_hash}" != "${current_hash}" ]; then
        need_rebuild=true
        echo "Changes detected (problem or source file changed). Rebuilding..."
    elif [ "${saved_configure}" != "${configure_args}" ]; then
        need_rebuild=true
        echo "Configure flags changed:"
        echo "  was: ${saved_configure:-<not recorded>}"
        echo "  now: ${configure_args}"
        echo "Rebuilding (a plain make would leave objects from both configurations)..."
    else
        echo "No changes detected. Skipping compilation..."
    fi
else
    need_rebuild=true
    echo "First build or state file missing. Building..."
fi

# 1. Configure and build
if [ "$need_rebuild" = true ]; then
    echo "Configuring: python3 configure.py ${configure_args}"
    if [ "$use_mpi" -eq 1 ]; then
        echo "MPI parallelization: ENABLED"
    else
        echo "MPI parallelization: DISABLED (single block mode)"
    fi
    python3 configure.py ${configure_args}

    echo "Building..."
    make clean
    make -j$(($(nproc) - 1))

    # Save current state
    echo "PROBLEM=${problem}" > "${build_state_file}"
    echo "HASH=${current_hash}" >> "${build_state_file}"
    echo "CONFIGURE=${configure_args}" >> "${build_state_file}"
    echo "Build completed successfully!"
else
    echo "Using existing build."
fi

# 2. Run directory: data/ for output, materials/ for the exact inputs used
mkdir -p "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"
mkdir -p "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}"

# 3. Record what this run was made with. A convergence study is only meaningful if every
# point in it can be shown to come from the same binary, so the provenance travels with
# the run: input, generator source, configure line and the working-tree commit.
run_materials="${results_directory}/${project_directory}/${sample_directory}/${materials_directory}"
cp "${repo_directory}/inputs/hydro/athinput.${input}" "${run_materials}/"
cp "${repo_directory}/src/pgen/${problem}.cpp" "${run_materials}/"
if [ -f "${repo_directory}/configure.log" ]; then
    cp "${repo_directory}/configure.log" "${run_materials}/"
fi
{
    echo "date        = $(date -Iseconds)"
    echo "host        = $(hostname)"
    echo "problem     = ${problem}"
    echo "configure   = ${configure_args}"
    echo "pgen_md5    = ${current_hash}"
    echo "git_sha     = $(git -C "${repo_directory}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty   = $(git -C "${repo_directory}" status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "mpi_procs   = $([ "$use_mpi" -eq 1 ] && echo "${num_mpi_procs}" || echo 1)"
} > "${run_materials}/provenance.txt"

# The visualization scripts live outside the repository (scripts/ is gitignored), so a
# fresh clone will not have them. They are a convenience copy, not something the run
# depends on - skip whatever is missing instead of aborting under `set -e`.
vis_scripts_dir="${repo_directory}/scripts/practice/vis"
for vis_script in athena_data.py vis1d.py vis2d.py vishst.py visforces.py; do
    if [ -f "${vis_scripts_dir}/${vis_script}" ]; then
        cp "${vis_scripts_dir}/${vis_script}" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
    else
        echo "Note: ${vis_script} not found in ${vis_scripts_dir} - skipping."
    fi
done

# 4. Athena++ writes into the current directory
cd "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"

# 5. Drop any output left from an earlier run in this directory
rm -f *.tab *.athdf *.athdf.xdmf *.hst

# 6. Run
if [ "$use_mpi" -eq 1 ]; then
    echo "Running with ${num_mpi_procs} MPI processes..."
    if [ "$show_progress" = "yes" ]; then
        mpirun -np ${num_mpi_procs} "${repo_directory}/bin/athena" -p -i "../${materials_directory}/athinput.${input}"
    else
        mpirun -np ${num_mpi_procs} "${repo_directory}/bin/athena" -i "../${materials_directory}/athinput.${input}"
    fi
else
    echo "Running in single block mode (no MPI)..."
    if [ "$show_progress" = "yes" ]; then
        "${repo_directory}/bin/athena" -p -i "../${materials_directory}/athinput.${input}"
    else
        "${repo_directory}/bin/athena" -i "../${materials_directory}/athinput.${input}"
    fi
fi

# 7. Report where everything went
# ls

echo ""
echo "Results saved to: ${results_directory}/${project_directory}/${sample_directory}/${data_directory}"
echo "Materials saved to: ${results_directory}/${project_directory}/${sample_directory}/${materials_directory}"