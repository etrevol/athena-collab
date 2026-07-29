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
#   python3 scripts/theory/make_athinput.py -o inputs/hydro/athinput.acc_disk_visc
#   python3 scripts/theory/verify_athinput.py inputs/hydro/athinput.acc_disk_visc
#
set -e  # abort on the first failing command

problem="acc_disk_visc"        # problem generator: src/pgen/<problem>.cpp
input="acc_disk_visc"          # input file: inputs/hydro/athinput.<input>
specification="-acc_disk_visc" # suffix appended to the run directory name

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

# 3. Record what this run was made with
cp "${repo_directory}/inputs/hydro/athinput.${input}" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
cp "${repo_directory}/src/pgen/${problem}.cpp" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"

# The visualization scripts live outside the repository (scripts/ is gitignored), so a
# fresh clone will not have them. They are a convenience copy, not something the run
# depends on - skip whatever is missing instead of aborting under `set -e`.
vis_scripts_dir="${repo_directory}/scripts/practice/vis"
for vis_script in vis1d.py vis2d.py vishst.py visforces.py; do
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