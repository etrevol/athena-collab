#!/bin/bash
set -e  # stop the script immediately if any command fails

problem="acc_disk_visc" # acc_disk_grav, acc_disk_temp, disk_2d_visc, acc_disk_temp_visc
input="acc_disk_visc"
specification="-acc_disk_visc"

use_mpi=0              # 1 = use MPI parallelization, 0 = single block (no MPI)
num_mpi_procs=4       # Number of MPI processes

show_progress="yes"

project_directory="fixed_tests"

repo_directory=$(pwd)

results_directory="${repo_directory}/results"
timestamp=$(date +"%Y%m%d-%H%M%S")
sample_directory="sample-${timestamp}${specification}"
data_directory="data"
materials_directory="materials"

# Check if we need to recompile
build_state_file=".build_state"
need_rebuild=false

# Calculate hash of the source file
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

# 1. Configuration and Build (only if needed)
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

# 2. Create working directories
mkdir -p "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"
mkdir -p "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}"

# 3. Copy files to materials directory
cp "${repo_directory}/inputs/hydro/athinput.${input}" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
cp "${repo_directory}/src/pgen/${problem}.cpp" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"

cp "${repo_directory}/scripts/vis1d.py" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
cp "${repo_directory}/scripts/vis2d.py" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
cp "${repo_directory}/scripts/vishst.py" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"
cp "${repo_directory}/scripts/visforces.py" "${results_directory}/${project_directory}/${sample_directory}/${materials_directory}/"

# 4. Change to the data directory
cd "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"

# 5. Clean up old files
rm -f *.tab *.athdf *.athdf.xdmf *.hst

# 6. Run Athena
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

# 7. List results
# ls

echo ""
echo "Results saved to: ${results_directory}/${project_directory}/${sample_directory}/${data_directory}"
echo "Materials saved to: ${results_directory}/${project_directory}/${sample_directory}/${materials_directory}"