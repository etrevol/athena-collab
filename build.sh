#!/bin/bash
set -e  # stop the script immediately if any command fails

problem="acc_disk_visc" # acc_disk_grav, acc_disk_temp, disk_2d_visc, acc_disk_temp_visc
input="acc_disk_visc"
specification="-acc_disk_visc"

use_mpi=0              # 1 = use MPI parallelization, 0 = single block (no MPI)
num_mpi_procs=4       # Number of MPI processes

use_hdf5=1             # 1 = build with HDF5 output (.athdf), 0 = tab/vtk only
                       # The visualization scripts read .athdf and .tab alike, but the
                       # input file now asks for hdf5, so this needs to stay on.

show_progress="yes"

project_directory="runs"

repo_directory=$(pwd)

# Locate the HDF5 headers. Debian/Ubuntu keep the serial build in a subdirectory;
# other distributions put it straight under /usr.
hdf5_flags=""
if [ "$use_hdf5" -eq 1 ]; then
    for candidate in /usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib64/hdf5 /usr/local /usr; do
        if [ -f "${candidate}/include/hdf5.h" ] || [ -f "/usr/include/hdf5/serial/hdf5.h" ]; then
            if [ -d "${candidate}/include" ]; then
                hdf5_flags="-hdf5 --hdf5_path=${candidate}"
            else
                hdf5_flags="-hdf5"
            fi
            break
        fi
    done
    if [ -z "${hdf5_flags}" ]; then
        echo "Error: use_hdf5=1 but no HDF5 headers found."
        echo "       Install them (Debian/Ubuntu: sudo apt install libhdf5-dev),"
        echo "       or set use_hdf5=0 and switch the output blocks in the input"
        echo "       file back to file_type = tab."
        exit 1
    fi
fi

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
    saved_hdf5=$(grep "^HDF5=" "${build_state_file}" | cut -d'=' -f2)

    # The configure flags have to be part of the state. Athena++'s Makefile does not
    # track header dependencies, so flipping a flag without a rebuild leaves stale
    # object files and a binary that mixes two configurations.
    if [ "${saved_problem}" != "${problem}" ] || [ "${saved_hash}" != "${current_hash}" ] \
       || [ "${saved_hdf5}" != "${use_hdf5}" ]; then
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
    if [ "$use_hdf5" -eq 1 ]; then
        echo "HDF5 output: ENABLED (${hdf5_flags})"
    else
        echo "HDF5 output: DISABLED"
    fi

    if [ "$use_mpi" -eq 1 ]; then
        echo "MPI parallelization: ENABLED"
        python3 configure.py --prob "${problem}" --coord=cylindrical -mpi ${hdf5_flags}
    else
        echo "MPI parallelization: DISABLED (single block mode)"
        python3 configure.py --prob "${problem}" --coord=cylindrical ${hdf5_flags}
    fi

    echo "Building..."
    make clean
    make -j$(($(nproc) - 1))
    
    # Save current state
    echo "PROBLEM=${problem}" > "${build_state_file}"
    echo "HASH=${current_hash}" >> "${build_state_file}"
    echo "HDF5=${use_hdf5}" >> "${build_state_file}"
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