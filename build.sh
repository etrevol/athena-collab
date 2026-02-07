#!/bin/bash
set -e  # stop the script immediately if any command fails

problem="pp_disk_2d_internal_visc"
input="pp_disk_2d_internal_visc"
specification="-athena_internal_viscosity"

project_directory="pp_disk_viscosity"

results_directory="/home/etrevol/results"
timestamp=$(date +"%Y%m%d-%H%M%S")
sample_directory="sample-${timestamp}${specification}"
data_directory="data"

repo_directory=$(pwd)

# 1. Configuration
python3 configure.py --prob "${problem}" --coord=cylindrical
# python3 configure.py --prob "${problem}" --coord=cartesian --coord=cylindrical

# 2. Build
make clean
make -j$(($(nproc) - 1))

# 3. Create working directories
mkdir -p "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"

# 4. Change to the data directory
cd "${results_directory}/${project_directory}/${sample_directory}/${data_directory}"

# 5. Clean up old files
rm -f *.tab

# 6. Copy the input file
cp "${repo_directory}/inputs/hydro/athinput.${input}" .
cp "${repo_directory}/src/pgen/${problem}.cpp" .

# 7. Copy the vizualizer script
cp "${repo_directory}/scripts/a-visualizer.py" .
cp "${repo_directory}/scripts/a-visualizer_2d.py" .
# cp "${repo_directory}/scripts/plot_history.py" .
# cp "${repo_directory}/scripts/a-visualizer-flux.py" .

# 8. Run Athena
"${repo_directory}/bin/athena" -i "athinput.${input}"

# 9. List results
ls

echo ""
echo "Results saved to: ${results_directory}/${project_directory}/${sample_directory}/${data_directory}"