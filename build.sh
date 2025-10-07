#!/bin/bash
set -e  # stop the script immediately if any command fails

problem="pp_disk"
input="pp_disk"

project_directory="pp_disk"

results_directory="results"
timestamp=$(date +"%Y%m%d-%H%M%S")
sample_directory="sample-${timestamp}"
data_directory="data"

# 1. Configuration
python3 configure.py --prob "${problem}" --coord=cylindrical
# python3 configure.py --prob "${problem}" --coord=cartesian --coord=cylindrical

# 2. Build
make clean
make

# 3. Create working directories
mkdir -p "./${results_directory}/${project_directory}/${sample_directory}/${data_directory}"

# 4. Change to the data directory
cd "./${results_directory}/${project_directory}/${sample_directory}/${data_directory}"

# 5. Clean up old files
rm -f *.tab

# 6. Copy the input file
cp "../../../../inputs/hydro/athinput.${input}" .

# 7. Run Athena
../../../../bin/athena -i "athinput.${input}"

# 8. List results
ls