#!/bin/bash
#
# Build bin/athena_visc_ring, then put bin/athena back the way it was.
#
# Athena++ compiles exactly one problem generator into the binary, so the LBP ring test
# needs its own. This builds it, renames it out of the way, and restores the production
# binary - leaving the tree in the state it started in, which matters because .build_state
# is what decides whether the next ./build.sh recompiles.
#
#   ./scripts/vv/build_ring.sh
#
set -e

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

jobs=$(( $(nproc) - 1 ))
main_problem="acc_disk_visc_vv"
main_configure="--prob ${main_problem} --coord=cylindrical"

echo "=== building visc_ring ==="
python3 configure.py --prob visc_ring --coord=cylindrical
make clean > /dev/null
make -j"${jobs}" > /dev/null
mv bin/athena bin/athena_visc_ring
echo "wrote bin/athena_visc_ring"

echo "=== restoring ${main_problem} ==="
python3 configure.py ${main_configure}
make clean > /dev/null
make -j"${jobs}" > /dev/null
{
    echo "PROBLEM=${main_problem}"
    echo "HASH=$(md5sum src/pgen/${main_problem}.cpp | cut -d' ' -f1)"
    echo "CONFIGURE=${main_configure}"
} > .build_state
echo "restored bin/athena"

ls -l bin/athena bin/athena_visc_ring
