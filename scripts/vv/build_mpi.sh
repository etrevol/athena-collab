#!/bin/bash
#
# Build bin/athena_mpi, then put bin/athena back the way it was.
#
# -mpi is a compile-time flag, so an MPI run cannot share the serial binary. This builds
# one, renames it out of the way, and restores the production binary - leaving the tree
# in the state it started in, which matters because .build_state is what decides whether
# the next ./build.sh recompiles.
#
#   ./scripts/vv/build_mpi.sh
#
set -e

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

if ! command -v mpicxx > /dev/null; then
    echo "mpicxx not found; the MPI cases will be skipped." >&2
    exit 1
fi

jobs=$(( $(nproc) - 1 ))
main_problem="acc_disk_visc_vv"
main_configure="--prob ${main_problem} --coord=cylindrical"

echo "=== building ${main_problem} with -mpi ==="
python3 configure.py ${main_configure} -mpi
make clean > /dev/null
make -j"${jobs}" > /dev/null
mv bin/athena bin/athena_mpi
echo "wrote bin/athena_mpi"

echo "=== restoring the serial ${main_problem} ==="
python3 configure.py ${main_configure}
make clean > /dev/null
make -j"${jobs}" > /dev/null
{
    echo "PROBLEM=${main_problem}"
    echo "HASH=$(md5sum src/pgen/${main_problem}.cpp | cut -d' ' -f1)"
    echo "CONFIGURE=${main_configure}"
} > .build_state
echo "restored bin/athena"

ls -l bin/athena bin/athena_mpi
