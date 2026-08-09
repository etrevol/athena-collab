#!/usr/bin/env bash
# Build every problem generator the vvkit studies need, each into its own binary.
#
# Athena++ compiles one problem generator per binary and its Makefile does not track
# header dependencies, so every configure change needs a full `make clean`. This runs in
# the artem-vvkit worktree, which has its own obj/ and bin/ -- the main checkout's
# bin/athena and .build_state are never touched.
set -e
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
jobs=$(( $(nproc) - 1 ))

build() {  # build <prob> <output-binary-name>
    echo "=== $1 -> bin/$2 ==="
    python3 configure.py --prob "$1" --coord=cylindrical
    make clean > /dev/null
    make -j"${jobs}" 2>&1 | tail -3
    mv bin/athena "bin/$2"
    echo "=== bin/$2 done ==="
}

build acc_disk_visc_vv athena_acc_disk
build visc_ring        athena_visc_ring
build acc_disk_mms     athena_mms

echo "ALL BUILDS OK"
ls -la bin/

# Restore .build_state so ./build.sh does not think the tree is stale. Building here
# reconfigures the shared tree, and build.sh keys its rebuild decision on the recorded
# problem, source hash and configure line -- leaving those describing the last study
# binary would make the next production run rebuild for no reason, or worse, believe a
# rebuild had already happened.
main_problem="acc_disk_visc_vv"
main_configure="--prob ${main_problem} --coord=cylindrical"
echo "=== restoring ${main_problem} build state ==="
python3 configure.py ${main_configure} > /dev/null
make clean > /dev/null
make -j"${jobs}" > /dev/null 2>&1
{
  echo "PROBLEM=${main_problem}"
  echo "HASH=$(md5sum "src/pgen/${main_problem}.cpp" | cut -d' ' -f1)"
  echo "CONFIGURE=${main_configure}"
} > .build_state
echo "=== bin/athena and .build_state back on ${main_problem} ==="
