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

echo "ALL BUILDS OK"
ls -la bin/
