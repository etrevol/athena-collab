#!/usr/bin/env python3
"""Collect a whole Athena++ run into one HDF5 file.

Athena++ writes one file per output time (and, for `.tab`, one per MeshBlock on top
of that), which is fine for the code and awkward for a human: a 100-orbit run is
thousands of files that no viewer will show you as a single object.

This walks a run directory, reads every frame through the same reader the plotting
scripts use, and writes a single `.h5` with time as the leading axis:

    /                       attrs: problem_id, source_format, coordinates, n_frames, ...
    /time                   (nt,)            simulation time of each frame
    /cycle                  (nt,)            cycle number of each frame
    /x1v, /x2v              (nx1,), (nx2,)   cell-centre coordinates
    /out1/rho               (nt, nx2, nx1)   one dataset per variable, per output id
    /out1/press             ...
    /out2/f_grav            ...              user output variables, if present
    /hst/<column>           (n,)             the history file, if present

The grid is written once, not once per frame, which is where most of the size goes.
Datasets are chunked per frame and gzip-compressed, so the result is typically far
smaller than the input and still opens instantly in a viewer.

The source can be `.tab` or `.athdf` - whichever the run produced. Reading goes
through scripts/practice/vis/athena_data.py, so a run made before the HDF5 work and
one made after collapse into exactly the same layout.

Usage
-----
    collect_hdf5.py RUNDIR [-o OUT.h5] [--output-ids 1 2] [--no-compress]
                           [--first N] [--last N] [--stride N] [--dry-run]

    RUNDIR may be the run directory itself or its data/ subdirectory.

Examples
--------
    collect_hdf5.py results/runs/sample-20260728-175850-acc_disk_visc
    collect_hdf5.py .../data -o /tmp/disk.h5 --stride 10
"""

import argparse
import os
import sys
import time as _time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "vis")))

import athena_data as ad  # noqa: E402

try:
    import h5py
except ImportError:                                            # pragma: no cover
    sys.exit("error: h5py is required.  pip install --user h5py")


def human(n):
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0


def resolve_data_dir(path):
    """Accept either the run directory or its data/ subdirectory."""
    if os.path.isdir(os.path.join(path, "data")):
        return os.path.join(path, "data"), path
    return path, os.path.dirname(os.path.abspath(path))


def find_hst(data_dir, run_dir):
    for d in (data_dir, run_dir):
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.endswith(".hst"):
                return os.path.join(d, n)
    return None


def collect(run_path, out_path=None, output_ids=(1, 2), compress=True,
            first=None, last=None, stride=1, dry_run=False):
    data_dir, run_dir = resolve_data_dir(os.path.abspath(run_path))
    if not os.path.isdir(data_dir):
        sys.exit(f"error: no such directory: {data_dir}")

    # which output ids actually exist
    found = {}
    for oid in output_ids:
        try:
            found[oid] = ad.discover(data_dir, output_id=oid)
        except FileNotFoundError:
            continue
    if not found:
        sys.exit(f"error: no .athdf or .tab output found in {data_dir}")

    if out_path is None:
        out_path = os.path.join(run_dir, os.path.basename(run_dir.rstrip("/")) + ".h5")

    # frames present in every requested output id, so the time axis is shared
    common = None
    for info in found.values():
        s = set(info["frames"])
        common = s if common is None else (common & s)
    frames = sorted(common)
    if first is not None:
        frames = [f for f in frames if f >= first]
    if last is not None:
        frames = [f for f in frames if f <= last]
    frames = frames[::max(1, stride)]
    if not frames:
        sys.exit("error: no frames left after filtering")

    print(f"Run      : {run_dir}")
    for oid, info in sorted(found.items()):
        print(f"  out{oid}   : {ad.describe(info)}")
    print(f"Frames   : {len(frames)} of {len(common)} "
          f"({frames[0]}..{frames[-1]}, stride {stride})")
    print(f"Output   : {out_path}")

    src_bytes = sum(os.path.getsize(p)
                    for info in found.values()
                    for f in frames for p in info["frames"][f])
    print(f"Source   : {human(src_bytes)}")

    if dry_run:
        print("\nDry run - nothing written.")
        return out_path

    t0 = _time.time()
    kw = dict(compression="gzip", compression_opts=4, shuffle=True) if compress else {}

    with h5py.File(out_path, "w") as h5:
        times, cycles = [], []
        grid_written = False
        coords = None

        for oid, info in sorted(found.items()):
            grp = h5.create_group(f"out{oid}")
            dsets = {}

            for i, fr in enumerate(frames):
                d = ad.read_frame(info["frames"][fr], info["format"])

                if oid == min(found):                 # time axis from the first id
                    times.append(d["time"])
                    cycles.append(d["cycle"])

                if not grid_written:
                    x1 = np.asarray(d["x1v"]).ravel()
                    h5.create_dataset("x1v", data=x1)
                    x2 = d.get("x2v")
                    if x2 is not None:
                        h5.create_dataset("x2v", data=np.asarray(x2).ravel())
                    coords = d.get("coordinates")
                    grid_written = True

                for k, v in d.items():
                    if k in ("time", "cycle", "x1v", "x2v", "coordinates"):
                        continue
                    arr = np.asarray(v, dtype=np.float32)
                    if k not in dsets:
                        dsets[k] = grp.create_dataset(
                            k, shape=(len(frames),) + arr.shape,
                            dtype=np.float32,
                            chunks=(1,) + arr.shape, **kw)
                    dsets[k][i] = arr

                if (i + 1) % 25 == 0 or i + 1 == len(frames):
                    print(f"\r  out{oid}: {i + 1}/{len(frames)} frames", end="", flush=True)
            print()

        h5.create_dataset("time", data=np.asarray(times, dtype=np.float64))
        h5.create_dataset("cycle", data=np.asarray(cycles, dtype=np.int64))
        h5.create_dataset("frame", data=np.asarray(frames, dtype=np.int64))

        hst = find_hst(data_dir, run_dir)
        if hst:
            try:
                import athena_read
                hd = athena_read.hst(hst)
                g = h5.create_group("hst")
                for k, v in hd.items():
                    g.create_dataset(k.replace("/", "_"), data=np.asarray(v))
                print(f"  hst  : {os.path.basename(hst)} "
                      f"({len(next(iter(hd.values())))} rows, {len(hd)} columns)")
            except Exception as exc:                   # a bad .hst must not lose the rest
                print(f"  hst  : skipped ({exc})")

        first_info = found[min(found)]
        h5.attrs["problem_id"] = first_info["base"]
        h5.attrs["source_format"] = first_info["format"]
        h5.attrs["source_dir"] = data_dir
        h5.attrs["n_frames"] = len(frames)
        h5.attrs["output_ids"] = sorted(found)
        h5.attrs["created"] = _time.strftime("%Y-%m-%d %H:%M:%S")
        if coords:
            h5.attrs["coordinates"] = coords

    dst = os.path.getsize(out_path)
    print(f"\nWrote {human(dst)} in {_time.time() - t0:.1f} s "
          f"({src_bytes / dst:.1f}x smaller than the source)"
          if dst else "")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Collect an Athena++ run into a single HDF5 file.")
    ap.add_argument("rundir", help="run directory, or its data/ subdirectory")
    ap.add_argument("-o", "--output", default=None, help="output .h5 (default: <run>.h5)")
    ap.add_argument("--output-ids", type=int, nargs="+", default=[1, 2],
                    metavar="N", help="output blocks to collect (default: 1 2)")
    ap.add_argument("--no-compress", action="store_true", help="store uncompressed")
    ap.add_argument("--first", type=int, default=None, help="first frame to include")
    ap.add_argument("--last", type=int, default=None, help="last frame to include")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth frame")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="report what would be collected, write nothing")
    a = ap.parse_args(argv)
    collect(a.rundir, a.output, a.output_ids, not a.no_compress,
            a.first, a.last, a.stride, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
