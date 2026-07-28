# Visualisation scripts

All four read **both** `.athdf` (HDF5) and `.tab`. The format is detected from what is in
the directory; when both are present `.athdf` wins, because it is one file per frame
instead of one per MeshBlock. Nothing about the figures changed in the move — same
layout, colours, colour maps and animations as the `.tab`-only versions.

| script | what it draws | reads |
|---|---|---|
| `vis1d.py` | radial profiles, 1D | out1 (`prim`) |
| `vis2d.py` | 2D maps, polar view, azimuthal and radial profiles, animations | out1 (`prim`) |
| `visforces.py` | radial force balance | out2 (`uov`) |
| `vishst.py` | history time series | `.hst` |
| `athena_data.py` | shared reader — not run directly | — |

## Getting HDF5 output

Configure Athena++ with HDF5 and ask the input file for it:

```bash
python3 configure.py --prob acc_disk_visc --coord=cylindrical -hdf5 \
        --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/serial
make clean && make -j$(nproc)
```

```
<output1>
file_type = hdf5
variable  = prim
dt        = 1.01227e-03
```

`make clean` is not optional here. The Athena++ Makefile does not track header
dependencies, so after changing any `configure.py` flag a plain `make` leaves stale object
files: the binary ends up mixing configurations, and the HDF5 metadata can report the
wrong coordinate system while the physics is built for another.

## Common arguments

```
--data_dir PATH     directory holding the output (default: ../data)
--format {athdf,tab}  force a format instead of auto-detecting
--output_id N       which <outputN> block to read
--title "..."       figure title, centred at the top
```

### `--title`

Overrides the title of every figure and animation the script produces. It is a format
string, so run metadata can go straight into it:

```bash
python3 vis2d.py --data_dir ../data --title "PP disk, α=0.01 — t={time:.4f}"
python3 vis1d.py --data_dir ../data --title "{base}, frame {frame}, cycle {cycle}"
python3 vishst.py --hst_file ../data/x.hst --title "PP disk — {label}"
```

| field | available in | meaning |
|---|---|---|
| `{time}` | vis1d, vis2d, visforces | simulation time of the frame |
| `{cycle}` | vis1d, vis2d, visforces | cycle number |
| `{frame}` | vis1d, vis2d, visforces | output frame index |
| `{base}` | vis1d, vis2d, visforces | `problem_id` from the file names |
| `{plot}` | vis2d | which panel set (`2d`, `polar`, `radial`, `azimuthal`) |
| `{label}`, `{var}`, `{file}` | vishst | quantity label, its column name, history file name |

A template that references a field which does not exist prints a warning naming the
available fields and falls back to the default title, rather than killing a long
animation run half way through.

## The `.tab` versions

The originals are kept unchanged in `../../practice-legacy/vis-tab/`. They are only
needed to reproduce an old figure exactly; for everything else the current scripts read
`.tab` too.
