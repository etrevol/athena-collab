#!/usr/bin/env python3
"""
=============================================================================
ATHENA++ VISUALIZATION - single entry point
=============================================================================

A thin dispatcher over the four standalone scripts in this directory. It does not
reimplement anything - `python3 vis.py 2d ...` just calls `vis2d.py`'s own argument
parser with the arguments after "2d". Every flag documented in vis1d.py / vis2d.py /
vishst.py / visforces.py works unchanged here; --help on a subcommand prints that
script's own --help.

This exists alongside the four scripts, not instead of them: they keep working
exactly as before, standalone, in a run directory or from the repo. Use whichever
is more convenient - `python3 vis2d.py --mode polar` and `python3 vis.py 2d --mode
polar` do the same thing.

USAGE:
    python3 vis.py <tool> [tool arguments...]
    python3 vis.py                 # lists the tools
    python3 vis.py <tool> --help   # that tool's own --help

TOOLS:
    2d       vis2d.py      heatmaps, polar view, radial/azimuthal profiles,
                            animations, the vector field overlay
    1d       vis1d.py      radial profiles + their time evolution
    hst      vishst.py     .hst history file time series (disk_mass, mdot_in, ...)
    forces   visforces.py  radial force balance (f_grav, f_centr, f_press, f_sum)

EXAMPLES:
    python3 vis.py 2d --mode polar --add-vectors --vec_frame full
    python3 vis.py 2d --mode vector_animation --vec_comp radial --subsample 20
    python3 vis.py 1d --mode profiles --frame 500
    python3 vis.py hst --mode all
    python3 vis.py forces --mode animation --log

    python3 vis.py 2d --help       # vis2d.py's own, full --help text

=============================================================================
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

TOOLS = {
    "2d":     ("vis2d",      "heatmaps, polar view, animations, vector field"),
    "1d":     ("vis1d",      "radial profiles + evolution animation"),
    "hst":    ("vishst",     ".hst history file time series"),
    "forces": ("visforces",  "radial force balance"),
}


def _usage():
    print(__doc__.strip("\n"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return 0 if argv else 1

    tool, rest = argv[0], argv[1:]
    if tool not in TOOLS:
        print(f"Unknown tool '{tool}'. Choose one of: {', '.join(TOOLS)}\n")
        _usage()
        return 1

    modname, _ = TOOLS[tool]
    # argparse's ArgumentParser captures prog=basename(argv[0]) at construction, i.e.
    # at import time - so the swap has to happen before the first import, or --help
    # inside a subcommand prints "usage: vis.py ..." instead of the tool it belongs
    # to. sys.modules is checked first since a second dispatch in the same process
    # (as the tests below do) must not re-run the module's import-time argparse setup.
    old_argv0, sys.argv[0] = sys.argv[0], f"{modname}.py"
    try:
        module = sys.modules.get(modname) or __import__(modname)
        # Each script's own main() reads its own argparse result. Two shapes exist:
        # vishst.main(argv) parses and runs in one call; the other three separate
        # _setup(argv) (parse + discover data) from main() (do the work).
        if hasattr(module, "_setup"):
            module._setup(rest)
            return module.main() or 0
        return module.main(rest) or 0
    finally:
        sys.argv[0] = old_argv0


if __name__ == "__main__":
    sys.exit(main())
