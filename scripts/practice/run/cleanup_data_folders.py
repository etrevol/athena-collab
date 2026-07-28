#!/usr/bin/env python3
"""Remove the bulky `data/` folders of Athena++ runs, keeping the history files.

Raw `.tab` output dominates disk usage (this results tree reached 66 GB), while the
`.hst` history files that every conservation check relies on are under a megabyte in
total.  Both live in the same `data/` folder, so a plain `rm -rf data` throws the
cheap and irreplaceable away together with the expensive and reproducible.

By default this script therefore moves each `*.hst` up one level, next to its run
directory, before deleting `data/`.  Pass --delete-hst if you really want them gone.

Usage
-----
    cleanup_data_folders.py [ROOT ...] [options]

    ROOT              directory to clean (default: the current directory)

    -n, --dry-run     list what would happen, change nothing
    -y, --yes         do not ask for confirmation
    -x, --exclude PAT skip any path containing PAT (repeatable)
        --delete-hst  delete the history files too, instead of preserving them
        --no-recurse  only look one level below ROOT (the old behaviour)

Examples
--------
    # clean everything except the newest sweep, keeping the history files
    cleanup_data_folders.py results -x sweep-20260728-103026

    # see what would go, without touching anything
    cleanup_data_folders.py results -n
"""

import argparse
import shutil
import sys
from pathlib import Path


def human(nbytes):
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0


def dir_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def find_data_dirs(root, recurse=True, excludes=()):
    """Every directory named 'data' under root, skipping excluded paths."""
    it = root.rglob("data") if recurse else root.glob("*/data")
    out = []
    for d in it:
        if not d.is_dir():
            continue
        if any(pat in str(d) for pat in excludes):
            continue
        out.append(d)
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Remove Athena++ `data/` folders, preserving the .hst history files.")
    ap.add_argument("roots", nargs="*", metavar="ROOT",
                    help="directories to clean (default: current directory)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="list what would happen, change nothing")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="do not ask for confirmation")
    ap.add_argument("-x", "--exclude", action="append", default=[], metavar="PAT",
                    help="skip any path containing PAT (repeatable)")
    ap.add_argument("--delete-hst", action="store_true",
                    help="delete the history files too")
    ap.add_argument("--no-recurse", action="store_true",
                    help="only look one level below ROOT")
    args = ap.parse_args(argv)

    roots = [Path(r).resolve() for r in (args.roots or ["."])]
    for r in roots:
        if not r.is_dir():
            sys.exit(f"error: not a directory: {r}")

    data_dirs = []
    for r in roots:
        data_dirs += find_data_dirs(r, recurse=not args.no_recurse,
                                    excludes=args.exclude)
    if not data_dirs:
        print("No 'data' folders found.")
        return 0

    total = 0
    n_hst = 0
    print(f"Found {len(data_dirs)} 'data' folder(s):\n")
    for d in data_dirs:
        size = dir_size(d)
        total += size
        hst = sorted(d.glob("*.hst"))
        n_hst += len(hst)
        note = ""
        if hst:
            where = "DELETED" if args.delete_hst else f"kept in {d.parent.name}/"
            note = f"   [{len(hst)} .hst -> {where}]"
        print(f"  {d}  ({human(size)}){note}")

    print(f"\nTotal to free: {human(total)}")
    if n_hst:
        print(f"History files affected: {n_hst} "
              f"({'deleted' if args.delete_hst else 'moved one level up'})")

    if args.dry_run:
        print("\nDry run - nothing changed.")
        return 0

    if not args.yes:
        try:
            resp = input("\nProceed with deletion? [y/N]: ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("Operation cancelled.")
            return 1

    deleted = kept = failed = 0
    for d in data_dirs:
        try:
            if not args.delete_hst:
                for h in sorted(d.glob("*.hst")):
                    dest = d.parent / h.name
                    if dest.exists():           # never clobber an existing file
                        dest = d.parent / f"{d.parent.name}_{h.name}"
                    if not dest.exists():
                        shutil.move(str(h), str(dest))
                        kept += 1
            shutil.rmtree(d)
            deleted += 1
        except Exception as e:                  # report and carry on
            print(f"  FAILED {d}: {e}")
            failed += 1

    print(f"\nDeleted {deleted} folder(s), freed ~{human(total)}.")
    if kept:
        print(f"Preserved {kept} history file(s) next to their run directory.")
    if failed:
        print(f"{failed} folder(s) could not be removed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
