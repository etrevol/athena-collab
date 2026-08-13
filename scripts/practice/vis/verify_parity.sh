#!/usr/bin/env bash
# ==============================================================================
#  verify_parity.sh - does athvis.py still render what the four scripts render?
# ==============================================================================
#
#  USAGE:
#    bash scripts/practice/vis/verify_parity.sh <run>/data [workdir]
#
#  Renders every mode of every command through both toolchains and compares the
#  images pixel by pixel; animations are compared on one extracted frame. Exit
#  status is 1 if anything differs.
#
#  Reading the two implementations does NOT catch drift - the same figure can be
#  built two ways, and a divergence shows up only in the output. Both real
#  divergences found so far were invisible in the source: `1d --mode profiles`
#  drawing vis2d's radial figure instead of vis1d's, and the polar animation
#  running its lower-row titles through the upper row's 270-degree tick label.
#
#  Needs ffmpeg for the animations. The run needs a uov output (id 2) and a .hst
#  for the forces and hst comparisons; missing ones are reported, not skipped.
#
# ==============================================================================
set -u

DATA=${1:-}
if [[ -z "$DATA" || ! -d "$DATA" ]]; then
  echo "usage: $0 <run>/data [workdir]" >&2
  exit 2
fi
DATA=$(cd "$DATA" && pwd)
B=${2:-$(mktemp -d)}
V=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Last frame, and a ten-frame window for the animations. Both are derived so the
# script works on any run length, not just the one it was written against.
mapfile -t FRAMES < <(ls "$DATA"/*.out1.*.tab 2>/dev/null |
                      sed -E 's/.*\.out1\.0*([0-9]+)\.tab/\1/' | sort -n | uniq)
if [[ ${#FRAMES[@]} -eq 0 ]]; then
  echo "no out1 .tab frames in $DATA" >&2
  exit 2
fi
LAST=${FRAMES[-1]}
FIRST_W=${FRAMES[$(( ${#FRAMES[@]} > 10 ? ${#FRAMES[@]} - 10 : 0 ))]}
HST=$(ls "$DATA"/*.hst 2>/dev/null | head -n1)

rm -rf "$B/old" "$B/new"; mkdir -p "$B/old" "$B/new"
F="--frame $LAST"
W="--start_frame $FIRST_W --end_frame $LAST"
VEC="--add-vectors --vec_frame full"

echo "run:     $DATA"
echo "frames:  ${#FRAMES[@]} (${FRAMES[0]}..$LAST), window $FIRST_W..$LAST"
echo "workdir: $B"
echo "rendering..."

run() {  # run <script-args...> ; stdout is noise, failures are caught by the diff
  python3 "$@" >/dev/null 2>&1
}

for m in heatmaps radial polar azimuthal; do
  run "$V/vis2d.py"     --mode $m $F --data_dir "$DATA" --output_dir "$B/old"
  run "$V/athvis.py" 2d --mode $m $F --data_dir "$DATA" --output_dir "$B/new"
done
run "$V/vis2d.py"     --mode polar $F --normalize azimuthal --data_dir "$DATA" --output_dir "$B/old"
run "$V/athvis.py" 2d --mode polar $F --normalize azimuthal --data_dir "$DATA" --output_dir "$B/new"
for m in heatmaps polar; do
  run "$V/vis2d.py"     --mode $m $F $VEC --data_dir "$DATA" --output_dir "$B/old"
  run "$V/athvis.py" 2d --mode $m $F $VEC --data_dir "$DATA" --output_dir "$B/new"
done
for m in animation polar_animation; do
  run "$V/vis2d.py"     --mode $m $W --data_dir "$DATA" --output_dir "$B/old"
  run "$V/athvis.py" 2d --mode $m $W --data_dir "$DATA" --output_dir "$B/new"
done
run "$V/vis2d.py"     --mode vector_animation $W $VEC --data_dir "$DATA" --output_dir "$B/old"
run "$V/athvis.py" 2d --mode vector_animation $W $VEC --data_dir "$DATA" --output_dir "$B/new"

for m in profiles animation; do
  run "$V/vis1d.py"     --mode $m $F $W --data_dir "$DATA" --output_dir "$B/old"
  run "$V/athvis.py" 1d --mode $m $F $W --data_dir "$DATA" --output_dir "$B/new"
done

if [[ -n "$HST" ]]; then
  run "$V/vishst.py"     --hst_file "$HST" --mode all --output_dir "$B/old"
  run "$V/athvis.py" hst --data_dir "$DATA" --mode all --output_dir "$B/new"
fi

for m in frame sum animation sum_animation; do
  run "$V/visforces.py"     --mode $m $F $W --data_dir "$DATA" --output_dir "$B/old"
  run "$V/athvis.py" forces --mode $m $F $W --data_dir "$DATA" --output_dir "$B/new"
done

python3 - "$B" <<'PY'
import os, subprocess, sys
import numpy as np
import matplotlib.image as mpimg

B = sys.argv[1]

def as_png(path):
    """Animations are compared on one frame; a whole-video diff would only tell
    us which frame differs, which the still comparisons already localise."""
    if not path.endswith(".mp4"):
        return path
    out = path[:-4] + "_f4.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path,
                    "-vf", r"select=eq(n\,4)", "-vframes", "1", out], check=True)
    return out

def counterpart(name):
    """athvis drops the redundant '_frame' from the still-figure filenames."""
    yield name
    yield name.replace("_frame_", "_")
    yield name.replace("_profile_frame_", "_")

old = sorted(os.listdir(f"{B}/old"))
bad = 0
for f in old:
    match = next((c for c in counterpart(f) if os.path.exists(f"{B}/new/{c}")), None)
    if match is None:
        print(f"  {f:38s} MISSING in athvis"); bad += 1; continue
    a = mpimg.imread(as_png(f"{B}/old/{f}"))[..., :3]
    b = mpimg.imread(as_png(f"{B}/new/{match}"))[..., :3]
    if a.shape != b.shape:
        print(f"  {f:38s} SHAPE {a.shape[:2]} vs {b.shape[:2]}"); bad += 1; continue
    # 0.02 in [0,1] is well below a visible step but above PNG rounding
    d = 100 * (np.abs(a - b).max(axis=2) > 0.02).mean()
    print(f"  {f:38s} {'identical' if d == 0 else f'diff {d:.3f}%'}")
    bad += d > 0

print(f"\n{len(old)} outputs compared, {bad} not identical")
sys.exit(1 if bad else 0)
PY
