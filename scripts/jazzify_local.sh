#!/usr/bin/env bash
set -euo pipefail

midi="$1"
repo="$(cd "$(dirname "$0")/.." && pwd)"
name="$(basename "${midi%.*}" | tr ' ' '_')"
cd "$repo"

for axis in raw nodesc; do
  styles=runs/base_v3_melodic/axes_heldout_both/styles.npz
  [ "$axis" = nodesc ] && styles=runs/base_v3_melodic/axes_heldout_both_nodesc/styles.npz
  out="jazzify_${name}_${axis}"
  .venv/bin/python scripts/jazzify.py --midi "$midi" \
    --ckpt runs/base_v3_melodic/best.pt --styles "$styles" \
    --data data/processed/train_v4.npz --out "$out" \
    --alpha-lo -3 --alpha-hi 3 --steps 7 --window -1
  .venv/bin/python scripts/render_audio.py "$out"
done
