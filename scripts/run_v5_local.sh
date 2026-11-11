#!/usr/bin/env bash
set -euo pipefail
RUN=runs/$1
DATA=data/processed/$2.npz
PY=.venv/bin/python
DESC="note_density pitch_range mean_interval rest_ratio syncopation chromaticism pc_entropy"
DEMO="--per-side 10 --alphas -3 -2 0 2 3 --bars 2 --bpm 80 --temperature 0.7 --top-p 0.92
      --key-bias 0 --register-span 12 --max-seed-range 12 --max-seed-chromaticism 0.12
      --match-candidates 24 --match-tol 3"

$PY scripts/analyse_axes.py --ckpt $RUN/best.pt --data $DATA --out $RUN/axes --field genre
$PY scripts/analyse_axes.py --ckpt $RUN/best.pt --data $DATA --out $RUN/axes_nodesc --field genre \
    --project-out-attr $DESC

for kind in raw nodesc; do
  styles=$RUN/axes/styles.npz
  [ $kind = nodesc ] && styles=$RUN/axes_nodesc/styles.npz
  out=$RUN/demos_${kind}_v10
  $PY scripts/build_demo.py --ckpt $RUN/best.pt --styles $styles --data $DATA --out $out $DEMO
  $PY scripts/render_audio.py $out --count-in 4
done
