#!/usr/bin/env bash
set -euo pipefail
S=${1:-}
PY=.venv/bin/python
RUN=base_v5$S
T=${S//_/-}

apply_v100() {
  kubectl get "job/mvae-${1//_/-}" >/dev/null 2>&1 && return 0
  $PY cluster/scripts/submit.py "$1" --dry-run | sed -e 's/gpu: a100/gpu: v100/' | kubectl apply -f -
}
wait_job() {
  kubectl wait --for=condition=complete "job/mvae-$1" --timeout=3000s
}

wait_job "base-v5$T"
apply_v100 "axes_v5$S"
apply_v100 "axes_v5${S}_nodesc"
wait_job "axes-v5$T"
wait_job "axes-v5$T-nodesc"

apply_v100 "demo_raw_v10$S"
apply_v100 "demo_nodesc_v10$S"
wait_job "demo-raw-v10$T"
wait_job "demo-nodesc-v10$T"

for kind in raw nodesc; do
  local_dir="demos_${kind}_v10$S"
  [ -d "${local_dir}_audio" ] && continue
  rm -rf "$local_dir"
  kubectl cp "mvae-helper:/storage/runs/$RUN/demos_${kind}_v10" "$local_dir" 2>&1 | grep -v "Unknown stream" || true
  $PY scripts/render_audio.py "$local_dir" --count-in 4
done
for a in axes axes_nodesc; do
  mkdir -p "runs/$RUN/$a"
  kubectl cp "mvae-helper:/storage/runs/$RUN/$a/axis_report.json" "runs/$RUN/$a/axis_report.json" 2>&1 | grep -v "Unknown stream" || true
done
echo "pipeline $RUN done"
