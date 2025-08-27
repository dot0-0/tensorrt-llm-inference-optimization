#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 <command...>"
  exit 1
fi
TMP=$(mktemp)
# sample every 200ms while the child runs
( nvidia-smi --query-gpu=timestamp,index,name,memory.used --format=csv -lms 200 > "$TMP" ) &
SAMPLER=$!
set +e
"${@}"
STATUS=$?
set -e
kill $SAMPLER || true
python scripts/summarize_nvidia_smi.py --csv "$TMP" --out data/peak_memory.csv
rm -f "$TMP"
exit $STATUS

