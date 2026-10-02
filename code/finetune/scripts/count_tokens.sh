#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=${1:-/workspace/dataset/music}
LOG_DIR=./count_token_logs
if [[ ! -d "$PARENT_DIR" ]]; then
    echo "Dataset directory does not exist: $PARENT_DIR" >&2
    exit 1
fi
mkdir -p -- "$LOG_DIR"

pids=()
while IFS= read -r -d '' mmap_path; do
    echo "Checking mmap file: $mmap_path"
    subdir=${mmap_path#"$PARENT_DIR"/}
    subdir=${subdir//\//_}
    nohup python "$SCRIPT_DIR/../tools/count_mmap_token.py" \
        --mmap_path "$mmap_path" > "$LOG_DIR/count.$subdir.log" 2>&1 &
    pids+=("$!")
done < <(find "$PARENT_DIR" -name '*.bin' -type f -print0)

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "Token counting failed; see $LOG_DIR for details." >&2
    exit "$status"
fi
echo "Token counting finished."
