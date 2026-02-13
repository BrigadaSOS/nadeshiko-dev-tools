#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/mnt/storage/nade-processed"
TARGET="prod"

if [ "${1:-}" = "--reset-history" ]; then
    first_dir=$(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d | head -1)
    if [ -n "$first_dir" ]; then
        uv run assets-uploader --target "$TARGET" --reset-history "$first_dir"
    else
        echo "No media folders found in $BASE_DIR"
        exit 1
    fi
    shift
fi

failed=0

for dir in "$BASE_DIR"/*/; do
    echo "=== Uploading: $(basename "$dir") ==="
    uv run assets-uploader --target "$TARGET" --storage r2 --apply --yes "$dir" || {
        echo "--- FAILED: $(basename "$dir") (will retry next run) ---"
        failed=$((failed + 1))
    }
done

if [ "$failed" -gt 0 ]; then
    echo "=== $failed media failed (check _upload_history_${TARGET}.json for details) ==="
fi
