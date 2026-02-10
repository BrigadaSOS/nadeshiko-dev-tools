#!/usr/bin/env bash

# Upload all media folders to production and move to nade-processed
# Usage: ./scripts/upload-all.sh [--dry-run] [--output-dir PATH]

set -euo pipefail

# Default paths
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/storage/nade-toprocess}"
DRY_RUN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--output-dir PATH]"
      echo ""
      echo "Options:"
      echo "  --dry-run       Preview without actually uploading/moving"
      echo "  --output-dir    Path to processed output folders (default: /mnt/storage/nade-toprocess)"
      echo ""
      echo "Environment variables:"
      echo "  OUTPUT_DIR      Override default output directory"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate directory exists
if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Error: Output directory does not exist: $OUTPUT_DIR"
  exit 1
fi

# Capture all folders at the start (not new ones added during execution)
mapfile -t folders < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

if [[ ${#folders[@]} -eq 0 ]]; then
  echo "No folders found in $OUTPUT_DIR"
  exit 0
fi

echo "Found ${#folders[@]} folders to process"
echo "Output directory: $OUTPUT_DIR"

if [[ -n "$DRY_RUN" ]]; then
  echo ""
  echo "DRY RUN MODE - no changes will be made"
  echo ""
fi

# Process each folder
for media_id in "${folders[@]}"; do
  echo ""
  echo "=========================================="
  echo "Processing: $media_id"
  echo "=========================================="

  output_path="$OUTPUT_DIR/$media_id"

  # Skip if output folder doesn't have _info.json
  if [[ ! -f "$output_path/_info.json" ]]; then
    echo "Skipping: No _info.json found in $output_path"
    continue
  fi

  # Build commands
  if [[ -n "$DRY_RUN" ]]; then
    upload_cmd="uv run ossets-uploader \"$output_path\" --target prod --storage r2"
    move_cmd="uv run post-upload \"$output_path\""
  else
    upload_cmd="uv run assets-uploader \"$output_path\" --target prod --storage r2 --apply --upload-r2"
    move_cmd="uv run post-upload \"$output_path\" --apply"
  fi

  # Upload to production
  echo "Running: $upload_cmd"
  if eval "$upload_cmd"; then
    echo "Upload successful for $media_id"

    # Move to nade-processed
    echo "Running: $move_cmd"
    if eval "$move_cmd"; then
      echo "Moved $media_id to nade-processed"
    else
      echo "Warning: Failed to move $media_id (but upload succeeded)"
    fi
  else
    echo "Error: Upload failed for $media_id - folder NOT moved"
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "All done! Processed ${#folders[@]} folders"
echo "=========================================="
