# Nadeshiko Dev Tools

`nadeshiko-dev-tools` is a collection of CLI utilities for Nadeshiko media workflows.

## Included Tools

- `media-sub-splitter`: Split anime episodes into subtitle-aligned media segments.
- `assets-uploader`: Upload generated media metadata and assets to Nadeshiko environments.
- `nsfw-tagger`: Classify segment screenshots for content rating (SAFE/SUGGESTIVE/QUESTIONABLE/EXPLICIT) and gore detection.
- `pitch-extractor`: Extract F0 pitch contours from audio segments with optional Demucs vocal separation.

## Setup

```bash
uv sync
cp .env.example .env
```

Configure the values in `.env` before running upload workflows.

## Tool Usage

### `media-sub-splitter`

```bash
uv run media-sub-splitter <input_folder> <output_folder> [OPTIONS]
```

Examples:

```bash
# Basic usage with DeepL translation
uv run media-sub-splitter ./anime ./output -t YOUR_DEEPL_TOKEN

# Dry run without generating segments
uv run media-sub-splitter ./anime ./output --dry-run

# Process selected episodes in parallel with verbose logs
uv run media-sub-splitter ./anime ./output -e 1,3,5 --parallel --verbose

# Skip ffsubsync subtitle sync
uv run media-sub-splitter ./anime ./output --no-sync
```

Common options:

- `-t, --token TOKEN`: DeepL token for subtitle translation.
- `-v, --verbose`: Enable debug output.
- `-d, --dry-run`: Parse subtitles only, without writing segments.
- `-x, --x`: Remove extra punctuation symbols like `・`.
- `-p, --parallel`: Process episodes in parallel.
- `-e, --episodes`: Comma-separated episode list (example: `1,3,5`).
- `--no-sync`: Skip subtitle syncing with `ffsubsync`.

### `assets-uploader`

```bash
uv run assets-uploader <media_folder> [OPTIONS]
```

Examples:

```bash
# Dry run against local API + local storage (default mode is dry-run)
uv run assets-uploader ./output/12345 --target local --storage local

# Apply upload to dev API + local storage
uv run assets-uploader ./output/12345 --target dev --storage local --apply

# Production upload with R2 storage and actual file upload
uv run assets-uploader ./output/12345 --target prod --storage r2 --apply --upload-r2

# Upload only one episode
uv run assets-uploader ./output/12345 --target dev --storage local --episode 1 --apply
```

Common options:

- `--target {local,dev,prod}`: API target environment.
- Use `--target dev` (not `--dev`).
- `--storage {local,r2}`: Storage backend to use in API metadata.
- `--episode N`: Upload a single episode number.
- `--apply`: Execute changes (without this, runs in dry-run mode).
- `--upload-r2` (alias: `--upload-to-r2`): Actually upload files to R2 (only valid with `--storage r2`).
- `--update-info`: Update media metadata only.

### `nsfw-tagger`

Classifies anime segment screenshots using WaifuDiffusion Tagger v3 (Danbooru tag predictor). Maps Danbooru ratings to content ratings and detects gore via content tags. Requires GPU dependencies for reasonable performance.

```bash
# Install GPU dependencies (requires CUDA)
uv pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/

# Or CPU-only (much slower)
uv sync --extra nsfw
```

```bash
# Classify all images in the processed archive
uv run nsfw-tagger classify

# Classify specific anime by AniList ID
uv run nsfw-tagger classify --media 100077 154587

# Resume interrupted classification (default behavior, skips already-done media)
uv run nsfw-tagger classify

# Force reclassify media that already have results
uv run nsfw-tagger classify --media 100077 --no-resume

# Review results for a specific anime
uv run nsfw-tagger review --media 100077

# Review overall summary
uv run nsfw-tagger review

# Export SQL update file for database migration
uv run nsfw-tagger export

# Export SQL for specific media only
uv run nsfw-tagger export --media 100077
```

Results are stored in `<archive>/_nsfw_results/` as per-media JSON files. The `export` command generates a SQL file that can be applied to the database to update `content_rating` and `content_analysis` columns.

Content rating mapping: `general` -> SAFE, `sensitive` -> SUGGESTIVE, `questionable` -> QUESTIONABLE, `explicit` -> EXPLICIT. Gore is detected separately via Danbooru violence tags (blood, gore, guro, etc.) and stored as a boolean flag.

### `pitch-extractor`

Extracts F0 pitch contours from audio segments using Parselmouth (Praat). Optionally separates vocals from BGM using Demucs before extraction for cleaner results.

```bash
# Install dependencies
uv sync --extra pitch
```

```bash
# Extract pitch contours for all media (with Demucs vocal separation)
uv run pitch-extractor extract

# Extract for specific anime
uv run pitch-extractor extract --media 100077 154587

# Extract without vocal separation (faster, less accurate)
uv run pitch-extractor extract --no-separation

# Save isolated vocals for debugging
uv run pitch-extractor extract --media 100077 --save-vocals

# Show extraction coverage statistics
uv run pitch-extractor stats

# Stats for specific media
uv run pitch-extractor stats --media 100077
```

Results are stored as `_pitch.json` files per episode in the archive directory.

## Tests

```bash
uv run pytest
```

## Linting

```bash
uv run ruff check .
uv run ruff check --fix . && uv run ruff format .
```
