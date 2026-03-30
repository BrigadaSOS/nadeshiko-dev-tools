# Nadeshiko Dev Tools

CLI utilities for Nadeshiko media workflows. The typical flow is:

1. **Split** anime episodes into segments with `media-sub-splitter`
2. **Upload** the generated segments to Nadeshiko with `assets-uploader`

## Setup

```bash
uv sync
uv run python -m unidic download
cp .env.example .env
```

Configure the values in `.env` before running upload workflows.

---

## Step 1 — Split: `media-sub-splitter`

Takes a folder of `.mkv` files and subtitle files, maps them to AniList media, and generates audio/video/screenshot segments with multi-language subtitles.

```
uv run media-sub-splitter <input_folder> <output_folder> [OPTIONS]
```

The input folder should contain one subfolder per anime, each with `.mkv` episode files. The tool will interactively prompt you to confirm AniList mappings and select audio/subtitle tracks on first run — these choices are saved to a config file and reused on subsequent runs.

### Examples

```bash
# Standard run — translate subtitles with DeepL
uv run media-sub-splitter ./input ./output -t YOUR_DEEPL_TOKEN

# Dry run — parse subtitles and validate everything without writing any files
uv run media-sub-splitter ./input ./output --dry-run

# Process specific episodes only
uv run media-sub-splitter ./input ./output -e 1,3,5 -t YOUR_DEEPL_TOKEN

# Process all episodes in parallel (faster on multi-core machines)
uv run media-sub-splitter ./input ./output -p -t YOUR_DEEPL_TOKEN

# Skip ffsubsync subtitle sync (useful when subtitles are already aligned)
uv run media-sub-splitter ./input ./output --no-sync -t YOUR_DEEPL_TOKEN

# Verbose output for debugging
uv run media-sub-splitter ./input ./output -v -t YOUR_DEEPL_TOKEN

# Reprocess specific episodes in parallel with verbose output
uv run media-sub-splitter ./input ./output -e 2,4 -p -v -t YOUR_DEEPL_TOKEN
```

### Options

| Flag | Description |
|------|-------------|
| `-t, --token TOKEN` | DeepL token for subtitle translation. Without it, only subtitles from existing files are used. |
| `-d, --dry-run` | Parse and validate subtitles without writing any segment files. |
| `-e, --episodes 1,3,5` | Process only the specified episode numbers. |
| `-p, --parallel` | Process episodes in parallel. |
| `-v, --verbose` | Print extra debug information. |
| `--no-sync` | Skip syncing external subtitles with the internal track via ffsubsync. |
| `-x` | Strip extra punctuation symbols like `・` (may reduce fidelity). |

---

## Step 2 — Upload: `assets-uploader`

Uploads the generated segments (metadata + media files) to the Nadeshiko API. Always defaults to dry-run — pass `--apply` to actually write anything.

```
uv run assets-uploader <media_folder> [OPTIONS]
```

The `<media_folder>` is the AniList ID folder inside your output directory (e.g. `./output/12345`).

### Examples

```bash
# Dry run against local API — inspect what would be uploaded
uv run assets-uploader ./output/12345 --target local --storage local

# Apply to local API with local storage
uv run assets-uploader ./output/12345 --target local --storage local --apply

# Dry run against dev API with R2 storage references (no files uploaded yet)
uv run assets-uploader ./output/12345 --target dev --storage r2

# Apply to dev API + upload files to R2
uv run assets-uploader ./output/12345 --target dev --storage r2 --apply --upload-r2

# Full production upload — apply to prod API + upload files to R2
uv run assets-uploader ./output/12345 --target prod --storage r2 --apply --upload-r2

# Upload a single episode only
uv run assets-uploader ./output/12345 --target prod --storage r2 --episode 3 --apply --upload-r2

# Update media/character info only, skip episodes and segments
uv run assets-uploader ./output/12345 --target prod --storage r2 --update-info --apply

# Skip the production confirmation prompt
uv run assets-uploader ./output/12345 --target prod --storage r2 --apply --upload-r2 -y
```

### Options

| Flag | Description |
|------|-------------|
| `--target {local,dev,prod}` | API environment to upload to. Defaults to `local`. |
| `--storage {local,r2}` | Storage backend for media file URLs. |
| `--apply` | Actually perform the upload. Without this, runs as dry-run. |
| `--upload-r2` | Upload media files to R2 (requires `--storage r2`). |
| `--episode N` | Upload only a specific episode number. |
| `--update-info` | Update media/character/list info only, skipping episodes and segments. |
| `-y, --yes` | Skip the confirmation prompt for production uploads. |

---

## Tests

```bash
uv run pytest
```

## Linting

```bash
uv run ruff check --fix . && uv run ruff format .
```
