# Nadeshiko Dev Tools

`nadeshiko-dev-tools` is a collection of CLI utilities for Nadeshiko media workflows.

## Included Tools

- `media-sub-splitter`: Split anime episodes into subtitle-aligned media segments.
- `assets-uploader`: Upload generated media metadata and assets to Nadeshiko environments.

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
uv run assets-uploader <media_folder> (--dev | --prod) [OPTIONS]
```

Examples:

```bash
# Dry run against local/dev (default mode is dry-run)
uv run assets-uploader ./output/12345 --dev

# Apply upload to local/dev
uv run assets-uploader ./output/12345 --dev --apply

# Production upload with R2 files
uv run assets-uploader ./output/12345 --prod --apply --upload-r2

# Upload only one episode
uv run assets-uploader ./output/12345 --dev --episode 1 --apply
```

Common options:

- `--episode N`: Upload a single episode number.
- `--apply`: Execute changes (without this, runs in dry-run mode).
- `--upload-r2`: Upload files to R2 storage.
- `--update-info`: Update media metadata only.

## Tests

```bash
uv run pytest
```

## Linting

```bash
uv run ruff check .
uv run ruff check --fix . && uv run ruff format .
```
