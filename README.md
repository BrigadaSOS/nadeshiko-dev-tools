# Media Sub Splitter

Split an input video onto separate audio segments with images.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run media-sub-splitter <input_folder> <output_folder> [OPTIONS]
```

**Examples:**

```bash
# Basic usage with DeepL translation
uv run media-sub-splitter ./anime ./output -t YOUR_DEEPL_TOKEN

# Dry run to check subtitles without generating segments
uv run media-sub-splitter ./anime ./output --dry-run

# Parallel processing with verbose output
uv run media-sub-splitter ./anime ./output -p -v
```

**Options:**
- `-t, --token TOKEN`: DeepL API token for translations
- `-v, --verbose`: Add extra debug information
- `-d, --dry-run`: Parse subtitles without generating segments
- `-x, --x`: Remove extra punctuation symbols like ・
- `-p, --parallel`: Generate segments in parallel

## Tests

```bash
uv run pytest
```

## Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues and format code
uv run ruff check --fix . && uv run ruff format .
```
