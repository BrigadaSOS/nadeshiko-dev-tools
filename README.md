# Nadeshiko Dev Tools

CLI tools for processing anime into language-learning segments for Nadeshiko.

## Setup

```bash
uv sync
uv run python -m unidic download
cp .env.example .env   # configure API keys, R2 credentials, etc.
```

## Pipeline

Each step is a separate command. Run them in order, checking output between steps:

```bash
# 1. Extract segments from MKVs (extracts ep1 first for validation, then rest)
uv run process-media --anilist-id 21804 --input ./mkv-folder --output ./output \
  --subtitle-indices 2,4 --parallel

# 2. Tokenize (Sudachi + UniDic POS analysis)
uv run tokenize-media ./output/21804

# 3. Tag (NSFW content classification, requires GPU)
uv run tag-media ./output/21804

# 4. Upload to dev
uv run assets-uploader ./output/21804 --target dev --storage r2 --upload-r2 --apply

# 5. Upload to prod + notify
uv run assets-uploader ./output/21804 --target prod --storage r2 --apply --yes
uv run notify-discord 21804
```

Each processing command (1-3) runs QC on its output and exits non-zero on failure.

## CLI Reference

| Command | Purpose |
|---------|---------|
| `process-media` | Extract segments from MKV files |
| `tokenize-media` | Batch Sudachi + UniDic tokenization |
| `tag-media` | Batch NSFW tagger (GPU) |
| `quality-check` | Standalone QC (ad-hoc) |
| `assets-uploader` | Upload to Nadeshiko API + R2 |
| `delete-media` | Remove media from API + R2 |
| `notify-discord` | Post Discord notification |

Run any command with `--help` for full options.

## Other Tools

```bash
# Find JP subtitles on jimaku.cc / kitsunekko
uv run python scripts/find_jp_subs.py --anilist-id 21804
```

## Tests

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
```
