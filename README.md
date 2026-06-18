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

# 3. Tag (NSFW content classification — local GPU, CPU fallback, or Modal GPU)
uv run tag-media ./output/21804

# 4. Upload to dev
uv run assets-uploader ./output/21804 --target dev --storage r2 --upload-r2 --apply

# 5. Upload to prod + notify
uv run assets-uploader ./output/21804 --target prod --storage r2 --apply --yes
uv run notify-discord 21804
```

Each processing command (1-3) runs QC on its output and exits non-zero on failure.

`tag-media` uses a local NVIDIA GPU, falls back to CPU automatically when none is
available, or offloads to a Modal GPU with `--modal` (needs `uv sync --extra modal`
and `uv run modal token new`). Add `--fallback-local` to retry locally if Modal fails.

## YouTube Pipeline

Same steps as the MKV pipeline, with a separate fetch step (the equivalent of
having the MKVs on disk). Steps 3-5 are the exact same commands.

```bash
# 1. Fetch subtitles + video (single video or whole channel)
uv run fetch-youtube https://www.youtube.com/@ChannelHandle --out ./output --browser chrome

# 2. Build segment data + media (validates the first video, then the rest, then QC)
uv run process-youtube ./output/UCxxxxxxxxxxxxxxx

# 3. Tokenize (Sudachi + UniDic POS analysis)
uv run tokenize-media ./output/UCxxxxxxxxxxxxxxx

# 4. Tag (NSFW content classification)
uv run tag-media ./output/UCxxxxxxxxxxxxxxx

# 5. Upload to dev
uv run assets-uploader ./output/UCxxxxxxxxxxxxxxx --target dev --storage r2 --upload-r2 --apply
```

Like `process-media`, `process-youtube` extracts the first video first for
validation (stopping if it produces no segments), then the rest, then runs QC and
exits non-zero on failure. Add `--discord-audit` to mirror progress to
`DISCORD_AUDIT_WEBHOOK_URL`.

By default, cues are grouped into sentence segments by an LLM that decides sentence
boundaries (cut positions only — it never emits text, so it can't corrupt content or
timing) from the Japanese alone, unioned with a deterministic Japanese sentence-end
signal, with a length/duration heuristic as a safety net for over-long groups. This
needs `OPENAI_API_KEY`. Pass `--no-llm-grouping` to use the Japanese punctuation
heuristic instead (no LLM calls); the LLM path also falls back to it automatically when
no API key is set.

`fetch-youtube` downloads the video (≤720p) so `process-youtube` can extract a
screenshot/audio/clip per segment. `--browser` is optional and exports cookies
to bypass YouTube's bot detection. Set `OPENAI_API_KEY` in `.env` to translate
missing EN/ES langs; without it, JA lines lacking an EN/ES cue are dropped to
`ignored_segments`.

## CLI Reference

| Command | Purpose |
|---------|---------|
| `process-media` | Extract segments from MKV files |
| `fetch-youtube` | Download YouTube subtitles + translate missing langs |
| `process-youtube` | Convert downloaded YouTube subs into segment data |
| `tokenize-media` | Batch Sudachi + UniDic tokenization |
| `tag-media` | Batch NSFW tagger (local GPU/CPU, or Modal GPU via `--modal`) |
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
