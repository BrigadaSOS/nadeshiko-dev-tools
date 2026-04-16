---
name: process-anime
description: Process anime MKV files into segments and upload to Nadeshiko. Use when source files are ready and user wants to extract, tokenize, tag, and upload.
---

# Process Anime

Process MKV files into language-learning segments for Nadeshiko.

**Default paths:**
- **MKV input**: `/mnt/storage/<anime-romaji-title>/`
- **Output**: `/mnt/storage/<anime-romaji-title>-output/`

## Input folder setup

Place all sources in a single input folder:
- **MKV files** — the video source. If multiple releases exist (e.g., one for video, another with named chapters for OP/ED detection), use `--source-pattern` to pick which one is the video source.
- **External subtitle files** (`.ass`, `.srt`) — auto-discovered by filename. Language is detected from the filename (e.g., `ep01.es.ass`, `[Trix] Show - 01.ja.srt`).
- `--subtitle-indices` is for *internal* MKV streams only.

## Pipeline

### Step 1: Probe subtitle streams
```bash
ffprobe -v error -select_streams s -show_entries stream=index,codec_name:stream_tags=language,title \
  -of csv=p=0 "<MKV_FILE>"
```

### Step 2: Extract
```bash
uv run process-media --anilist-id <ID> --input <MKV_FOLDER> \
  --output <OUTPUT> --subtitle-indices <INDICES> \
  --source-pattern "<GROUP>" --episode-range 1-<LAST_EP> --discord-audit
```

Key flags:
- `--source-pattern "[Trix]"` — only MKVs matching this are used as video source. Others in the folder are still scanned for OP/ED chapter metadata.
- `--episode-range 1-22` — skip OVAs/specials/recaps.
- `--episodes 3,7` — process only specific episodes (combines with `--episode-range`).

Behavior:
- Segments by JA subtitle lines, then matches EN/ES translations by temporal overlap.
- OP/ED filtered automatically if any MKV in the input folder has named chapters ("Opening", "Ending").
- Extracts ep1 first as validation. If it fails, stops.
- Idempotent: completed episodes are skipped, partial extractions are auto-cleaned.
- Sequential processing, ~8 min/episode. Safe to interrupt and re-run.

### Step 3: Tokenize
```bash
uv run tokenize-media <OUTPUT>/<ANILIST_ID>
```
Prints `RESULT: PASS` or `RESULT: FAIL` at the end. Must pass.

### Step 4: Tag
```bash
uv run tag-media <OUTPUT>/<ANILIST_ID>
```
Prints `RESULT: PASS` or `RESULT: FAIL` at the end. Must pass.

### Step 5: Final QC
```bash
uv run quality-check <OUTPUT>/<ANILIST_ID>
```
Runs all checks (segments + tokenizer + tagger). Look for `RESULT: PASS` at the end.
- If `RESULT: FAIL` — check the `ERRORS:` section for which episodes failed, then follow the Recovery section below.
- Warnings are informational and don't block uploading.

### Step 6: Upload to dev
Only after `RESULT: PASS` from step 5.
```bash
uv run assets-uploader <OUTPUT>/<ANILIST_ID> --target dev --storage r2 --upload-r2 --apply
```

### Promote to prod (only when requested)
```bash
uv run assets-uploader <OUTPUT>/<ANILIST_ID> --target prod --storage r2 --apply --yes
uv run notify-discord <ANILIST_ID>
```

## Recovery

Delete broken episode folders and re-run (good episodes are skipped):
```bash
rm -rf <OUTPUT>/<ANILIST_ID>/<EPISODE_NUM>
uv run process-media [same flags as step 2]
uv run tokenize-media <OUTPUT>/<ANILIST_ID>
uv run tag-media <OUTPUT>/<ANILIST_ID>
uv run quality-check <OUTPUT>/<ANILIST_ID>
```

Remove a previous upload: `uv run delete-media <ANILIST_ID> --target dev -y`

## Error handling

When QC reports `RESULT: FAIL`, check the `ERRORS:` section and act based on the error type:

**Extraction errors** (re-extractable — delete episode folder and re-run step 2):
- `_data.json missing but N mp4 files` — extraction crashed mid-episode
- `N segments in _data.json have NO media files` — extraction was interrupted
- `N missing media files` / `N zero-size files` — ffmpeg failed on some segments
- `0 segments generated` — wrong `--subtitle-indices` or no subtitle match. Re-probe the MKV.

**Subtitle source errors** (need different source — escalate to user):
- Most segments have `no en/es subtitle match` — subtitles are from a different source/timing. Need better-matching subs.
- `Only N segments (unusually low)` on most episodes — wrong subtitle language or filtered out. Check stream indices.
- High `missing required languages` count — the ES or EN subtitle source doesn't cover all dialogue. May need a different release.

**Tokenizer/tagger errors** (re-runnable — just re-run step 3 or 4):
- `N missing pos_analysis` — tokenizer didn't run or crashed. Re-run `tokenize-media`.
- `N missing content_rating` — tagger didn't run or crashed. Re-run `tag-media`.
- `content_analysis is None` — GPU issue. Check: `uv run python -c "import onnxruntime as rt; print(rt.get_available_providers())"`
- UniDic RuntimeError — run `uv run python -m unidic download`

## CLI Reference

| Command | Purpose |
|---------|---------|
| `process-media` | Extract segments from MKVs |
| `tokenize-media` | Batch Sudachi + UniDic tokenization |
| `tag-media` | Batch NSFW tagger (GPU) |
| `quality-check` | Standalone QC |
| `assets-uploader` | Upload to Nadeshiko API + R2 |
| `delete-media` | Remove from API + R2 |
| `notify-discord` | Post Discord notification |
