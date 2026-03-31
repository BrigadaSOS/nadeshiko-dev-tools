---
name: process-anime
description: Process anime MKV files into segments and upload to Nadeshiko. Use when source files are ready and user wants to extract, tokenize, tag, and upload.
---

# Process Anime

Process MKV files into language-learning segments for Nadeshiko. Assumes source files are already downloaded (use `/procure-source` first if needed).

**Default paths:**
- **MKV input**: `/mnt/storage/<anime-romaji-title>/`
- **Output**: `/mnt/storage/<anime-romaji-title>-output/`

## Pipeline

Each command does one thing + validates its output. Read the QC report after each step before proceeding.

### Step 1: Extract episode 1 (validate release)
```bash
uv run process-media --anilist-id <ID> --input <MKV_FOLDER> \
  --output <OUTPUT> --subtitle-indices <EN>,<ES> --episodes 1 --discord-audit
```
If segment count is good (>100) and no errors, continue.

### Step 2: Extract all episodes
```bash
uv run process-media --anilist-id <ID> --input <MKV_FOLDER> \
  --output <OUTPUT> --subtitle-indices <EN>,<ES> --parallel --discord-audit
```
Check all episodes have >100 segments.

### Step 3: Tokenize
```bash
uv run tokenize-media <OUTPUT>/<ANILIST_ID>
```

### Step 4: Tag
```bash
uv run tag-media <OUTPUT>/<ANILIST_ID>
```

### Step 5: Upload to dev
```bash
uv run assets-uploader <OUTPUT>/<ANILIST_ID> --target dev --storage r2 --upload-r2 --apply
```

### Promote to prod (only when requested)
```bash
uv run assets-uploader <OUTPUT>/<ANILIST_ID> --target prod --storage r2 --apply --yes
uv run notify-discord <ANILIST_ID>
```

### Recovery

If QC fails due to wrong stream indices:
```bash
rm -rf <OUTPUT>/<ANILIST_ID>/<EPISODE_NUM>
uv run process-media --anilist-id <ID> --input <MKV_FOLDER> --output <OUTPUT> \
  --subtitle-indices <EN>,<CORRECT_ES> --episodes <NUMS>
```

To remove a previous upload: `uv run delete-media <ANILIST_ID> --target dev -y`

## Final Report

After all steps, present:
```
## Processing Report: <Anime Title>
**AniList ID**: <id>
**Source**: <release — group, resolution, sub languages>
**Episodes**: <count> | **Segments**: <count>
**Upload**: dev

### Next steps
- Verify on dev.nadeshiko.co
- When ready, ask me to upload to prod
```

## CLI Reference

| Command | Purpose |
|---------|---------|
| `process-media` | Extract segments from MKVs + QC |
| `tokenize-media` | Batch Sudachi + UniDic tokenization + QC |
| `tag-media` | Batch NSFW tagger (GPU) + QC |
| `quality-check` | Standalone QC (ad-hoc) |
| `assets-uploader` | Upload to Nadeshiko API + R2 |
| `delete-media` | Remove from API + R2 |
| `notify-discord` | Post Discord notification |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Already borrowed" | Use `--parallel` (tokenizer runs separately now) |
| `content_analysis` is None | Check GPU: `uv run python -c "import onnxruntime as rt; print(rt.get_available_providers())"` |
| UniDic RuntimeError | `uv run python -m unidic download` |
| 0 segments | Check stream index, probe all episodes |
