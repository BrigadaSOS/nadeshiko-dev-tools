#!/bin/bash
# Recovery script for Clannad episodes with subtitle mismatches
# Run this from the media-sub-splitter directory

echo "=== Clannad Subtitle Recovery ==="
echo ""
echo "PROBLEM: Judas MKV embedded EN subtitle missing prologue (starts at 01:30)"
echo "JA and ES subtitles have prologue (00:00) - causing no-match segments"
echo ""

ANIME_FOLDER="/mnt/storage/clannad-test/[Judas] Clannad (Season 1-2 + OVAs) [BD1080p][HEVC x265 10bit][Dual-Audio][Eng-Subs]/[Judas] Clannad S1"
OUTPUT_BASE="/mnt/storage/clannad-test-output"

# Episodes needing recovery (<90% valid)
EPISODES="3 16"

echo "Episodes to recover: $EPISODES"
echo ""

for EP in $EPISODES; do
    echo "--- Processing Episode $EP ---"
    
    MKV_FILE="$ANIME_FOLDER/[Judas] Clannad - S01E$(printf "%02d" $EP).mkv"
    
    if [ ! -f "$MKV_FILE" ]; then
        echo "ERROR: MKV file not found: $MKV_FILE"
        continue
    fi
    
    # Check current subtitle streams
    echo "Current subtitle streams in MKV:"
    ffprobe -v error -select_streams s -show_entries stream=index,codec_name:stream_tags=language,title -of csv=p=0 "$MKV_FILE" 2>/dev/null | while read line; do
        echo "  $line"
    done
    
    echo ""
    echo "Option 1: Try to find external English subtitle from alternative source"
    echo "  - Check kitsunekko.net for matching BD English subtitle"
    echo "  - Look for '[Coalgirls]' or other BD release with full prologue"
    echo ""
    
    echo "Option 2: Use DeepL machine translation for missing EN segments"
    echo "  - Re-process with DeepL token to auto-translate missing EN"
    echo ""
    
    echo "Option 3: Download different release with proper subtitles"
    echo "  - Coalgirls 720p BD (has matching JP/EN/ES from same source)"
    echo "  - Erai-raws (if available with all 3 languages)"
    echo ""
done

echo ""
echo "=== RECOMMENDED RECOVERY ==="
echo ""
echo "For episodes E3 and E16:"
echo "1. Find English subtitle that matches jimaku.cc timing:"
echo "   - Search kitsunekko.net for 'Clannad' episode $EP"
echo "   - Download .ass/.srt with matching scene timing"
echo ""
echo "2. Place external EN subtitle alongside MKV:"
echo "   [Judas] Clannad - S01E$EP.en.ass"
echo ""
echo "3. Delete processed episode and re-run:"
echo "   rm -rf $OUTPUT_BASE/2167/$EP"
echo "   rm -rf $OUTPUT_BASE/2167/tmp_ep$EP"
echo ""
echo "4. Re-process with --episodes $EP"
echo ""
