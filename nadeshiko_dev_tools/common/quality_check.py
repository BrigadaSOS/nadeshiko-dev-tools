#!/usr/bin/env python3
"""Quality check script for processed segments.

Check groups (controlled by --checks or the `checks` param in run_qc()):
  segments  — counts, ignored ratio, content, media files, translations, deep analysis
  tokenizer — pos_analysis (sudachi + unidic) present
  tagger    — content_rating + content_analysis present

Usage:
    uv run python scripts/quality_check.py /mnt/storage/output/21804
    uv run python scripts/quality_check.py /mnt/storage/output/21804 --episodes 1
    uv run python scripts/quality_check.py /mnt/storage/output/21804 --checks segments
"""

import argparse
import json
import os
import random
import sys

from nadeshiko_dev_tools.common.file_utils import discover_data_dirs

ALL_CHECKS = {"segments", "tokenizer", "tagger"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quality check for processed segments. "
        "Runs all checks by default: segments, tokenizer, tagger, and deep analysis."
    )
    parser.add_argument("media_folder", help="Path to media folder (e.g. output/21804)")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode numbers")
    parser.add_argument(
        "--sample-size", type=int, default=5, help="Translation samples per episode"
    )
    parser.add_argument(
        "--checks",
        default=None,
        help="Comma-separated check groups to run: segments,tokenizer,tagger (default: all)",
    )
    return parser.parse_args()


class SegmentStats:
    """Per-folder segment statistics, keyed by folder_id (E<n> or YouTube video id)."""

    def __init__(self, folder_id: str):
        self.folder_id = folder_id
        self.valid_segments = 0
        self.ignored_segments = 0
        self.no_match_segments = 0

    def summary(self) -> str:
        total = self.valid_segments + self.ignored_segments
        pct_valid = (self.valid_segments / total * 100) if total > 0 else 0
        return (
            f"{self.folder_id}: "
            f"{self.valid_segments} valid ({pct_valid:.0f}%), "
            f"{self.ignored_segments} skipped, "
            f"{self.no_match_segments} no-match"
        )


class QualityReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.stats: dict[str, SegmentStats] = {}

    def error(self, msg):
        self.errors.append(msg)
        print(f"  ❌ {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  ⚠️  {msg}")

    def ok(self, msg):
        self.info.append(msg)
        print(f"  ✅ {msg}")

    def get_stats(self, folder_id: str) -> SegmentStats:
        if folder_id not in self.stats:
            self.stats[folder_id] = SegmentStats(folder_id)
        return self.stats[folder_id]

    def summary(self):
        print(f"\n{'=' * 60}")
        print("QUALITY CHECK SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Errors:   {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Passed:   {len(self.info)}")
        if self.stats:
            print(f"\n{'=' * 60}")
            print("BREAKDOWN")
            for folder_id in self.stats:
                print(f"  {self.stats[folder_id].summary()}")
        if self.errors:
            print(f"\n{'=' * 60}")
            print("ERRORS:")
            for e in self.errors:
                print(f"  ❌ {e}")
        if self.warnings:
            print(f"\n{'=' * 60}")
            print("WARNINGS:")
            for w in self.warnings:
                print(f"  ⚠️  {w}")

        print(f"\n{'=' * 60}")
        print("RESULT: PASS" if not self.errors else "RESULT: FAIL")
        return len(self.errors) == 0


def check_segments(folder: str, folder_id: str, data: dict, report: QualityReport, sample_size: int):
    """Check segment counts, content, media files, translations."""
    segments = data.get("segments", [])
    ignored = data.get("ignored_segments", [])
    metadata = data.get("metadata", {})

    seg_count = len(segments)
    ign_count = len(ignored)
    no_match_count = sum(1 for ig in ignored if "no en/es subtitle match" in ig.get("reason", ""))
    total = seg_count + ign_count

    stats = report.get_stats(folder_id)
    stats.valid_segments = seg_count
    stats.ignored_segments = ign_count
    stats.no_match_segments = no_match_count

    pct_valid = (seg_count / total * 100) if total > 0 else 0

    if seg_count == 0:
        report.error(f"{folder_id}: 0 segments generated")
    elif seg_count < 100:
        report.warn(f"{folder_id}: Only {seg_count} segments (unusually low)")
    elif pct_valid < 90:
        report.warn(
            f"{folder_id}: {seg_count} valid ({pct_valid:.0f}%), "
            f"{ign_count} skipped, {no_match_count} no-match — LOW RATIO, RECHECK NEEDED"
        )
    else:
        report.ok(
            f"{folder_id}: {seg_count} valid ({pct_valid:.0f}%), "
            f"{ign_count} skipped, {no_match_count} no-match"
        )

    if total > 0 and ign_count > 0:
        from collections import Counter

        reasons = Counter(ig.get("reason", "unknown") for ig in ignored)
        no_match = sum(v for k, v in reasons.items() if "no" in k and "match" in k)
        over_joined = sum(v for k, v in reasons.items() if "too many" in k)
        ign_ratio = ign_count / total

        if ign_count > 20 or ign_ratio > 0.1:
            report.info.append(f"{folder_id}: Skip reasons — {dict(reasons)}")
            print(f"    Skip breakdown: {dict(reasons)}")

        if ign_ratio > 0.5:
            report.error(
                f"{folder_id}: {ign_ratio:.0%} segments ignored ({ign_count}/{total}) "
                f"— likely sync issue (no_match={no_match}, over_joined={over_joined})"
            )
        elif ign_ratio > 0.3:
            report.warn(
                f"{folder_id}: {ign_ratio:.0%} segments ignored ({ign_count}/{total}) "
                f"— no_match={no_match}, over_joined={over_joined}"
            )

        early_nomatch = [
            ig
            for ig in ignored
            if "no en/es subtitle match" in ig.get("reason", "")
            and ig.get("start_ms", 9999999) < 120000  # within first 2 minutes
        ]
        if early_nomatch and pct_valid < 95:
            report.warn(
                f"{folder_id}: {len(early_nomatch)} 'no-match' segments in first 2min "
                f"— likely subtitle source mismatch (wrong subtitle files)"
            )

    version = metadata.get("version")
    if version != "6":
        report.warn(f"{folder_id}: Unexpected format version: {version}")

    duration_ms = metadata.get("duration_ms", 0)
    if duration_ms == 0:
        report.warn(f"{folder_id}: Duration is 0 in metadata")
    else:
        report.ok(f"{folder_id}: Duration {duration_ms / 60000:.1f} min")

    # One scandir for the whole folder: {filename: size}, reused for every file check below.
    file_sizes = {e.name: e.stat().st_size for e in os.scandir(folder) if e.is_file()}

    # File integrity: every segment that declares a media file must have it on disk.
    # Segments that declare no files (e.g. media extraction skipped) are skipped here.
    expected_hashes = {
        seg["segment_hash"]
        for seg in segments
        if seg.get("segment_hash") and seg.get("files", {}).get("video")
    }
    actual_mp4s = {name[:-4] for name in file_sizes if name.endswith(".mp4")}
    missing_from_disk = expected_hashes - actual_mp4s
    orphan_on_disk = actual_mp4s - expected_hashes

    if missing_from_disk:
        report.error(
            f"{folder_id}: {len(missing_from_disk)}/{len(expected_hashes)} segments in "
            f"_data.json have NO media files on disk — extraction was interrupted"
        )
    if orphan_on_disk:
        report.warn(f"{folder_id}: {len(orphan_on_disk)} orphan mp4 files on disk not in _data.json")

    missing_files = 0
    zero_size_files = 0
    empty_content = 0
    long_content = 0
    long_duration = 0
    mt_es_count = 0
    mt_en_count = 0
    duration_stats = []

    for seg in segments:
        ja = seg.get("content_ja", "")
        if not ja:
            empty_content += 1
        if len(ja or "") > 300:
            long_content += 1

        dur = seg.get("duration_ms", 0)
        duration_stats.append(dur)
        if dur > 30000:
            long_duration += 1

        if seg.get("is_mt_es"):
            mt_es_count += 1
        if seg.get("is_mt_en"):
            mt_en_count += 1

        for ftype in ("audio", "screenshot", "video"):
            fname = seg.get("files", {}).get(ftype)
            if not fname:
                continue
            if fname not in file_sizes:
                missing_files += 1
            elif file_sizes[fname] == 0:
                zero_size_files += 1

    if missing_files > 0:
        report.error(f"{folder_id}: {missing_files} missing media files")
    elif not missing_from_disk:
        report.ok(f"{folder_id}: All media files present")

    if zero_size_files > 0:
        report.error(f"{folder_id}: {zero_size_files} zero-size files")
    if empty_content > 0:
        report.error(f"{folder_id}: {empty_content} segments with empty Japanese content")
    if long_content > 0:
        report.warn(f"{folder_id}: {long_content} segments with JP content > 300 chars")
    if long_duration > 0:
        report.warn(f"{folder_id}: {long_duration} segments > 30s duration")

    if duration_stats:
        avg_dur = sum(duration_stats) / len(duration_stats)
        report.ok(
            f"{folder_id}: Duration stats — avg={avg_dur / 1000:.1f}s, "
            f"min={min(duration_stats) / 1000:.1f}s, max={max(duration_stats) / 1000:.1f}s"
        )

    if mt_es_count > 0 or mt_en_count > 0:
        report.warn(f"{folder_id}: Machine-translated — ES: {mt_es_count}, EN: {mt_en_count}")
    else:
        report.ok(f"{folder_id}: No machine translations (all from subs)")

    if segments and sample_size > 0:
        print(f"\n  Translation samples ({folder_id}):")
        for i, seg in enumerate(random.sample(segments, min(sample_size, len(segments))), 1):
            dur = seg.get("duration_ms", 0) / 1000
            print(f"    [{i}] ({dur:.1f}s)")
            print(f"        JA: {seg.get('content_ja', '')}")
            print(f"        EN: {seg.get('content_en', '')}")
            print(f"        ES: {seg.get('content_es', '')}")


def check_tokenizer(folder_id: str, data: dict, report: QualityReport):
    """Check pos_analysis (sudachi + unidic) present on all segments."""
    segments = data.get("segments", [])
    missing_pos = 0
    missing_sudachi = 0
    missing_unidic = 0

    for seg in segments:
        pos = seg.get("pos_analysis")
        if pos is None:
            missing_pos += 1
        else:
            if not pos.get("sudachi"):
                missing_sudachi += 1
            if not pos.get("unidic"):
                missing_unidic += 1

    if missing_pos > 0:
        report.error(f"{folder_id}: {missing_pos}/{len(segments)} missing pos_analysis")
    else:
        report.ok(f"{folder_id}: All segments have pos_analysis")

    if missing_sudachi > 0:
        report.warn(f"{folder_id}: {missing_sudachi} segments missing sudachi tokens")
    if missing_unidic > 0:
        report.warn(f"{folder_id}: {missing_unidic} segments missing unidic tokens")


def check_tagger(folder_id: str, data: dict, report: QualityReport):
    """Check content_rating + content_analysis present on all segments."""
    segments = data.get("segments", [])
    missing_rating = 0
    missing_analysis = 0
    ratings: dict[str, int] = {}

    for seg in segments:
        cr = seg.get("content_rating")
        if cr is None:
            missing_rating += 1
        else:
            ratings[cr] = ratings.get(cr, 0) + 1
        if seg.get("content_analysis") is None:
            missing_analysis += 1

    if missing_rating > 0:
        report.error(f"{folder_id}: {missing_rating}/{len(segments)} missing content_rating")
    else:
        rating_summary = ", ".join(f"{k}: {v}" for k, v in sorted(ratings.items()))
        report.ok(f"{folder_id}: Content ratings — {rating_summary}")

    if missing_analysis > 0:
        report.warn(f"{folder_id}: {missing_analysis}/{len(segments)} missing content_analysis")
    else:
        report.ok(f"{folder_id}: All segments have content_analysis")


def deep_analysis(loaded: list[tuple[str, dict]], report: QualityReport):
    """Deep analysis: over-joined segments, translation ratio mismatches."""
    print(f"\n{'=' * 60}")
    print("DEEP ANALYSIS")
    print(f"{'=' * 60}")

    all_joined = []
    all_ratio = []

    for folder_id, data in loaded:
        for seg in data.get("segments", []):
            ja_lines = seg.get("subtitles", {}).get("ja", [])
            if len(ja_lines) >= 3:
                all_joined.append((folder_id, seg, len(ja_lines)))

            ja = seg.get("content_ja", "")
            en = seg.get("content_en", "")
            es = seg.get("content_es", "")
            if ja and en and es:
                ja_len, en_len, es_len = len(ja), len(en), len(es)
                if ja_len > 20 and (en_len < ja_len * 0.3 or es_len < ja_len * 0.3):
                    all_ratio.append((folder_id, seg, "target_too_short"))
                if ja_len < 10 and (en_len > 80 or es_len > 80):
                    all_ratio.append((folder_id, seg, "target_too_long"))

    print(f"\n  Segments with >=3 JP lines joined: {len(all_joined)}")
    for folder_id, seg, n_lines in sorted(all_joined, key=lambda x: -x[2])[:10]:
        dur = seg["duration_ms"] / 1000
        ja_texts = [line["text"] for line in seg["subtitles"]["ja"]]
        print(f"    {folder_id} ({dur:.1f}s, {n_lines} lines): {ja_texts}")
        print(f"      JA: {seg['content_ja']}")
        print(f"      EN: {seg['content_en']}\n")

    if len(all_joined) > 50:
        report.warn(f"High count of over-joined segments: {len(all_joined)}")
    else:
        report.ok(f"Over-joined segments (3+ JP lines): {len(all_joined)}")

    print(f"  Translation ratio suspects: {len(all_ratio)}")
    for folder_id, seg, reason in all_ratio[:10]:
        ja, en, es = seg["content_ja"], seg["content_en"], seg["content_es"]
        print(f"    {folder_id} [{reason}] JA({len(ja)})={ja}")
        print(f"      EN({len(en)})={en}")
        print(f"      ES({len(es)})={es}\n")

    if len(all_ratio) > 20:
        report.warn(f"High count of translation ratio suspects: {len(all_ratio)}")
    else:
        report.ok(f"Translation ratio suspects: {len(all_ratio)}")


def run_qc(
    media_folder: str,
    episodes: set[int] | None = None,
    checks: set[str] | None = None,
    sample_size: int = 5,
) -> QualityReport:
    """Run quality checks and return the report.

    Args:
        media_folder: Output folder (e.g. output/21804 or output/UCxxxx).
        episodes: Episode numbers to check (anime only). None = all.
        checks: Check groups to run: {"segments", "tokenizer", "tagger"}. None = all.
        sample_size: Translation samples per folder (segments check only).
    """
    if checks is None:
        checks = ALL_CHECKS

    report = QualityReport()
    media_folder = os.path.abspath(media_folder)

    if not os.path.isdir(media_folder):
        report.error(f"Not a directory: {media_folder}")
        return report

    print(f"Quality check: {media_folder}")
    print(f"  Checks: {', '.join(sorted(checks))}")
    print(f"{'=' * 60}")

    info_path = os.path.join(media_folder, "_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        # Anime stores a nested title dict; YouTube stores a flat channel name.
        title = info.get("name") or info.get("title", {}).get("romaji", "Unknown")
        report.ok(f"info.json present — {title}")
    else:
        report.error("info.json missing")

    data_dirs = discover_data_dirs(media_folder, episodes)
    if not data_dirs:
        report.error("No folders with _data.json found")
        return report

    report.ok(f"Found {len(data_dirs)}: {[folder_id for folder_id, _ in data_dirs]}")

    loaded: list[tuple[str, dict]] = []
    for folder_id, path in data_dirs:
        print(f"\n--- {folder_id} ---")
        with open(os.path.join(path, "_data.json")) as f:
            data = json.load(f)
        loaded.append((folder_id, data))

        if "segments" in checks:
            check_segments(path, folder_id, data, report, sample_size)
        if "tokenizer" in checks:
            check_tokenizer(folder_id, data, report)
        if "tagger" in checks:
            check_tagger(folder_id, data, report)

    if "segments" in checks:
        deep_analysis(loaded, report)

    return report


def main():
    args = parse_args()

    checks = None
    if args.checks:
        checks = {c.strip() for c in args.checks.split(",")}
        invalid = checks - ALL_CHECKS
        if invalid:
            print(f"Error: Unknown check groups: {invalid}. Valid: {ALL_CHECKS}")
            return 1

    ep_filter = None
    if args.episodes:
        ep_filter = {int(e.strip()) for e in args.episodes.split(",")}

    report = run_qc(
        media_folder=args.media_folder,
        episodes=ep_filter,
        checks=checks,
        sample_size=args.sample_size,
    )

    passed = report.summary()
    return 0 if passed else 1


if __name__ == "__main__":
    random.seed(42)
    sys.exit(main() or 0)
