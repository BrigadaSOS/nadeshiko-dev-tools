#!/usr/bin/env python3
"""Quality check script for processed anime segments.

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


class QualityReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def error(self, msg):
        self.errors.append(msg)
        print(f"  ❌ {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  ⚠️  {msg}")

    def ok(self, msg):
        self.info.append(msg)
        print(f"  ✅ {msg}")

    def summary(self):
        print(f"\n{'='*60}")
        print("QUALITY CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"  Errors:   {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Passed:   {len(self.info)}")
        if self.errors:
            print(f"\n{'='*60}")
            print("ERRORS:")
            for e in self.errors:
                print(f"  ❌ {e}")
        if self.warnings:
            print(f"\n{'='*60}")
            print("WARNINGS:")
            for w in self.warnings:
                print(f"  ⚠️  {w}")
        return len(self.errors) == 0


def check_episode_segments(
    episode_folder: str, episode_num: int, report: QualityReport, sample_size: int
):
    """Check segment counts, content, media files, translations."""
    data_path = os.path.join(episode_folder, "_data.json")
    if not os.path.exists(data_path):
        report.error(f"E{episode_num}: _data.json missing")
        return

    with open(data_path) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    ignored = data.get("ignored_segments", [])
    metadata = data.get("metadata", {})

    # Segment count and ignored ratio
    seg_count = len(segments)
    ign_count = len(ignored)
    total = seg_count + ign_count
    if seg_count == 0:
        report.error(f"E{episode_num}: 0 segments generated")
    elif seg_count < 100:
        report.warn(f"E{episode_num}: Only {seg_count} segments (unusually low)")
    else:
        report.ok(f"E{episode_num}: {seg_count} segments, {ign_count} ignored")

    # Ignored segment ratio
    if total > 0 and ign_count > 0:
        ign_ratio = ign_count / total
        from collections import Counter

        reasons = Counter(ig.get("reason", "unknown") for ig in ignored)
        no_match = sum(v for k, v in reasons.items() if "no" in k and "match" in k)
        over_joined = sum(v for k, v in reasons.items() if "too many" in k)

        if ign_ratio > 0.5:
            report.error(
                f"E{episode_num}: {ign_ratio:.0%} segments ignored ({ign_count}/{total}) "
                f"— likely sync issue (no_match={no_match}, over_joined={over_joined})"
            )
        elif ign_ratio > 0.3:
            report.warn(
                f"E{episode_num}: {ign_ratio:.0%} segments ignored ({ign_count}/{total}) "
                f"— no_match={no_match}, over_joined={over_joined}"
            )

    # Metadata check
    version = metadata.get("version")
    if version != "6":
        report.warn(f"E{episode_num}: Unexpected format version: {version}")

    duration_ms = metadata.get("duration_ms", 0)
    if duration_ms == 0:
        report.warn(f"E{episode_num}: Duration is 0 in metadata")
    else:
        duration_min = duration_ms / 60000
        report.ok(f"E{episode_num}: Duration {duration_min:.1f} min")

    # Per-segment checks
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

        files = seg.get("files", {})
        if files:
            for ftype in ("audio", "screenshot", "video"):
                fname = files.get(ftype)
                if fname:
                    fpath = os.path.join(episode_folder, fname)
                    if not os.path.exists(fpath):
                        missing_files += 1
                    elif os.path.getsize(fpath) == 0:
                        zero_size_files += 1

    if missing_files > 0:
        report.error(f"E{episode_num}: {missing_files} missing media files")
    else:
        report.ok(f"E{episode_num}: All media files present")

    if zero_size_files > 0:
        report.error(f"E{episode_num}: {zero_size_files} zero-size files")

    if empty_content > 0:
        report.error(f"E{episode_num}: {empty_content} segments with empty Japanese content")

    if long_content > 0:
        report.warn(f"E{episode_num}: {long_content} segments with JP content > 300 chars")

    if long_duration > 0:
        report.warn(f"E{episode_num}: {long_duration} segments > 30s duration")

    if duration_stats:
        avg_dur = sum(duration_stats) / len(duration_stats)
        max_dur = max(duration_stats)
        min_dur = min(duration_stats)
        report.ok(
            f"E{episode_num}: Duration stats — "
            f"avg={avg_dur/1000:.1f}s, min={min_dur/1000:.1f}s, max={max_dur/1000:.1f}s"
        )

    if mt_es_count > 0 or mt_en_count > 0:
        report.warn(
            f"E{episode_num}: Machine-translated — ES: {mt_es_count}, EN: {mt_en_count}"
        )
    else:
        report.ok(f"E{episode_num}: No machine translations (all from subs)")

    # Translation samples
    if segments and sample_size > 0:
        print(f"\n  Translation samples (E{episode_num}):")
        sample = random.sample(segments, min(sample_size, len(segments)))
        for i, seg in enumerate(sample, 1):
            ja = seg.get("content_ja", "")
            en = seg.get("content_en", "")
            es = seg.get("content_es", "")
            dur = seg.get("duration_ms", 0) / 1000
            print(f"    [{i}] ({dur:.1f}s)")
            print(f"        JA: {ja}")
            print(f"        EN: {en}")
            print(f"        ES: {es}")


def check_episode_tokenizer(episode_folder: str, episode_num: int, report: QualityReport):
    """Check pos_analysis (sudachi + unidic) present on all segments."""
    data_path = os.path.join(episode_folder, "_data.json")
    if not os.path.exists(data_path):
        return

    with open(data_path) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    seg_count = len(segments)
    missing_pos = 0
    missing_pos_sudachi = 0
    missing_pos_unidic = 0

    for seg in segments:
        pos = seg.get("pos_analysis")
        if pos is None:
            missing_pos += 1
        else:
            if not pos.get("sudachi"):
                missing_pos_sudachi += 1
            if not pos.get("unidic"):
                missing_pos_unidic += 1

    if missing_pos > 0:
        report.error(f"E{episode_num}: {missing_pos}/{seg_count} missing pos_analysis")
    else:
        report.ok(f"E{episode_num}: All segments have pos_analysis")

    if missing_pos_sudachi > 0:
        report.warn(f"E{episode_num}: {missing_pos_sudachi} segments missing sudachi tokens")
    if missing_pos_unidic > 0:
        report.warn(f"E{episode_num}: {missing_pos_unidic} segments missing unidic tokens")


def check_episode_tagger(episode_folder: str, episode_num: int, report: QualityReport):
    """Check content_rating + content_analysis present on all segments."""
    data_path = os.path.join(episode_folder, "_data.json")
    if not os.path.exists(data_path):
        return

    with open(data_path) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    seg_count = len(segments)
    missing_content_rating = 0
    missing_content_analysis = 0
    content_ratings = {}

    for seg in segments:
        cr = seg.get("content_rating")
        if cr is None:
            missing_content_rating += 1
        else:
            content_ratings[cr] = content_ratings.get(cr, 0) + 1

        ca = seg.get("content_analysis")
        if ca is None:
            missing_content_analysis += 1

    if missing_content_rating > 0:
        report.error(
            f"E{episode_num}: {missing_content_rating}/{seg_count} missing content_rating"
        )
    else:
        rating_summary = ", ".join(f"{k}: {v}" for k, v in sorted(content_ratings.items()))
        report.ok(f"E{episode_num}: Content ratings — {rating_summary}")

    if missing_content_analysis > 0:
        report.warn(
            f"E{episode_num}: {missing_content_analysis}/{seg_count} missing content_analysis"
        )
    else:
        report.ok(f"E{episode_num}: All segments have content_analysis")


def deep_analysis(media_folder: str, episode_dirs: list, report: QualityReport):
    """Deep analysis: over-joined segments, translation ratio mismatches."""
    print(f"\n{'='*60}")
    print("DEEP ANALYSIS")
    print(f"{'='*60}")

    all_joined = []
    all_ratio = []

    for episode_num, episode_path in episode_dirs:
        data_path = os.path.join(episode_path, "_data.json")
        if not os.path.exists(data_path):
            continue

        with open(data_path) as f:
            data = json.load(f)

        for seg in data.get("segments", []):
            ja_lines = seg.get("subtitles", {}).get("ja", [])
            if len(ja_lines) >= 3:
                all_joined.append((episode_num, seg, len(ja_lines)))

            ja = seg.get("content_ja", "")
            en = seg.get("content_en", "")
            es = seg.get("content_es", "")
            if ja and en and es:
                ja_len, en_len, es_len = len(ja), len(en), len(es)
                if ja_len > 20 and (en_len < ja_len * 0.3 or es_len < ja_len * 0.3):
                    all_ratio.append((episode_num, seg, "target_too_short"))
                if ja_len < 10 and (en_len > 80 or es_len > 80):
                    all_ratio.append((episode_num, seg, "target_too_long"))

    print(f"\n  Segments with >=3 JP lines joined: {len(all_joined)}")
    if all_joined:
        for ep, seg, n_lines in sorted(all_joined, key=lambda x: -x[2])[:10]:
            dur = seg["duration_ms"] / 1000
            ja_texts = [line["text"] for line in seg["subtitles"]["ja"]]
            print(f"    E{ep} ({dur:.1f}s, {n_lines} lines): {ja_texts}")
            print(f"      JA: {seg['content_ja']}")
            print(f"      EN: {seg['content_en']}")
            print()

    if len(all_joined) > 50:
        report.warn(f"High count of over-joined segments: {len(all_joined)}")
    else:
        report.ok(f"Over-joined segments (3+ JP lines): {len(all_joined)}")

    print(f"  Translation ratio suspects: {len(all_ratio)}")
    if all_ratio:
        for ep, seg, reason in all_ratio[:10]:
            ja, en, es = seg["content_ja"], seg["content_en"], seg["content_es"]
            print(f"    E{ep} [{reason}] JA({len(ja)})={ja}")
            print(f"      EN({len(en)})={en}")
            print(f"      ES({len(es)})={es}")
            print()

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
        media_folder: Path to the anime output folder (e.g. output/21804).
        episodes: Episode numbers to check. None = all.
        checks: Check groups to run: {"segments", "tokenizer", "tagger"}. None = all.
        sample_size: Translation samples per episode (segments check only).
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
    print(f"{'='*60}")

    # Check info.json
    info_path = os.path.join(media_folder, "_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        title = info.get("title", {}).get("romaji", "Unknown")
        report.ok(f"info.json present — {title}")
    else:
        report.error("info.json missing")

    # Discover episodes
    episode_dirs = []
    for entry in sorted(os.listdir(media_folder)):
        entry_path = os.path.join(media_folder, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            episode_dirs.append((int(entry), entry_path))

    if episodes:
        episode_dirs = [(n, p) for n, p in episode_dirs if n in episodes]

    if not episode_dirs:
        report.error("No episode directories found")
        return report

    report.ok(f"Found {len(episode_dirs)} episode(s): {[n for n, _ in episode_dirs]}")

    # Run requested check groups per episode
    for episode_num, episode_path in episode_dirs:
        print(f"\n--- Episode {episode_num} ---")

        if "segments" in checks:
            check_episode_segments(episode_path, episode_num, report, sample_size)

        if "tokenizer" in checks:
            check_episode_tokenizer(episode_path, episode_num, report)

        if "tagger" in checks:
            check_episode_tagger(episode_path, episode_num, report)

    # Deep analysis (only with segments check)
    if "segments" in checks:
        deep_analysis(media_folder, episode_dirs, report)

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
