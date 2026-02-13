"""Mora timing alignment and F0 slicing.

Uses voiced regions from F0 data to place mora where speech actually occurs.
Silent gaps in the audio produce gaps in mora timing — no advancing during silence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analyzer import WordMora


@dataclass
class _VoicedRegion:
    """A contiguous region of voiced audio."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _find_voiced_regions(
    f0: list[int],
    sample_ms: int,
    merge_gap_ms: int = 40,
    min_region_ms: int = 30,
) -> list[_VoicedRegion]:
    """Detect contiguous voiced regions from F0, merging small gaps.

    Args:
        f0: F0 values (0 = unvoiced).
        sample_ms: Time step per F0 frame.
        merge_gap_ms: Merge voiced regions separated by gaps smaller than this.
        min_region_ms: Discard regions shorter than this.
    """
    regions: list[_VoicedRegion] = []
    i = 0
    while i < len(f0):
        if f0[i] > 0:
            start = i
            while i < len(f0) and f0[i] > 0:
                i += 1
            regions.append(_VoicedRegion(
                start_ms=start * sample_ms,
                end_ms=i * sample_ms,
            ))
        else:
            i += 1

    if not regions:
        return []

    # Merge regions separated by small gaps
    merged: list[_VoicedRegion] = [regions[0]]
    for r in regions[1:]:
        gap = r.start_ms - merged[-1].end_ms
        if gap <= merge_gap_ms:
            merged[-1].end_ms = r.end_ms
        else:
            merged.append(r)

    # Discard very short regions (noise)
    return [r for r in merged if r.duration_ms >= min_region_ms]


def assign_mora_timing(
    words: list[WordMora],
    start_ms: int,
    end_ms: int,
    f0: list[int] | None = None,
    sample_ms: int = 10,
) -> None:
    """Distribute mora timing to match voiced regions in the F0 data.

    Strategy:
    1. Detect voiced regions from F0
    2. Distribute mora across voiced regions proportional to their duration
    3. Within each region, spread mora by phoneme duration ratios
    4. Silence gaps between regions → gaps in mora timing (no karaoke advance)

    Falls back to proportional distribution if no F0 data is provided.
    """
    all_mora = [m for w in words for m in w.mora]
    if not all_mora:
        return

    total_phoneme_dur = sum(m.duration for m in all_mora)
    if total_phoneme_dur <= 0:
        total_phoneme_dur = float(len(all_mora))
        for m in all_mora:
            m.duration = 1.0

    # If no F0, fall back to simple proportional distribution
    if not f0:
        _distribute_proportional(all_mora, start_ms, end_ms, total_phoneme_dur)
        return

    regions = _find_voiced_regions(f0, sample_ms)

    if not regions:
        # No voiced audio — fall back
        _distribute_proportional(all_mora, start_ms, end_ms, total_phoneme_dur)
        return

    # Calculate total voiced duration
    total_voiced_ms = sum(r.duration_ms for r in regions)

    # Distribute mora across regions proportionally to region duration.
    # Each region gets a share of mora based on (region_duration / total_voiced).
    mora_idx = 0
    for ri, region in enumerate(regions):
        region_share = region.duration_ms / total_voiced_ms

        if ri == len(regions) - 1:
            # Last region gets all remaining mora
            region_mora = all_mora[mora_idx:]
        else:
            # Calculate how many mora fit in this region
            target_dur = region_share * total_phoneme_dur
            accumulated = 0.0
            count = 0
            for m in all_mora[mora_idx:]:
                if accumulated + m.duration > target_dur + 0.01 and count > 0:
                    break
                accumulated += m.duration
                count += 1
            region_mora = all_mora[mora_idx : mora_idx + count]

        if region_mora:
            _distribute_proportional(
                region_mora,
                region.start_ms,
                region.end_ms,
                sum(m.duration for m in region_mora),
            )
            mora_idx += len(region_mora)

    # Safety: any unassigned mora get placed at the end
    if mora_idx < len(all_mora) and regions:
        last = regions[-1]
        remaining = all_mora[mora_idx:]
        _distribute_proportional(
            remaining,
            last.start_ms,
            last.end_ms,
            sum(m.duration for m in remaining),
        )


def _distribute_proportional(
    mora_list: list,
    start_ms: int,
    end_ms: int,
    total_dur: float,
) -> None:
    """Spread mora across a time window proportionally to their durations."""
    if not mora_list or total_dur <= 0:
        return

    span_ms = end_ms - start_ms
    current = float(start_ms)

    for i, mora in enumerate(mora_list):
        ratio = mora.duration / total_dur
        mora_span = ratio * span_ms

        mora._start_ms = int(round(current))  # noqa: SLF001
        if i == len(mora_list) - 1:
            mora._end_ms = end_ms  # noqa: SLF001
        else:
            mora._end_ms = int(round(current + mora_span))  # noqa: SLF001

        current += mora_span


def slice_f0(
    mora_start_ms: int,
    mora_end_ms: int,
    f0: list[int],
    sample_ms: int,
    segment_start_ms: int = 0,
) -> tuple[list[int], float | None]:
    """Extract F0 values for a mora's time range.

    Args:
        mora_start_ms: Absolute start time of mora in ms.
        mora_end_ms: Absolute end time of mora in ms.
        f0: Full F0 array for the segment.
        sample_ms: Time step between F0 samples.
        segment_start_ms: Absolute start of the segment (to convert to relative).

    Returns:
        Tuple of (f0_values, f0_mean). f0_mean is None if all unvoiced.
    """
    rel_start = mora_start_ms - segment_start_ms
    rel_end = mora_end_ms - segment_start_ms

    frame_start = max(0, rel_start // sample_ms)
    frame_end = min(len(f0), (rel_end + sample_ms - 1) // sample_ms)

    if frame_start >= frame_end or frame_start >= len(f0):
        return [], None

    f0_slice = f0[frame_start:frame_end]

    voiced = [v for v in f0_slice if v > 0]
    f0_mean = round(sum(voiced) / len(voiced), 1) if voiced else None

    return f0_slice, f0_mean
