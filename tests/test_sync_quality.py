"""Tests for _compute_overlap_score in main.py."""

from dataclasses import dataclass

from nadeshiko_dev_tools.media_sub_splitter.main import _compute_overlap_score


@dataclass
class FakeEvent:
    """Minimal stand-in for a pysubs2 SSAEvent with start/end in ms."""

    start: int
    end: int
    type: str = "Dialogue"


def _make_track(timings: list[tuple[int, int]], event_type: str = "Dialogue") -> list:
    return [FakeEvent(start=s, end=e, type=event_type) for s, e in timings]


class TestIdenticalTracks:
    def test_perfect_overlap(self):
        track = _make_track([(0, 1000), (2000, 3000), (4000, 5000)])
        overlap, offset = _compute_overlap_score(track, track)
        assert overlap == 1.0
        assert offset == 0.0


class TestDisjointTracks:
    def test_no_overlap(self):
        track_a = _make_track([(0, 1000), (2000, 3000), (4000, 5000)])
        track_b = _make_track([(10000, 11000), (12000, 13000), (14000, 15000)])
        overlap, offset = _compute_overlap_score(track_a, track_b)
        assert overlap == 0.0


class TestSlightlyOffsetTracks:
    def test_small_offset_high_overlap(self):
        track_a = _make_track([(0, 2000), (3000, 5000), (6000, 8000)])
        # Shift by 200ms — still overlapping
        track_b = _make_track([(200, 2200), (3200, 5200), (6200, 8200)])
        overlap, offset = _compute_overlap_score(track_a, track_b)
        assert overlap == 1.0
        assert 150 <= offset <= 250  # mean offset ≈ 200ms

    def test_large_offset_partial_overlap(self):
        track_a = _make_track([(0, 1000), (2000, 3000), (4000, 5000)])
        # Shift by 800ms — still overlapping (each event is 1000ms wide)
        track_b = _make_track([(800, 1800), (2800, 3800), (4800, 5800)])
        overlap, offset = _compute_overlap_score(track_a, track_b)
        assert overlap == 1.0
        assert 750 <= offset <= 850


class TestEmptyTracks:
    def test_empty_track_a(self):
        track_b = _make_track([(0, 1000)])
        overlap, offset = _compute_overlap_score([], track_b)
        assert overlap == 0.0
        assert offset == 0.0

    def test_empty_track_b(self):
        track_a = _make_track([(0, 1000)])
        overlap, offset = _compute_overlap_score(track_a, [])
        assert overlap == 0.0
        assert offset == 0.0

    def test_both_empty(self):
        overlap, offset = _compute_overlap_score([], [])
        assert overlap == 0.0
        assert offset == 0.0


class TestNonDialogueFiltered:
    def test_comments_are_filtered(self):
        track_a = _make_track([(0, 1000)], event_type="Comment")
        track_b = _make_track([(0, 1000)])
        overlap, offset = _compute_overlap_score(track_a, track_b)
        assert overlap == 0.0


class TestSampling:
    def test_large_track_still_works(self):
        """Ensure sampling works for tracks larger than sample_size."""
        n = 200
        track = _make_track([(i * 2000, i * 2000 + 1000) for i in range(n)])
        overlap, offset = _compute_overlap_score(track, track, sample_size=50)
        assert overlap == 1.0
        assert offset == 0.0
