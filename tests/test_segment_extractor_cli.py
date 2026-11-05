import sys

from nadeshiko_dev_tools.segment_extractor import cli, splitter


def test_drop_nonpositive_duration_lines():
    lines = [
        {"start": 1000, "end": 1000, "sentence": "zero"},
        {"start": 2000, "end": 1999, "sentence": "negative"},
        {"start": 3000, "end": 3001, "sentence": "valid"},
    ]

    kept, dropped = splitter._drop_nonpositive_duration_lines(lines)

    assert kept == [lines[2]]
    assert dropped == 2


def test_parse_args_can_disable_external_subtitle_sync(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process-media",
            "--anilist-id",
            "11061",
            "--input",
            "/tmp/input",
            "--output",
            "/tmp/output",
            "--subtitle-indices",
            "6,5",
            "--no-sync-external-subs",
        ],
    )

    args = cli.parse_args()

    assert args.sync_external_subs is False
