import json
import sys

from nadeshiko_dev_tools.segment_extractor import cli


def _args(monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process-media",
            "--anilist-id",
            "877",
            "--input",
            "/in",
            "--output",
            "/out",
            "--subtitle-indices",
            "2",
            *extra,
        ],
    )
    return cli.parse_args()


def test_jobs_defaults_to_sequential(monkeypatch):
    args = _args(monkeypatch)
    assert args.jobs == 1
    assert args.worker is False


def test_split_round_robin_keeps_groups_disjoint_and_complete():
    groups = cli.split_round_robin(list(range(2, 48)), 3)
    assert len(groups) == 3
    assert sorted(e for g in groups for e in g) == list(range(2, 48))
    assert cli.split_round_robin([5], 4) == [[5]]


def test_worker_argv_passes_sources_and_options(monkeypatch):
    args = _args(
        monkeypatch,
        "--jobs",
        "3",
        "--audio-index",
        "1",
        "--source-pattern",
        "[BD]",
        "--no-sync-external-subs",
    )
    argv = cli.worker_argv(args, [4, 7], "/in", "/out")
    joined = " ".join(argv)
    assert "--episodes 4,7" in joined
    assert "--worker" in argv
    assert "--audio-index 1" in joined
    assert "--source-pattern [BD]" in joined
    assert "--no-sync-external-subs" in argv
    assert "--jobs" not in argv
    assert "--discord-audit" not in argv


def test_episode_count_on_disk(tmp_path):
    ep = tmp_path / "3"
    ep.mkdir()
    assert cli.episode_count_on_disk(str(tmp_path), 3) is None
    segments = [{"segment_hash": "a"}, {"segment_hash": "b"}]
    (ep / "_data.json").write_text(json.dumps({"segments": segments}))
    (ep / "a.mp4").write_bytes(b"")
    assert cli.episode_count_on_disk(str(tmp_path), 3) is None  # b.mp4 missing
    (ep / "b.mp4").write_bytes(b"")
    assert cli.episode_count_on_disk(str(tmp_path), 3) == 2
