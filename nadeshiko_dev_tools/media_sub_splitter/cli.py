import argparse
import pathlib


def command_args():
    parser = argparse.ArgumentParser(
        description="Split anime .mkv files into audio segments with multi-language subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process anime episodes with DeepL translation:
    %(prog)s ./input ./output -t YOUR_DEEPL_TOKEN

  Dry run to check subtitles without generating segments:
    %(prog)s ./input ./output --dry-run

  Process episodes in parallel with verbose output:
    %(prog)s ./input ./output -p -v

  Reprocess only specific episodes:
    %(prog)s ./input ./output -e 1,3,5

  Skip subtitle sync with ffsubsync:
    %(prog)s ./input ./output --no-sync
    """,
    )
    parser.add_argument(
        "input", type=pathlib.Path, help="Input folder with .mkv files and subtitles"
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="Output folder",
    )
    parser.add_argument(
        "-t",
        "--token",
        dest="token",
        type=str,
        help=(
            "DeepL token for translating subtitles. If not provided, the only generated "
            "subtitles will be taken from existing subtitle files"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add extra debug information to the execution",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        dest="dryrun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Execute and parse subtitles, but without generating the segments",
    )
    parser.add_argument(
        "-x",
        "--x",
        dest="extra_punctuation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove other common punctuation symbols like ・. This might cause certain"
        "subtitles to lose fidelity.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate segments for episodes in parallel",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        dest="episodes",
        type=str,
        help="Comma-separated list of episode numbers to process (e.g., '1,3,5'). "
        "If not provided, all episodes will be processed.",
    )
    parser.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        default=False,
        help="Skip syncing external subtitles with internal track using ffsubsync",
    )
    return parser.parse_args()
