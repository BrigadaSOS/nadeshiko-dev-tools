"""Export content rating results as SQL update statements.

Reads classification results from {media_folder}/_nsfw_results/results.json
and generates a SQL file that can be applied directly to the database.
"""

import json
from pathlib import Path

from rich.console import Console

from .batch import RESULTS_DIR_NAME, RESULTS_FILE

console = Console()


def export_sql(media_folder: Path):
    """Generate SQL file from classification results.

    Args:
        media_folder: Path to media folder (must contain _nsfw_results/)
    """
    results_dir = media_folder / RESULTS_DIR_NAME
    result_file = results_dir / RESULTS_FILE

    if not result_file.exists():
        console.print(f"[red]Results not found: {result_file}[/red]")
        console.print("Run the batch classifier first.")
        return

    console.print("[bold]Content Rating SQL Export[/bold]")
    console.print(f"Media: {media_folder.name}")
    console.print()

    with open(result_file) as f:
        data = json.load(f)

    stats: dict[str, int] = {"SAFE": 0, "SUGGESTIVE": 0, "QUESTIONABLE": 0, "EXPLICIT": 0}
    all_segments = []

    for ep_data in data.values():
        for hashed_id, seg_data in ep_data.items():
            cr = seg_data["content_rating"]
            stats[cr] = stats.get(cr, 0) + 1

            content_analysis = {
                "scores": seg_data["scores"],
                "tags": seg_data["tags"],
            }
            all_segments.append((hashed_id, cr, content_analysis))

    total = sum(stats.values())
    console.print(f"Total segments: {total:,}")
    console.print(f"Content ratings: {stats}")
    console.print()

    # Generate SQL
    output_file = results_dir / "update.sql"
    with open(output_file, "w") as f:
        f.write("-- Content rating bulk update\n")
        f.write(
            f"-- Generated from media {media_folder.name},"
            f" {total} segments\n"
        )
        f.write(f"-- Ratings: {stats}\n\n")

        f.write("BEGIN;\n\n")

        f.write(f"-- Update {len(all_segments)} segments\n")
        for hashed_id, cr, content_analysis in all_segments:
            analysis_json = json.dumps(
                content_analysis, separators=(",", ":")
            )
            analysis_json_escaped = analysis_json.replace("'", "''")
            f.write(
                f'UPDATE "Segment" SET content_rating = \'{cr}\', '
                f"rating_analysis = '{analysis_json_escaped}' "
                f"WHERE hashed_id = '{hashed_id}';\n"
            )

        f.write("\nCOMMIT;\n")

    console.print(f"[bold green]SQL file written: {output_file}[/bold green]")
    console.print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
