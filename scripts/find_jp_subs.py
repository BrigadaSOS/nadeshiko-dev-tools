#!/usr/bin/env python3
"""Find available Japanese subtitles for an anime.

Searches jimaku.cc (primary, uses AniList ID) then kitsunekko.net (fallback).
Outputs structured info so the caller can decide which source to download.

Usage:
    uv run python scripts/find_jp_subs.py --anilist-id 128547
    uv run python scripts/find_jp_subs.py --anilist-id 20812 --sample
    uv run python scripts/find_jp_subs.py --name "Odd Taxi" --sample
    uv run python scripts/find_jp_subs.py --anilist-id 128547 --json
"""

import argparse
import json
import os
import re
import sys
import urllib.parse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

KITSUNEKKO_BASE = "https://kitsunekko.net/subtitles/japanese"
JIMAKU_API = "https://jimaku.cc/api"


# ---------------------------------------------------------------------------
# Source classification (shared by both backends)
# ---------------------------------------------------------------------------

def classify_source(filename: str) -> str:
    fl = filename.lower()
    if "bluray" in fl or "bdremux" in fl or "bdrip" in fl or "[bd]" in fl or ".bd." in fl:
        return "BD"
    if "netflix" in fl:
        return "Netflix"
    if "prime" in fl or "amazon" in fl:
        return "Prime Video"
    if "crunchyroll" in fl or "cr " in fl:
        return "Crunchyroll"
    if "funimation" in fl:
        return "Funimation"
    if "hidive" in fl:
        return "HIDIVE"
    if "anime time" in fl or "animetime" in fl:
        return "Anime Time"
    if "retimed" in fl:
        m = re.search(r"[Rr]etimed\s+for\s+(.+?)(?:\.|$)", filename)
        target = m.group(1).strip() if m else "?"
        return f"Retimed ({target})"
    if "web" in fl:
        return "WEB"
    return "Unknown"


def classify_filetype(filename: str) -> str:
    fl = filename.lower()
    for ext in (".zip", ".7z", ".rar"):
        if fl.endswith(ext):
            return "archive"
    if fl.endswith(".srt"):
        return "srt"
    if fl.endswith(".ass"):
        return "ass"
    return "other"


# ---------------------------------------------------------------------------
# jimaku.cc (primary)
# ---------------------------------------------------------------------------

def jimaku_search(anilist_id: int | None, name: str | None, api_key: str) -> list[dict]:
    """Search jimaku for an anime entry. Returns list of entries."""
    headers = {"Authorization": api_key}
    params = {}
    if anilist_id:
        params["anilist_id"] = anilist_id
    elif name:
        params["query"] = name
    else:
        return []

    resp = requests.get(f"{JIMAKU_API}/entries/search", params=params, headers=headers, timeout=10)
    if resp.status_code != 200:
        return []
    return resp.json()


def jimaku_list_files(entry_id: int, api_key: str) -> list[dict]:
    """List subtitle files for a jimaku entry."""
    headers = {"Authorization": api_key}
    resp = requests.get(f"{JIMAKU_API}/entries/{entry_id}/files", headers=headers, timeout=10)
    if resp.status_code != 200:
        return []

    files = []
    for f in resp.json():
        filename = f["name"]
        # Skip bilingual (ja-en) and non-Japanese subs
        fl = filename.lower()
        if "ja-en" in fl or ".en." in fl or ".chs" in fl:
            continue
        # Skip audio drama and specials
        if "audio drama" in fl or "drama cd" in fl:
            continue

        files.append({
            "filename": filename,
            "download_url": f["url"],
            "source": classify_source(filename),
            "filetype": classify_filetype(filename),
            "size": f.get("size", 0),
        })
    return files


def search_jimaku(anilist_id: int | None, name: str | None) -> tuple[list[dict], str | None]:
    """Search jimaku. Returns (files, provider_info) or ([], None)."""
    api_key = os.getenv("JIMAKU_API_KEY")
    if not api_key:
        return [], None

    entries = jimaku_search(anilist_id, name, api_key)
    if not entries:
        return [], None

    entry = entries[0]
    entry_id = entry["id"]
    entry_name = entry.get("name", "?")
    info = f"jimaku.cc entry #{entry_id} ({entry_name})"

    files = jimaku_list_files(entry_id, api_key)
    return files, info


# ---------------------------------------------------------------------------
# kitsunekko.net (fallback)
# ---------------------------------------------------------------------------

def kitsunekko_find_directory(anime_name: str) -> str | None:
    """Find the kitsunekko directory URL for an anime."""
    # Try direct URL
    url = f"{KITSUNEKKO_BASE}/{urllib.parse.quote(anime_name, safe='')}/"
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return url
    except requests.RequestException:
        pass

    # Browse parent directory
    try:
        resp = requests.get(f"{KITSUNEKKO_BASE}/", timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href", "")
                text = link.get_text(strip=True)
                if anime_name.lower() in text.lower() or anime_name.lower() in href.lower():
                    if href.startswith("http"):
                        return href
                    if href.startswith("/"):
                        return f"https://kitsunekko.net{href}"
                    return f"{KITSUNEKKO_BASE}/{href}"
    except requests.RequestException:
        pass
    return None


def kitsunekko_list_files(directory_url: str) -> list[dict]:
    """List subtitle files in a kitsunekko directory."""
    resp = requests.get(directory_url, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    for link in soup.find_all("a"):
        text = link.get_text(strip=True)
        href = link.get("href", "")
        if not text or text == ".." or href == "../":
            continue
        if not any(text.lower().endswith(ext) for ext in (".srt", ".ass", ".zip", ".7z", ".rar")):
            continue

        if href.startswith("http"):
            download_url = href
        elif href.startswith("/"):
            download_url = f"https://kitsunekko.net{href}"
        else:
            download_url = f"{directory_url}{urllib.parse.quote(text)}"

        files.append({
            "filename": text,
            "download_url": download_url,
            "source": classify_source(text),
            "filetype": classify_filetype(text),
        })
    return files


def search_kitsunekko(name: str) -> tuple[list[dict], str | None]:
    """Search kitsunekko. Returns (files, provider_info) or ([], None)."""
    dir_url = kitsunekko_find_directory(name)
    if not dir_url:
        return [], None
    files = kitsunekko_list_files(dir_url)
    return files, f"kitsunekko.net ({dir_url})"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def group_by_source(files: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for f in files:
        groups.setdefault(f["source"], []).append(f)
    return groups


def sample_srt_timing(url: str, headers: dict | None = None, num_lines: int = 3) -> list[dict] | None:
    """Download an SRT and return first few entries with timing."""
    try:
        resp = requests.get(url, timeout=15, headers=headers or {})
        resp.raise_for_status()
        entries = []
        for block in re.split(r"\n\n+", resp.text.strip())[:num_lines]:
            lines = block.strip().split("\n")
            if len(lines) >= 3 and "-->" in lines[1]:
                entries.append({"timing": lines[1], "text": " ".join(lines[2:])})
        return entries or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find Japanese subtitles (jimaku.cc primary, kitsunekko fallback)"
    )
    parser.add_argument("--anilist-id", type=int, help="AniList media ID (preferred for jimaku)")
    parser.add_argument("--name", help="Anime title (used for kitsunekko and jimaku fallback)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-sample", action="store_true", help="Skip downloading sample timing from SRTs")
    args = parser.parse_args()

    if not args.anilist_id and not args.name:
        parser.error("Provide --anilist-id and/or --name")

    # Search jimaku first (needs API key + anilist_id or name)
    files, provider = search_jimaku(args.anilist_id, args.name)
    if files:
        print(f"Found on {provider}", file=sys.stderr)
    else:
        # Fallback to kitsunekko (needs name)
        name = args.name
        if not name and args.anilist_id:
            # Fetch name from AniList
            resp = requests.post(
                "https://graphql.anilist.co",
                json={"query": "{ Media(id: %d) { title { romaji } } }" % args.anilist_id},
            )
            name = resp.json()["data"]["Media"]["title"]["romaji"]
            print(f"Resolved AniList {args.anilist_id} -> {name}", file=sys.stderr)

        if name:
            print(f"Not on jimaku, trying kitsunekko for: {name}", file=sys.stderr)
            files, provider = search_kitsunekko(name)

    if not files:
        print("No Japanese subtitles found on jimaku or kitsunekko", file=sys.stderr)
        return 1

    # Group and optionally sample
    groups = group_by_source(files)

    if not args.no_sample:
        api_key = os.getenv("JIMAKU_API_KEY", "")
        headers = {"Authorization": api_key} if "jimaku" in (provider or "") else {}
        for source_files in groups.values():
            srt_files = [f for f in source_files if f["filetype"] == "srt"]
            if srt_files:
                sample = sample_srt_timing(srt_files[0]["download_url"], headers=headers)
                if sample:
                    srt_files[0]["sample_timing"] = sample

    # Output
    if args.json:
        print(json.dumps({
            "provider": provider,
            "sources": groups,
            "total_files": len(files),
        }, indent=2, ensure_ascii=False))
    else:
        print(f"\nAvailable JP subs ({provider})")
        print(f"Total files: {len(files)}\n")

        for source, source_files in sorted(groups.items()):
            type_counts = {}
            for f in source_files:
                type_counts[f["filetype"]] = type_counts.get(f["filetype"], 0) + 1
            summary = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()))
            print(f"  [{source}] ({summary})")

            for f in source_files:
                size_str = f" ({f['size']//1024}KB)" if f.get("size") else ""
                print(f"    {f['filename']}{size_str}")
                print(f"      {f['download_url']}")
                if f.get("sample_timing"):
                    print(f"      First lines:")
                    for entry in f["sample_timing"]:
                        print(f"        {entry['timing']}  {entry['text'][:60]}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
