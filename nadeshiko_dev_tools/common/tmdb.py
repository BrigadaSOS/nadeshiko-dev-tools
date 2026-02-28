import logging
import os

import requests

logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/original"

# TMDB status -> normalized status strings (matching AniList style)
STATUS_MAP = {
    "Returning Series": "RELEASING",
    "Ended": "FINISHED",
    "Canceled": "CANCELLED",
    "In Production": "NOT_YET_RELEASED",
    "Planned": "NOT_YET_RELEASED",
    "Pilot": "NOT_YET_RELEASED",
    # Movie statuses
    "Released": "FINISHED",
    "Post Production": "NOT_YET_RELEASED",
    "Rumored": "NOT_YET_RELEASED",
}


class TmdbFuzzyDate:
    """Mimics AniList FuzzyDate with year/month/day attributes."""

    def __init__(self, date_str: str | None):
        self.year = None
        self.month = None
        self.day = None
        if date_str:
            parts = date_str.split("-")
            if len(parts) >= 1 and parts[0]:
                self.year = int(parts[0])
            if len(parts) >= 2 and parts[1]:
                self.month = int(parts[1])
            if len(parts) >= 3 and parts[2]:
                self.day = int(parts[2])

    def __bool__(self):
        return self.year is not None


class TmdbTitle:
    """Mimics AniList title object with romaji/english/native."""

    def __init__(self, romaji: str | None, english: str | None, native: str | None):
        self.romaji = romaji
        self.english = english
        self.native = native


class TmdbCover:
    """Mimics AniList coverImage object."""

    def __init__(self, poster_path: str | None):
        self.extra_large = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None


class TmdbStudioNode:
    """Single studio/production company."""

    def __init__(self, name: str):
        self.name = name


class TmdbStudios:
    """Mimics AniList studios object with nodes list."""

    def __init__(self, companies: list[dict]):
        self.nodes = [TmdbStudioNode(c["name"]) for c in companies] if companies else []


class TmdbMediaData:
    """Wrapper for TMDB data that mirrors AnimeData attribute access patterns.

    Provides the same interface that save_info_json() in file_utils.py expects:
    .id, .title.romaji/english/native, .format, .status, .episodes, .source,
    .genres, .synonyms, .start_date, .end_date, .season, .season_year,
    .cover.extra_large, .banner, .studios.nodes[0].name,
    .relations, .characters, .streaming_episodes
    """

    def __init__(
        self,
        en_data: dict,
        ja_data: dict,
        media_type: str = "tv",
        season_data: dict | None = None,
    ):
        self._en = en_data
        self._ja = ja_data
        self._media_type = media_type
        self._season_data = season_data

        self.id = en_data["id"]
        self.tmdb_season_number = None

        # Title: romaji and native from Japanese response, english from English response
        ja_name = ja_data.get("name") or ja_data.get("title")
        en_name = en_data.get("name") or en_data.get("title")
        original_name = en_data.get("original_name") or en_data.get("original_title")

        # If season data available, append season name to titles
        if season_data:
            self.tmdb_season_number = season_data.get("season_number")
            season_name = season_data.get("name", "")
            if season_name and en_name:
                en_name = f"{en_name} - {season_name}"

        self.title = TmdbTitle(
            romaji=original_name or ja_name,
            english=en_name,
            native=original_name or ja_name,
        )

        # Format
        if media_type == "movie":
            self.format = "MOVIE"
        else:
            self.format = "TV"

        # Status
        raw_status = en_data.get("status", "")
        self.status = STATUS_MAP.get(raw_status, raw_status)

        # Episodes: use season-specific count if available
        if season_data:
            self.episodes = len(season_data.get("episodes", []))
        else:
            self.episodes = en_data.get("number_of_episodes")

        # Source
        self.source = "DRAMA"

        # Genres
        self.genres = [g["name"] for g in en_data.get("genres", [])]

        # Synonyms - not available from TMDB
        self.synonyms = []

        # Dates: use season-specific dates if available
        if season_data and season_data.get("air_date"):
            self.start_date = TmdbFuzzyDate(season_data["air_date"])
            # End date: use last episode air_date if available
            episodes = season_data.get("episodes", [])
            last_ep_date = episodes[-1].get("air_date") if episodes else None
            self.end_date = TmdbFuzzyDate(last_ep_date or season_data["air_date"])
            first_date = season_data["air_date"]
        else:
            first_date_key = "first_air_date" if media_type == "tv" else "release_date"
            last_date_key = "last_air_date" if media_type == "tv" else "release_date"
            self.start_date = TmdbFuzzyDate(en_data.get(first_date_key))
            self.end_date = TmdbFuzzyDate(en_data.get(last_date_key))
            first_date = en_data.get(first_date_key, "")

        # Season info - derive from air date month
        if first_date and len(first_date) >= 7:
            month = int(first_date.split("-")[1])
            self.season = (
                "WINTER" if month <= 3
                else "SPRING" if month <= 6
                else "SUMMER" if month <= 9
                else "FALL"
            )
            self.season_year = int(first_date.split("-")[0])
        else:
            self.season = None
            self.season_year = None

        # Cover image: prefer season poster, fall back to show poster
        season_poster = season_data.get("poster_path") if season_data else None
        self.cover = TmdbCover(season_poster or en_data.get("poster_path"))

        # Banner image (no season-specific banner, use show-level)
        backdrop = en_data.get("backdrop_path")
        self.banner = f"{TMDB_IMAGE_BASE}{backdrop}" if backdrop else None

        # Studios / production companies
        self.studios = TmdbStudios(en_data.get("production_companies", []))

        # Not supported for TMDB
        self.relations = None
        self.characters = None
        self.streaming_episodes = None


class CachedTmdb:
    """TMDB API client with caching, mirroring CachedAnilist interface."""

    def __init__(self):
        self.id_cache: dict[tuple[str, int, int | None], TmdbMediaData] = {}
        self._session = requests.Session()

        # Prefer bearer token, fall back to API key as query param
        self._bearer_token = os.environ.get("TMDB_API_TOKEN")
        self._api_key = os.environ.get("TMDB_API_KEY")

        if self._bearer_token:
            self._session.headers["Authorization"] = f"Bearer {self._bearer_token}"
        elif not self._api_key:
            logger.warning(
                "Neither TMDB_API_TOKEN nor TMDB_API_KEY is set. TMDB requests will fail."
            )

    def _request(self, endpoint: str, params: dict | None = None) -> dict:
        """Make an authenticated request to the TMDB API."""
        url = f"{TMDB_BASE_URL}{endpoint}"
        if params is None:
            params = {}

        # If no bearer token, use API key as query param
        if not self._bearer_token and self._api_key:
            params["api_key"] = self._api_key

        response = self._session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_media_with_id(
        self, tmdb_id: int, media_type: str = "tv", tmdb_season: int | None = None,
    ) -> TmdbMediaData:
        """Fetch media details by TMDB ID, optionally for a specific season.

        Args:
            tmdb_id: The TMDB ID of the media.
            media_type: Either "tv" or "movie".
            tmdb_season: TMDB season number (1-based). If provided, fetches
                season-specific data (episode count, dates, poster).

        Returns:
            TmdbMediaData with attributes compatible with save_info_json().
        """
        cache_key = (media_type, tmdb_id, tmdb_season)
        if cache_key in self.id_cache:
            return self.id_cache[cache_key]

        try:
            endpoint = f"/{media_type}/{tmdb_id}"

            # Fetch English details
            en_data = self._request(endpoint, {"language": "en-US"})

            # Fetch Japanese details for native names
            ja_data = self._request(endpoint, {"language": "ja-JP"})

            # Fetch season-specific data if requested
            season_data = None
            if tmdb_season is not None and media_type == "tv":
                try:
                    season_data = self._request(
                        f"/tv/{tmdb_id}/season/{tmdb_season}",
                        {"language": "en-US"},
                    )
                except requests.HTTPError as se:
                    logger.warning(
                        f"Could not fetch season {tmdb_season} for TMDB {tmdb_id}: "
                        f"{se.response.status_code}"
                    )

            result = TmdbMediaData(en_data, ja_data, media_type, season_data)
            self.id_cache[cache_key] = result
            return result

        except requests.HTTPError as e:
            # Auto-detect: if TV lookup 404s, try movie (and vice versa)
            if e.response.status_code == 404:
                alt_type = "movie" if media_type == "tv" else "tv"
                alt_key = (alt_type, tmdb_id, tmdb_season)
                if alt_key not in self.id_cache:
                    try:
                        alt_endpoint = f"/{alt_type}/{tmdb_id}"
                        en_data = self._request(alt_endpoint, {"language": "en-US"})
                        ja_data = self._request(alt_endpoint, {"language": "ja-JP"})
                        result = TmdbMediaData(en_data, ja_data, alt_type)
                        self.id_cache[alt_key] = result
                        logger.info(
                            f"TMDB ID {tmdb_id} not found as {media_type}, "
                            f"resolved as {alt_type}"
                        )
                        return result
                    except requests.HTTPError:
                        pass  # Fall through to original error
                else:
                    return self.id_cache[alt_key]

            raise Exception(
                f"Failed to fetch TMDB {media_type} ID {tmdb_id}: {e.response.status_code} "
                f"{e.response.text}"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to fetch TMDB {media_type} ID {tmdb_id}: {e}") from e

    def search(self, query: str, media_type: str = "tv") -> list[dict]:
        """Search TMDB for media matching the query.

        Args:
            query: Search string.
            media_type: Either "tv" or "movie".

        Returns:
            List of result dicts from TMDB search API.
        """
        try:
            data = self._request(
                f"/search/{media_type}",
                {"query": query, "language": "en-US"},
            )
            return data.get("results", [])
        except requests.HTTPError as e:
            logger.error(f"TMDB search failed: {e.response.status_code} {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"TMDB search failed: {e}")
            return []
