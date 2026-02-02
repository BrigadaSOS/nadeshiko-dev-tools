import logging

import requests
from anilist import Client

logger = logging.getLogger(__name__)

# Custom GraphQL query with all needed fields
ANIME_QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    id
    title {
      romaji
      english
      native
    }
    format
    status
    genres
    episodes
    season
    seasonYear
    synonyms
    source
    startDate {
      year
      month
      day
    }
    endDate {
      year
      month
      day
    }
    coverImage {
      extraLarge
      large
      medium
    }
    bannerImage
    studios(isMain: true) {
      nodes {
        id
        name
        isAnimationStudio
        siteUrl
      }
    }
    relations {
      edges {
        relationType
        node {
          id
          type
          title {
            romaji
            english
            native
          }
        }
      }
    }
    characters(sort: ROLE, perPage: 25) {
      edges {
        role
        node {
          id
          name {
            full
            native
          }
          image {
            medium
          }
        }
        voiceActors(language: JAPANESE) {
          id
          name {
            full
            native
          }
          image {
            medium
          }
        }
      }
    }
  }
}
"""


class AnimeData:
    """Wrapper class for anime data from AniList GraphQL response."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str):
        # Handle special attribute names with underscores
        if name == "start_date":
            return FuzzyDate(self._data.get("startDate"))
        elif name == "end_date":
            return FuzzyDate(self._data.get("endDate"))
        elif name == "season_year":
            return self._data.get("seasonYear")
        elif name == "cover":
            cover_data = self._data.get("coverImage")
            if cover_data:
                return type("Cover", (), {"extra_large": cover_data.get("extraLarge")})()
            return None
        elif name == "banner":
            return self._data.get("bannerImage")

        # Handle nested objects
        value = self._data.get(name)
        if isinstance(value, dict):
            return NestedObject(value)
        elif isinstance(value, list):
            return [NestedObject(item) if isinstance(item, dict) else item for item in value]
        return value


class FuzzyDate:
    """Wrapper for AniList fuzzy date format."""

    def __init__(self, date_data: dict | None):
        self.year = date_data.get("year") if date_data else None
        self.month = date_data.get("month") if date_data else None
        self.day = date_data.get("day") if date_data else None


class NestedObject:
    """Generic wrapper for nested objects."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str):
        value = self._data.get(name)
        if isinstance(value, dict):
            return NestedObject(value)
        elif isinstance(value, list):
            return [NestedObject(item) if isinstance(item, dict) else item for item in value]
        return value


class CachedAnilist:
    def __init__(self):
        self.client = Client()
        self.cached_results = {}
        self.id_cache = {}
        self.graphql_url = "https://graphql.anilist.co"

    def get_anime_with_id(self, anilist_id: int):
        if anilist_id in self.id_cache:
            return self.id_cache[anilist_id]

        try:
            # Make direct GraphQL request with custom query
            response = requests.post(
                self.graphql_url,
                json={"query": ANIME_QUERY, "variables": {"id": anilist_id}},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                raise Exception(f"GraphQL errors: {result['errors']}")

            anime_data = result["data"]["Media"]
            anime_result = AnimeData(anime_data)

            self.id_cache[anilist_id] = anime_result
            return anime_result
        except Exception as e:
            raise Exception(f"Failed to fetch Anilist ID {anilist_id}: {e}") from e

    def search(self, search_query: str):
        return self.client.search(search_query)
