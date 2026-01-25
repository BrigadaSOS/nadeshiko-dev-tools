import logging

from anilist import Client

logger = logging.getLogger(__name__)


class CachedAnilist:
    def __init__(self):
        self.client = Client()
        self.cached_results = {}
        self.id_cache = {}

    def get_anime_with_id(self, anilist_id: int):
        if anilist_id in self.id_cache:
            return self.id_cache[anilist_id]

        try:
            anime_result = self.client.get_anime(anilist_id)
            self.id_cache[anilist_id] = anime_result
            return anime_result
        except Exception as e:
            raise Exception(f"Failed to fetch Anilist ID {anilist_id}: {e}") from e

    def search(self, search_query: str):
        return self.client.search(search_query)
