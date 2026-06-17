"""Shared translator helpers for the YouTube + anime pipelines (fetch + process)."""

import logging
import os

import deepl as deepl_lib

logger = logging.getLogger(__name__)

# Map our internal language codes to DeepL target-language codes (also understood by the
# LLM backend, which translates these codes to language names for its prompt).
DEEPL_LANG = {"en": "EN-US", "es": "ES"}


def get_translator():
    """Return a translator for the configured backend, or None if it can't be built.

    TRANSLATOR=openai → OpenAI LLM backend (needs OPENAI_API_KEY).
    TRANSLATOR=deepl (default) → DeepL (needs TOKEN).
    """
    backend = os.getenv("TRANSLATOR", "deepl").lower()
    if backend == "openai":
        from nadeshiko_dev_tools.common.llm_translator import get_openai_translator

        return get_openai_translator()

    token = os.getenv("TOKEN")
    return deepl_lib.Translator(token) if token else None
