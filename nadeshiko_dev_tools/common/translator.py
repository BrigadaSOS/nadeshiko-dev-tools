"""Shared translator helper for the YouTube + anime pipelines (fetch + process)."""

# Internal language codes mapped to the target-language codes the LLM translator
# turns into language names for its prompt.
LANG_CODES = {"en": "EN-US", "es": "ES"}


def get_translator():
    """Return the OpenAI-backed translator, or None if it can't be built (no API key)."""
    from nadeshiko_dev_tools.common.llm_translator import get_openai_translator

    return get_openai_translator()
