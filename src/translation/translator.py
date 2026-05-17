"""
translator.py — Translates text between Indian languages and English.

Uses deep-translator (Google Translate) as the translation backend.
This is the lightweight fallback — IndicTrans2 will replace this
in a future week for higher quality Indic language translation.

Why deep-translator:
  - No API key needed (uses Google Translate free tier)
  - Supports all Indian languages
  - Simple pip install, no model download
  - Good enough quality for our current stage

Future: Replace with AI4Bharat IndicTrans2 for:
  - Higher accuracy on Indic languages
  - No rate limits
  - Offline capability
  - Better handling of code-mixed text
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound, RequestError

from .detector import detect_language, get_google_code, is_english

logger = logging.getLogger(__name__)

# Cache translations to avoid re-translating the same text
# Simple dict cache — good enough for session-level caching
_translation_cache: dict = {}

# Rate limiting — Google Translate free tier has limits
TRANSLATION_DELAY = 0.5  # seconds between translation requests


class SchemeTranslator:
    """
    Translates questions and answers for YojanaGPT.

    The translation pattern:
      1. Detect language of user question
      2. If not English, translate question → English
      3. Run RAG pipeline with English question
      4. Translate English answer → user's language
      5. Return translated answer

    Usage:
        translator = SchemeTranslator()

        # Translate to English for RAG
        english_q, lang = translator.to_english("PM Kisan ke liye kaun eligible hai?")
        # english_q = "Who is eligible for PM Kisan?"
        # lang = "hi"

        # Translate answer back
        hindi_answer = translator.from_english(english_answer, target_lang="hi")
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._last_request_time = 0.0
        logger.info("SchemeTranslator initialised | cache=%s", use_cache)

    def to_english(self, text: str) -> tuple[str, str]:
        """
        Translate text to English. Detects source language automatically.

        Args:
            text: Input text in any supported language.

        Returns:
            Tuple of (translated_text, source_language_code).
            If text is already English, returns (original_text, "en").

        Examples:
            to_english("PM Kisan ke liye kaun eligible hai?")
            → ("Who is eligible for PM Kisan?", "hi")

            to_english("Who is eligible for PM Kisan?")
            → ("Who is eligible for PM Kisan?", "en")
        """
        if not text or not text.strip():
            return text, "en"

        # Detect language
        source_lang = detect_language(text)

        # Skip translation if already English
        if source_lang == "en":
            logger.debug("Text already in English, skipping translation")
            return text, "en"

        logger.info("Translating %s → English | text: %s...", source_lang, text[:50])

        translated = self._translate(
            text=text,
            source=get_google_code(source_lang),
            target="en",
        )

        return translated, source_lang

    def from_english(self, text: str, target_lang: str) -> str:
        """
        Translate English text to the target language.

        Args:
            text:        English text to translate.
            target_lang: Target language code (e.g. "hi", "ta", "bn").

        Returns:
            Translated text in target language.
            Returns original text if target is English or translation fails.

        Examples:
            from_english("All landholding farmers are eligible.", "hi")
            → "सभी भूमिधारक किसान पात्र हैं।"
        """
        if not text or not text.strip():
            return text

        # Skip if target is English
        if target_lang == "en":
            return text

        logger.info("Translating English → %s | text: %s...", target_lang, text[:50])

        translated = self._translate(
            text=text,
            source="en",
            target=get_google_code(target_lang),
        )

        return translated

    def _translate(self, text: str, source: str, target: str) -> str:
        """
        Core translation method with caching and rate limiting.

        Args:
            text:   Text to translate.
            source: Source language code (Google format).
            target: Target language code (Google format).

        Returns:
            Translated text, or original text if translation fails.
        """
        # Check cache first
        cache_key = f"{source}:{target}:{text[:100]}"
        if self.use_cache and cache_key in _translation_cache:
            logger.debug("Cache hit for translation")
            return _translation_cache[cache_key]

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < TRANSLATION_DELAY:
            time.sleep(TRANSLATION_DELAY - elapsed)

        try:
            translator = GoogleTranslator(source=source, target=target)

            # Google Translate has a 5000 char limit per request
            # Split long texts and translate in chunks
            if len(text) > 4500:
                translated = self._translate_long_text(translator, text)
            else:
                translated = translator.translate(text)

            self._last_request_time = time.time()

            # Cache the result
            if self.use_cache:
                _translation_cache[cache_key] = translated

            return translated or text

        except TranslationNotFound:
            logger.warning("Translation not found for text: %s...", text[:50])
            return text

        except RequestError as e:
            logger.warning("Translation request failed: %s", e)
            return text

        except Exception as e:
            logger.warning("Unexpected translation error: %s", e)
            return text

    def _translate_long_text(self, translator: GoogleTranslator, text: str) -> str:
        """
        Translate text longer than 4500 chars by splitting into chunks.
        Splits on sentence boundaries to preserve meaning.
        """
        # Split on newlines first, then by length
        chunks = []
        current = ""

        for line in text.split("\n"):
            if len(current) + len(line) < 4500:
                current += line + "\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = line + "\n"

        if current:
            chunks.append(current.strip())

        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            try:
                translated = translator.translate(chunk)
                translated_chunks.append(translated or chunk)
                time.sleep(TRANSLATION_DELAY)
            except Exception:
                translated_chunks.append(chunk)

        return "\n".join(translated_chunks)