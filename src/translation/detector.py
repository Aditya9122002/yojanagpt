"""
detector.py — Detects the language of a given text.

Uses langdetect library which supports 55 languages including
most Indian languages written in their native scripts.

Supported Indian languages (with their codes):
  hi — Hindi
  bn — Bengali
  te — Telugu
  mr — Marathi
  ta — Tamil
  gu — Gujarati
  kn — Kannada
  ml — Malayalam
  pa — Punjabi
  or — Odia
  ur — Urdu
  en — English (Hinglish falls here mostly)

Note on code-mixed text (Hinglish, Tanglish):
  "PM Kisan ke liye kaun eligible hai?" — langdetect sees this as Hindi
  "PM Kisan eligibility kya hai yaar" — may detect as English or Hindi
  We default to English for code-mixed text since our chunks are in English
  and the LLM handles Hinglish queries well without translation.
"""

from __future__ import annotations

import logging
from typing import Optional

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)

# Make langdetect deterministic — without this it gives different results
# for the same input on different runs
DetectorFactory.seed = 0

# Languages we support for translation
# Maps langdetect code → our internal language name → Google Translate code
SUPPORTED_LANGUAGES = {
    "hi": {"name": "Hindi",      "google_code": "hi"},
    "bn": {"name": "Bengali",    "google_code": "bn"},
    "te": {"name": "Telugu",     "google_code": "te"},
    "mr": {"name": "Marathi",    "google_code": "mr"},
    "ta": {"name": "Tamil",      "google_code": "ta"},
    "gu": {"name": "Gujarati",   "google_code": "gu"},
    "kn": {"name": "Kannada",    "google_code": "kn"},
    "ml": {"name": "Malayalam",  "google_code": "ml"},
    "pa": {"name": "Punjabi",    "google_code": "pa"},
    "or": {"name": "Odia",       "google_code": "or"},
    "ur": {"name": "Urdu",       "google_code": "ur"},
    "en": {"name": "English",    "google_code": "en"},
}

# If detection confidence is low or language unsupported, default to English
DEFAULT_LANGUAGE = "en"

# Minimum text length for reliable detection
MIN_TEXT_LENGTH = 10


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text: Input text in any language.

    Returns:
        Language code string — one of the keys in SUPPORTED_LANGUAGES,
        or "en" as fallback if detection fails or language unsupported.

    Examples:
        detect_language("PM Kisan ke liye kaun eligible hai?") → "hi"
        detect_language("Who is eligible for PM Kisan?") → "en"
        detect_language("நான் யார்?") → "ta"
    """
    if not text or not text.strip():
        return DEFAULT_LANGUAGE

    # Short text — detection unreliable, default to English
    if len(text.strip()) < MIN_TEXT_LENGTH:
        logger.debug("Text too short for reliable detection, defaulting to English")
        return DEFAULT_LANGUAGE

    try:
        detected = detect(text)
        logger.debug("Detected language: %s for text: %s...", detected, text[:50])

        # Check if we support this language
        if detected in SUPPORTED_LANGUAGES:
            return detected
        else:
            logger.debug(
                "Detected language '%s' not in supported list, defaulting to English",
                detected,
            )
            return DEFAULT_LANGUAGE

    except LangDetectException as e:
        logger.warning("Language detection failed: %s — defaulting to English", e)
        return DEFAULT_LANGUAGE


def is_english(text: str) -> bool:
    """
    Quick check if text is in English.
    Used to skip translation when not needed.
    """
    return detect_language(text) == "en"


def get_language_name(lang_code: str) -> str:
    """
    Get the human readable name for a language code.

    Args:
        lang_code: Language code like "hi", "ta", "en"

    Returns:
        Language name like "Hindi", "Tamil", "English"
    """
    lang_info = SUPPORTED_LANGUAGES.get(lang_code, {})
    return lang_info.get("name", "Unknown")


def get_google_code(lang_code: str) -> str:
    """
    Get the Google Translate code for a language.
    Most codes are the same but some differ.
    """
    lang_info = SUPPORTED_LANGUAGES.get(lang_code, {})
    return lang_info.get("google_code", "en")