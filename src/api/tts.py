"""
tts.py — Text-to-speech using gTTS (Google Text-to-Speech).

Converts LLM answer text to audio in the user's detected language.
Returns audio as bytes that FastAPI streams back to the browser.

Supported Indian languages via gTTS:
  hi = Hindi, bn = Bengali, te = Telugu, mr = Marathi,
  ta = Tamil, gu = Gujarati, kn = Kannada, ml = Malayalam,
  pa = Punjabi, en = English (fallback)
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# gTTS language code mapping
# Maps langdetect codes → gTTS language codes
LANG_MAP = {
    "hi": "hi",   # Hindi
    "bn": "bn",   # Bengali
    "te": "te",   # Telugu
    "mr": "mr",   # Marathi
    "ta": "ta",   # Tamil
    "gu": "gu",   # Gujarati
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "pa": "pa",   # Punjabi
    "en": "en",   # English
    "ur": "ur",   # Urdu
    # Fallback for unsupported languages
    "or": "hi",   # Odia → Hindi fallback
    "as": "bn",   # Assamese → Bengali fallback
    "ne": "ne",   # Nepali
}

DEFAULT_LANG = "hi"  # Default to Hindi if detection fails


def text_to_speech(text: str, lang_code: str = "en") -> bytes:
    """
    Convert text to speech audio bytes using gTTS.

    Args:
        text:      The text to convert (LLM answer).
        lang_code: Language code from langdetect (e.g. 'hi', 'ta', 'en').

    Returns:
        MP3 audio as bytes, ready to stream to browser.

    Raises:
        ImportError: If gtts is not installed.
        Exception:   If gTTS API call fails.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError(
            "gtts not installed. Run: pip install gtts"
        )

    # Map detected language to gTTS code
    gtts_lang = LANG_MAP.get(lang_code, "hi")

    # Truncate very long answers to avoid timeout
    # gTTS works best under ~500 words
    words = text.split()
    if len(words) > 200:
        text = " ".join(words[:200]) + "..."
        logger.info("Text truncated to 200 words for TTS.")

    logger.info("Generating TTS | lang=%s | gtts_lang=%s | chars=%d",
                lang_code, gtts_lang, len(text))

    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()

    except Exception as e:
        # If language not supported, fall back to English
        logger.warning("gTTS failed for lang=%s, falling back to English: %s", gtts_lang, e)
        tts = gTTS(text=text, lang="en", slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()