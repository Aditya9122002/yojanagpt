"""
llm.py — Calls the Groq API and returns the answer.

Uses Groq's free tier which provides:
  - 14,400 requests/day free
  - Very fast inference (runs Llama 3 on Groq hardware)
  - Simple OpenAI-compatible API

Model: llama-3.3-70b-versatile
  - Strong multilingual support including Indian languages
  - Good at following instructions
  - Free on Groq
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Groq API constants
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2


class GeminiClient:
    """
    LLM client using Groq API (OpenAI-compatible).
    Named GeminiClient for backward compatibility with pipeline.py.

    Usage:
        client = GeminiClient()
        answer = client.generate("What is PM Kisan scheme?")
        print(answer)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY in your .env file. "
                "Get a free key at https://console.groq.com"
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.endpoint = f"{GROQ_API_BASE}/chat/completions"

        logger.info("LLM client initialised | model=%s | max_tokens=%d", model, max_tokens)

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Groq and return the text response.

        Groq uses the OpenAI chat completions format:
          POST /chat/completions
          { "model": "...", "messages": [{"role": "user", "content": "..."}] }

        Args:
            prompt: The full prompt string (system + context + question).

        Returns:
            The model's text response as a string.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt passed to generate()")

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.debug("Sending request to Groq | prompt_length=%d chars", len(prompt))

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.endpoint,
                json=request_body,
                headers=headers,
            )

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(
                "Groq API error | status=%d | response=%s",
                response.status_code,
                error_text,
            )
            raise ValueError(
                f"Groq API returned status {response.status_code}: {error_text}"
            )

        data = response.json()

        # OpenAI-compatible response format:
        # { "choices": [{ "message": { "content": "..." } }] }
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error("Unexpected Groq response structure: %s", data)
            raise ValueError(f"Could not extract text from Groq response: {e}")

        # Log token usage
        usage = data.get("usage", {})
        if usage:
            logger.info(
                "Groq usage | prompt_tokens=%d | completion_tokens=%d | total=%d",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

        return text.strip()