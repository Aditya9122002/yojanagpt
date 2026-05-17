"""
client.py — HTTP client for myscheme.gov.in API.

All network requests in the scraper go through this file.
Handles sessions, retries, rate limiting, and error logging.

Two public methods:
  - fetch_scheme_list(offset, language)  → raw list API response
  - fetch_scheme_detail(slug, language)  → raw detail API response

API discovery notes (confirmed via Chrome DevTools, May 2025):
  LIST   → https://api.myscheme.gov.in/search/v6/schemes
  DETAIL → https://api.myscheme.gov.in/schemes/v6/public/schemes
  All requests require X-Api-Key header and browser-like User-Agent.
  Without these the server returns 403.

IMPORTANT — q param encoding issue:
  The list API requires q=%5B%5D (URL-encoded empty array []).
  If we pass q="[]" via requests params={}, requests encodes [] → %5B%5D,
  then its internal URL builder encodes again → %255B%255D, causing 500.
  Fix: build the list URL as a complete string with q=%5B%5D already in it,
  and pass NO params dict. This way requests sends the URL as-is.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://api.myscheme.gov.in"

# Returns paginated list of schemes (search/browse endpoint)
LIST_ENDPOINT = "/search/v6/schemes"

# Returns full details for a single scheme
# Params: slug, lang
DETAIL_ENDPOINT = "/schemes/v6/public/schemes"

# These headers are mandatory — confirmed via Chrome DevTools.
# The API returns 403 without the X-Api-Key and matching Origin/Referer.
DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Origin": "https://www.myscheme.gov.in",
    "Referer": "https://www.myscheme.gov.in/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Mobile Safari/537.36"
    ),
    "X-Api-Key": "tYTy5eEhlu9rFjyxuCr7ra7ACp4dv1RH8gWuHTDc",
}

# Seconds to wait between requests — be polite to the government server
REQUEST_DELAY = 1.0


# ── Session Factory ───────────────────────────────────────────────────────────

def _build_session(max_retries: int = 3) -> requests.Session:
    """
    Build a requests Session with connection-level retry logic.

    This handles low-level retries (connection refused, DNS failure).
    Application-level retries (500 errors, timeouts) are handled by tenacity
    decorators on the fetch methods below.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(DEFAULT_HEADERS)

    return session


# ── Main Client Class ─────────────────────────────────────────────────────────

class MySchemeClient:
    """
    HTTP client for the myscheme.gov.in API.

    Usage:
        client = MySchemeClient()
        raw = client.fetch_scheme_list(offset=0)
        detail = client.fetch_scheme_detail(slug="pm-kisan")
    """

    def __init__(
        self,
        max_retries: int = 3,
        request_delay: float = REQUEST_DELAY,
        timeout: int = 30,
    ):
        self.session = _build_session(max_retries=max_retries)
        self.request_delay = request_delay
        self.timeout = timeout
        self._last_request_time: float = 0.0

        logger.info(
            "MySchemeClient initialised | delay=%.1fs | timeout=%ds | retries=%d",
            request_delay,
            timeout,
            max_retries,
        )

    # ── Rate Limiting ─────────────────────────────────────────────

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        remaining = self.request_delay - elapsed
        if remaining > 0:
            logger.debug("Rate limit wait: %.2fs", remaining)
            time.sleep(remaining)

    def _record_request_time(self) -> None:
        self._last_request_time = time.time()

    # ── Core Request Method ───────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a single GET request with retry logic.

        Args:
            url:    Full URL to request. For list API this includes all params
                    already embedded in the URL string (to avoid double-encoding).
            params: Optional query parameters dict. Only used for detail API
                    where params don't have encoding issues.

        Returns:
            Parsed JSON response as a dict.
        """
        self._wait_for_rate_limit()
        logger.debug("GET %s | params=%s", url, params)

        response = self.session.get(url, params=params, timeout=self.timeout)
        self._record_request_time()
        response.raise_for_status()

        return response.json()

    # ── Public Methods ────────────────────────────────────────────

    def fetch_scheme_list(
        self,
        offset: int = 0,
        size: int = 10,
        language: str = "en",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch one page of scheme summaries from the list API.

        The list API uses Elasticsearch under the hood.
        Results are paginated via `from` (offset) and `size` parameters.
        Total scheme count is in response["data"]["summary"]["total"].

        IMPORTANT: We build the URL as a complete string instead of using
        params={} because the q=[] parameter gets double-encoded by requests
        otherwise, causing a 500 error from the server.

        Args:
            offset:   Pagination offset (0, 10, 20, ...).
            size:     Number of schemes per page.
            language: Language code for response content.

        Returns:
            Raw API response dict on success, None on failure.

        Response structure:
            {
              "data": {
                "hits": {
                  "hits": [
                    { "_source": { "slug": "pm-kisan", ... } },
                    ...
                  ]
                },
                "summary": { "total": 4696 }
              }
            }
        """
        # Build URL with all params embedded as a string.
        # q=%5B%5D is the URL-encoded form of [] (empty filters array).
        # We must keep it pre-encoded here — do NOT pass via params={}.
        url = (
            f"{BASE_URL}{LIST_ENDPOINT}"
            f"?lang={language}"
            f"&q=%5B%5D"
            f"&keyword="
            f"&sort="
            f"&from={offset}"
            f"&size={size}"
        )

        try:
            # Pass params=None — all params are already in the URL string
            data = self._get(url, params=None)

            hits = data.get("data", {}).get("hits", {}).get("hits", [])
            logger.info(
                "Fetched scheme list | offset=%d | schemes_in_page=%d",
                offset,
                len(hits),
            )
            return data

        except requests.HTTPError as e:
            logger.error(
                "HTTP error fetching list at offset=%d | status=%d | %s",
                offset,
                e.response.status_code,
                str(e),
            )
            return None

        except (requests.Timeout, requests.ConnectionError) as e:
            logger.error(
                "Network error fetching list at offset=%d | %s",
                offset,
                str(e),
            )
            return None

    def fetch_scheme_detail(
        self,
        slug: str,
        language: str = "en",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch full detail for a single scheme by its slug.

        The detail API returns all scheme data nested under data.en.*
        Structure confirmed:
          data.en.basicDetails       — name, ministry, tags, dates
          data.en.schemeContent      — description, benefits_md, exclusions_md
          data.en.applicationProcess — how to apply steps
          data.en.eligibilityCriteria — eligibility rules
          data.en.schemeDefinitions  — key terms

        Args:
            slug:     URL slug for the scheme (e.g. "pm-kisan", "nos-swd").
            language: Language code for response content.

        Returns:
            Raw API response dict on success, None on failure.
        """
        url = BASE_URL + DETAIL_ENDPOINT
        params = {
            "slug": slug,
            "lang": language,
        }

        try:
            data = self._get(url, params=params)
            logger.info("Fetched detail for slug=%s", slug)
            return data

        except requests.HTTPError as e:
            status = e.response.status_code
            if status == 404:
                logger.warning("Scheme not found (404) | slug=%s", slug)
            else:
                logger.error(
                    "HTTP error fetching detail | slug=%s | status=%d",
                    slug,
                    status,
                )
            return None

        except (requests.Timeout, requests.ConnectionError) as e:
            logger.error(
                "Network error fetching detail | slug=%s | %s",
                slug,
                str(e),
            )
            return None

    def close(self) -> None:
        """Close the underlying HTTP session and free resources."""
        self.session.close()
        logger.info("MySchemeClient session closed")

    def __enter__(self) -> "MySchemeClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()