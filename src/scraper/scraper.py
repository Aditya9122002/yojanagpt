"""
scraper.py — Orchestrates the full myscheme.gov.in scraping pipeline.

Responsibilities:
  - Paginate through all schemes via the list API
  - Fetch detail for each scheme
  - Save results to disk as JSONL with checkpointing
  - Track and save failed slugs for retry

Usage (via cli.py):
  python -m src.scraper.cli scrape --test
  python -m src.scraper.cli scrape --full
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .client import MySchemeClient
from .models import SchemeDetail, SchemeListItem, ScrapeResult
from .parser import parse_scheme_detail, parse_scheme_list

logger = logging.getLogger(__name__)

# ── Default Paths ─────────────────────────────────────────────────────────────

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_LIST_FILE = DEFAULT_RAW_DIR / "schemes_list.jsonl"
DEFAULT_DETAIL_FILE = DEFAULT_RAW_DIR / "schemes_detail.jsonl"
DEFAULT_FAILED_FILE = DEFAULT_RAW_DIR / "failed_slugs.txt"

# How often to flush results to disk
CHECKPOINT_EVERY = 100


# ── Main Scraper Class ────────────────────────────────────────────────────────

class YojanaGPTScraper:
    """
    Orchestrates the full scraping pipeline for myscheme.gov.in.

    Usage:
        scraper = YojanaGPTScraper(test_mode=True)
        scraper.run()
    """

    def __init__(
        self,
        test_mode: bool = True,
        test_limit: int = 20,
        raw_dir: Path = DEFAULT_RAW_DIR,
        checkpoint_every: int = CHECKPOINT_EVERY,
    ):
        """
        Initialise the scraper.

        Args:
            test_mode:        If True, only scrape first test_limit schemes.
            test_limit:       Number of schemes to scrape in test mode.
            raw_dir:          Directory to save scraped data.
            checkpoint_every: Save progress to disk every N schemes.
        """
        self.test_mode = test_mode
        self.test_limit = test_limit
        self.raw_dir = Path(raw_dir)
        self.checkpoint_every = checkpoint_every

        # File paths
        self.list_file = self.raw_dir / "schemes_list.jsonl"
        self.detail_file = self.raw_dir / "schemes_detail.jsonl"
        self.failed_file = self.raw_dir / "failed_slugs.txt"

        # Runtime state
        self.client = MySchemeClient()
        self.scraped_slugs: Set[str] = set()
        self.failed_slugs: List[str] = []
        self.results_buffer: List[SchemeDetail] = []

        # Ensure output directory exists
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        mode_label = f"TEST (limit={test_limit})" if test_mode else "FULL"
        logger.info("YojanaGPTScraper initialised | mode=%s | output=%s", mode_label, raw_dir)

    # ── Step 1: Scrape Scheme List ────────────────────────────────

    def scrape_list(self) -> List[SchemeListItem]:
        """
        Paginate through the list API and collect all scheme slugs.

        Returns:
            List of SchemeListItem objects — one per scheme found.
        """
        logger.info("Starting scheme list scrape...")

        all_items: List[SchemeListItem] = []
        offset = 0
        page_size = 10
        total_schemes = self.test_limit if self.test_mode else 4676

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Fetching scheme list..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("list", total=total_schemes)

            while offset < total_schemes:
                # In test mode, don't fetch more than we need
                current_size = min(page_size, total_schemes - offset)

                raw = self.client.fetch_scheme_list(
                    offset=offset,
                    size=current_size,
                )

                if raw is None:
                    logger.warning("Got None response at offset=%d, skipping page", offset)
                    offset += page_size
                    continue

                items = parse_scheme_list(raw)

                if not items:
                    logger.info("Empty page at offset=%d — likely end of data", offset)
                    break

                all_items.extend(items)
                progress.advance(task, len(items))

                logger.debug("offset=%d | collected=%d total", offset, len(all_items))
                offset += page_size

        # Save the list to disk
        self._save_list(all_items)

        logger.info("Scheme list scrape complete | total=%d", len(all_items))
        return all_items

    # ── Step 2: Scrape Scheme Details ─────────────────────────────

    def scrape_details(self, items: List[SchemeListItem]) -> None:
        """
        Fetch full detail for each scheme in the list.
        Saves results to disk with checkpointing.

        Args:
            items: List of SchemeListItem from scrape_list()
        """
        logger.info("Starting detail scrape | total_schemes=%d", len(items))

        # Load already-scraped slugs to support resuming
        self._load_existing_slugs()

        # Filter out already done
        remaining = [i for i in items if i.slug not in self.scraped_slugs]
        logger.info(
            "Resuming | already_done=%d | remaining=%d",
            len(items) - len(remaining),
            len(remaining),
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Scraping scheme details..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("details", total=len(remaining))

            for i, item in enumerate(remaining):
                result = self._scrape_one(item.slug)

                if result.success and result.data:
                    self.results_buffer.append(result.data)
                    self.scraped_slugs.add(item.slug)
                else:
                    self.failed_slugs.append(item.slug)

                progress.advance(task, 1)

                # Checkpoint — flush buffer to disk periodically
                if len(self.results_buffer) >= self.checkpoint_every:
                    self._flush_buffer()
                    logger.info(
                        "Checkpoint saved | progress=%d/%d",
                        i + 1,
                        len(remaining),
                    )

            # Final flush for whatever is left in the buffer
            if self.results_buffer:
                self._flush_buffer()

        # Save failed slugs for retry
        if self.failed_slugs:
            self._save_failed_slugs()

        logger.info(
            "Detail scrape complete | success=%d | failed=%d",
            len(self.scraped_slugs),
            len(self.failed_slugs),
        )

    # ── Full Pipeline ─────────────────────────────────────────────

    def run(self) -> None:
        """
        Run the complete scraping pipeline — list then details.
        This is the main entry point called by cli.py.
        """
        start_time = datetime.utcnow()
        logger.info("=" * 60)
        logger.info("YojanaGPT Scraper started | %s", start_time.isoformat())
        logger.info("=" * 60)

        try:
            # Step 1 — collect all slugs
            items = self.scrape_list()

            if not items:
                logger.error("No schemes found in list scrape. Aborting.")
                return

            # Step 2 — fetch details for each slug
            self.scrape_details(items)

        finally:
            self.client.close()

        end_time = datetime.utcnow()
        duration = (end_time - start_time).seconds
        logger.info("=" * 60)
        logger.info("Scraper finished | duration=%ds", duration)
        logger.info("Output: %s", self.detail_file)
        logger.info("=" * 60)

    # ── Private Helpers ───────────────────────────────────────────

    def _scrape_one(self, slug: str) -> ScrapeResult:
        """
        Scrape detail for a single scheme slug.
        Always returns a ScrapeResult — never raises.

        Args:
            slug: Scheme URL slug.

        Returns:
            ScrapeResult with success=True and data, or success=False and error.
        """
        try:
            raw = self.client.fetch_scheme_detail(slug=slug)

            if raw is None:
                return ScrapeResult(
                    slug=slug,
                    success=False,
                    error="Client returned None — HTTP or network error",
                )

            detail = parse_scheme_detail(raw, slug=slug)

            if detail is None:
                return ScrapeResult(
                    slug=slug,
                    success=False,
                    error="Parser returned None — could not extract fields",
                )

            return ScrapeResult(slug=slug, success=True, data=detail)

        except Exception as e:
            logger.error("Unexpected error scraping slug=%s | %s", slug, str(e))
            return ScrapeResult(slug=slug, success=False, error=str(e))

    def _save_list(self, items: List[SchemeListItem]) -> None:
        """Save scheme list to JSONL file."""
        with open(self.list_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(item.json() + "\n")
        logger.info("Saved scheme list | file=%s | count=%d", self.list_file, len(items))

    def _flush_buffer(self) -> None:
        """
        Append buffered SchemeDetail results to the detail JSONL file.
        Clears the buffer after writing.
        """
        with open(self.detail_file, "a", encoding="utf-8") as f:
            for detail in self.results_buffer:
                f.write(detail.json() + "\n")
        logger.debug("Flushed %d records to disk", len(self.results_buffer))
        self.results_buffer.clear()

    def _load_existing_slugs(self) -> None:
        """
        Load slugs already saved in the detail file.
        Enables resuming a interrupted scrape without re-scraping done schemes.
        """
        if not self.detail_file.exists():
            return

        count = 0
        with open(self.detail_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    slug = obj.get("scheme_id")
                    if slug:
                        self.scraped_slugs.add(slug)
                        count += 1
                except json.JSONDecodeError:
                    continue

        logger.info("Loaded %d existing scraped slugs from disk", count)

    def _save_failed_slugs(self) -> None:
        """Save failed slugs to a text file — one slug per line."""
        with open(self.failed_file, "w", encoding="utf-8") as f:
            for slug in self.failed_slugs:
                f.write(slug + "\n")
        logger.warning(
            "Saved %d failed slugs to %s",
            len(self.failed_slugs),
            self.failed_file,
        )