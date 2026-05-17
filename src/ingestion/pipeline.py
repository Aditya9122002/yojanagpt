"""
pipeline.py — Orchestrates the full ingestion pipeline.

Connects chunker → embedder → vectorstore into one clean flow.
Reads scraped JSONL data, processes it, stores in ChromaDB.

Usage:
    pipeline = IngestionPipeline()
    pipeline.run(input_file="data/raw/schemes_detail.jsonl")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.scraper.models import SchemeDetail
from src.ingestion.chunker import chunk_schemes
from src.ingestion.embedder import SchemeEmbedder
from src.ingestion.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# How many schemes to process in one pipeline batch
# Keeps memory usage under control
DEFAULT_BATCH_SIZE = 50

# Default input file — output of the scraper
DEFAULT_INPUT_FILE = "data/raw/schemes_detail.jsonl"

# Default ChromaDB storage directory
DEFAULT_CHROMA_DIR = "data/chromadb"


# ── Pipeline Class ────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Full ingestion pipeline: JSONL → chunks → embeddings → ChromaDB.

    Processes schemes in batches to keep memory usage manageable.
    Safe to re-run — upsert logic prevents duplicates.

    Usage:
        pipeline = IngestionPipeline()
        pipeline.run("data/raw/schemes_detail.jsonl")
    """

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialise the pipeline components.

        Args:
            chroma_dir:       Where ChromaDB stores its data.
            batch_size:       Schemes to process per batch.
            embedding_model:  Override the default embedding model name.
        """
        self.batch_size = batch_size

        # Initialise components
        logger.info("Initialising ingestion pipeline...")

        embedder_kwargs = {}
        if embedding_model:
            embedder_kwargs["model_name"] = embedding_model

        self.embedder = SchemeEmbedder(**embedder_kwargs)
        self.vectorstore = VectorStore(persist_dir=chroma_dir)

        logger.info(
            "Pipeline ready | batch_size=%d | chroma_dir=%s",
            batch_size,
            chroma_dir,
        )

    # ── Main Entry Point ──────────────────────────────────────────

    def run(
        self,
        input_file: str = DEFAULT_INPUT_FILE,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full ingestion pipeline.

        Args:
            input_file: Path to the scraped JSONL file.
            limit:      Optional — process only first N schemes (for testing).

        Returns:
            Summary dict with counts of processed, chunked, embedded, stored.
        """
        start_time = datetime.utcnow()
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_file}\n"
                f"Run the scraper first: python -m src.scraper.cli scrape --test"
            )

        logger.info("=" * 60)
        logger.info("Ingestion pipeline started | file=%s", input_file)
        logger.info("=" * 60)

        # Step 1 — load schemes from JSONL
        schemes = self._load_schemes(input_path, limit=limit)
        if not schemes:
            logger.error("No schemes loaded. Aborting.")
            return {"success": False, "error": "No schemes loaded"}

        # Step 2 — process in batches
        total_chunks_stored = 0
        total_schemes_processed = 0
        total_failed = 0

        batches = self._make_batches(schemes, self.batch_size)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Ingesting schemes..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("ingest", total=len(schemes))

            for batch_num, batch in enumerate(batches):
                logger.debug(
                    "Processing batch %d | size=%d",
                    batch_num + 1,
                    len(batch),
                )

                try:
                    chunks_stored = self._process_batch(batch)
                    total_chunks_stored += chunks_stored
                    total_schemes_processed += len(batch)

                except Exception as e:
                    logger.error(
                        "Batch %d failed | error=%s",
                        batch_num + 1,
                        str(e),
                    )
                    total_failed += len(batch)

                progress.advance(task, len(batch))

        # Step 3 — summary
        duration = (datetime.utcnow() - start_time).seconds
        info = self.vectorstore.get_collection_info()

        summary = {
            "success":                  True,
            "schemes_processed":        total_schemes_processed,
            "schemes_failed":           total_failed,
            "chunks_stored":            total_chunks_stored,
            "total_chunks_in_store":    info["total_chunks"],
            "duration_seconds":         duration,
        }

        logger.info("=" * 60)
        logger.info("Ingestion complete")
        logger.info("  Schemes processed : %d", total_schemes_processed)
        logger.info("  Schemes failed    : %d", total_failed)
        logger.info("  Chunks stored     : %d", total_chunks_stored)
        logger.info("  Total in ChromaDB : %d", info["total_chunks"])
        logger.info("  Duration          : %ds", duration)
        logger.info("=" * 60)

        return summary

    # ── Private Methods ───────────────────────────────────────────

    def _load_schemes(
        self,
        path: Path,
        limit: Optional[int] = None,
    ) -> List[SchemeDetail]:
        """
        Load SchemeDetail objects from a JSONL file.

        Args:
            path:  Path to the JSONL file.
            limit: Load only first N schemes if set.

        Returns:
            List of validated SchemeDetail objects.
        """
        schemes = []
        failed = 0

        logger.info("Loading schemes from %s", path)

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if limit and len(schemes) >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    scheme = SchemeDetail(**data)
                    schemes.append(scheme)

                except Exception as e:
                    logger.warning(
                        "Failed to load scheme at line %d | error=%s",
                        line_num,
                        str(e),
                    )
                    failed += 1
                    continue

        logger.info(
            "Loaded %d schemes | failed=%d | file=%s",
            len(schemes),
            failed,
            path,
        )
        return schemes

    def _process_batch(self, batch: List[SchemeDetail]) -> int:
        """
        Process one batch of schemes through the full pipeline.

        Steps: chunk → embed → store

        Args:
            batch: List of SchemeDetail objects.

        Returns:
            Number of chunks stored for this batch.
        """
        # Step 1 — chunk
        chunks = chunk_schemes(batch)
        if not chunks:
            logger.warning("No chunks produced for batch")
            return 0

        # Step 2 — embed
        embedded_chunks = self.embedder.embed_chunks(chunks)
        if not embedded_chunks:
            logger.warning("No embeddings produced for batch")
            return 0

        # Step 3 — store
        stored = self.vectorstore.add_chunks(embedded_chunks)
        return stored

    def _make_batches(
        self,
        items: List[Any],
        batch_size: int,
    ) -> List[List[Any]]:
        """
        Split a list into batches of a given size.

        Args:
            items:      List to split.
            batch_size: Max items per batch.

        Returns:
            List of batches.
        """
        return [
            items[i: i + batch_size]
            for i in range(0, len(items), batch_size)
        ]