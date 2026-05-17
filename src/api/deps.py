"""
deps.py — Shared dependencies for FastAPI routes.

The RAG pipeline is expensive to initialise:
  - Loads the 471MB embedding model into memory
  - Connects to ChromaDB
  - Loads the translation components

We do this ONCE at startup and reuse the same instance for all requests.
This is the standard FastAPI pattern for expensive shared resources.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from src.retrieval.pipeline import YojanaRAGPipeline

logger = logging.getLogger(__name__)

# Module-level singleton — created once, reused forever
_pipeline: Optional[YojanaRAGPipeline] = None


def get_pipeline() -> YojanaRAGPipeline:
    """
    Returns the shared RAG pipeline instance.
    Creates it on first call, reuses on all subsequent calls.

    This is used as a FastAPI dependency:
        @app.post("/ask")
        def ask(request: AskRequest, pipeline: YojanaRAGPipeline = Depends(get_pipeline)):
            ...
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising RAG pipeline (first request)...")
        _pipeline = YojanaRAGPipeline(
            chroma_dir="data/chromadb",
            top_k=5,
            enable_translation=True,
        )
        logger.info("RAG pipeline ready")
    return _pipeline


def initialise_pipeline() -> None:
    """
    Eagerly initialise the pipeline at startup.
    Called from the FastAPI lifespan event so the first request
    doesn't have to wait for model loading.
    """
    get_pipeline()