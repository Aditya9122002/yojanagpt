"""
startup.py — Downloads ChromaDB from HuggingFace Dataset before API starts.

Run once at container startup. If data/chromadb/ already exists, skips download.

Usage:
    python startup.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CHROMA_DIR = Path("data/chromadb")

# Set these as HuggingFace Space secrets:
#   HF_TOKEN      — your HuggingFace read token
#   HF_DATASET_ID — e.g. "Adi12340/yojanagpt-chromadb"
HF_DATASET_ID = os.getenv("HF_DATASET_ID", "Adi12340/yojanagpt-chromadb")
HF_TOKEN      = os.getenv("HF_TOKEN", "")


def download_chromadb() -> None:
    """Download ChromaDB snapshot from HuggingFace Dataset."""

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        logger.info("ChromaDB already exists at %s — skipping download.", CHROMA_DIR)
        return

    logger.info("ChromaDB not found. Downloading from HuggingFace Dataset: %s", HF_DATASET_ID)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    if not HF_TOKEN:
        logger.warning(
            "HF_TOKEN not set. Download will fail if dataset is private. "
            "Set HF_TOKEN as a Space secret."
        )

    CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)

    try:
        local_dir = snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            local_dir=str(CHROMA_DIR),
            token=HF_TOKEN or None,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        logger.info("ChromaDB downloaded successfully to: %s", local_dir)

    except Exception as e:
        logger.error("Failed to download ChromaDB: %s", e)
        logger.error(
            "Make sure:\n"
            "  1. HF_DATASET_ID is correct (e.g. 'Adi12340/yojanagpt-chromadb')\n"
            "  2. HF_TOKEN is set as a Space secret\n"
            "  3. The dataset exists on HuggingFace"
        )
        sys.exit(1)


def verify_chromadb() -> None:
    """Quick sanity check — make sure ChromaDB is readable."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection("yojanagpt_schemes")
        count = collection.count()
        logger.info("ChromaDB verified — %d chunks loaded.", count)
    except Exception as e:
        logger.error("ChromaDB verification failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    logger.info("=== YojanaGPT Startup ===")
    download_chromadb()
    verify_chromadb()
    logger.info("=== Startup complete — API ready to launch ===")