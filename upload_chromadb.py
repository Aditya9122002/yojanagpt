"""
upload_chromadb.py — One-time script to upload ChromaDB to HuggingFace Dataset.

Run this ONCE from your local machine before deploying to HF Spaces.

Usage:
    python upload_chromadb.py

Requirements:
    pip install huggingface-hub
    HF_TOKEN must be set in .env or environment
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config — change these if needed ──────────────────────────────
CHROMA_DIR   = Path("data/chromadb")
DATASET_ID = "Adi12340/yojanagpt-chromadb"   # Will be created if it doesn't exist
HF_TOKEN     = os.getenv("HF_TOKEN", "")
# ─────────────────────────────────────────────────────────────────


def upload() -> None:
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not set.\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Create a token with WRITE access\n"
            "3. Add HF_TOKEN=your_token to your .env file"
        )

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}.\n"
            "Run ingestion first: python -m src.ingestion.cli ingest"
        )

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    # Create the dataset repo if it doesn't exist
    logger.info("Creating/verifying HF Dataset: %s", DATASET_ID)
    api.create_repo(
        repo_id=DATASET_ID,
        repo_type="dataset",
        private=True,          # Keep private — your vector DB
        exist_ok=True,
    )
    logger.info("Dataset repo ready.")

    # Upload entire chromadb folder
    logger.info("Uploading %s → HF Dataset (this may take a few minutes)…", CHROMA_DIR)
    api.upload_folder(
        folder_path=str(CHROMA_DIR),
        repo_id=DATASET_ID,
        repo_type="dataset",
        commit_message="Upload ChromaDB snapshot — YojanaGPT",
    )
    logger.info("Upload complete! Dataset: https://huggingface.co/datasets/%s", DATASET_ID)
    logger.info(
        "\nNext steps:\n"
        "  1. Go to your HF Space settings\n"
        "  2. Add these secrets:\n"
        "       GROQ_API_KEY = your_groq_key\n"
        "       HF_TOKEN     = your_hf_token\n"
        "       HF_DATASET_ID = %s\n"
        "  3. Push your code to trigger a rebuild",
        DATASET_ID,
    )


if __name__ == "__main__":
    upload()