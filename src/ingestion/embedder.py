"""
embedder.py — Converts text chunks into vector embeddings.

Uses sentence-transformers with a multilingual model that supports
all 22 Indian languages needed for YojanaGPT.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - Supports 50+ languages including all Indian languages
  - 118MB — small enough to run on CPU
  - Downloads automatically on first use to ~/.cache/huggingface/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# This model supports all Indian languages and runs on CPU
# Downloads automatically on first use (~118MB)
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# How many chunks to embed in one batch
# Larger = faster but more memory usage
# 64 is safe for a machine without a GPU
DEFAULT_BATCH_SIZE = 64


# ── Embedder Class ────────────────────────────────────────────────────────────

class SchemeEmbedder:
    """
    Converts text chunks into vector embeddings using sentence-transformers.

    Usage:
        embedder = SchemeEmbedder()
        chunks_with_embeddings = embedder.embed_chunks(chunks)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialise the embedder and load the model.

        The model downloads automatically on first use.
        Subsequent runs load from local cache — no internet needed.

        Args:
            model_name: HuggingFace model name for sentence-transformers.
            batch_size: Number of chunks to embed per batch.
        """
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add vector embeddings to a list of chunks.

        Takes chunks from chunker.py (each with 'text' and 'metadata')
        and adds an 'embedding' key containing the vector.

        Args:
            chunks: List of chunk dicts from chunk_scheme().

        Returns:
            Same list with 'embedding' added to each chunk.
            Failed chunks are skipped and logged.

        Example:
            Input:  [{"text": "Eligibility: SC/ST students", "metadata": {...}}]
            Output: [{"text": "...", "metadata": {...}, "embedding": [0.1, 0.3, ...]}]
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list")
            return []

        # Extract just the text for batch embedding
        texts = [chunk["text"] for chunk in chunks]

        logger.info(
            "Embedding %d chunks in batches of %d | model=%s",
            len(texts),
            self.batch_size,
            self.model_name,
        )

        try:
            # encode() returns a numpy array of shape (num_chunks, embedding_dim)
            # show_progress_bar=True prints a progress bar for large batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        except Exception as e:
            logger.error("Embedding failed: %s", str(e))
            return []

        # Attach each embedding back to its chunk
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            enriched = {
                **chunk,
                "embedding": embedding.tolist(),
            }
            result.append(enriched)

        logger.info("Embedding complete | total=%d vectors", len(result))
        return result

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string for retrieval.

        Used at query time — takes the user's question and returns
        a vector that can be compared against stored chunk vectors.

        Args:
            query: The user's question in any language.

        Returns:
            Embedding vector as a list of floats.

        Example:
            >>> embedder = SchemeEmbedder()
            >>> vec = embedder.embed_query("मुझे किसान योजना चाहिए")
            >>> print(len(vec))  # 384
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    @property
    def embedding_dimension(self) -> int:
        """
        Return the dimension of the embedding vectors this model produces.
        Used when setting up ChromaDB collection.
        """
        return self.model.get_sentence_embedding_dimension()