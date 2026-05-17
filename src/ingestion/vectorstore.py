"""
vectorstore.py — ChromaDB vector store for YojanaGPT.

Handles storing and retrieving scheme chunks using vector similarity.
All data is persisted to disk — survives restarts.

Usage:
    store = VectorStore()
    store.add_chunks(embedded_chunks)
    results = store.query(query_vector, n_results=5)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Where ChromaDB persists data on disk
DEFAULT_PERSIST_DIR = "data/chromadb"

# Name of the collection inside ChromaDB
COLLECTION_NAME = "yojanagpt_schemes"

# How many results to return by default
DEFAULT_N_RESULTS = 5


# ── Vector Store Class ────────────────────────────────────────────────────────

class VectorStore:
    """
    ChromaDB-backed vector store for scheme chunks.

    Persists data to disk so you only need to ingest once.
    Subsequent runs load from disk instantly.

    Usage:
        store = VectorStore()
        store.add_chunks(embedded_chunks)
        results = store.query(query_embedding, n_results=5)
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Initialise ChromaDB client and get or create the collection.

        Args:
            persist_dir:      Directory to store ChromaDB data on disk.
            collection_name:  Name of the ChromaDB collection.
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        # Ensure the storage directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialise persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
        )

        # Get existing collection or create a new one
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "YojanaGPT scheme chunks for RAG retrieval",
                "hnsw:space": "cosine",
            },
        )

        logger.info(
            "VectorStore ready | dir=%s | collection=%s | chunks=%d",
            persist_dir,
            collection_name,
            self.collection.count(),
        )

    # ── Write Operations ──────────────────────────────────────────

    def add_chunks(
        self,
        embedded_chunks: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        """
        Add embedded chunks to the ChromaDB collection.

        Skips chunks that are already stored — safe to call multiple
        times without creating duplicates.

        Args:
            embedded_chunks: List of chunks with 'text', 'metadata',
                             and 'embedding' keys.
            batch_size:      How many chunks to insert per ChromaDB call.

        Returns:
            Number of chunks successfully added.
        """
        if not embedded_chunks:
            logger.warning("add_chunks called with empty list")
            return 0

        # Build unique IDs for each chunk
        # Format: scheme_id__field__index
        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(embedded_chunks):
            scheme_id = chunk["metadata"].get("scheme_id", f"unknown_{i}")
            field = chunk["metadata"].get("field", "unknown")
            chunk_id = f"{scheme_id}__{field}__{i}"

            ids.append(chunk_id)
            texts.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])

        # Insert in batches to avoid memory issues with large datasets
        total_added = 0
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))

            batch_ids = ids[start:end]
            batch_texts = texts[start:end]
            batch_embeddings = embeddings[start:end]
            batch_metadatas = metadatas[start:end]

            try:
                # upsert = insert if not exists, update if exists
                # This makes the operation idempotent — safe to re-run
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
                total_added += len(batch_ids)
                logger.debug(
                    "Upserted batch | start=%d | end=%d | total_so_far=%d",
                    start,
                    end,
                    total_added,
                )

            except Exception as e:
                logger.error(
                    "Failed to upsert batch | start=%d | error=%s",
                    start,
                    str(e),
                )
                continue

        logger.info(
            "add_chunks complete | added=%d | collection_total=%d",
            total_added,
            self.collection.count(),
        )
        return total_added

    # ── Read Operations ───────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        n_results: int = DEFAULT_N_RESULTS,
        filter_state: Optional[str] = None,
        filter_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the most relevant chunks for a query embedding.

        Args:
            query_embedding:  Vector from embedder.embed_query().
            n_results:        Number of results to return.
            filter_state:     Optional — filter by state name or 'Central'.
            filter_category:  Optional — filter by category like 'Education'.

        Returns:
            List of result dicts, each containing:
              - text:      the chunk text
              - metadata:  scheme_id, field, name, state, category, source_url
              - distance:  similarity score (lower = more similar)
        """
        if not query_embedding:
            logger.error("query called with empty embedding")
            return []

        # Build optional metadata filter
        where_filter = self._build_filter(filter_state, filter_category)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        except Exception as e:
            logger.error("ChromaDB query failed: %s", str(e))
            return []

        # Reformat ChromaDB's response into clean dicts
        formatted = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            formatted.append({
                "text":     text,
                "metadata": metadata,
                "distance": round(distance, 4),
            })

        logger.debug(
            "Query returned %d results | top_distance=%.4f",
            len(formatted),
            distances[0] if distances else 0,
        )
        return formatted

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Return basic info about the collection.

        Returns:
            Dict with collection name and total chunk count.
        """
        return {
            "collection_name": self.collection_name,
            "total_chunks":    self.collection.count(),
            "persist_dir":     str(self.persist_dir),
        }

    def delete_collection(self) -> None:
        """
        Delete the entire collection and all its data.
        Used for resetting the vector store during development.
        """
        self.client.delete_collection(self.collection_name)
        logger.warning("Collection deleted: %s", self.collection_name)

    # ── Private Helpers ───────────────────────────────────────────

    def _build_filter(
        self,
        filter_state: Optional[str],
        filter_category: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Build a ChromaDB metadata filter dict.

        Returns None if no filters are specified.
        Returns a $and filter if both are specified.
        Returns a single filter if only one is specified.
        """
        conditions = []

        if filter_state:
            conditions.append({"state": {"$eq": filter_state}})
        if filter_category:
            conditions.append({"category": {"$eq": filter_category}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}