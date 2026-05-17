"""
retriever.py — Searches ChromaDB and returns the most relevant chunks.

Uses hybrid search:
  1. Semantic search — embed the question and find similar chunks
  2. Keyword search — extract scheme names/abbreviations from the question
     and do exact metadata filter lookups
  3. Merge both result sets and return the best combined results

This handles cases like "PM Kisan" → "Pradhan Mantri Kisan Samman Nidhi"
where the embedding model fails to match abbreviations to full names.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "yojanagpt_schemes"
DEFAULT_CHROMA_DIR = "data/chromadb"
DEFAULT_TOP_K = 5

# Common scheme abbreviations and keywords mapped to slug fragments
# Add more as we discover them from user queries
SCHEME_KEYWORDS = {
    "pm kisan": "pm-kisan",
    "pmkisan": "pm-kisan",
    "kisan samman": "pm-kisan",
    "pm-kisan": "pm-kisan",
    "pmsby": "pmsby",
    "pmjdy": "pmjdy",
    "pmay": "pmay",
    "pm awas": "pmay",
    "ujjwala": "pmuy",
    "pmuy": "pmuy",
    "ayushman": "pmjay",
    "pmjay": "pmjay",
    "mudra": "mudra",
    "pm mudra": "mudra",
    "standup india": "sui",
    "startup india": "startup-india",
    "skill india": "pmkvy",
    "pmkvy": "pmkvy",
    "nos-swd": "nos-swd",
    "national overseas scholarship": "nos-swd",
}


@dataclass
class RetrievedChunk:
    """A single chunk returned from ChromaDB search."""
    chunk_id: str
    text: str
    scheme_id: str
    scheme_name: str
    chunk_type: str
    source_url: str
    distance: float


class SchemeRetriever:
    """
    Searches ChromaDB for scheme chunks relevant to a user question.
    Uses hybrid search (semantic + keyword) for best results.

    Usage:
        retriever = SchemeRetriever()
        chunks = retriever.search("PM Kisan ke liye kaun eligible hai?", top_k=5)
        for chunk in chunks:
            print(chunk.scheme_name, chunk.chunk_type)
    """

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.top_k = top_k

        logger.info("Loading embedding model: %s", embedding_model)
        self.model = SentenceTransformer(embedding_model)

        logger.info("Connecting to ChromaDB at: %s", chroma_dir)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(name=collection_name)
        count = self.collection.count()
        logger.info("ChromaDB ready | chunks=%d", count)

    def search(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Hybrid search — semantic + keyword combined.

        Args:
            question: User's question in any language.
            top_k:    Number of chunks to return.

        Returns:
            List of RetrievedChunk objects, best results first.
        """
        k = top_k or self.top_k

        if not question or not question.strip():
            return []

        # Step 1 — semantic search
        semantic_chunks = self._semantic_search(question, top_k=k)

        # Step 2 — keyword search (finds schemes by name/abbreviation)
        keyword_chunks = self._keyword_search(question, top_k=k)

        # Step 3 — merge, deduplicate, return top k
        merged = self._merge_results(semantic_chunks, keyword_chunks, top_k=k)

        logger.info(
            "Hybrid search | semantic=%d | keyword=%d | merged=%d | question='%s...'",
            len(semantic_chunks),
            len(keyword_chunks),
            len(merged),
            question[:50],
        )
        return merged

    def _semantic_search(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """Embed the question and search ChromaDB by vector similarity."""
        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_query_results(results)

    def _keyword_search(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """
        Look for known scheme names/abbreviations in the question.
        If found, fetch chunks for those schemes directly by scheme_id.

        This handles abbreviations like "PM Kisan" → scheme_id "pm-kisan"
        which semantic search misses because the embedding distance is too large.
        """
        question_lower = question.lower()
        matched_slugs = set()

        # Check against known keyword mappings
        for keyword, slug in SCHEME_KEYWORDS.items():
            if keyword in question_lower:
                matched_slugs.add(slug)
                logger.debug("Keyword match: '%s' → '%s'", keyword, slug)

        if not matched_slugs:
            return []

        # Fetch chunks for each matched scheme
        all_chunks = []
        for slug in matched_slugs:
            try:
                results = self.collection.get(
                    where={"scheme_id": slug},
                    include=["documents", "metadatas"],
                    limit=top_k,
                )
                chunks = self._parse_get_results(results, distance=0.0)
                all_chunks.extend(chunks)
                logger.debug("Keyword fetch: slug=%s → %d chunks", slug, len(chunks))
            except Exception as e:
                logger.warning("Keyword search failed for slug=%s: %s", slug, e)

        return all_chunks

    def _merge_results(
        self,
        semantic: List[RetrievedChunk],
        keyword: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Merge semantic and keyword results.

        Keyword results go first (distance=0.0, highest priority).
        Semantic results fill the remaining slots.
        Deduplication by chunk_id.
        """
        seen_ids = set()
        merged = []

        # Keyword results first — highest priority
        for chunk in keyword:
            if chunk.chunk_id not in seen_ids:
                merged.append(chunk)
                seen_ids.add(chunk.chunk_id)

        # Semantic results fill remaining slots
        for chunk in semantic:
            if chunk.chunk_id not in seen_ids:
                merged.append(chunk)
                seen_ids.add(chunk.chunk_id)

        return merged[:top_k]

    def _parse_query_results(self, results: dict) -> List[RetrievedChunk]:
        """Parse ChromaDB query() results into RetrievedChunk objects."""
        chunks = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunks.append(self._make_chunk(chunk_id, text, metadata, float(distance)))

        return chunks

    def _parse_get_results(self, results: dict, distance: float = 0.0) -> List[RetrievedChunk]:
        """Parse ChromaDB get() results into RetrievedChunk objects."""
        chunks = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            chunks.append(self._make_chunk(chunk_id, text, metadata, distance))

        return chunks

    def _make_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: dict,
        distance: float,
    ) -> RetrievedChunk:
        """Build a RetrievedChunk from raw ChromaDB data."""
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text or "",
            scheme_id=metadata.get("scheme_id", ""),
            scheme_name=metadata.get("name", metadata.get("scheme_name", "Unknown Scheme")),
            chunk_type=metadata.get("field", metadata.get("chunk_type", "unknown")),
            source_url=metadata.get("source_url", ""),
            distance=distance,
        )