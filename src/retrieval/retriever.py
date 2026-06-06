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
    # PM Kisan
    "pm kisan": "pm-kisan",
    "pmkisan": "pm-kisan",
    "kisan samman": "pm-kisan",
    "pm-kisan": "pm-kisan",
    "kisan nidhi": "pm-kisan",
    # PM Kusum
    "pm kusum": "pm-kusum",
    "kusum yojana": "pm-kusum",
    "kusum": "pm-kusum",
    "pm-kusum": "pm-kusum",
    "kisan urja": "pm-kusum",
    "solar pump farmer": "pm-kusum",
    # Ayushman Bharat
    "ayushman": "pmjay",
    "pmjay": "pmjay",
    "pm jay": "pmjay",
    "jan arogya": "pmjay",
    "ayushman bharat": "pmjay",
    # PM Awas
    "pmay": "pmay",
    "pm awas": "pmay",
    "awas yojana": "pmay",
    # Ujjwala
    "ujjwala": "pmuy",
    "pmuy": "pmuy",
    "pm ujjwala": "pmuy",
    "free lpg": "pmuy",
    # Mudra
    "mudra": "mudra",
    "pm mudra": "mudra",
    "mudra loan": "mudra",
    "pmmy": "mudra",
    # Jan Dhan
    "pmjdy": "pmjdy",
    "jan dhan": "pmjdy",
    # PMFBY
    "pmfby": "pmfby",
    "fasal bima": "pmfby",
    "crop insurance": "pmfby",
    # Skill India
    "pmkvy": "pmkvy",
    "skill india": "pmkvy",
    "kaushal vikas": "pmkvy",
    # Sukanya
    "sukanya": "ssy",
    "sukanya samriddhi": "ssy",
    # Beti Bachao
    "beti bachao": "bbbp",
    "beti padhao": "bbbp",
    "bbbp": "bbbp",
    # Atal Pension
    "atal pension": "apy",
    "apy": "apy",
    # Startup / Standup
    "standup india": "sui",
    "startup india": "startup-india",
    # Scholarships
    "nos-swd": "nos-swd",
    "national overseas scholarship": "nos-swd",
    "nsp": "nsp",
    # PMSBY / PMJJBY
    "pmsby": "pmsby",
    "suraksha bima": "pmsby",
    "pmjjby": "pmjjby",
    "jeevan jyoti": "pmjjby",
    # PM Vishwakarma
    "vishwakarma": "pm-vishwakarma",
    "pm vishwakarma": "pm-vishwakarma",
    # SVANidhi
    "svanidhi": "pm-svanidhi",
    "street vendor": "pm-svanidhi",
    # Swachh Bharat
    "swachh bharat": "sbm",
    "sbm": "sbm",
    # e-Shram
    "e shram": "e-shram",
    "eshram": "e-shram",
    # Garib Kalyan
    "garib kalyan": "pmgkay",
    "pmgkay": "pmgkay",
    "free ration": "pmgkay",
}

# Full name expansions for semantic search boost
QUERY_EXPANSION = {
    "pm kusum": "Pradhan Mantri Kisan Urja Suraksha Evam Utthaan Mahabhiyan PM-KUSUM solar pump farmer",
    "kusum yojana": "Pradhan Mantri Kisan Urja Suraksha Evam Utthaan Mahabhiyan solar pump",
    "kusum": "Pradhan Mantri Kisan Urja Suraksha Evam Utthaan Mahabhiyan PM-KUSUM solar pump",
    "pm kisan": "Pradhan Mantri Kisan Samman Nidhi PM-KISAN farmer income support 6000",
    "kisan samman": "Pradhan Mantri Kisan Samman Nidhi PM-KISAN farmer",
    "ayushman bharat": "Pradhan Mantri Jan Arogya Yojana PM-JAY health insurance 5 lakh",
    "ayushman": "Pradhan Mantri Jan Arogya Yojana PM-JAY health insurance hospital",
    "pm awas": "Pradhan Mantri Awas Yojana PMAY housing scheme home",
    "awas yojana": "Pradhan Mantri Awas Yojana PMAY housing scheme",
    "pmfby": "Pradhan Mantri Fasal Bima Yojana crop insurance farmer premium",
    "fasal bima": "Pradhan Mantri Fasal Bima Yojana crop insurance farmer",
    "mudra loan": "Pradhan Mantri Mudra Yojana PMMY micro enterprise small business loan",
    "mudra yojana": "Pradhan Mantri Mudra Yojana PMMY small business loan shishu kishor tarun",
    "sukanya": "Sukanya Samriddhi Yojana girl child savings account interest deposit",
    "sukanya samriddhi": "Sukanya Samriddhi Yojana SSY girl child savings scheme interest",
    "ujjwala": "Pradhan Mantri Ujjwala Yojana PMUY free LPG gas connection BPL women",
    "pm ujjwala": "Pradhan Mantri Ujjwala Yojana free LPG gas connection below poverty line",
    "skill india": "Pradhan Mantri Kaushal Vikas Yojana PMKVY skill training certification job",
    "kaushal vikas": "Pradhan Mantri Kaushal Vikas Yojana PMKVY skill development training",
    "jan dhan": "Pradhan Mantri Jan Dhan Yojana PMJDY zero balance bank account financial inclusion",
    "pmjdy": "Pradhan Mantri Jan Dhan Yojana zero balance savings account",
    "atal pension": "Atal Pension Yojana APY retirement pension unorganised sector subscriber",
    "vishwakarma": "PM Vishwakarma Yojana artisan craftsman traditional skills toolkit loan training",
    "pm vishwakarma": "PM Vishwakarma Yojana artisan craftsman skills collateral free loan",
    "svanidhi": "PM SVANidhi street vendor micro credit loan working capital",
    "street vendor": "PM SVANidhi Pradhan Mantri Street Vendor AtmaNirbhar Nidhi micro loan",
    "beti bachao": "Beti Bachao Beti Padhao BBBP girl child education welfare scheme",
    "beti padhao": "Beti Bachao Beti Padhao girl child education scheme",
    "swachh bharat": "Swachh Bharat Mission SBM toilet construction individual household sanitation ODF",
    "toilet scheme": "Swachh Bharat Mission toilet construction gram panchayat",
    "standup india": "Stand Up India SC ST women entrepreneur bank loan greenfield enterprise",
    "startup india": "Startup India scheme fund of funds innovation startup recognition",
    "garib kalyan": "Pradhan Mantri Garib Kalyan Anna Yojana PMGKAY free food grain ration",
    "free ration": "Pradhan Mantri Garib Kalyan Anna Yojana free food grain PDS",
    "e shram": "e-Shram card unorganised worker registration social security database",
    "eshram": "e-Shram unorganised worker construction agriculture domestic worker",
    "jeevan jyoti": "Pradhan Mantri Jeevan Jyoti Bima Yojana PMJJBY life insurance death benefit",
    "suraksha bima": "Pradhan Mantri Suraksha Bima Yojana PMSBY accident insurance disability",
    "pmsby": "Pradhan Mantri Suraksha Bima Yojana accidental death disability insurance",
    "pmjjby": "Pradhan Mantri Jeevan Jyoti Bima Yojana life insurance renewable",
    "national scholarship": "National Scholarship Portal NSP student merit scholarship education",
    "nsp scholarship": "National Scholarship Portal NSP pre matric post matric scholarship",
    "solar pump": "PM KUSUM Pradhan Mantri Kisan Urja Suraksha solar irrigation pump farmer",
    "kisan urja": "PM KUSUM Pradhan Mantri Kisan Urja Suraksha Evam Utthaan solar pump",
    "vikas bharat": "Vikas Bharat Rozgar Yojana employment scheme rural development",
}


def expand_query(query: str) -> str:
    """Expand short/common scheme names to full names for better semantic matching."""
    query_lower = query.lower().strip()
    expansions = []
    for short_name, full_name in QUERY_EXPANSION.items():
        if short_name in query_lower:
            expansions.append(full_name)
    if expansions:
        expanded = query + " " + " ".join(expansions)
        return expanded
    return query




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
        # Expand query with full scheme names for better matching
        expanded = expand_query(question)
        query_embedding = self.model.encode(
            expanded,
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