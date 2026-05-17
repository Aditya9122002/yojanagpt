"""
test_vectorstore.py — Unit tests for src/ingestion/vectorstore.py

Uses an in-memory ChromaDB client so tests run fast
and leave no files on disk.

Run with:
  pytest tests/unit/test_vectorstore.py -v
"""

from __future__ import annotations

import pytest
import chromadb
from unittest.mock import patch
from src.ingestion.vectorstore import VectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def in_memory_store():
    """
    VectorStore backed by an in-memory ChromaDB client.
    Fast, isolated, leaves no files on disk.
    Patched so VectorStore uses in-memory client instead of persistent.
    """
    with patch("src.ingestion.vectorstore.chromadb.PersistentClient") as mock:
        mock.return_value = chromadb.Client()
        store = VectorStore(persist_dir="test_tmp", collection_name="test_collection")
        yield store


@pytest.fixture
def sample_chunks():
    """Three embedded chunks from two different schemes."""
    return [
        {
            "text": "Eligibility: SC/ST students who secured admission abroad.",
            "metadata": {
                "scheme_id":   "nos-swd",
                "name":        "National Overseas Scholarship",
                "field":       "eligibility",
                "field_label": "Eligibility",
                "state":       "Central",
                "category":    "Education",
                "ministry":    "Ministry of Social Justice",
                "source_url":  "https://www.myscheme.gov.in/schemes/nos-swd",
            },
            # Real embedding dimension for paraphrase-multilingual-MiniLM-L12-v2 is 384
            # We use 3 dimensions here for simplicity in tests
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "text": "Benefits: Tuition fee and living allowance covered.",
            "metadata": {
                "scheme_id":   "nos-swd",
                "name":        "National Overseas Scholarship",
                "field":       "benefit",
                "field_label": "Benefits",
                "state":       "Central",
                "category":    "Education",
                "ministry":    "Ministry of Social Justice",
                "source_url":  "https://www.myscheme.gov.in/schemes/nos-swd",
            },
            "embedding": [0.4, 0.5, 0.6],
        },
        {
            "text": "Eligibility: Farmers with less than 2 hectares of land.",
            "metadata": {
                "scheme_id":   "pmkisan",
                "name":        "PM Kisan Samman Nidhi",
                "field":       "eligibility",
                "field_label": "Eligibility",
                "state":       "Central",
                "category":    "Agriculture",
                "ministry":    "Ministry of Agriculture",
                "source_url":  "https://www.myscheme.gov.in/schemes/pmkisan",
            },
            "embedding": [0.7, 0.8, 0.9],
        },
    ]


# ── Basic Operations Tests ────────────────────────────────────────────────────

class TestVectorStoreBasic:

    def test_store_initialises(self, in_memory_store):
        """VectorStore should initialise without errors."""
        assert in_memory_store is not None

    def test_empty_store_count(self, in_memory_store):
        """Fresh store should have zero chunks."""
        info = in_memory_store.get_collection_info()
        assert info["total_chunks"] == 0

    def test_add_chunks_returns_count(self, in_memory_store, sample_chunks):
        """add_chunks should return the number of chunks added."""
        count = in_memory_store.add_chunks(sample_chunks)
        assert count == len(sample_chunks)

    def test_collection_count_after_add(self, in_memory_store, sample_chunks):
        """Collection count should reflect added chunks."""
        in_memory_store.add_chunks(sample_chunks)
        info = in_memory_store.get_collection_info()
        assert info["total_chunks"] == len(sample_chunks)

    def test_add_empty_chunks(self, in_memory_store):
        """Adding empty list should return 0 without errors."""
        count = in_memory_store.add_chunks([])
        assert count == 0

    def test_upsert_no_duplicates(self, in_memory_store, sample_chunks):
        """
        Adding the same chunks twice should not create duplicates.
        Upsert logic means second run updates not inserts.
        """
        in_memory_store.add_chunks(sample_chunks)
        in_memory_store.add_chunks(sample_chunks)
        info = in_memory_store.get_collection_info()
        assert info["total_chunks"] == len(sample_chunks)


# ── Query Tests ───────────────────────────────────────────────────────────────

class TestVectorStoreQuery:

    def test_query_returns_list(self, in_memory_store, sample_chunks):
        """query() should return a list."""
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=2,
        )
        assert isinstance(results, list)

    def test_query_returns_correct_count(self, in_memory_store, sample_chunks):
        """query() should return at most n_results results."""
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=2,
        )
        assert len(results) <= 2

    def test_query_result_has_text(self, in_memory_store, sample_chunks):
        """Each result should have a text field."""
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=1,
        )
        assert "text" in results[0]
        assert isinstance(results[0]["text"], str)

    def test_query_result_has_metadata(self, in_memory_store, sample_chunks):
        """Each result should have a metadata dict."""
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=1,
        )
        assert "metadata" in results[0]
        assert isinstance(results[0]["metadata"], dict)

    def test_query_result_has_distance(self, in_memory_store, sample_chunks):
        """Each result should have a distance score."""
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=1,
        )
        assert "distance" in results[0]
        assert isinstance(results[0]["distance"], float)

    def test_query_empty_store(self):
        """Querying an empty store should return empty list."""
        with patch("src.ingestion.vectorstore.chromadb.PersistentClient") as mock:
             mock.return_value = chromadb.Client()
             fresh_store = VectorStore(
                persist_dir="test_tmp_fresh",
                collection_name="test_empty_collection",
             )
             results = fresh_store.query(
                 query_embedding=[0.1, 0.2, 0.3],
                 n_results=5,
            )
             assert results == []

    def test_query_empty_embedding(self, in_memory_store, sample_chunks):
        """Querying with empty embedding should return empty list."""
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(query_embedding=[])
        assert results == []


# ── Filter Tests ──────────────────────────────────────────────────────────────

class TestVectorStoreFilters:

    def test_filter_by_category(self, in_memory_store, sample_chunks):
        """
        Filtering by category=Agriculture should only return
        PM Kisan chunks, not Education chunks.
        """
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(
            query_embedding=[0.7, 0.8, 0.9],
            n_results=3,
            filter_category="Agriculture",
        )
        for result in results:
            assert result["metadata"]["category"] == "Agriculture"

    def test_filter_by_state(self, in_memory_store, sample_chunks):
        """Filtering by state=Central should return only central schemes."""
        in_memory_store.add_chunks(sample_chunks)
        results = in_memory_store.query(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=3,
            filter_state="Central",
        )
        for result in results:
            assert result["metadata"]["state"] == "Central"


# ── Collection Info Tests ─────────────────────────────────────────────────────

class TestCollectionInfo:

    def test_info_has_collection_name(self, in_memory_store):
        """get_collection_info should return collection name."""
        info = in_memory_store.get_collection_info()
        assert "collection_name" in info
        assert info["collection_name"] == "test_collection"

    def test_info_has_total_chunks(self, in_memory_store):
        """get_collection_info should return total chunk count."""
        info = in_memory_store.get_collection_info()
        assert "total_chunks" in info
        assert isinstance(info["total_chunks"], int)

    def test_info_has_persist_dir(self, in_memory_store):
        """get_collection_info should return persist directory."""
        info = in_memory_store.get_collection_info()
        assert "persist_dir" in info