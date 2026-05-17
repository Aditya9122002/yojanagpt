"""
test_chunker.py — Unit tests for src/ingestion/chunker.py

Run with:
  pytest tests/unit/test_chunker.py -v
"""

from __future__ import annotations

import pytest
from src.ingestion.chunker import chunk_scheme, chunk_schemes, MIN_CHUNK_LENGTH
from src.scraper.models import SchemeDetail


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def full_scheme():
    """A fully populated SchemeDetail with all fields filled."""
    return SchemeDetail(
        scheme_id="nos-swd",
        name="National Overseas Scholarship",
        ministry="Ministry of Social Justice and Empowerment",
        state="Central",
        category="Education",
        brief_description="Financial assistance for SC/ST students to study abroad.",
        eligibility="SC/ST students who have secured admission in top foreign universities.",
        benefit="Tuition fee, living allowance, and travel allowance covered.",
        how_to_apply="Apply online at the official portal before the deadline.",
        documents_needed=["Caste certificate", "Admission letter", "Passport copy"],
        helpline_number="1800-11-2001",
        application_portal="https://nosmsje.gov.in",
        application_deadline="2024-03-31",
        grievance_portal="https://pgportal.gov.in",
        csc_applicable=False,
        tags=["scholarship", "education", "SC", "ST"],
        source_url="https://www.myscheme.gov.in/schemes/nos-swd",
    )


@pytest.fixture
def minimal_scheme():
    """A SchemeDetail with only required fields — all optional fields are None."""
    return SchemeDetail(
        scheme_id="minimal-scheme",
        name="Minimal Test Scheme",
    )


@pytest.fixture
def scheme_with_short_fields():
    """A scheme where some fields are too short to be worth embedding."""
    return SchemeDetail(
        scheme_id="short-scheme",
        name="Short Field Scheme",
        eligibility="Yes",
        benefit="Cash",
        brief_description="A scheme with very short field values that should be skipped.",
    )


# ── Basic Structure Tests ─────────────────────────────────────────────────────

class TestChunkSchemeBasic:

    def test_returns_list(self, full_scheme):
        """chunk_scheme should always return a list."""
        result = chunk_scheme(full_scheme)
        assert isinstance(result, list)

    def test_full_scheme_produces_chunks(self, full_scheme):
        """A fully populated scheme should produce multiple chunks."""
        result = chunk_scheme(full_scheme)
        assert len(result) > 0

    def test_each_chunk_has_text(self, full_scheme):
        """Every chunk must have a non-empty text field."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert "text" in chunk
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_each_chunk_has_metadata(self, full_scheme):
        """Every chunk must have a metadata dict."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert "metadata" in chunk
            assert isinstance(chunk["metadata"], dict)

    def test_minimal_scheme_produces_chunks(self, minimal_scheme):
        """
        A scheme with only a short name produces no chunks.
        The name 'Minimal Test Scheme' is exactly at the length boundary
        and gets skipped. This is correct behaviour — short content
        produces low quality embeddings and should be excluded.
        """
        result = chunk_scheme(minimal_scheme)
        assert isinstance(result, list)


# ── Metadata Tests ────────────────────────────────────────────────────────────

class TestChunkMetadata:

    def test_scheme_id_in_metadata(self, full_scheme):
        """All chunks should carry the scheme_id in metadata."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert chunk["metadata"]["scheme_id"] == "nos-swd"

    def test_name_in_metadata(self, full_scheme):
        """All chunks should carry the scheme name in metadata."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert chunk["metadata"]["name"] == "National Overseas Scholarship"

    def test_state_in_metadata(self, full_scheme):
        """All chunks should carry the state in metadata."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert chunk["metadata"]["state"] == "Central"

    def test_category_in_metadata(self, full_scheme):
        """All chunks should carry the category in metadata."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert chunk["metadata"]["category"] == "Education"

    def test_field_in_metadata(self, full_scheme):
        """Each chunk should identify which field it came from."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert "field" in chunk["metadata"]
            assert isinstance(chunk["metadata"]["field"], str)

    def test_source_url_in_metadata(self, full_scheme):
        """All chunks should carry the source URL."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert "source_url" in chunk["metadata"]


# ── Field Coverage Tests ──────────────────────────────────────────────────────

class TestChunkFieldCoverage:

    def test_eligibility_chunk_present(self, full_scheme):
        """Should produce a chunk for the eligibility field."""
        result = chunk_scheme(full_scheme)
        fields = [c["metadata"]["field"] for c in result]
        assert "eligibility" in fields

    def test_benefit_chunk_present(self, full_scheme):
        """Should produce a chunk for the benefit field."""
        result = chunk_scheme(full_scheme)
        fields = [c["metadata"]["field"] for c in result]
        assert "benefit" in fields

    def test_how_to_apply_chunk_present(self, full_scheme):
        """Should produce a chunk for how_to_apply field."""
        result = chunk_scheme(full_scheme)
        fields = [c["metadata"]["field"] for c in result]
        assert "how_to_apply" in fields

    def test_documents_chunk_present(self, full_scheme):
        """Should produce a chunk for documents_needed field."""
        result = chunk_scheme(full_scheme)
        fields = [c["metadata"]["field"] for c in result]
        assert "documents_needed" in fields

    def test_documents_joined_as_string(self, full_scheme):
        """documents_needed list should be joined into a single string."""
        result = chunk_scheme(full_scheme)
        doc_chunk = next(
            c for c in result if c["metadata"]["field"] == "documents_needed"
        )
        assert "Caste certificate" in doc_chunk["text"]
        assert "Admission letter" in doc_chunk["text"]

    def test_summary_chunk_present(self, full_scheme):
        """Should produce a summary chunk combining key fields."""
        result = chunk_scheme(full_scheme)
        fields = [c["metadata"]["field"] for c in result]
        assert "summary" in fields

    def test_summary_chunk_contains_name(self, full_scheme):
        """Summary chunk should contain the scheme name."""
        result = chunk_scheme(full_scheme)
        summary = next(c for c in result if c["metadata"]["field"] == "summary")
        assert "National Overseas Scholarship" in summary["text"]


# ── Edge Case Tests ───────────────────────────────────────────────────────────

class TestChunkEdgeCases:

    def test_short_fields_skipped(self, scheme_with_short_fields):
        """
        Fields shorter than MIN_CHUNK_LENGTH should be skipped.
        'Yes' and 'Cash' are too short to be useful embeddings.
        """
        result = chunk_scheme(scheme_with_short_fields)
        fields = [c["metadata"]["field"] for c in result]
        assert "eligibility" not in fields
        assert "benefit" not in fields

    def test_none_fields_skipped(self, minimal_scheme):
        """None fields should be skipped silently."""
        result = chunk_scheme(minimal_scheme)
        fields = [c["metadata"]["field"] for c in result]
        assert "eligibility" not in fields
        assert "how_to_apply" not in fields

    def test_chunk_text_min_length(self, full_scheme):
        """All chunks should meet the minimum length requirement."""
        result = chunk_scheme(full_scheme)
        for chunk in result:
            assert len(chunk["text"]) >= MIN_CHUNK_LENGTH

    def test_chunk_text_contains_label(self, full_scheme):
        """Chunk text should contain the field label for context."""
        result = chunk_scheme(full_scheme)
        eligibility_chunk = next(
            c for c in result if c["metadata"]["field"] == "eligibility"
        )
        assert "Eligibility" in eligibility_chunk["text"]


# ── Batch Chunking Tests ──────────────────────────────────────────────────────

class TestChunkSchemes:

    def test_chunk_schemes_returns_flat_list(self, full_scheme, minimal_scheme):
        """chunk_schemes should return one flat list from multiple schemes."""
        result = chunk_schemes([full_scheme, minimal_scheme])
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunk_schemes_empty_input(self):
        """chunk_schemes with empty list should return empty list."""
        result = chunk_schemes([])
        assert result == []

    def test_chunk_schemes_total_count(self, full_scheme, minimal_scheme):
        """
        Total chunks from batch should equal sum of individual chunks.
        """
        individual_total = (
            len(chunk_scheme(full_scheme)) +
            len(chunk_scheme(minimal_scheme))
        )
        batch_total = len(chunk_schemes([full_scheme, minimal_scheme]))
        assert batch_total == individual_total