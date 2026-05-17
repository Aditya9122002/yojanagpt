"""
chunker.py — Splits SchemeDetail objects into text chunks for embedding.

Strategy: field-based chunking — each meaningful field becomes its own
chunk with metadata attached. This preserves semantic boundaries and
gives precise retrieval results.

Each chunk is a dict with:
  - text:      the actual text to embed
  - metadata:  scheme_id, field, name, state, category, source_url
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.scraper.models import SchemeDetail

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum characters a chunk must have to be worth embedding
# Chunks shorter than this are skipped — they add noise without value
MIN_CHUNK_LENGTH = 30

# Fields we want to create chunks from, in priority order
# Each tuple is (field_name, human_readable_label)
CHUNK_FIELDS = [
    ("name",                "Scheme Name"),
    ("brief_description",   "Description"),
    ("eligibility",         "Eligibility"),
    ("benefit",             "Benefits"),
    ("how_to_apply",        "How to Apply"),
    ("documents_needed",    "Documents Required"),
    ("helpline_number",     "Helpline"),
    ("application_portal",  "Apply Online"),
    ("grievance_portal",    "Grievance Portal"),
    ("state_nodal_contact", "Nodal Contact"),
]


# ── Main Chunker Function ─────────────────────────────────────────────────────

def chunk_scheme(scheme: SchemeDetail) -> List[Dict[str, Any]]:
    """
    Convert a single SchemeDetail into a list of text chunks.

    Each chunk is a dict with 'text' and 'metadata' keys.
    Empty or very short fields are skipped.

    Args:
        scheme: A validated SchemeDetail object.

    Returns:
        List of chunk dicts ready for embedding.
        Empty list if scheme has no usable content.

    Example output:
        [
            {
                "text": "Eligibility: SC/ST students who secured admission abroad.",
                "metadata": {
                    "scheme_id": "nos-swd",
                    "field": "eligibility",
                    "name": "National Overseas Scholarship",
                    "state": "Central",
                    "category": "Education",
                    "source_url": "https://www.myscheme.gov.in/schemes/nos-swd"
                }
            },
            ...
        ]
    """
    chunks = []

    # Base metadata attached to every chunk from this scheme
    base_metadata = {
        "scheme_id":  scheme.scheme_id,
        "name":       scheme.name,
        "state":      scheme.state or "Unknown",
        "category":   scheme.category or "Unknown",
        "ministry":   scheme.ministry or "Unknown",
        "source_url": scheme.source_url or "",
    }

    for field_name, field_label in CHUNK_FIELDS:
        value = getattr(scheme, field_name, None)

        if not value:
            continue

        # Convert list fields (like documents_needed) to readable text
        if isinstance(value, list):
            if not value:
                continue
            text_value = ", ".join(str(v) for v in value if v)
        else:
            text_value = str(value).strip()

        # Skip chunks that are too short to be meaningful
        if len(text_value) < MIN_CHUNK_LENGTH:
            logger.debug(
                "Skipping short chunk | scheme=%s | field=%s | len=%d",
                scheme.scheme_id,
                field_name,
                len(text_value),
            )
            continue

        # Format the chunk text — label + content
        # This format helps the embedding model understand context
        chunk_text = f"{field_label}: {text_value}"

        chunk = {
            "text": chunk_text,
            "metadata": {
                **base_metadata,
                "field": field_name,
                "field_label": field_label,
            },
        }
        chunks.append(chunk)

    # Also create a combined summary chunk for broad queries
    # This helps when someone asks a general question about the scheme
    summary_chunk = _build_summary_chunk(scheme, base_metadata)
    if summary_chunk:
        chunks.append(summary_chunk)

    logger.debug(
        "Chunked scheme | id=%s | chunks=%d",
        scheme.scheme_id,
        len(chunks),
    )
    return chunks


def chunk_schemes(schemes: List[SchemeDetail]) -> List[Dict[str, Any]]:
    """
    Chunk a list of SchemeDetail objects into a flat list of chunks.

    Args:
        schemes: List of SchemeDetail objects.

    Returns:
        Flat list of all chunks from all schemes.
    """
    all_chunks = []
    failed = 0

    for scheme in schemes:
        try:
            chunks = chunk_scheme(scheme)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(
                "Failed to chunk scheme | id=%s | error=%s",
                scheme.scheme_id,
                str(e),
            )
            failed += 1
            continue

    logger.info(
        "Chunking complete | schemes=%d | total_chunks=%d | failed=%d",
        len(schemes),
        len(all_chunks),
        failed,
    )
    return all_chunks


# ── Private Helpers ───────────────────────────────────────────────────────────

def _build_summary_chunk(
    scheme: SchemeDetail,
    base_metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Build a combined summary chunk for broad/general queries.

    Combines name, description, eligibility, and benefit into one chunk.
    This is useful when a user asks something like
    "what schemes are available for farmers" — a broad query that
    benefits from a holistic view of the scheme.

    Args:
        scheme:        The SchemeDetail object.
        base_metadata: Base metadata dict shared across all chunks.

    Returns:
        Summary chunk dict, or None if not enough content.
    """
    parts = []

    if scheme.name:
        parts.append(f"Scheme: {scheme.name}")
    if scheme.ministry:
        parts.append(f"Ministry: {scheme.ministry}")
    if scheme.state:
        parts.append(f"Level: {scheme.state}")
    if scheme.category:
        parts.append(f"Category: {scheme.category}")
    if scheme.brief_description:
        parts.append(f"About: {scheme.brief_description}")
    if scheme.eligibility:
        parts.append(f"Eligible: {scheme.eligibility}")
    if scheme.benefit:
        parts.append(f"Benefit: {scheme.benefit}")

    if len(parts) < 2:
        return None

    summary_text = " | ".join(parts)

    if len(summary_text) < MIN_CHUNK_LENGTH:
        return None

    return {
        "text": summary_text,
        "metadata": {
            **base_metadata,
            "field": "summary",
            "field_label": "Summary",
        },
    }