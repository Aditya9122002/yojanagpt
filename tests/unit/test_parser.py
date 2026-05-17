"""
test_parser.py — Unit tests for src/scraper/parser.py

Run with:
  pytest tests/unit/test_parser.py -v

Run with coverage:
  pytest tests/unit/test_parser.py -v --cov=src.scraper.parser
"""

from __future__ import annotations

import pytest
from src.scraper.parser import parse_scheme_list, parse_scheme_detail
from src.scraper.models import SchemeListItem, SchemeDetail


# ── Fixtures — Sample API Responses ──────────────────────────────────────────
# These mimic the REAL myscheme.gov.in API response structure (confirmed May 2025).
#
# LIST API structure:
#   data → hits → items[] → fields → scheme fields
#
# DETAIL API structure:
#   data → en → basicDetails / schemeContent / eligibilityCriteria / applicationProcess

@pytest.fixture
def valid_list_response():
    """Mimics real paginated list API response with two schemes."""
    return {
        "status": "Success",
        "statusCode": 200,
        "data": {
            "summary": {"total": 4696},
            "hits": {
                "items": [
                    {
                        "_id": "abc123",
                        "fields": {
                            "slug": "nos-swd",
                            "schemeName": "National Overseas Scholarship",
                            "schemeShortTitle": "NOS",
                            "briefDescription": "Scholarship for students going abroad.",
                            "tags": ["scholarship", "education", "overseas"],
                            "level": "Central",
                            "nodalMinistryName": "Ministry of Social Justice",
                            "schemeCategory": ["Education"],
                            "beneficiaryState": ["All"],
                            "schemeCloseDate": "",
                        }
                    },
                    {
                        "_id": "def456",
                        "fields": {
                            "slug": "pmay",
                            "schemeName": "Pradhan Mantri Awas Yojana",
                            "schemeShortTitle": "PMAY",
                            "briefDescription": "Housing for all.",
                            "tags": ["housing", "urban", "rural"],
                            "level": "Central",
                            "nodalMinistryName": "Ministry of Housing",
                            "schemeCategory": ["Housing"],
                            "beneficiaryState": ["All"],
                            "schemeCloseDate": "2024-12-31",
                        }
                    },
                ]
            }
        }
    }


@pytest.fixture
def valid_detail_response():
    """Mimics real scheme detail API response with all fields present."""
    return {
        "status": "Success",
        "statusCode": 200,
        "data": {
            "slug": "nos-swd",
            "en": {
                "basicDetails": {
                    "schemeName": "National Overseas Scholarship",
                    "nodalMinistryName": {
                        "value": 712,
                        "label": "Ministry Of Social Justice and Empowerment"
                    },
                    "level": {"value": "central", "label": "Central"},
                    "schemeCategory": [
                        {"value": "edu-001", "label": "Education"}
                    ],
                    "tags": ["scholarship", "education", "SC", "ST", "overseas"],
                    "schemeOpenDate": "2023-01-01",
                    "schemeCloseDate": "2024-03-31",
                    "helplineNumber": "1800-11-2001",
                    "applicationPortal": "https://nosmsje.gov.in",
                    "grievancePortal": "https://pgportal.gov.in",
                    "cscApplicable": "false",
                },
                "schemeContent": {
                    "briefDescription": "Provides financial assistance to students from SC/ST communities for studies abroad.",
                    "benefits_md": "Tuition fee, living allowance, travel allowance covered.",
                },
                "eligibilityCriteria": {
                    "eligibilityDescription_md": "SC/ST students who have secured admission in top foreign universities.",
                },
                "applicationProcess": [
                    {
                        "type": "paragraph",
                        "children": [{"text": "Apply online at the official portal before the deadline."}]
                    }
                ],
            }
        }
    }


@pytest.fixture
def minimal_detail_response():
    """Mimics a detail response where most optional fields are missing."""
    return {
        "status": "Success",
        "statusCode": 200,
        "data": {
            "slug": "minimal-scheme",
            "en": {
                "basicDetails": {
                    "schemeName": "Minimal Test Scheme",
                },
                "schemeContent": {},
            }
        }
    }


# ── List Parser Tests ─────────────────────────────────────────────────────────

class TestParseSchemeList:

    def test_parse_scheme_list_valid(self, valid_list_response):
        """Valid response should return a list of two SchemeListItem objects."""
        result = parse_scheme_list(valid_list_response)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, SchemeListItem) for item in result)

    def test_first_item_fields(self, valid_list_response):
        """First item should have correct field values."""
        result = parse_scheme_list(valid_list_response)
        first = result[0]

        assert first.slug == "nos-swd"
        assert first.scheme_name == "National Overseas Scholarship"
        assert first.level == "Central"
        assert first.nodal_ministry_name == "Ministry of Social Justice"
        assert "scholarship" in first.tags

    def test_parse_scheme_list_empty_response(self):
        """Empty dict should return an empty list without raising."""
        result = parse_scheme_list({})
        assert result == []

    def test_parse_scheme_list_none_response(self):
        """None response should return an empty list without raising."""
        result = parse_scheme_list(None)
        assert result == []

    def test_parse_scheme_list_missing_data_key(self):
        """Response missing 'data' key should return empty list."""
        result = parse_scheme_list({"error": "something went wrong"})
        assert result == []

    def test_parse_scheme_list_empty_items(self):
        """Response with empty items list should return empty list."""
        result = parse_scheme_list({
            "data": {
                "hits": {"items": []},
                "summary": {"total": 0}
            }
        })
        assert result == []

    def test_parse_scheme_list_partial_failure(self):
        """
        If one scheme in the list is malformed, the rest should still parse.
        The bad one gets skipped, not the whole batch.
        """
        response = {
            "data": {
                "hits": {
                    "items": [
                        {
                            "_id": "good",
                            "fields": {
                                "slug": "good-scheme",
                                "schemeName": "Good Scheme",
                                "level": "Central",
                            }
                        },
                        {
                            "_id": "bad",
                            "fields": {
                                # Missing required slug field
                                "schemeName": "Bad Scheme With No Slug",
                            }
                        },
                    ]
                }
            }
        }
        result = parse_scheme_list(response)
        assert len(result) == 1
        assert result[0].slug == "good-scheme"


# ── Detail Parser Tests ───────────────────────────────────────────────────────

class TestParseSchemeDetail:

    def test_parse_scheme_detail_valid(self, valid_detail_response):
        """Valid detail response should return a SchemeDetail object."""
        result = parse_scheme_detail(valid_detail_response, slug="nos-swd")

        assert result is not None
        assert isinstance(result, SchemeDetail)

    def test_detail_core_fields(self, valid_detail_response):
        """Core fields should be extracted correctly."""
        result = parse_scheme_detail(valid_detail_response, slug="nos-swd")

        assert result.scheme_id == "nos-swd"
        assert result.name == "National Overseas Scholarship"
        assert result.ministry == "Ministry Of Social Justice and Empowerment"
        assert result.category == "Education"

    def test_detail_eligibility_and_benefit(self, valid_detail_response):
        """Eligibility and benefit fields should be extracted."""
        result = parse_scheme_detail(valid_detail_response, slug="nos-swd")

        assert result.eligibility is not None
        assert "SC/ST" in result.eligibility
        assert result.benefit is not None
        assert "Tuition fee" in result.benefit

    def test_detail_documents_list(self, valid_detail_response):
        """Documents needed should be an empty list when not in response."""
        result = parse_scheme_detail(valid_detail_response, slug="nos-swd")
        assert isinstance(result.documents_needed, list)

    def test_detail_helpline_and_portal(self, valid_detail_response):
        """Contact fields should be extracted correctly."""
        result = parse_scheme_detail(valid_detail_response, slug="nos-swd")

        assert result.helpline_number == "1800-11-2001"
        assert result.application_portal == "https://nosmsje.gov.in"
        assert result.grievance_portal == "https://pgportal.gov.in"

    def test_detail_csc_applicable_false(self, valid_detail_response):
        """cscApplicable string 'false' should be parsed to Python False."""
        result = parse_scheme_detail(valid_detail_response, slug="nos-swd")
        assert result.csc_applicable is False

    def test_detail_source_url_auto_built(self, valid_detail_response):
        """Source URL should be auto-built from scheme_id."""
        result = parse_scheme_detail(valid_detail_response, slug="nos-swd")
        assert result.source_url == "https://www.myscheme.gov.in/schemes/nos-swd"

    def test_detail_minimal_response(self, minimal_detail_response):
        """Response with only required fields should parse successfully."""
        result = parse_scheme_detail(minimal_detail_response, slug="minimal-scheme")

        assert result is not None
        assert result.scheme_id == "minimal-scheme"
        assert result.name == "Minimal Test Scheme"
        assert result.eligibility is None
        assert result.benefit is None
        assert result.documents_needed == []
        assert result.helpline_number is None

    def test_detail_empty_response(self):
        """Empty response should return None without raising."""
        result = parse_scheme_detail({}, slug="test-slug")
        assert result is None

    def test_detail_none_response(self):
        """None response should return None without raising."""
        result = parse_scheme_detail(None, slug="test-slug")
        assert result is None


# ── Boolean Parsing Tests ─────────────────────────────────────────────────────

class TestBooleanParsing:
    """Tests for _parse_bool helper via parse_scheme_detail."""

    def _make_response(self, csc_value):
        return {
            "data": {
                "slug": "test-scheme",
                "en": {
                    "basicDetails": {
                        "schemeName": "Test Scheme",
                        "cscApplicable": csc_value,
                    },
                    "schemeContent": {},
                }
            }
        }

    def test_bool_true_string(self):
        result = parse_scheme_detail(self._make_response("true"), slug="test")
        assert result.csc_applicable is True

    def test_bool_false_string(self):
        result = parse_scheme_detail(self._make_response("false"), slug="test")
        assert result.csc_applicable is False

    def test_bool_yes_string(self):
        result = parse_scheme_detail(self._make_response("yes"), slug="test")
        assert result.csc_applicable is True

    def test_bool_one_string(self):
        result = parse_scheme_detail(self._make_response("1"), slug="test")
        assert result.csc_applicable is True

    def test_bool_actual_true(self):
        result = parse_scheme_detail(self._make_response(True), slug="test")
        assert result.csc_applicable is True

    def test_bool_none(self):
        result = parse_scheme_detail(self._make_response(None), slug="test")
        assert result.csc_applicable is None


# ── Tags Normalisation Tests ──────────────────────────────────────────────────

class TestTagsNormalisation:
    """Tags can come from the API as a list, a string, or None."""

    def _make_response(self, tags_value):
        return {
            "data": {
                "slug": "test-scheme",
                "en": {
                    "basicDetails": {
                        "schemeName": "Test Scheme",
                        "tags": tags_value,
                    },
                    "schemeContent": {},
                }
            }
        }

    def test_tags_as_list(self):
        result = parse_scheme_detail(
            self._make_response(["education", "scholarship"]), slug="test"
        )
        assert result.tags == ["education", "scholarship"]

    def test_tags_as_string(self):
        result = parse_scheme_detail(
            self._make_response("education"), slug="test"
        )
        assert isinstance(result.tags, list)
        assert "education" in result.tags

    def test_tags_as_none(self):
        result = parse_scheme_detail(
            self._make_response(None), slug="test"
        )
        assert result.tags == []