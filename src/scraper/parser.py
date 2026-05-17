"""
parser.py — Converts raw API JSON into clean Pydantic model objects.

Two public functions:
  - parse_scheme_list(raw)   → List[SchemeListItem]
  - parse_scheme_detail(raw) → Optional[SchemeDetail]

All parsing errors are logged as warnings — never raised.
The scraper continues even if one scheme fails to parse.

Confirmed API response structures (May 2025):

LIST response — Elasticsearch style, hits nested under data.hits.hits[]:
  {
    "data": {
      "hits": {
        "hits": [
          { "_source": { "slug": "pm-kisan", "schemeName": "...", ... } },
          ...
        ]
      },
      "summary": { "total": 4696 }
    }
  }

DETAIL response — all content under data.en.* (language key):
  {
    "data": {
      "en": {
        "basicDetails": {
          "schemeName": "...",
          "tags": ["Farmers", ...],
          "nodalMinistryName": { "value": 498, "label": "Ministry Of ..." },
          "level": { "value": "central", "label": "Central" },
          "schemeCategory": [{ "value": "...", "label": "..." }],
          "schemeOpenDate": "2019-02-24",
          "schemeCloseDate": null,
          ...
        },
        "schemeContent": {
          "briefDescription": "plain text summary",
          "benefits_md": "markdown text — BEST for RAG",
          "exclusions_md": "markdown text",
          "detailedDescription_md": "markdown text",
          "references": [{ "title": "...", "url": "..." }],
          "benefitTypes": { "value": "Cash", "label": "Cash" },
          ...
        },
        "applicationProcess": [...rich text nodes...],
        "eligibilityCriteria": [...rich text nodes...],
        "schemeDefinitions": [...rich text nodes...]
      },
      "slug": "pm-kisan"
    }
  }

Design decision — use _md fields for RAG:
  The API provides content in two formats: rich-text JSON trees (nested
  children arrays) and pre-rendered markdown strings (*_md fields).
  We use the _md fields because they are clean plain text, ideal for
  embedding and retrieval. The rich-text trees require recursive traversal
  and produce identical content.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .models import SchemeDetail, SchemeListItem

logger = logging.getLogger(__name__)


# ── List Parser ───────────────────────────────────────────────────────────────

def parse_scheme_list(raw: Dict[str, Any]) -> List[SchemeListItem]:
    """
    Parse the raw list API response into a list of SchemeListItem objects.

    Navigates: raw → data → hits → hits[] → _source → scheme fields

    Args:
        raw: Raw JSON response dict from client.fetch_scheme_list()

    Returns:
        List of validated SchemeListItem objects.
        Empty list if response is malformed or contains no schemes.
    """
    if not raw:
        logger.warning("parse_scheme_list received empty response")
        return []

    data = raw.get("data", {})
    if not data:
        logger.warning("parse_scheme_list: 'data' key missing from response")
        return []

    hits_wrapper = data.get("hits", {})
    if not hits_wrapper:
        logger.warning("parse_scheme_list: 'hits' key missing from data")
        return []

    hits = hits_wrapper.get("items", hits_wrapper.get("hits", []))
    if not hits:
        logger.warning("parse_scheme_list: hits list is empty or missing")
        return []

    parsed = []
    for i, hit in enumerate(hits):
        # Each hit: { "_index": "...", "_id": "...", "_source": { scheme fields } }
        scheme_raw = hit.get("fields", hit.get("_source", {}))
        if not scheme_raw:
            logger.warning("parse_scheme_list: hit at index %d has no fields", i)
            continue

        try:
            item = SchemeListItem(**scheme_raw)
            parsed.append(item)
        except Exception as e:
            slug = scheme_raw.get("slug", f"index_{i}")
            logger.warning(
                "Failed to parse list item | slug=%s | error=%s",
                slug,
                str(e),
            )
            continue

    logger.debug("parse_scheme_list: parsed %d/%d schemes", len(parsed), len(hits))
    return parsed


# ── Detail Parser ─────────────────────────────────────────────────────────────

def parse_scheme_detail(
    raw: Dict[str, Any],
    slug: str,
) -> Optional[SchemeDetail]:
    """
    Parse the raw detail API response into a SchemeDetail object.

    Navigates: raw → data → en → {basicDetails, schemeContent, ...}

    Args:
        raw:  Raw JSON response dict from client.fetch_scheme_detail()
        slug: The scheme slug — used as fallback scheme_id and for logging

    Returns:
        Validated SchemeDetail object on success, None on failure.
    """
    if not raw:
        logger.warning("parse_scheme_detail received empty response | slug=%s", slug)
        return None

    try:
        data = raw.get("data", {})
        if not data:
            logger.warning("parse_scheme_detail: 'data' missing | slug=%s", slug)
            return None

        lang_data = data.get("en", {})
        if not lang_data:
            logger.warning("parse_scheme_detail: 'en' block missing | slug=%s", slug)
            return None

        # ── Extract the four main sections ────────────────────────
        basic   = lang_data.get("basicDetails", {})
        content = lang_data.get("schemeContent", {})
        app_process = lang_data.get("applicationProcess", [])
        eligibility_raw = lang_data.get("eligibilityCriteria", {})

        # ── Identity ──────────────────────────────────────────────

        scheme_id = data.get("slug") or slug

        name = basic.get("schemeName") or basic.get("schemeShortTitle") or "Unknown Scheme"

        # nodalMinistryName → { "value": 498, "label": "Ministry Of ..." }
        ministry = _label(basic.get("nodalMinistryName"))

        # level → { "value": "central", "label": "Central" }
        # For state schemes this will be state name
        level = _label(basic.get("level"))
        state_raw = basic.get("beneficiaryState")
        state = _label(state_raw) if state_raw else level

        # schemeCategory → list of { "value": "...", "label": "..." }
        category_list = basic.get("schemeCategory", [])
        if isinstance(category_list, list) and category_list:
            category = ", ".join(_label(c) for c in category_list if _label(c))
        else:
            category = _label(category_list) if category_list else None

        # ── Description ───────────────────────────────────────────
        # briefDescription is a clean plain-text summary — perfect for RAG
        brief_description = (
            content.get("briefDescription")
            or content.get("detailedDescription_md")
            or None
        )
        if brief_description:
            brief_description = _clean_html(brief_description)

        # ── Benefits ──────────────────────────────────────────────
        # benefits_md is pre-rendered markdown — cleanest format for RAG
        # ── Benefits ──────────────────────────────────────────────────────────────────
        benefit = (
            content.get("benefits_md")
             or _extract_richtext(content.get("benefits", []))
             or None
        )
        if benefit:
            benefit = _clean_html(benefit).strip() or None

        # ── Eligibility ───────────────────────────────────────────
        # eligibilityCriteria is a rich-text tree — extract text from it
        eligibility = (
            eligibility_raw.get("eligibilityDescription_md")
            if isinstance(eligibility_raw, dict)
            else _extract_richtext(eligibility_raw)
        )
        if eligibility:
            eligibility = _clean_html(eligibility).strip() or None
        # ── How to apply ──────────────────────────────────────────
        # applicationProcess is also a rich-text tree
        how_to_apply = _extract_richtext(app_process)

        # ── Documents ─────────────────────────────────────────────
        # Not a separate section in this API — often embedded in applicationProcess
        # We leave as empty list; can be enhanced later
        documents_needed: List[str] = []

        # ── Contact and support ───────────────────────────────────
        helpline_number = (
            basic.get("helplineNumber")
            or basic.get("helpline")
            or basic.get("tollFreeNumber")
            or None
        )
        application_portal = (
            basic.get("applicationPortal")
            or basic.get("applyLink")
            or basic.get("onlineApplicationLink")
            or None
        )
        application_deadline = basic.get("schemeCloseDate") or None

        state_nodal_contact = (
            basic.get("stateNodalContact")
            or basic.get("nodalContact")
            or None
        )
        grievance_portal = basic.get("grievancePortal") or None

        # ── Flags ─────────────────────────────────────────────────
        csc_applicable = _parse_bool(basic.get("cscApplicable"))

        # ── Tags ──────────────────────────────────────────────────
        # tags is a plain list of strings in basicDetails
        tags_raw = basic.get("tags", [])
        tags = [str(t).strip() for t in tags_raw if t] if tags_raw else []

        # ── Source URL ────────────────────────────────────────────
        source_url = f"https://www.myscheme.gov.in/schemes/{scheme_id}"

        # ── Build model ───────────────────────────────────────────
        detail = SchemeDetail(
            scheme_id=scheme_id,
            name=name,
            ministry=ministry,
            state=state,
            category=category,
            brief_description=brief_description,
            eligibility=eligibility,
            benefit=benefit,
            how_to_apply=how_to_apply,
            documents_needed=documents_needed,
            helpline_number=helpline_number,
            application_portal=application_portal,
            application_deadline=application_deadline,
            state_nodal_contact=state_nodal_contact,
            grievance_portal=grievance_portal,
            csc_applicable=csc_applicable,
            tags=tags,
            source_url=source_url,
        )

        logger.debug("Parsed detail | slug=%s | name=%s", slug, name)
        return detail

    except Exception as e:
        logger.error(
            "Failed to parse scheme detail | slug=%s | error=%s", slug, str(e)
        )
        return None


# ── Private Helpers ───────────────────────────────────────────────────────────

def _label(value: Any) -> Optional[str]:
    """
    Extract display string from a label/value dict or plain string.

    API returns lookup fields as: { "value": 498, "label": "Ministry Of ..." }
    We always want the human-readable "label".

    Examples:
        _label({"value": 498, "label": "Ministry Of Agriculture"})
        → "Ministry Of Agriculture"

        _label("Central")
        → "Central"

        _label(None)
        → None
    """
    if value is None:
        return None
    if isinstance(value, dict):
        result = value.get("label") or value.get("name") or str(value.get("value", ""))
        return result.strip() if result else None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _extract_richtext(nodes: Any) -> Optional[str]:
    """
    Recursively extract plain text from the API's rich-text node tree.

    The API stores content as a Slate.js-style tree:
      [
        { "type": "paragraph", "children": [{ "text": "Must be Indian citizen" }] },
        { "type": "ul_list", "children": [
            { "type": "list_item", "children": [{ "text": "Age 18+" }] }
        ]}
      ]

    We walk the tree and collect all "text" leaf values,
    joining them with newlines to produce readable plain text.

    Args:
        nodes: List of rich-text node dicts, or a plain string, or None.

    Returns:
        Extracted plain text string, or None if empty.
    """
    if not nodes:
        return None

    # Already a plain string
    if isinstance(nodes, str):
        return nodes.strip() or None

    if not isinstance(nodes, list):
        return None

    parts = []
    _walk_nodes(nodes, parts)
    text = "\n".join(p for p in parts if p.strip())
    return text.strip() or None


def _walk_nodes(nodes: List[Any], parts: List[str]) -> None:
    """
    Recursive walker for rich-text node trees.
    Appends text leaf values to the parts list.
    """
    for node in nodes:
        if not isinstance(node, dict):
            continue

        text = node.get("text")
        if text and str(text).strip():
            parts.append(str(text).strip())

        children = node.get("children")
        if children and isinstance(children, list):
            _walk_nodes(children, parts)


def _clean_html(text: str) -> str:
    """
    Remove HTML tags and decode common HTML entities from text.

    The API sometimes returns HTML-encoded content in the _md fields,
    e.g. &amp;quot; instead of " or &amp;amp; instead of &.

    Args:
        text: Raw string possibly containing HTML tags and entities.

    Returns:
        Cleaned plain text string.
    """
    if not text:
        return text

    # Decode double-encoded HTML entities first (&amp;quot; → &quot; → ")
    text = text.replace("&amp;quot;", '"')
    text = text.replace("&amp;amp;", "&")
    text = text.replace("&amp;lt;", "<")
    text = text.replace("&amp;gt;", ">")

    # Decode standard HTML entities
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")

    # Remove <br> tags
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Remove any remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _parse_bool(value: Any) -> Optional[bool]:
    """
    Convert various truthy/falsy representations to bool.
    Handles: True/False, "true"/"false", "yes"/"no", "1"/"0", None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
    return None


# ── Kept for backward compatibility ──────────────────────────────────────────

def _extract_text_field(data: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if not value:
            continue
        if isinstance(value, dict):
            value = value.get("text") or value.get("value") or value.get("label")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = " ".join(str(v) for v in value if v)
            if text.strip():
                return text.strip()
    return None


def _extract_list_field(data: Dict[str, Any], keys: List[str]) -> List[str]:
    for key in keys:
        value = data.get(key)
        if not value:
            continue
        if isinstance(value, list):
            return [str(v).strip() for v in value if v]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []