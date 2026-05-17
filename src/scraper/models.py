"""
models.py — Pydantic data models for YojanaGPT scraper.

Defines the shape of data at every stage of the scraping pipeline:
  - SchemeListItem: lightweight record from the list API
  - SchemeDetail:   full record after detail page scraping
  - ScrapeResult:   wrapper tracking success/failure of each scrape attempt
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator


# ── Scheme List Item ──────────────────────────────────────────────────────────
# This is what we get from the paginated list API.
# We use this to build our queue of schemes to scrape in detail.

class SchemeListItem(BaseModel):
    """
    Lightweight scheme record returned by the list API.

    The list API returns fields under a 'fields' key per item.
    Some fields come as lists (even when they contain a single value)
    because the API uses Elasticsearch field format.
    Validators normalise everything to clean scalar types.
    """

    # Unique identifier — same as slug, used to build detail URL
    id: Optional[str] = Field(None, description="Unique scheme ID from API")

    # URL slug — used to construct the detail page URL
    # Example: 'nos-swd' → https://www.myscheme.gov.in/schemes/nos-swd
    slug: str = Field(..., description="URL slug for the scheme")

    # Human-readable scheme name
    scheme_name: Optional[str] = Field(None, alias="schemeName")

    # Short title / abbreviation
    scheme_short_title: Optional[str] = Field(None, alias="schemeShortTitle")

    # One or two sentence summary
    brief_description: Optional[str] = Field(None, alias="briefDescription")

    # Searchable tags list
    tags: Optional[List[str]] = Field(default_factory=list)

    # Central or State level scheme
    level: Optional[str] = Field(None, description="Central or State")

    # Ministry responsible for this scheme
    nodal_ministry_name: Optional[str] = Field(None, alias="nodalMinistryName")

    # Category: Agriculture, Education, Health, Housing, etc.
    # API may send as a string OR a list of strings — validator handles both
    scheme_category: Optional[str] = Field(None, alias="schemeCategory")

    # Which state benefits from this scheme (empty for central schemes)
    # API may send as a string OR a list of strings — validator handles both
    beneficiary_state: Optional[str] = Field(None, alias="beneficiaryState")

    # Application closing date if scheme has a deadline
    scheme_close_date: Optional[str] = Field(None, alias="schemeCloseDate")

    # ── Validators ────────────────────────────────────────────────

    @validator("scheme_category", pre=True, always=True)
    def normalise_scheme_category(cls, v):
        """
        schemeCategory arrives as a list in the list API response.
        Example: ["Agriculture,Rural & Environment", "Social welfare"]
        We join multiple values with a comma into a single string.
        """
        if v is None:
            return None
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if x]
            return ", ".join(parts) if parts else None
        return str(v).strip() or None

    @validator("beneficiary_state", pre=True, always=True)
    def normalise_beneficiary_state(cls, v):
        """
        beneficiaryState arrives as a list in the list API response.
        Example: ["All"] or ["Gujarat", "Maharashtra"]
        We join multiple values with a comma into a single string.
        """
        if v is None:
            return None
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if x]
            return ", ".join(parts) if parts else None
        return str(v).strip() or None

    @validator("tags", pre=True, always=True)
    def normalise_tags(cls, v):
        """Tags can be None, a string, or a list — normalise to list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(t).strip() for t in v if t]

    @validator("nodal_ministry_name", pre=True, always=True)
    def normalise_ministry(cls, v):
        """Ministry name may arrive as a list — take first item."""
        if v is None:
            return None
        if isinstance(v, list):
            return str(v[0]).strip() if v else None
        return str(v).strip() or None

    @validator("level", pre=True, always=True)
    def normalise_level(cls, v):
        """Level may arrive as a list — take first item."""
        if v is None:
            return None
        if isinstance(v, list):
            return str(v[0]).strip() if v else None
        return str(v).strip() or None

    class Config:
        # Allow both alias and field name when creating the model
        allow_population_by_field_name = True
        # Ignore extra fields from API we don't care about
        extra = "ignore"


# ── Full Scheme Detail ────────────────────────────────────────────────────────
# This is our complete data model — what gets stored in ChromaDB later.
# Maps to the COMPLETE DATA MODEL defined in the project spec.

class SchemeDetail(BaseModel):
    """
    Complete scheme record with all fields needed for RAG and eligibility checking.
    This is the final shape of data stored in our vector database.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    scheme_id: str = Field(..., description="Unique slug from URL — primary key")
    name: str = Field(..., description="Full scheme name")
    ministry: Optional[str] = Field(None, description="Ministry that runs this scheme")
    state: Optional[str] = Field(
        None,
        description="'Central' for central schemes, state name for state schemes"
    )
    category: Optional[str] = Field(
        None,
        description="Agriculture, Education, Health, Housing, etc."
    )

    # ── Core Scheme Information ───────────────────────────────────────────────
    brief_description: Optional[str] = Field(
        None,
        description="Short summary of the scheme"
    )
    eligibility: Optional[str] = Field(
        None,
        description="Who can apply — income, caste, age, occupation criteria"
    )
    benefit: Optional[str] = Field(
        None,
        description="What the scheme gives you — money, service, subsidy, etc."
    )

    # ── Application Information ───────────────────────────────────────────────
    how_to_apply: Optional[str] = Field(
        None,
        description="Step by step application process"
    )
    documents_needed: Optional[List[str]] = Field(
        default_factory=list,
        description="List of documents required to apply"
    )
    application_portal: Optional[str] = Field(
        None,
        description="Direct URL to apply online"
    )
    application_deadline: Optional[str] = Field(
        None,
        description="Closing date if scheme has a deadline"
    )

    # ── Contact and Support ───────────────────────────────────────────────────
    helpline_number: Optional[str] = Field(
        None,
        description="Toll-free helpline number for this scheme"
    )
    state_nodal_contact: Optional[str] = Field(
        None,
        description="State-level office contact details"
    )
    grievance_portal: Optional[str] = Field(
        None,
        description="URL or process to raise a complaint"
    )

    # ── Flags and Tags ────────────────────────────────────────────────────────
    csc_applicable: Optional[bool] = Field(
        None,
        description="True if you can apply at a Common Service Centre"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Searchable keywords for this scheme"
    )

    # ── Source and Freshness ──────────────────────────────────────────────────
    source_url: Optional[str] = Field(
        None,
        description="Original myscheme.gov.in URL for this scheme"
    )
    scraped_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this record was scraped"
    )
    last_verified: Optional[datetime] = Field(
        None,
        description="UTC timestamp of last manual or automated verification"
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @validator("source_url", pre=True, always=True)
    def build_source_url(cls, v, values):
        """
        If source_url is not provided, auto-build it from scheme_id.
        """
        if v:
            return v
        scheme_id = values.get("scheme_id")
        if scheme_id:
            return f"https://www.myscheme.gov.in/schemes/{scheme_id}"
        return None

    @validator("tags", pre=True, always=True)
    def ensure_tags_list(cls, v):
        """
        Tags can come in as None, a string, or a list.
        Normalise everything to a list of strings.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(t).strip() for t in v if t]

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# ── Scrape Result ─────────────────────────────────────────────────────────────
# Wraps every scrape attempt so we can track success, failure, and errors
# without crashing the whole pipeline when one scheme fails.

class ScrapeResult(BaseModel):
    """
    Tracks the outcome of scraping a single scheme.
    Successful scrapes carry a SchemeDetail object.
    Failed scrapes carry an error message for logging and retry.
    """

    slug: str
    success: bool
    data: Optional[SchemeDetail] = None
    error: Optional[str] = None
    http_status: Optional[int] = None
    attempted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "ignore"