"""
models.py — Pydantic models for API request and response validation.

These define the exact shape of data coming in and going out of the API.
FastAPI uses these automatically for:
  - Input validation (rejects bad requests with clear error messages)
  - Response serialization (converts Python objects to JSON)
  - Auto-generated API documentation at /docs
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """
    Request body for POST /ask

    Example:
        {
            "question": "PM Kisan ke liye kaun eligible hai?",
            "top_k": 5
        }
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question about any government scheme in any Indian language",
        example="PM Kisan ke liye kaun eligible hai?",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of scheme chunks to retrieve (1-20)",
    )


class EligibilityRequest(BaseModel):
    """
    Request body for POST /eligibility

    Example:
        {
            "question": "Kya mujhe PM Kisan mil sakta hai?",
            "profile": {
                "age": "45",
                "state": "Maharashtra",
                "occupation": "Farmer",
                "income": "80000"
            }
        }
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Eligibility question in any language",
        example="Kya mujhe koi scholarship mil sakti hai?",
    )
    profile: dict = Field(
        default_factory=dict,
        description="User profile with keys: age, state, income, caste, occupation, gender, disability, bpl",
        example={
            "age": "25",
            "state": "Maharashtra",
            "caste": "OBC",
            "income": "150000",
            "occupation": "Student",
        },
    )
    top_k: int = Field(default=5, ge=1, le=20)


# ── Response Models ───────────────────────────────────────────────────────────

class SchemeSource(BaseModel):
    """A single scheme cited as a source in the answer."""
    scheme_id: str
    scheme_name: str
    source_url: str


class AskResponse(BaseModel):
    """
    Response body for POST /ask and POST /eligibility

    Example:
        {
            "answer": "PM-KISAN ke liye sabhi zameen dharkar kisan...",
            "detected_language": "hi",
            "language_name": "Hindi",
            "sources": [
                {
                    "scheme_id": "pm-kisan",
                    "scheme_name": "Pradhan Mantri Kisan Samman Nidhi",
                    "source_url": "https://www.myscheme.gov.in/schemes/pm-kisan"
                }
            ],
            "chunks_retrieved": 5
        }
    """
    answer: str = Field(description="Answer in the same language as the question")
    detected_language: str = Field(description="Detected language code e.g. 'hi', 'ta', 'en'")
    language_name: str = Field(description="Human readable language name e.g. 'Hindi'")
    sources: List[SchemeSource] = Field(description="Schemes cited in the answer")
    chunks_retrieved: int = Field(description="Number of scheme chunks used to generate the answer")


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status: str
    chunks_in_db: int
    model: str
    translation_enabled: bool


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None