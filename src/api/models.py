"""
models.py — Pydantic request/response models for YojanaGPT API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared response pieces ─────────────────────────────────────────────────────

class SchemeSource(BaseModel):
    scheme_id: str
    scheme_name: str
    source_url: str


class AskResponse(BaseModel):
    answer: str
    detected_language: str
    language_name: str
    sources: List[SchemeSource]
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    chunks_in_db: int
    model: str
    translation_enabled: bool


# ── Request models ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "PM Kisan ke liye kaun eligible hai?",
                "top_k": 5,
            }
        }


class EligibilityRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    profile: Dict[str, str] = Field(default_factory=dict)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Am I eligible for any farmer scheme?",
                "profile": {
                    "age": "35",
                    "state": "Maharashtra",
                    "caste": "OBC",
                    "income": "120000",
                    "occupation": "Farmer",
                    "gender": "Male",
                },
                "top_k": 5,
            }
        }


class DocumentsRequest(BaseModel):
    """Request documents checklist for a scheme."""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What documents do I need for PM Kisan?",
                "top_k": 5,
            }
        }


class ApplyGuideRequest(BaseModel):
    """Request step-by-step application guide for a scheme."""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I apply for PM Awas Yojana?",
                "top_k": 5,
            }
        }


class CompareRequest(BaseModel):
    """Request side-by-side comparison of two or more schemes."""
    question: str = Field(..., min_length=1, max_length=1000)
    scheme_names: List[str] = Field(
        ...,
        min_length=2,
        description="List of scheme names to compare. Must have at least 2.",
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Compare PM Kisan and PMFBY",
                "scheme_names": ["PM Kisan", "PMFBY"],
                "top_k": 5,
            }
        }


class ContactRequest(BaseModel):
    """Request contact details and helpline info for a scheme."""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the helpline number for PM Kisan?",
                "top_k": 5,
            }
        }
        
class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    lang_code: str = Field(default="en")