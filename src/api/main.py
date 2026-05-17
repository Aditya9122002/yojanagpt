"""
main.py — FastAPI application for YojanaGPT.

Routes:
  GET  /          → Welcome message
  GET  /health    → Health check with DB stats
  POST /ask       → Ask a question about any scheme
  POST /eligibility → Check eligibility with user profile

Run with:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Then visit:
  http://localhost:8000/docs    → Interactive API documentation
  http://localhost:8000/health  → Health check
"""


from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .deps import get_pipeline, initialise_pipeline
from .models import (
    AskRequest,
    AskResponse,
    EligibilityRequest,
    HealthResponse,
    SchemeSource,
)
from src.retrieval.pipeline import YojanaRAGPipeline
from src.translation.detector import get_language_name

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
# FastAPI lifespan runs startup and shutdown code.
# We use it to pre-load the pipeline so the first request is fast.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline at startup, clean up at shutdown."""
    logger.info("YojanaGPT API starting up...")
    initialise_pipeline()
    logger.info("YojanaGPT API ready")
    yield
    logger.info("YojanaGPT API shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YojanaGPT API",
    description=(
        "AI-powered API to help Indian citizens find and understand "
        "government schemes in any of 22 Indian languages. "
        "Ask questions in Hindi, Tamil, Bengali, Telugu, or any Indian language."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for now (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """Welcome message and API info."""
    return {
        "name": "YojanaGPT",
        "description": "AI assistant for Indian government schemes",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "languages": "Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, English",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Health check endpoint.
    Returns database stats and model info.
    Used by monitoring systems to verify the API is running.
    """
    try:
        # Check ChromaDB
        client = chromadb.PersistentClient(path="data/chromadb")
        collection = client.get_collection("yojanagpt_schemes")
        chunk_count = collection.count()

        return HealthResponse(
            status="ok",
            chunks_in_db=chunk_count,
            model="llama-3.3-70b-versatile (Groq)",
            translation_enabled=True,
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["Schemes"])
def ask_question(
    request: AskRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Ask any question about Indian government schemes.

    Supports all 22 Indian languages — ask in Hindi, Tamil, Bengali, etc.
    The answer will be in the same language as your question.

    Examples:
    - "PM Kisan ke liye kaun eligible hai?"
    - "What documents do I need for NOS-SWD scholarship?"
    - "நான் PM கிசான் திட்டத்திற்கு தகுதியானவரா?"
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = pipeline.ask(
            question=request.question,
            top_k=request.top_k,
        )

        return AskResponse(
            answer=response.answer,
            detected_language=response.detected_language,
            language_name=get_language_name(response.detected_language),
            sources=[
                SchemeSource(
                    scheme_id=s["scheme_id"],
                    scheme_name=s["scheme_name"],
                    source_url=s["source_url"],
                )
                for s in response.sources
            ],
            chunks_retrieved=len(response.chunks_used),
        )

    except Exception as e:
        logger.error("Error processing question: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


@app.post("/eligibility", response_model=AskResponse, tags=["Schemes"])
def check_eligibility(
    request: EligibilityRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Check eligibility for government schemes based on user profile.

    Provide your profile (age, state, income, caste, occupation) and
    ask which schemes you qualify for. The answer will be in the same
    language as your question.

    Example profile:
        {
            "age": "35",
            "state": "Maharashtra",
            "caste": "OBC",
            "income": "120000",
            "occupation": "Farmer"
        }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = pipeline.check_eligibility(
            question=request.question,
            user_profile=request.profile,
            top_k=request.top_k,
        )

        return AskResponse(
            answer=response.answer,
            detected_language=response.detected_language,
            language_name=get_language_name(response.detected_language),
            sources=[
                SchemeSource(
                    scheme_id=s["scheme_id"],
                    scheme_name=s["scheme_name"],
                    source_url=s["source_url"],
                )
                for s in response.sources
            ],
            chunks_retrieved=len(response.chunks_used),
        )

    except Exception as e:
        logger.error("Error checking eligibility: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error checking eligibility: {str(e)}",
        )