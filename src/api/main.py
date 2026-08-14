"""
main.py — FastAPI application for YojanaGPT.

Routes:
  GET  /            → Welcome message
  GET  /health      → Health check with DB stats
  POST /ask         → General scheme Q&A
  POST /eligibility → Eligibility check with user profile
  POST /documents   → Document checklist for a scheme
  POST /apply       → Step-by-step application guide
  POST /compare     → Side-by-side scheme comparison
  POST /contact     → Contact details and helpline numbers
  POST /speak       → Text-to-speech (MP3)

Run with:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
import io
import logging
from contextlib import asynccontextmanager

from fastapi.responses import FileResponse, StreamingResponse

from .models import SpeakRequest  # add SpeakRequest to existing import
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import chromadb
from gtts import gTTS
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from .deps import get_pipeline, initialise_pipeline
from .models import (
    AskRequest,
    AskResponse,
    ApplyGuideRequest,
    CompareRequest,
    ContactRequest,
    DocumentsRequest,
    EligibilityRequest,
    HealthResponse,
    SchemeSource,
)
from src.retrieval.pipeline import YojanaRAGPipeline
from src.translation.detector import get_language_name

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        "government schemes in any of 22 Indian languages."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/ui", tags=["General"])
def serve_ui():
    """Serve the frontend UI."""
    return FileResponse("frontend/index.html")


# ── Shared helper ─────────────────────────────────────────────────────────────

def _to_ask_response(response) -> AskResponse:
    """Convert a RAGResponse to the AskResponse schema."""
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "name": "YojanaGPT",
        "description": "AI assistant for Indian government schemes",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ask": "POST /ask — General Q&A",
            "eligibility": "POST /eligibility — Eligibility check",
            "documents": "POST /documents — Document checklist",
            "apply": "POST /apply — Step-by-step guide",
            "compare": "POST /compare — Compare schemes",
            "contact": "POST /contact — Helpline & contact info",
            "speak": "POST /speak — Text to speech",
        },
        "languages": "Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu, English",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    try:
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
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/ask", response_model=AskResponse, tags=["Schemes"])
@limiter.limit("15/minute")
def ask_question(
    request: Request,
    body: AskRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Ask any question about Indian government schemes.
    Supports all 22 Indian languages.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = pipeline.ask(question=body.question, top_k=body.top_k)
        return _to_ask_response(response)
    except Exception as e:
        logger.error("Error in /ask: %s", e)
        raise HTTPException(status_code=500, detail="Error processing your question. Please try again.")


@app.post("/eligibility", response_model=AskResponse, tags=["Schemes"])
@limiter.limit("15/minute")
def check_eligibility(
    request: Request,
    body: EligibilityRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Check eligibility for government schemes based on your profile.

    Provide profile fields: age, state, income, caste, occupation, gender, disability, bpl.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = pipeline.check_eligibility(
            question=body.question,
            user_profile=body.profile,
            top_k=body.top_k,
        )
        return _to_ask_response(response)
    except Exception as e:
        logger.error("Error in /eligibility: %s", e)
        raise HTTPException(status_code=500, detail="Error checking eligibility. Please try again.")


@app.post("/documents", response_model=AskResponse, tags=["Schemes"])
@limiter.limit("15/minute")
def get_documents(
    request: Request,
    body: DocumentsRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Get a complete document checklist for applying to a scheme.

    Returns a numbered, categorised list of all documents required,
    with a note on why each document is needed.

    Example: "What documents do I need for PM Kisan?"
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = pipeline.get_documents(question=body.question, top_k=body.top_k)
        return _to_ask_response(response)
    except Exception as e:
        logger.error("Error in /documents: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching documents. Please try again.")


@app.post("/apply", response_model=AskResponse, tags=["Schemes"])
@limiter.limit("15/minute")
def get_apply_guide(
    request: Request,
    body: ApplyGuideRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Get a step-by-step application guide for a scheme.

    Covers online/offline application paths, portal URLs,
    which office to visit, and approximate timelines.

    Example: "How do I apply for PM Awas Yojana?"
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = pipeline.get_apply_guide(question=body.question, top_k=body.top_k)
        return _to_ask_response(response)
    except Exception as e:
        logger.error("Error in /apply: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching application guide. Please try again.")


@app.post("/compare", response_model=AskResponse, tags=["Schemes"])
@limiter.limit("15/minute")
def compare_schemes(
    request: Request,
    body: CompareRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Compare two or more government schemes side by side.

    Provide a list of scheme_names (at least 2). The response covers:
    benefits, eligibility, application process, key differences,
    and a recommendation on who should apply for which.

    Example:
        {
            "question": "Compare PM Kisan and PMFBY",
            "scheme_names": ["PM Kisan", "PMFBY"]
        }
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(body.scheme_names) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 scheme names to compare")
    try:
        response = pipeline.compare_schemes(
            question=body.question,
            scheme_names=body.scheme_names,
            top_k=body.top_k,
        )
        return _to_ask_response(response)
    except Exception as e:
        logger.error("Error in /compare: %s", e)
        raise HTTPException(status_code=500, detail="Error comparing schemes. Please try again.")


@app.post("/contact", response_model=AskResponse, tags=["Schemes"])
@limiter.limit("15/minute")
def get_contact(
    request: Request,
    body: ContactRequest,
    pipeline: YojanaRAGPipeline = Depends(get_pipeline),
):
    """
    Get helpline numbers and contact details for a scheme.

    Returns: helpline numbers, official portal URL, email,
    nodal ministry, and grievance redressal portal.

    Example: "What is the helpline for PM Kisan?"
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = pipeline.get_contact(question=body.question, top_k=body.top_k)
        return _to_ask_response(response)
    except Exception as e:
        logger.error("Error in /contact: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching contact details. Please try again.")


@app.post("/speak", tags=["General"])
@limiter.limit("15/minute")
def speak(request: Request, body: SpeakRequest):
    """
    Convert text to speech and stream back an MP3.
    Uses gTTS — lang_code should be a gTTS-supported language code
    (e.g. 'hi' for Hindi, 'ta' for Tamil, 'en' for English).
    """
    try:
        tts = gTTS(text=body.text, lang=body.lang_code)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/mpeg")
    except ValueError as e:
        logger.error("Unsupported language in /speak: %s", e)
        raise HTTPException(status_code=400, detail="Unsupported language code")
    except Exception as e:
        logger.error("Error in /speak: %s", e)
        raise HTTPException(status_code=500, detail="Error generating speech")