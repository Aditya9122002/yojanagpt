"""
pipeline.py — Orchestrates the full RAG pipeline with translation.

Flow for non-English queries:
  1. Detect language of user question
  2. Translate question → English
  3. Retrieve relevant chunks from ChromaDB (English)
  4. Build prompt with English chunks + original question
  5. LLM generates answer (tries to answer in user's language)
  6. If detected language is not English/Hindi, translate answer back
     to user's language using Google Translate as a guaranteed fallback
  7. Return answer with source attribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .llm import GeminiClient
from .prompt import (
    build_prompt,
    build_eligibility_prompt,
    build_documents_prompt,
    build_apply_guide_prompt,
    build_compare_prompt,
    build_contact_prompt,
)
from .retriever import RetrievedChunk, SchemeRetriever

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_CHROMA_DIR = "data/chromadb"


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    question: str
    detected_language: str = "en"
    english_question: str = ""
    chunks_used: List[RetrievedChunk] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)

    def __post_init__(self):
        seen = set()
        for chunk in self.chunks_used:
            if chunk.scheme_id not in seen:
                self.sources.append({
                    "scheme_id": chunk.scheme_id,
                    "scheme_name": chunk.scheme_name,
                    "source_url": chunk.source_url,
                })
                seen.add(chunk.scheme_id)


class YojanaRAGPipeline:
    """
    The main RAG pipeline for YojanaGPT with multilingual support.

    Supports:
      - General scheme Q&A
      - Eligibility checking with user profile
      - Document checklist extraction
      - Step-by-step application guide
      - Side-by-side scheme comparison
      - Contact details extraction
    """

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        top_k: int = DEFAULT_TOP_K,
        groq_api_key: Optional[str] = None,
        enable_translation: bool = True,
    ):
        logger.info("Initialising YojanaRAGPipeline...")

        self.retriever = SchemeRetriever(
            chroma_dir=chroma_dir,
            top_k=top_k,
        )
        self.llm = GeminiClient(api_key=groq_api_key)
        self.top_k = top_k
        self.enable_translation = enable_translation

        self._translator = None
        if enable_translation:
            self._load_translator()

        logger.info("YojanaRAGPipeline ready | translation=%s", enable_translation)

    def _load_translator(self):
        """Lazy load the translator."""
        try:
            from src.translation.translator import SchemeTranslator
            self._translator = SchemeTranslator()
            logger.info("Translation layer loaded")
        except ImportError as e:
            logger.warning(
                "Translation not available: %s. Processing in original language only.", e
            )
            self.enable_translation = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _translate_to_english(self, question: str) -> tuple[str, str]:
        """
        Translate question to English for retrieval.
        Returns (english_question, detected_language).
        Falls back to original question if translation fails.
        """
        if not self.enable_translation or not self._translator:
            return question, "en"
        try:
            english_question, detected_lang = self._translator.to_english(question)
            logger.info("Language: %s | English query: %s", detected_lang, english_question[:80])
            return english_question, detected_lang
        except Exception as e:
            logger.warning("Translation failed, using original: %s", e)
            return question, "en"

    def _translate_answer(self, answer: str, detected_lang: str) -> str:
        """
        Translate LLM answer back to user's language if needed.

        The LLM tries to answer in the user's language but often fails
        for non-Hindi Indian languages (Tamil, Telugu, Kannada etc.).
        This guarantees the answer is in the correct language by
        translating back using Google Translate as a fallback.

        Languages where LLM is reliable (skip translation):
          - English (en)
          - Hindi (hi) — LLM is strong in Hindi

        All other Indian languages get Google Translate fallback.
        """
        # LLM handles these well — skip translation
        if detected_lang in ("en", "hi"):
            return answer

        if not self.enable_translation or not self._translator:
            return answer

        try:
            translated = self._translator.from_english(answer, target_lang=detected_lang)
            logger.info("Answer translated to %s", detected_lang)
            return translated
        except Exception as e:
            logger.warning("Answer translation failed, returning original: %s", e)
            return answer

    def _retrieve(self, english_question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve chunks from ChromaDB using an English query."""
        chunks = self.retriever.search(english_question, top_k=top_k or self.top_k)
        logger.info("Retrieved %d chunks", len(chunks))
        return chunks

    def _build_response(
        self,
        answer: str,
        question: str,
        detected_lang: str,
        english_question: str,
        chunks: List[RetrievedChunk],
    ) -> RAGResponse:
        # Translate answer back to user's language if needed
        answer = self._translate_answer(answer, detected_lang)
        return RAGResponse(
            answer=answer,
            question=question,
            detected_language=detected_lang,
            english_question=english_question,
            chunks_used=chunks,
        )

    # ── Public methods ────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a general question about government schemes.
        Supports any language — auto-detects and handles translation.
        """
        if not question or not question.strip():
            return RAGResponse(
                answer="Please ask a question about government schemes.",
                question=question,
            )

        logger.info("Processing question: %s", question[:100])
        english_question, detected_lang = self._translate_to_english(question)
        chunks = self._retrieve(english_question, top_k)
        prompt = build_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return self._build_response(answer, question, detected_lang, english_question, chunks)

    def check_eligibility(
        self,
        question: str,
        user_profile: dict,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Check scheme eligibility based on user profile.

        Args:
            question:     User's eligibility question.
            user_profile: Dict with age, income, caste, state, occupation etc.
        """
        logger.info("Checking eligibility | profile keys=%s", list(user_profile.keys()))
        english_question, detected_lang = self._translate_to_english(question)
        chunks = self._retrieve(english_question, top_k)
        prompt = build_eligibility_prompt(question, chunks, user_profile)
        answer = self.llm.generate(prompt)
        return self._build_response(answer, question, detected_lang, english_question, chunks)

    def get_documents(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Return a document checklist for applying to a scheme.

        The LLM extracts all required documents from the scheme's context
        and formats them as a numbered, categorised checklist.

        Args:
            question: e.g. "What documents do I need for PM Kisan?"
        """
        logger.info("Document checklist request: %s", question[:100])
        english_question, detected_lang = self._translate_to_english(question)
        chunks = self._retrieve(english_question, top_k)
        prompt = build_documents_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return self._build_response(answer, question, detected_lang, english_question, chunks)

    def get_apply_guide(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Return a step-by-step application guide for a scheme.

        Covers online and offline application paths, portal URLs,
        office to visit, and expected timeline.

        Args:
            question: e.g. "How do I apply for PM Awas Yojana?"
        """
        logger.info("Application guide request: %s", question[:100])
        english_question, detected_lang = self._translate_to_english(question)
        chunks = self._retrieve(english_question, top_k)
        prompt = build_apply_guide_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return self._build_response(answer, question, detected_lang, english_question, chunks)

    def compare_schemes(
        self,
        question: str,
        scheme_names: List[str],
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Compare two or more schemes side by side.

        Retrieves chunks for all named schemes and asks the LLM to
        produce a structured comparison table covering benefits,
        eligibility, application process, and who should apply.

        Args:
            question:     User's comparison question.
            scheme_names: List of scheme names to compare (e.g. ["PM Kisan", "PMFBY"])
        """
        logger.info("Comparing schemes: %s", scheme_names)

        # Translate the question for retrieval
        english_question, detected_lang = self._translate_to_english(question)

        # Build a combined search query that includes all scheme names
        # so retriever fetches chunks for all of them
        search_queries = scheme_names if scheme_names else [english_question]
        all_chunks: List[RetrievedChunk] = []
        seen_ids = set()

        for sq in search_queries:
            chunks = self._retrieve(sq, top_k=top_k or self.top_k)
            for chunk in chunks:
                if chunk.scheme_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.scheme_id)

        # Also retrieve using the full question in case the names alone miss context
        for chunk in self._retrieve(english_question, top_k=top_k or self.top_k):
            if chunk.scheme_id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk.scheme_id)

        prompt = build_compare_prompt(question, all_chunks, scheme_names)
        answer = self.llm.generate(prompt)
        return self._build_response(answer, question, detected_lang, english_question, all_chunks)

    def get_contact(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Extract contact details and helpline info for a scheme.

        Returns helpline numbers, portal URLs, ministry name,
        email addresses, and grievance portal if available.

        Args:
            question: e.g. "What is the helpline for PM Kisan?"
        """
        logger.info("Contact details request: %s", question[:100])
        english_question, detected_lang = self._translate_to_english(question)
        chunks = self._retrieve(english_question, top_k)
        prompt = build_contact_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return self._build_response(answer, question, detected_lang, english_question, chunks)