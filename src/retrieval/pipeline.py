"""
pipeline.py — Orchestrates the full RAG pipeline with translation.

Flow for non-English queries:
  1. Detect language of user question
  2. Translate question → English
  3. Retrieve relevant chunks from ChromaDB (English)
  4. Build prompt with English chunks + original question
  5. LLM generates answer (in user's language — the prompt instructs this)
  6. Return answer with source attribution

Note on translation strategy:
  We translate the QUESTION to English for better retrieval,
  but we include the ORIGINAL question in the prompt so the LLM
  knows what language to answer in. This gives better results than
  translating the answer after the fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .llm import GeminiClient
from .prompt import build_prompt, build_eligibility_prompt
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

    Usage:
        pipeline = YojanaRAGPipeline()

        # Works in any language
        response = pipeline.ask("PM Kisan ke liye kaun eligible hai?")
        response = pipeline.ask("PM கிசான் திட்டம் என்ன?")  # Tamil
        response = pipeline.ask("Who is eligible for PM Kisan?")  # English

        print(response.answer)
        print(response.detected_language)
        print(response.sources)
    """

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        top_k: int = DEFAULT_TOP_K,
        gemini_api_key: Optional[str] = None,
        enable_translation: bool = True,
    ):
        logger.info("Initialising YojanaRAGPipeline...")

        self.retriever = SchemeRetriever(
            chroma_dir=chroma_dir,
            top_k=top_k,
        )
        self.llm = GeminiClient(api_key=gemini_api_key)
        self.top_k = top_k
        self.enable_translation = enable_translation

        # Lazy load translator — only import if translation enabled
        # This avoids loading langdetect if not needed
        self._translator = None
        if enable_translation:
            self._load_translator()

        logger.info("YojanaRAGPipeline ready | translation=%s", enable_translation)

    def _load_translator(self):
        """Lazy load the translator to avoid import errors if not installed."""
        try:
            from src.translation.translator import SchemeTranslator
            self._translator = SchemeTranslator()
            logger.info("Translation layer loaded")
        except ImportError as e:
            logger.warning(
                "Translation not available (missing packages): %s. "
                "Queries will be processed in original language only.", e
            )
            self.enable_translation = False

    def ask(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a user's question about government schemes.
        Supports any language — automatically detects and handles translation.

        Args:
            question: User's question in any language.
            top_k:    Number of chunks to retrieve.

        Returns:
            RAGResponse with answer in the user's language.
        """
        if not question or not question.strip():
            return RAGResponse(
                answer="Please ask a question about government schemes.",
                question=question,
            )

        logger.info("Processing question: %s", question[:100])

        # Step 1 — detect language and translate to English for retrieval
        detected_lang = "en"
        english_question = question

        if self.enable_translation and self._translator:
            try:
                english_question, detected_lang = self._translator.to_english(question)
                logger.info(
                    "Language: %s | English query: %s",
                    detected_lang,
                    english_question[:80],
                )
            except Exception as e:
                logger.warning("Translation failed, using original: %s", e)
                english_question = question
                detected_lang = "en"

        # Step 2 — retrieve using English question (better embedding match)
        chunks = self.retriever.search(english_question, top_k=top_k or self.top_k)
        logger.info("Retrieved %d chunks", len(chunks))

        # Step 3 — build prompt
        # Pass ORIGINAL question (not translated) so LLM answers in user's language
        prompt = build_prompt(question, chunks)

        # Step 4 — generate answer
        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            question=question,
            detected_language=detected_lang,
            english_question=english_question,
            chunks_used=chunks,
        )

    def check_eligibility(
        self,
        question: str,
        user_profile: dict,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Check eligibility for schemes based on user profile.

        Args:
            question:     User's eligibility question.
            user_profile: Dict with age, income, caste, state, occupation etc.
            top_k:        Number of chunks to retrieve.

        Returns:
            RAGResponse with eligibility assessment.
        """
        logger.info("Checking eligibility | profile=%s", user_profile)

        # Translate question for retrieval
        english_question = question
        detected_lang = "en"

        if self.enable_translation and self._translator:
            try:
                english_question, detected_lang = self._translator.to_english(question)
            except Exception:
                pass

        chunks = self.retriever.search(english_question, top_k=top_k or self.top_k)
        prompt = build_eligibility_prompt(question, chunks, user_profile)
        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            question=question,
            detected_language=detected_lang,
            english_question=english_question,
            chunks_used=chunks,
        )