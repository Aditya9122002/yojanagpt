"""
prompt.py — Builds the prompt sent to Gemini from retrieved chunks.

Takes the user's question and the top-k chunks from ChromaDB,
and formats them into a clear prompt that tells Gemini:
  - What context to use
  - How to answer
  - What language to answer in
  - What NOT to do (hallucinate, go off-topic)
"""

from __future__ import annotations

from typing import List

from .retriever import RetrievedChunk


# ── System prompt ─────────────────────────────────────────────────────────────
# This tells Gemini its role and rules.
# Kept concise — longer system prompts waste tokens and confuse smaller models.

SYSTEM_PROMPT = """You are YojanaGPT, an AI assistant that helps Indian citizens find and understand government schemes.

Your rules:
1. Answer ONLY based on the scheme information provided in the context below.
2. If the context does not contain enough information to answer, say so clearly. Do not guess.
3. Answer in the SAME LANGUAGE the user asked in. If they asked in Hindi, answer in Hindi. If English, answer in English.
4. Be concise and helpful. Use simple language that any citizen can understand.
5. When mentioning eligibility criteria, list them clearly.
6. Always mention the source scheme name in your answer.
7. If multiple schemes are relevant, mention all of them briefly."""


def build_prompt(
    question: str,
    chunks: List[RetrievedChunk],
) -> str:
    """
    Build the full prompt to send to Gemini.

    Format:
      [System instructions]
      [Context — scheme chunks]
      [User question]

    Args:
        question: The user's original question.
        chunks:   Retrieved chunks from ChromaDB.

    Returns:
        Complete prompt string ready to send to Gemini.
    """
    if not chunks:
        # No context found — ask Gemini to say so honestly
        return f"""{SYSTEM_PROMPT}

No relevant scheme information was found for this question.

User question: {question}

Please tell the user you could not find relevant information and suggest they visit myscheme.gov.in directly."""

    # Build the context block from chunks
    context_parts = []
    seen_schemes = set()

    for i, chunk in enumerate(chunks, 1):
        # Group chunks by scheme for readability
        scheme_header = ""
        if chunk.scheme_id not in seen_schemes:
            scheme_header = f"\n--- Scheme: {chunk.scheme_name} ---\n"
            seen_schemes.add(chunk.scheme_id)

        context_parts.append(
            f"{scheme_header}"
            f"[{chunk.chunk_type.upper()}]\n"
            f"{chunk.text.strip()}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT (Government Scheme Information) ---
{context}
--- END CONTEXT ---

User question: {question}

Answer:"""

    return prompt


def build_eligibility_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    user_profile: dict,
) -> str:
    """
    Build a prompt specifically for eligibility checking.

    Takes the user's profile (age, income, caste, state, etc.) and
    asks Gemini to check eligibility against the scheme criteria.

    Args:
        question:     The user's question.
        chunks:       Retrieved chunks from ChromaDB.
        user_profile: Dict with keys like age, income, caste, state, occupation.

    Returns:
        Complete eligibility-checking prompt.
    """
    # Format user profile as readable text
    profile_lines = []
    field_labels = {
        "age": "Age",
        "income": "Annual income",
        "caste": "Caste category",
        "state": "State",
        "occupation": "Occupation",
        "gender": "Gender",
        "disability": "Disability status",
        "bpl": "BPL card holder",
    }
    for key, label in field_labels.items():
        value = user_profile.get(key)
        if value:
            profile_lines.append(f"  - {label}: {value}")

    profile_text = "\n".join(profile_lines) if profile_lines else "  - (No profile information provided)"

    # Build context
    context_parts = []
    for chunk in chunks:
        if chunk.chunk_type in ("eligibility", "description", "benefit"):
            context_parts.append(
                f"Scheme: {chunk.scheme_name}\n"
                f"[{chunk.chunk_type.upper()}]\n"
                f"{chunk.text.strip()}"
            )

    context = "\n\n".join(context_parts) if context_parts else "No eligibility information found."

    prompt = f"""{SYSTEM_PROMPT}

You are checking eligibility for a user with this profile:
{profile_text}

--- SCHEME ELIGIBILITY INFORMATION ---
{context}
--- END ---

User question: {question}

Based on the user's profile and the scheme eligibility criteria above, tell them:
1. Which schemes they are likely eligible for
2. Which criteria they meet
3. Which criteria they may not meet (if any)
4. Next steps to apply

Answer:"""

    return prompt