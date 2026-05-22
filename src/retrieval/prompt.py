"""
prompt.py — Builds the prompt sent to the LLM from retrieved chunks.

Takes the user's question and the top-k chunks from ChromaDB,
and formats them into a clear prompt that tells the LLM:
  - What context to use
  - How to answer
  - What language to answer in
  - What NOT to do (hallucinate, go off-topic)
"""

from __future__ import annotations

from typing import List

from .retriever import RetrievedChunk


# ── System prompt ─────────────────────────────────────────────────────────────

# ── REPLACE SYSTEM_PROMPT in src/retrieval/prompt.py ─────────────
# Find the existing SYSTEM_PROMPT constant and replace it with this:

SYSTEM_PROMPT = """You are YojanaGPT, an AI assistant that helps Indian citizens find and understand government schemes.

Your rules:
1. Answer ONLY based on the scheme information provided in the context below.
2. If the context does not contain enough information to answer, say so clearly. Do not guess.
3. LANGUAGE RULE — This is critical: You MUST answer in the EXACT SAME LANGUAGE the user asked in.
   - If the user asked in Tamil (தமிழ்), answer entirely in Tamil.
   - If the user asked in Telugu (తెలుగు), answer entirely in Telugu.
   - If the user asked in Bengali (বাংলা), answer entirely in Bengali.
   - If the user asked in Marathi (मराठी), answer entirely in Marathi.
   - If the user asked in Gujarati (ગુજરાતી), answer entirely in Gujarati.
   - If the user asked in Kannada (ಕನ್ನಡ), answer entirely in Kannada.
   - If the user asked in Malayalam (മലയാളം), answer entirely in Malayalam.
   - If the user asked in Hindi or Hinglish, answer in Hindi.
   - If the user asked in English, answer in English.
   - NEVER answer in a different language than the one used in the question.
4. Be concise and helpful. Use simple language that any citizen can understand.
5. When mentioning eligibility criteria, list them clearly.
6. Always mention the source scheme name in your answer.
7. If multiple schemes are relevant, mention all of them briefly."""


def build_prompt(
    question: str,
    chunks: List[RetrievedChunk],
) -> str:
    """
    Build the full prompt for a general scheme question.

    Args:
        question: The user's original question.
        chunks:   Retrieved chunks from ChromaDB.

    Returns:
        Complete prompt string ready to send to the LLM.
    """
    if not chunks:
        return f"""{SYSTEM_PROMPT}

No relevant scheme information was found for this question.

User question: {question}

Please tell the user you could not find relevant information and suggest they visit myscheme.gov.in directly."""

    context_parts = []
    seen_schemes = set()

    for i, chunk in enumerate(chunks, 1):
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
    Build a prompt for eligibility checking against a user profile.

    Args:
        question:     The user's question.
        chunks:       Retrieved chunks from ChromaDB.
        user_profile: Dict with keys like age, income, caste, state, occupation.

    Returns:
        Complete eligibility-checking prompt.
    """
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

    profile_text = (
        "\n".join(profile_lines)
        if profile_lines
        else "  - (No profile information provided)"
    )

    context_parts = []
    for chunk in chunks:
        if chunk.chunk_type in ("eligibility", "description", "benefit"):
            context_parts.append(
                f"Scheme: {chunk.scheme_name}\n"
                f"[{chunk.chunk_type.upper()}]\n"
                f"{chunk.text.strip()}"
            )

    context = (
        "\n\n".join(context_parts)
        if context_parts
        else "No eligibility information found."
    )

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


def build_documents_prompt(
    question: str,
    chunks: List[RetrievedChunk],
) -> str:
    """
    Build a prompt that extracts required documents for a scheme.

    Instructs the LLM to produce a clean numbered checklist of documents
    the applicant must gather before applying.

    Args:
        question: User's question (e.g. "What documents do I need for PM Kisan?")
        chunks:   Retrieved chunks from ChromaDB.

    Returns:
        Prompt string focused on document requirements.
    """
    if not chunks:
        return f"""{SYSTEM_PROMPT}

No relevant scheme information was found.

User question: {question}

Tell the user you could not find document requirements and suggest they visit myscheme.gov.in."""

    context_parts = []
    seen_schemes = set()

    for chunk in chunks:
        scheme_header = ""
        if chunk.scheme_id not in seen_schemes:
            scheme_header = f"\n--- Scheme: {chunk.scheme_name} ---\n"
            seen_schemes.add(chunk.scheme_id)
        context_parts.append(
            f"{scheme_header}[{chunk.chunk_type.upper()}]\n{chunk.text.strip()}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT (Government Scheme Information) ---
{context}
--- END CONTEXT ---

User question: {question}

Your task: Extract and list ALL documents the applicant needs to apply for this scheme.

Format your answer as a numbered checklist. For each document write:
  - Document name
  - Why it is needed (one short sentence)

Group documents by category if possible (Identity proof, Income proof, Bank details, etc.).
End with a note about where to submit these documents.

Answer in the same language as the question.

Answer:"""

    return prompt


def build_apply_guide_prompt(
    question: str,
    chunks: List[RetrievedChunk],
) -> str:
    """
    Build a prompt that generates a step-by-step application guide.

    Instructs the LLM to produce clear, actionable steps a citizen
    can follow to actually apply for the scheme — online or offline.

    Args:
        question: User's question (e.g. "How do I apply for PM Kisan?")
        chunks:   Retrieved chunks from ChromaDB.

    Returns:
        Prompt string focused on application procedure.
    """
    if not chunks:
        return f"""{SYSTEM_PROMPT}

No relevant scheme information was found.

User question: {question}

Tell the user you could not find application steps and suggest they visit myscheme.gov.in."""

    context_parts = []
    seen_schemes = set()

    for chunk in chunks:
        scheme_header = ""
        if chunk.scheme_id not in seen_schemes:
            scheme_header = f"\n--- Scheme: {chunk.scheme_name} ---\n"
            seen_schemes.add(chunk.scheme_id)
        context_parts.append(
            f"{scheme_header}[{chunk.chunk_type.upper()}]\n{chunk.text.strip()}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT (Government Scheme Information) ---
{context}
--- END CONTEXT ---

User question: {question}

Your task: Provide a clear step-by-step guide to apply for this scheme.

Format:
Step 1: [Action] — [Brief explanation]
Step 2: ...
...

Include:
- Whether application is online, offline, or both
- Official website or portal URL if mentioned in the context
- Which government office to visit for offline applications
- Approximate timeline or processing time if mentioned
- Helpline number or contact if mentioned

Keep language simple. A first-time applicant with no prior experience should be able to follow these steps.
Answer in the same language as the question.

Answer:"""

    return prompt


def build_compare_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    scheme_names: List[str],
) -> str:
    """
    Build a prompt that compares two or more schemes side by side.

    Instructs the LLM to produce a structured comparison covering
    benefits, eligibility, application process, and who should apply.

    Args:
        question:     User's comparison question.
        chunks:       Retrieved chunks covering all schemes to compare.
        scheme_names: Names of schemes being compared (for clarity).

    Returns:
        Prompt string focused on scheme comparison.
    """
    if not chunks:
        return f"""{SYSTEM_PROMPT}

No relevant scheme information was found for comparison.

User question: {question}

Tell the user you could not find enough information and suggest they visit myscheme.gov.in."""

    context_parts = []
    seen_schemes = set()

    for chunk in chunks:
        scheme_header = ""
        if chunk.scheme_id not in seen_schemes:
            scheme_header = f"\n--- Scheme: {chunk.scheme_name} ---\n"
            seen_schemes.add(chunk.scheme_id)
        context_parts.append(
            f"{scheme_header}[{chunk.chunk_type.upper()}]\n{chunk.text.strip()}"
        )

    context = "\n\n".join(context_parts)
    schemes_list = " vs ".join(scheme_names) if scheme_names else "the requested schemes"

    prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT (Government Scheme Information) ---
{context}
--- END CONTEXT ---

User question: {question}

Your task: Compare {schemes_list} side by side.

Use this structure:

**Overview**
[One sentence about each scheme's purpose]

**Benefits**
[What each scheme provides — money, subsidy, loan, insurance, etc.]

**Who is eligible**
[Key eligibility criteria for each scheme]

**How to apply**
[Application process for each scheme — online/offline, portal]

**Key differences**
[Most important differences a citizen should know when choosing]

**Recommendation**
[Which scheme suits which type of person — be specific]

Answer in the same language as the question.

Answer:"""

    return prompt


def build_contact_prompt(
    question: str,
    chunks: List[RetrievedChunk],
) -> str:
    """
    Build a prompt that extracts contact details and helpline info for a scheme.

    Args:
        question: User's question about contacts/helpline.
        chunks:   Retrieved chunks from ChromaDB.

    Returns:
        Prompt string focused on contact extraction.
    """
    if not chunks:
        return f"""{SYSTEM_PROMPT}

No relevant scheme information was found.

User question: {question}

Tell the user you could not find contact details and suggest they visit myscheme.gov.in or call 1800-11-8080 (MyScheme helpline)."""

    context_parts = []
    seen_schemes = set()

    for chunk in chunks:
        scheme_header = ""
        if chunk.scheme_id not in seen_schemes:
            scheme_header = f"\n--- Scheme: {chunk.scheme_name} ---\n"
            seen_schemes.add(chunk.scheme_id)
        context_parts.append(
            f"{scheme_header}[{chunk.chunk_type.upper()}]\n{chunk.text.strip()}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT (Government Scheme Information) ---
{context}
--- END CONTEXT ---

User question: {question}

Your task: Extract all contact information for this scheme.

List:
- Helpline numbers (toll-free if available)
- Official website / portal URL
- Email addresses
- Nodal ministry or department name
- State-level contact points if mentioned
- Grievance redressal portal if mentioned

If a specific piece of information is not in the context, say "Not specified in available data" rather than guessing.
End with: "For general scheme queries you can also call MyScheme helpline: 1800-11-8080"

Answer in the same language as the question.

Answer:"""

    return prompt