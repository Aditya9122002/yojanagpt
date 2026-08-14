"""
Run this from D:\\yojanagpt:
    python fix_retriever.py
"""

QUERY_EXPANSION_CODE = '''
# Full name expansions for semantic search boost
QUERY_EXPANSION = {
    "pm kusum": "Pradhan Mantri Kisan Urja Suraksha Evam Utthaan Mahabhiyan PM-KUSUM solar pump farmer",
    "kusum yojana": "Pradhan Mantri Kisan Urja Suraksha Evam Utthaan Mahabhiyan solar pump",
    "kusum": "Pradhan Mantri Kisan Urja Suraksha Evam Utthaan Mahabhiyan PM-KUSUM solar pump",
    "pm kisan": "Pradhan Mantri Kisan Samman Nidhi PM-KISAN farmer income support 6000",
    "kisan samman": "Pradhan Mantri Kisan Samman Nidhi PM-KISAN farmer",
    "ayushman bharat": "Pradhan Mantri Jan Arogya Yojana PM-JAY health insurance 5 lakh",
    "ayushman": "Pradhan Mantri Jan Arogya Yojana PM-JAY health insurance hospital",
    "pm awas": "Pradhan Mantri Awas Yojana PMAY housing scheme home",
    "awas yojana": "Pradhan Mantri Awas Yojana PMAY housing scheme",
    "pmfby": "Pradhan Mantri Fasal Bima Yojana crop insurance farmer premium",
    "fasal bima": "Pradhan Mantri Fasal Bima Yojana crop insurance farmer",
    "mudra loan": "Pradhan Mantri Mudra Yojana PMMY micro enterprise small business loan",
    "mudra yojana": "Pradhan Mantri Mudra Yojana PMMY small business loan shishu kishor tarun",
    "sukanya": "Sukanya Samriddhi Yojana girl child savings account interest deposit",
    "sukanya samriddhi": "Sukanya Samriddhi Yojana SSY girl child savings scheme interest",
    "ujjwala": "Pradhan Mantri Ujjwala Yojana PMUY free LPG gas connection BPL women",
    "pm ujjwala": "Pradhan Mantri Ujjwala Yojana free LPG gas connection below poverty line",
    "skill india": "Pradhan Mantri Kaushal Vikas Yojana PMKVY skill training certification job",
    "kaushal vikas": "Pradhan Mantri Kaushal Vikas Yojana PMKVY skill development training",
    "jan dhan": "Pradhan Mantri Jan Dhan Yojana PMJDY zero balance bank account financial inclusion",
    "pmjdy": "Pradhan Mantri Jan Dhan Yojana zero balance savings account",
    "atal pension": "Atal Pension Yojana APY retirement pension unorganised sector subscriber",
    "vishwakarma": "PM Vishwakarma Yojana artisan craftsman traditional skills toolkit loan training",
    "pm vishwakarma": "PM Vishwakarma Yojana artisan craftsman skills collateral free loan",
    "svanidhi": "PM SVANidhi street vendor micro credit loan working capital",
    "street vendor": "PM SVANidhi Pradhan Mantri Street Vendor AtmaNirbhar Nidhi micro loan",
    "beti bachao": "Beti Bachao Beti Padhao BBBP girl child education welfare scheme",
    "beti padhao": "Beti Bachao Beti Padhao girl child education scheme",
    "swachh bharat": "Swachh Bharat Mission SBM toilet construction individual household sanitation ODF",
    "toilet scheme": "Swachh Bharat Mission toilet construction gram panchayat",
    "standup india": "Stand Up India SC ST women entrepreneur bank loan greenfield enterprise",
    "startup india": "Startup India scheme fund of funds innovation startup recognition",
    "garib kalyan": "Pradhan Mantri Garib Kalyan Anna Yojana PMGKAY free food grain ration",
    "free ration": "Pradhan Mantri Garib Kalyan Anna Yojana free food grain PDS",
    "e shram": "e-Shram card unorganised worker registration social security database",
    "eshram": "e-Shram unorganised worker construction agriculture domestic worker",
    "jeevan jyoti": "Pradhan Mantri Jeevan Jyoti Bima Yojana PMJJBY life insurance death benefit",
    "suraksha bima": "Pradhan Mantri Suraksha Bima Yojana PMSBY accident insurance disability",
    "pmsby": "Pradhan Mantri Suraksha Bima Yojana accidental death disability insurance",
    "pmjjby": "Pradhan Mantri Jeevan Jyoti Bima Yojana life insurance renewable",
    "national scholarship": "National Scholarship Portal NSP student merit scholarship education",
    "nsp scholarship": "National Scholarship Portal NSP pre matric post matric scholarship",
    "solar pump": "PM KUSUM Pradhan Mantri Kisan Urja Suraksha solar irrigation pump farmer",
    "kisan urja": "PM KUSUM Pradhan Mantri Kisan Urja Suraksha Evam Utthaan solar pump",
    "vikas bharat": "Vikas Bharat Rozgar Yojana employment scheme rural development",
}


def expand_query(query: str) -> str:
    """Expand short/common scheme names to full names for better semantic matching."""
    query_lower = query.lower().strip()
    expansions = []
    for short_name, full_name in QUERY_EXPANSION.items():
        if short_name in query_lower:
            expansions.append(full_name)
    if expansions:
        expanded = query + " " + " ".join(expansions)
        return expanded
    return query

'''

EXPANDED_SCHEME_KEYWORDS = '''SCHEME_KEYWORDS = {
    # PM Kisan
    "pm kisan": "pm-kisan",
    "pmkisan": "pm-kisan",
    "kisan samman": "pm-kisan",
    "pm-kisan": "pm-kisan",
    "kisan nidhi": "pm-kisan",
    # PM Kusum
    "pm kusum": "pm-kusum",
    "kusum yojana": "pm-kusum",
    "kusum": "pm-kusum",
    "pm-kusum": "pm-kusum",
    "kisan urja": "pm-kusum",
    "solar pump farmer": "pm-kusum",
    # Ayushman Bharat
    "ayushman": "pmjay",
    "pmjay": "pmjay",
    "pm jay": "pmjay",
    "jan arogya": "pmjay",
    "ayushman bharat": "pmjay",
    # PM Awas
    "pmay": "pmay",
    "pm awas": "pmay",
    "awas yojana": "pmay",
    # Ujjwala
    "ujjwala": "pmuy",
    "pmuy": "pmuy",
    "pm ujjwala": "pmuy",
    "free lpg": "pmuy",
    # Mudra
    "mudra": "mudra",
    "pm mudra": "mudra",
    "mudra loan": "mudra",
    "pmmy": "mudra",
    # Jan Dhan
    "pmjdy": "pmjdy",
    "jan dhan": "pmjdy",
    # PMFBY
    "pmfby": "pmfby",
    "fasal bima": "pmfby",
    "crop insurance": "pmfby",
    # Skill India
    "pmkvy": "pmkvy",
    "skill india": "pmkvy",
    "kaushal vikas": "pmkvy",
    # Sukanya
    "sukanya": "ssy",
    "sukanya samriddhi": "ssy",
    # Beti Bachao
    "beti bachao": "bbbp",
    "beti padhao": "bbbp",
    "bbbp": "bbbp",
    # Atal Pension
    "atal pension": "apy",
    "apy": "apy",
    # Startup / Standup
    "standup india": "sui",
    "startup india": "startup-india",
    # Scholarships
    "nos-swd": "nos-swd",
    "national overseas scholarship": "nos-swd",
    "nsp": "nsp",
    # PMSBY / PMJJBY
    "pmsby": "pmsby",
    "suraksha bima": "pmsby",
    "pmjjby": "pmjjby",
    "jeevan jyoti": "pmjjby",
    # PM Vishwakarma
    "vishwakarma": "pm-vishwakarma",
    "pm vishwakarma": "pm-vishwakarma",
    # SVANidhi
    "svanidhi": "pm-svanidhi",
    "street vendor": "pm-svanidhi",
    # Swachh Bharat
    "swachh bharat": "sbm",
    "sbm": "sbm",
    # e-Shram
    "e shram": "e-shram",
    "eshram": "e-shram",
    # Garib Kalyan
    "garib kalyan": "pmgkay",
    "pmgkay": "pmgkay",
    "free ration": "pmgkay",
}'''

OLD_SCHEME_KEYWORDS = '''SCHEME_KEYWORDS = {
    "pm kisan": "pm-kisan",
    "pmkisan": "pm-kisan",
    "kisan samman": "pm-kisan",
    "pm-kisan": "pm-kisan",
    "pmsby": "pmsby",
    "pmjdy": "pmjdy",
    "pmay": "pmay",
    "pm awas": "pmay",
    "ujjwala": "pmuy",
    "pmuy": "pmuy",
    "ayushman": "pmjay",
    "pmjay": "pmjay",
    "mudra": "mudra",
    "pm mudra": "mudra",
    "standup india": "sui",
    "startup india": "startup-india",
    "skill india": "pmkvy",
    "pmkvy": "pmkvy",
    "nos-swd": "nos-swd",
    "national overseas scholarship": "nos-swd",
}'''

OLD_SEMANTIC = '''    def _semantic_search(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """Embed the question and search ChromaDB by vector similarity."""
        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()'''

NEW_SEMANTIC = '''    def _semantic_search(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """Embed the question and search ChromaDB by vector similarity."""
        # Expand query with full scheme names for better matching
        expanded = expand_query(question)
        query_embedding = self.model.encode(
            expanded,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()'''


def main():
    path = "src/retrieval/retriever.py"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Original file: {len(content)} chars")

    # Step 1 - Replace SCHEME_KEYWORDS
    if OLD_SCHEME_KEYWORDS in content:
        content = content.replace(OLD_SCHEME_KEYWORDS,
                                  EXPANDED_SCHEME_KEYWORDS + "\n" + QUERY_EXPANSION_CODE)
        print("✅ SCHEME_KEYWORDS expanded + QUERY_EXPANSION added")
    else:
        print("❌ Could not find SCHEME_KEYWORDS — check manually")
        return

    # Step 2 - Update _semantic_search to use expand_query
    if OLD_SEMANTIC in content:
        content = content.replace(OLD_SEMANTIC, NEW_SEMANTIC)
        print("✅ _semantic_search updated to use expand_query")
    else:
        print("❌ Could not find _semantic_search — check manually")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Done! New file: {len(content)} chars")
    print("\nNow test with:")
    print('  python -m src.retrieval.cli "PM Kusum Yojana"')
    print('  python -m src.retrieval.cli "solar pump scheme for farmers"')
    print('  python -m src.retrieval.cli "vikas bharat rozgar"')


if __name__ == "__main__":
    main()