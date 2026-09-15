"""Curated, auditable aliases for refined OpenSearch title retrieval.

The registry deliberately contains only relationships that are safe to expand in
both directions.  Aliases that identify more than one entity are detected and
excluded from automatic OpenSearch synonym rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.data_processing import simple_query_tokens, simple_title_tokens


@dataclass(frozen=True)
class SearchAliasEntity:
    entity_id: str
    category: str
    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class AliasAmbiguity:
    alias: str
    candidates: tuple[str, ...]


# Keep this registry conservative.  A false equivalence can distort traffic
# totals, so additions should be reviewed by an editor before release.
SEARCH_ALIAS_ENTITIES: tuple[SearchAliasEntity, ...] = (
    # Geographic aliases and renamed locations.
    SearchAliasEntity("geo_uttar_pradesh", "geographic", "Uttar Pradesh", ("UP", "U.P.")),
    SearchAliasEntity("geo_delhi", "geographic", "Delhi", ("DL",)),
    SearchAliasEntity("geo_madhya_pradesh", "geographic", "Madhya Pradesh", ("MP", "M.P.")),
    SearchAliasEntity("geo_himachal_pradesh", "geographic", "Himachal Pradesh", ("HP", "H.P.")),
    SearchAliasEntity(
        "geo_jammu_kashmir",
        "geographic",
        "Jammu and Kashmir",
        ("J&K", "JK", "J and K"),
    ),
    SearchAliasEntity("geo_andhra_pradesh", "geographic", "Andhra Pradesh", ("AP", "A.P.")),
    SearchAliasEntity("geo_arunachal_pradesh", "geographic", "Arunachal Pradesh", ("AP", "A.P.", "AR")),
    SearchAliasEntity("geo_maharashtra", "geographic", "Maharashtra", ("MH",)),
    SearchAliasEntity("geo_rajasthan", "geographic", "Rajasthan", ("RJ",)),
    SearchAliasEntity("geo_karnataka", "geographic", "Karnataka", ("KA",)),
    SearchAliasEntity("geo_kerala", "geographic", "Kerala", ("KL",)),
    SearchAliasEntity("geo_tamil_nadu", "geographic", "Tamil Nadu", ("TN",)),
    SearchAliasEntity("geo_telangana", "geographic", "Telangana", ("TG", "TS")),
    SearchAliasEntity("geo_west_bengal", "geographic", "West Bengal", ("WB",)),
    SearchAliasEntity("geo_chhattisgarh", "geographic", "Chhattisgarh", ("CG",)),
    SearchAliasEntity("geo_gujarat", "geographic", "Gujarat", ("GJ",)),
    SearchAliasEntity("geo_haryana", "geographic", "Haryana", ("HR",)),
    SearchAliasEntity("geo_jharkhand", "geographic", "Jharkhand", ("JH",)),
    SearchAliasEntity("geo_odisha", "geographic", "Odisha", ("OD",)),
    # Government institutions and ministries.
    SearchAliasEntity("gov_rbi", "government", "Reserve Bank of India", ("RBI",)),
    SearchAliasEntity("gov_eci", "government", "Election Commission of India", ("ECI",)),
    SearchAliasEntity("gov_cbi", "government", "Central Bureau of Investigation", ("CBI",)),
    SearchAliasEntity("gov_nia", "government", "National Investigation Agency", ("NIA",)),
    SearchAliasEntity(
        "gov_sebi",
        "government",
        "Securities and Exchange Board of India",
        ("SEBI",),
    ),
    SearchAliasEntity("gov_mha", "government", "Ministry of Home Affairs", ("MHA",)),
    SearchAliasEntity("gov_mea", "government", "Ministry of External Affairs", ("MEA",)),
    SearchAliasEntity("gov_mod", "government", "Ministry of Defence", ("MoD",)),
    # Political parties and alliances.
    SearchAliasEntity("party_bjp", "political", "Bharatiya Janata Party", ("BJP",)),
    SearchAliasEntity("party_inc", "political", "Indian National Congress", ("INC",)),
    SearchAliasEntity("party_aap", "political", "Aam Aadmi Party", ("AAP",)),
    SearchAliasEntity("party_tmc", "political", "Trinamool Congress", ("TMC",)),
    SearchAliasEntity("alliance_nda", "political", "National Democratic Alliance", ("NDA",)),
    SearchAliasEntity("alliance_upa", "political", "United Progressive Alliance", ("UPA",)),
    # Public-office abbreviations.  MP intentionally makes the alias ambiguous.
    SearchAliasEntity("office_prime_minister", "public_office", "Prime Minister", ("PM",)),
    SearchAliasEntity("office_chief_minister", "public_office", "Chief Minister", ("CM",)),
    SearchAliasEntity("office_member_parliament", "public_office", "Member of Parliament", ("MP",)),
    SearchAliasEntity(
        "office_mla",
        "public_office",
        "Member of Legislative Assembly",
        ("MLA",),
    ),
    # Frequently used news acronyms.
    SearchAliasEntity("news_gst", "news_acronym", "Goods and Services Tax", ("GST",)),
    SearchAliasEntity("news_gdp", "news_acronym", "Gross Domestic Product", ("GDP",)),
    SearchAliasEntity("news_fir", "news_acronym", "First Information Report", ("FIR",)),
    SearchAliasEntity("news_ai", "news_acronym", "Artificial Intelligence", ("AI",)),
    # Alternate spellings, transliterations, and renamed cities.
    SearchAliasEntity("place_bengaluru", "spelling_transliteration", "Bengaluru", ("Bangalore",)),
    SearchAliasEntity("place_mumbai", "spelling_transliteration", "Mumbai", ("Bombay",)),
    SearchAliasEntity("place_kolkata", "spelling_transliteration", "Kolkata", ("Calcutta",)),
    SearchAliasEntity("place_chennai", "spelling_transliteration", "Chennai", ("Madras",)),
    SearchAliasEntity("place_odisha", "spelling_transliteration", "Odisha", ("Orissa",)),
    SearchAliasEntity("place_prayagraj", "spelling_transliteration", "Prayagraj", ("Allahabad",)),
    SearchAliasEntity("place_gurugram", "spelling_transliteration", "Gurugram", ("Gurgaon",)),
    # Organization and company aliases.
    SearchAliasEntity("org_sbi", "organization_company", "State Bank of India", ("SBI",)),
    SearchAliasEntity("org_lic", "organization_company", "Life Insurance Corporation of India", ("LIC",)),
    SearchAliasEntity("org_ongc", "organization_company", "Oil and Natural Gas Corporation", ("ONGC",)),
    SearchAliasEntity("org_tcs", "organization_company", "Tata Consultancy Services", ("TCS",)),
    SearchAliasEntity("org_ril", "organization_company", "Reliance Industries Limited", ("RIL",)),
    SearchAliasEntity("org_hul", "organization_company", "Hindustan Unilever Limited", ("HUL",)),
)


_INITIALISM_AMPERSAND = re.compile(r"\b([A-Za-z])\s*&\s*([A-Za-z])\b")


def normalize_refined_alias_text(text: str) -> str:
    """Preserve punctuation-heavy initialisms before the normal token cleaner.

    The existing tokenizer already collapses dotted forms such as U.P.  This
    additional normalization makes J&K behave like the searchable token JK.
    """
    if not isinstance(text, str):
        return ""
    return _INITIALISM_AMPERSAND.sub(r"\1\2", text)


def refined_alias_title_text(title: str) -> str:
    """Build the text stored in the alias-aware OpenSearch subfield."""
    return " ".join(simple_title_tokens(normalize_refined_alias_text(title)))


def opensearch_synonym_rules() -> tuple[str, ...]:
    """Return safe Solr-format equivalence rules for ``synonym_graph``."""
    normalized_alias_targets = _normalized_alias_targets()
    ambiguous = {
        alias
        for alias, entity_ids in normalized_alias_targets.items()
        if len(entity_ids) > 1
    }
    rules = []
    for entity in SEARCH_ALIAS_ENTITIES:
        phrases = [_normalized_title_phrase(entity.canonical)]
        phrases.extend(
            _normalized_title_phrase(alias)
            for alias in entity.aliases
            if _normalized_title_phrase(alias) not in ambiguous
        )
        phrases = list(dict.fromkeys(phrase for phrase in phrases if phrase))
        if len(phrases) > 1:
            rules.append(", ".join(phrases))
    return tuple(dict.fromkeys(rules))


def query_uses_alias_expansion(normalized_query: str) -> bool:
    """Return whether a safe synonym rule applies to the normalized query."""
    query_tokens = str(normalized_query).split()
    for rule in opensearch_synonym_rules():
        for phrase in rule.split(", "):
            if _contains_token_phrase(query_tokens, phrase.split()):
                return True
    return False


def find_ambiguous_aliases(query: str) -> tuple[AliasAmbiguity, ...]:
    """Identify aliases that require an explicit full-form choice."""
    query_tokens = simple_query_tokens(normalize_refined_alias_text(query))
    targets = _normalized_alias_targets()
    entities = {entity.entity_id: entity for entity in SEARCH_ALIAS_ENTITIES}
    matches = []
    for alias, entity_ids in targets.items():
        if len(entity_ids) < 2 or not _contains_token_phrase(query_tokens, alias.split()):
            continue
        candidates = tuple(entities[entity_id].canonical for entity_id in sorted(entity_ids))
        matches.append(AliasAmbiguity(alias=alias.upper(), candidates=candidates))
    return tuple(matches)


def _normalized_alias_targets() -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    for entity in SEARCH_ALIAS_ENTITIES:
        for alias in entity.aliases:
            normalized = _normalized_title_phrase(alias)
            if normalized:
                targets.setdefault(normalized, set()).add(entity.entity_id)
    return targets


def _normalized_title_phrase(value: str) -> str:
    return " ".join(simple_title_tokens(normalize_refined_alias_text(value)))


def _contains_token_phrase(tokens: list[str], phrase: list[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    return any(tokens[index : index + width] == phrase for index in range(len(tokens) - width + 1))
