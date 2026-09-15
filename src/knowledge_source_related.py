from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Collection
from urllib.parse import quote

import pandas as pd
import requests

from src.open_knowledge_providers import (
    fetch_conceptnet_relationships,
    fetch_gdelt_relationships,
)


SOURCE_PROFILE_SCHEMA_VERSION = "2026-09-04-evidence-ranked-profile-v15"
SOURCE_RETRIEVAL_VERSION = "2026-09-04-evidence-ranked-retrieval-v15"
DEFAULT_SOURCE_PROFILE_MAX_AGE_HOURS = 24 * 7
DEFAULT_MAX_RELATIONSHIPS = 40
DEFAULT_MAX_RESULTS = 500
DEFAULT_RELATIONSHIP_DENSITY_THRESHOLD = 4
DEFAULT_EFFECTIVE_COVERAGE_THRESHOLD = 3
DEFAULT_MAX_CORPUS_RELATIONSHIPS = 12
DEFAULT_MAX_WIKIDATA_PATH_RELATIONSHIPS = 18
DEFAULT_MAX_WIKIDATA_GRAPH_HOPS = 3
CORPUS_RECENCY_HALF_LIFE_MONTHS = 3.0
DEFAULT_ENTITY_ACCEPTANCE_SCORE = 0.62
DEFAULT_ENTITY_ACCEPTANCE_MARGIN = 0.08
DEFAULT_MAX_QUERY_VARIANTS = 6
DEFAULT_MAX_WIKIPEDIA_CANDIDATES = 5
DEFAULT_RRF_K = 60
DEFAULT_MIN_TITLE_SUBJECT_CENTRALITY = 0.72
DEFAULT_MIN_SEMANTIC_SUBJECT_SCORE = 0.50
DEFAULT_SEMANTIC_RECOVERY_FLOOR = 0.32
DEFAULT_MAX_SEMANTIC_RECOVERY_RESULTS = 5
DEFAULT_SOURCE_SEMANTIC_MODEL = "BAAI/bge-m3"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "knowledge_sources"
    / "source_profiles.db"
)


class KnowledgeSourceError(RuntimeError):
    """Raised when no configured knowledge source can resolve the query."""


@dataclass(frozen=True)
class WikidataPropertyRule:
    label: str
    family: str
    relationship_class: str
    role: str
    can_retrieve_standalone: bool
    confidence: float
    bridge_template: str


@dataclass(frozen=True)
class QueryVariant:
    text: str
    strategy: str
    penalty: float = 0.0


@dataclass(frozen=True)
class QuerySpan:
    text: str
    start: int
    end: int
    strategy: str


WIKIDATA_PROPERTY_RULES: dict[str, WikidataPropertyRule] = {
    # The grounded research prompt treats editorially material immediate family
    # as a first-class PERSON relationship family.  These properties make that
    # same taxonomy executable through Wikidata instead of merely copying the
    # prompt's output field names into the deterministic source flow.
    "P22": WikidataPropertyRule(
        "father",
        "immediate family",
        "CORE_RELATED",
        "IMMEDIATE_FAMILY",
        True,
        0.93,
        "{related} is the father of {query}.",
    ),
    "P25": WikidataPropertyRule(
        "mother",
        "immediate family",
        "CORE_RELATED",
        "IMMEDIATE_FAMILY",
        True,
        0.93,
        "{related} is the mother of {query}.",
    ),
    "P26": WikidataPropertyRule(
        "spouse",
        "immediate family",
        "CORE_RELATED",
        "IMMEDIATE_FAMILY",
        True,
        0.92,
        "{related} is the spouse of {query}.",
    ),
    "P40": WikidataPropertyRule(
        "child",
        "immediate family",
        "CORE_RELATED",
        "IMMEDIATE_FAMILY",
        True,
        0.92,
        "{related} is a child of {query}.",
    ),
    "P3373": WikidataPropertyRule(
        "sibling",
        "immediate family",
        "CORE_RELATED",
        "IMMEDIATE_FAMILY",
        True,
        0.91,
        "{related} is a sibling of {query}.",
    ),
    "P1038": WikidataPropertyRule(
        "relative",
        "extended family",
        "CONTEXTUAL",
        "FAMILY_NETWORK",
        False,
        0.76,
        "{related} is a recorded relative of {query}.",
    ),
    "P1056": WikidataPropertyRule(
        "product or material produced",
        "product",
        "CORE_RELATED",
        "QUERY_SPECIFIC_MANIFESTATION",
        True,
        0.92,
        "{related} is a product or material produced by {query}.",
    ),
    "P355": WikidataPropertyRule(
        "subsidiary",
        "corporate structure",
        "CORE_RELATED",
        "QUERY_SPECIFIC_MANIFESTATION",
        True,
        0.86,
        "{related} is a subsidiary of {query}.",
    ),
    "P527": WikidataPropertyRule(
        "has part",
        "composition",
        "CORE_RELATED",
        "QUERY_SPECIFIC_MANIFESTATION",
        True,
        0.84,
        "{related} is a specific part of {query}.",
    ),
    "P1344": WikidataPropertyRule(
        "participant in",
        "event participation",
        "CORE_RELATED",
        "NAMED_EVENT",
        True,
        0.88,
        "{query} participated in {related}.",
    ),
    "P607": WikidataPropertyRule(
        "conflict",
        "named conflict",
        "CORE_RELATED",
        "NAMED_EVENT",
        True,
        0.90,
        "{query} is directly associated with the conflict {related}.",
    ),
    "P793": WikidataPropertyRule(
        "significant event",
        "significant event",
        "CORE_RELATED",
        "NAMED_EVENT",
        True,
        0.87,
        "{related} is a significant event involving {query}.",
    ),
    "P1542": WikidataPropertyRule(
        "has effect",
        "specific consequence",
        "CORE_RELATED",
        "SPECIFIC_CONSEQUENCE",
        True,
        0.88,
        "{related} is a stated effect of {query}.",
    ),
    "P828": WikidataPropertyRule(
        "has cause",
        "specific cause",
        "CORE_RELATED",
        "SPECIFIC_CONSEQUENCE",
        True,
        0.86,
        "{related} is a stated cause of {query}.",
    ),
    "P361": WikidataPropertyRule(
        "part of",
        "composition",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.78,
        "{query} is part of {related}.",
    ),
    "P710": WikidataPropertyRule(
        "participant",
        "event participant",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.82,
        "{related} participated in {query}.",
    ),
    "P112": WikidataPropertyRule(
        "founded by",
        "founder",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.82,
        "{query} was founded by {related}.",
    ),
    "P169": WikidataPropertyRule(
        "chief executive officer",
        "executive",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.84,
        "{related} is recorded as a chief executive of {query}.",
    ),
    "P127": WikidataPropertyRule(
        "owned by",
        "ownership",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.82,
        "{query} is owned by {related}.",
    ),
    "P749": WikidataPropertyRule(
        "parent organization",
        "corporate structure",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.84,
        "{related} is the parent organization of {query}.",
    ),
    "P176": WikidataPropertyRule(
        "manufacturer",
        "manufacturer",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.84,
        "{query} is manufactured by {related}.",
    ),
    "P178": WikidataPropertyRule(
        "developer",
        "developer",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.84,
        "{query} was developed by {related}.",
    ),
    "P50": WikidataPropertyRule(
        "author",
        "creator",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.80,
        "{related} is an author of {query}.",
    ),
    "P57": WikidataPropertyRule(
        "director",
        "creator",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.80,
        "{related} directed {query}.",
    ),
    "P161": WikidataPropertyRule(
        "cast member",
        "cast",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.76,
        "{related} is a cast member of {query}.",
    ),
    "P108": WikidataPropertyRule(
        "employer",
        "employment",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.78,
        "{query} is employed by {related}.",
    ),
    "P1416": WikidataPropertyRule(
        "affiliation",
        "formal affiliation",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.76,
        "{query} is formally affiliated with {related}.",
    ),
    "P1830": WikidataPropertyRule(
        "owner of",
        "ownership",
        "CORE_RELATED",
        "INSTITUTIONAL_BRIDGE",
        True,
        0.86,
        "{query} owns {related}.",
    ),
    "P463": WikidataPropertyRule(
        "member of",
        "membership",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.76,
        "{query} is a member of {related}.",
    ),
    "P39": WikidataPropertyRule(
        "position held",
        "public office",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.80,
        "{query} holds or held the position {related}.",
    ),
    "P102": WikidataPropertyRule(
        "political party",
        "political affiliation",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.78,
        "{query} is affiliated with {related}.",
    ),
    "P54": WikidataPropertyRule(
        "member of sports team",
        "team membership",
        "CORE_RELATED",
        "INSTITUTIONAL_BRIDGE",
        True,
        0.86,
        "{query} is or was a member of {related}.",
    ),
    "P664": WikidataPropertyRule(
        "organizer",
        "event organizer",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.80,
        "{related} organized {query}.",
    ),
    "P276": WikidataPropertyRule(
        "location",
        "geographic context",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.72,
        "{query} is associated with the location {related}.",
    ),
    "P131": WikidataPropertyRule(
        "located in administrative entity",
        "geographic context",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.70,
        "{query} is located in {related}.",
    ),
    "P17": WikidataPropertyRule(
        "country",
        "geographic context",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.68,
        "{query} is associated with the country {related}.",
    ),
    "P452": WikidataPropertyRule(
        "industry",
        "industry classification",
        "CONTEXTUAL",
        "GENERIC_TOPIC",
        False,
        0.68,
        "{query} operates in the {related} industry.",
    ),
    "P101": WikidataPropertyRule(
        "field of work",
        "field classification",
        "CONTEXTUAL",
        "GENERIC_TOPIC",
        False,
        0.66,
        "{query} is associated with the field {related}.",
    ),
    "P31": WikidataPropertyRule(
        "instance of",
        "classification and provenance",
        "CONTEXTUAL",
        "GENERIC_TOPIC",
        False,
        0.72,
        "{query} is classified as {related}.",
    ),
    "P279": WikidataPropertyRule(
        "subclass of",
        "classification and provenance",
        "CONTEXTUAL",
        "GENERIC_TOPIC",
        False,
        0.70,
        "{query} is a subclass of {related}.",
    ),
    "P106": WikidataPropertyRule(
        "occupation",
        "office and role",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.74,
        "{query} has the occupation {related}.",
    ),
    "P166": WikidataPropertyRule(
        "award received",
        "recognition",
        "CONTEXTUAL",
        "QUERY_SPECIFIC_MANIFESTATION",
        False,
        0.78,
        "{query} received {related}.",
    ),
    "P6": WikidataPropertyRule(
        "head of government",
        "government leadership",
        "CORE_RELATED",
        "DIRECT_PARTICIPANT",
        True,
        0.88,
        "{related} is recorded as head of government for {query}.",
    ),
    "P35": WikidataPropertyRule(
        "head of state",
        "government leadership",
        "CORE_RELATED",
        "DIRECT_PARTICIPANT",
        True,
        0.88,
        "{related} is recorded as head of state for {query}.",
    ),
    "P488": WikidataPropertyRule(
        "chairman",
        "organizational leadership",
        "CORE_RELATED",
        "DIRECT_PARTICIPANT",
        True,
        0.84,
        "{related} is recorded as chair of {query}.",
    ),
    "P1037": WikidataPropertyRule(
        "director or manager",
        "organizational leadership",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.80,
        "{related} directs or manages {query}.",
    ),
    "P3320": WikidataPropertyRule(
        "board member",
        "organizational governance",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.78,
        "{related} is a board member of {query}.",
    ),
    "P137": WikidataPropertyRule(
        "operator",
        "operational responsibility",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.82,
        "{related} operates {query}.",
    ),
    "P159": WikidataPropertyRule(
        "headquarters location",
        "geographic context",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.70,
        "{query} has its headquarters in {related}.",
    ),
    "P150": WikidataPropertyRule(
        "contains administrative entity",
        "geographic containment",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.70,
        "{query} contains the administrative entity {related}.",
    ),
    "P47": WikidataPropertyRule(
        "shares border with",
        "geographic proximity",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.68,
        "{query} shares a border with {related}.",
    ),
    "P530": WikidataPropertyRule(
        "diplomatic relation",
        "diplomacy",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.72,
        "{query} has a recorded diplomatic relationship with {related}.",
    ),
    "P1365": WikidataPropertyRule(
        "replaces",
        "succession",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.82,
        "{query} replaced {related}.",
    ),
    "P1366": WikidataPropertyRule(
        "replaced by",
        "succession",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.82,
        "{query} was replaced by {related}.",
    ),
    "P155": WikidataPropertyRule(
        "follows",
        "temporal sequence",
        "CONTEXTUAL",
        "NAMED_EVENT",
        False,
        0.78,
        "{query} follows {related} in an established sequence.",
    ),
    "P156": WikidataPropertyRule(
        "followed by",
        "temporal sequence",
        "CONTEXTUAL",
        "NAMED_EVENT",
        False,
        0.78,
        "{query} is followed by {related} in an established sequence.",
    ),
    "P144": WikidataPropertyRule(
        "based on",
        "adaptation and provenance",
        "CORE_RELATED",
        "QUERY_SPECIFIC_MANIFESTATION",
        True,
        0.86,
        "{query} is based on {related}.",
    ),
    "P170": WikidataPropertyRule(
        "creator",
        "creator",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.82,
        "{related} created {query}.",
    ),
    "P175": WikidataPropertyRule(
        "performer",
        "performer",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.78,
        "{related} performed {query}.",
    ),
    "P162": WikidataPropertyRule(
        "producer",
        "production",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.78,
        "{related} produced {query}.",
    ),
    "P272": WikidataPropertyRule(
        "production company",
        "production",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.80,
        "{related} is a production company for {query}.",
    ),
    "P123": WikidataPropertyRule(
        "publisher",
        "distribution",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.80,
        "{related} published {query}.",
    ),
    "P1346": WikidataPropertyRule(
        "winner",
        "competition outcome",
        "CORE_RELATED",
        "DIRECT_PARTICIPANT",
        True,
        0.86,
        "{related} won {query}.",
    ),
    "P286": WikidataPropertyRule(
        "head coach",
        "sports leadership",
        "CORE_RELATED",
        "DIRECT_PARTICIPANT",
        True,
        0.84,
        "{related} is recorded as head coach of {query}.",
    ),
    "P118": WikidataPropertyRule(
        "league or competition",
        "sports competition",
        "CONTEXTUAL",
        "NAMED_EVENT",
        False,
        0.80,
        "{query} participates in the league or competition {related}.",
    ),
    "P115": WikidataPropertyRule(
        "home venue",
        "sports venue",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.76,
        "{related} is the home venue of {query}.",
    ),
    "P1923": WikidataPropertyRule(
        "participating team",
        "event participant",
        "CONTEXTUAL",
        "DIRECT_PARTICIPANT",
        False,
        0.82,
        "{related} participated as a team in {query}.",
    ),
    "P1001": WikidataPropertyRule(
        "applies to jurisdiction",
        "legal and regulatory jurisdiction",
        "CONTEXTUAL",
        "GEOGRAPHIC_CONTEXT",
        False,
        0.76,
        "{query} applies within the jurisdiction {related}.",
    ),
    "P467": WikidataPropertyRule(
        "legislated by",
        "legislation and governance",
        "CORE_RELATED",
        "INSTITUTIONAL_BRIDGE",
        True,
        0.86,
        "{query} was legislated by {related}.",
    ),
    "P194": WikidataPropertyRule(
        "legislative body",
        "legislation and governance",
        "CONTEXTUAL",
        "INSTITUTIONAL_BRIDGE",
        False,
        0.80,
        "{related} is the legislative body governing {query}.",
    ),
    "P921": WikidataPropertyRule(
        "main subject",
        "subject and provenance",
        "CONTEXTUAL",
        "GENERIC_TOPIC",
        False,
        0.68,
        "{related} is a main subject of {query}.",
    ),
}


# The allowlist describes what an edge means. This second policy describes how
# specific that edge is for title retrieval. Keeping the policy separate avoids
# treating a true but generic fact (for example, Reliance produces petroleum) as
# proof that every title about the endpoint is related to the query.
WIKIDATA_PROPERTY_SPECIFICITY: dict[str, str] = {
    "P22": "standalone",
    "P25": "standalone",
    "P26": "standalone",
    "P40": "standalone",
    "P3373": "standalone",
    "P54": "standalone",
    "P1038": "context_required",
    "P1056": "context_required",
    "P452": "context_required",
    "P101": "context_required",
    "P112": "standalone",
    "P169": "standalone",
    "P108": "standalone",
    "P3320": "standalone",
    "P1037": "standalone",
    "P1830": "standalone",
    "P127": "standalone",
    "P749": "standalone",
    "P355": "standalone",
    "P6": "standalone",
    "P35": "standalone",
    "P488": "standalone",
    "P144": "standalone",
    "P1346": "standalone",
    "P286": "standalone",
    "P467": "standalone",
}


# These bounded property groups operationalize the type-appropriate relationship
# families in the grounded research prompt.  They are intentionally narrower
# than the full Wikidata graph: a true path is not automatically an editorially
# useful route.
_IMMEDIATE_FAMILY_PROPERTIES = frozenset({"P22", "P25", "P26", "P40", "P3373"})
_PERSON_ORGANIZATION_PROPERTIES = frozenset(
    {"P108", "P1416", "P463", "P3320", "P1037", "P1830"}
)
_CORPORATE_STRUCTURE_PROPERTIES = frozenset(
    {"P355", "P527", "P127", "P749", "P112", "P169", "P488", "P1037", "P3320", "P137"}
)
_INCOMING_WIKIDATA_PROPERTIES = frozenset(
    {
        "P40",   # a returned subject is a child of the query person
        "P112",  # the query founded the returned organization
        "P169",  # the query is CEO of the returned organization
        "P488",  # the query chairs the returned organization
        "P1037", # the query directs or manages the returned organization
        "P3320", # the query is a board member of the returned organization
        "P137",  # the query operates the returned subject
        "P57",   # the query directed the returned work
        "P161",  # the query is a cast member of the returned work
        "P170",  # the query created the returned work
        "P175",  # the query performed the returned work
        "P162",  # the query produced the returned work
        "P664",  # the query organized the returned event
        "P710",  # the query participated in the returned event
    }
)

_INVERSE_BRIDGE_TEMPLATES: dict[str, str] = {
    "P40": "{query} is a child of {related}.",
    "P112": "{query} founded {related}.",
    "P169": "{query} is recorded as chief executive of {related}.",
    "P488": "{query} is recorded as chair of {related}.",
    "P1037": "{query} directs or manages {related}.",
    "P3320": "{query} is a board member of {related}.",
    "P137": "{query} operates {related}.",
    "P57": "{query} directed {related}.",
    "P161": "{query} is a cast member of {related}.",
    "P170": "{query} created {related}.",
    "P175": "{query} performed {related}.",
    "P162": "{query} produced {related}.",
    "P664": "{query} organized {related}.",
    "P710": "{query} participated in {related}.",
}

_RESEARCH_TAXONOMY_BRANCHES: dict[str, dict[str, frozenset[str]]] = {
    "person": {
        "immediate family": _IMMEDIATE_FAMILY_PROPERTIES,
        "roles and organizations": _PERSON_ORGANIZATION_PROPERTIES,
        "ventures and governance": frozenset({"P112", "P1830", "P3320", "P1037"}),
        "awards": frozenset({"P166"}),
        "public events": frozenset({"P1344", "P793", "P710"}),
    },
    "organization": {
        "leaders and founders": frozenset({"P112", "P169", "P488", "P1037"}),
        "ownership and corporate structure": frozenset({"P127", "P749", "P355", "P527"}),
        "products and operations": frozenset({"P1056", "P137"}),
        "industry and jurisdiction": frozenset({"P452", "P17", "P159"}),
    },
    "event": {
        "participants and organizers": frozenset({"P710", "P664", "P1923"}),
        "place and jurisdiction": frozenset({"P276", "P131", "P17"}),
        "causes and consequences": frozenset({"P828", "P1542"}),
        "sequence and follow-on events": frozenset({"P155", "P156", "P793"}),
    },
    "place": {
        "government and institutions": frozenset({"P6", "P35", "P194"}),
        "administrative geography": frozenset({"P131", "P150", "P47"}),
        "diplomacy and jurisdiction": frozenset({"P530", "P1001"}),
    },
    "creative_work": {
        "creators and performers": frozenset({"P50", "P57", "P161", "P170", "P175", "P162"}),
        "production and distribution": frozenset({"P272", "P123"}),
        "source works and awards": frozenset({"P144", "P166"}),
    },
}


_EVENT_QUERY_TERMS = frozenset(
    {
        "attack",
        "campaign",
        "conflict",
        "controversy",
        "crisis",
        "disaster",
        "election",
        "festival",
        "movement",
        "protest",
        "row",
        "strike",
        "war",
    }
)
_EVENT_DESCRIPTION_MARKERS = (
    "protest",
    "demonstration",
    "uprising",
    "event",
    "election",
    "conflict",
    "war",
    "campaign",
    "festival",
    "disaster",
    "strike",
    "movement",
)
_DISAMBIGUATION_MARKERS = (
    "disambiguation page",
    "wikimedia disambiguation",
)


_CORPUS_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from
    had has have having he her hers him his how i if in into is it its may might
    more most new no nor not of on or our ours out over says she should so than
    that the their theirs them then there these they this those through to under
    up us was we were what when where which while who why will with would you your
    after against amid among before between during following latest live news
    today tomorrow yesterday update updates watch know details report reports
    vs
    """.split()
)
_CORPUS_GENERIC_TERMS = frozenset(
    {
        "2025",
        "2026",
        "date",
        "day",
        "debut",
        "details",
        "fast",
        "hindi",
        "katha",
        "latest",
        "live",
        "match",
        "muhurat",
        "news",
        "paran",
        "parana",
        "puja",
        "report",
        "shubh",
        "significance",
        "time",
        "today",
        "tension",
        "tensions",
        "rise",
        "rises",
        "update",
        "updates",
        "video",
        "vidhi",
        "vrat",
    }
)
_CORPUS_ACRONYMS = frozenset({"bcci", "icc", "ipl", "nba", "nfl", "rr"})


_NAMED_ENTITY_DESCRIPTION_MARKERS = (
    "company",
    "corporation",
    "organisation",
    "organization",
    "business",
    "politician",
    "actor",
    "actress",
    "singer",
    "person",
    "city",
    "country",
    "state",
    "district",
    "film",
    "television",
    "album",
    "song",
    "product",
    "software",
    "newspaper",
    "university",
    "political party",
    "sports team",
)


def get_knowledge_source_user_agent() -> str:
    return os.getenv(
        "KNOWLEDGE_SOURCE_USER_AGENT",
        "TrafficPrediction/1.0 (source-based related-story research)",
    ).strip() or "TrafficPrediction/1.0 (source-based related-story research)"


def _source_mode_cache_value(value: bool | None) -> str:
    if value is None:
        return "auto"
    return "enabled" if value else "disabled"


def build_source_related_result(
    *,
    keyword_query: str,
    title_summary: pd.DataFrame,
    excluded_story_ids: Collection[str] = (),
    selected_wikidata_qid: str = "",
    include_wikipedia: bool = False,
    include_wordnet: bool | None = None,
    include_conceptnet: bool | None = None,
    include_gdelt: bool | None = None,
    force_refresh: bool = False,
    cache_path: Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    use_semantic_centrality: bool = True,
    semantic_model_name: str = DEFAULT_SOURCE_SEMANTIC_MODEL,
    session: requests.Session | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Resolve a query, extract source relationships, and match local titles."""
    started_at = time.perf_counter()
    timings: list[dict[str, object]] = []

    def run_stage(name: str, operation: Callable[[], Any]) -> Any:
        if progress_callback is not None:
            progress_callback(name)
        stage_started_at = time.perf_counter()
        value = operation()
        timings.append(
            {
                "step": name,
                "duration_seconds": max(0.0, time.perf_counter() - stage_started_at),
            }
        )
        return value

    profile = run_stage(
        "Resolve query and extract source relationships",
        lambda: discover_source_profile(
            keyword_query=keyword_query,
            title_summary=title_summary,
            selected_wikidata_qid=selected_wikidata_qid,
            include_wikipedia=include_wikipedia,
            include_wordnet=include_wordnet,
            include_conceptnet=include_conceptnet,
            include_gdelt=include_gdelt,
            force_refresh=force_refresh,
            cache_path=cache_path,
            session=session,
        ),
    )
    stories, diagnostics = run_stage(
        "Match relationships against the title corpus",
        lambda: retrieve_source_related_stories(
            profile=profile,
            title_summary=title_summary,
            excluded_story_ids=excluded_story_ids,
            max_results=max_results,
            use_semantic_centrality=use_semantic_centrality,
            semantic_model_name=semantic_model_name,
        ),
    )
    return {
        "profile": profile,
        "stories": stories,
        "diagnostics": diagnostics,
        "process_timings": timings,
        "total_elapsed_seconds": max(0.0, time.perf_counter() - started_at),
    }


def discover_source_profile(
    *,
    keyword_query: str,
    title_summary: pd.DataFrame | None = None,
    selected_wikidata_qid: str = "",
    include_wikipedia: bool = False,
    include_wordnet: bool | None = None,
    include_conceptnet: bool | None = None,
    include_gdelt: bool | None = None,
    force_refresh: bool = False,
    cache_path: Path | None = None,
    session: requests.Session | None = None,
) -> dict[str, object]:
    query = _clean_text(keyword_query)
    if not query:
        raise KnowledgeSourceError("A non-empty query is required.")
    qid = _normalize_qid(selected_wikidata_qid)
    selected_cache_path = cache_path or DEFAULT_CACHE_PATH
    corpus_fingerprint = _title_corpus_fingerprint(title_summary)
    cache_key = hashlib.sha256(
        (
            f"{SOURCE_PROFILE_SCHEMA_VERSION}|{query.casefold()}|{qid}|"
            f"wikipedia={int(include_wikipedia)}|"
            f"wordnet={_source_mode_cache_value(include_wordnet)}|"
            f"conceptnet={_source_mode_cache_value(include_conceptnet)}|"
            f"gdelt={_source_mode_cache_value(include_gdelt)}|"
            f"corpus={corpus_fingerprint or 'none'}"
        ).encode("utf-8")
    ).hexdigest()
    if not force_refresh:
        cached = _load_cached_profile(selected_cache_path, cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    if session is None:
        http = requests.Session()
        # The host process can inherit placeholder proxy variables (for example,
        # http://127.0.0.1:9) even though public knowledge APIs are reachable
        # directly.  Do not route this module's internally managed requests
        # through those ambient proxies.  Caller-provided sessions are left
        # untouched so deployments that require an explicit proxy can supply one.
        http.trust_env = False
    else:
        http = session
    warnings: list[str] = []
    source_status: dict[str, dict[str, str]] = {}
    interpretations: list[dict[str, object]] = []
    resolved_entity: dict[str, object] = {}
    resolved_entities: list[dict[str, object]] = []
    relationships: list[dict[str, object]] = []
    resolution: dict[str, object] = {}
    wikidata_graph_expansion: dict[str, object] = {}
    query_structure = _analyze_query_structure(query, title_summary=title_summary)

    try:
        resolution = _resolve_wikimedia_query(
            http,
            query=query,
            title_summary=title_summary,
            selected_wikidata_qid=qid,
            include_wikipedia=include_wikipedia,
        )
        raw_interpretations = resolution.get("candidates", [])
        if isinstance(raw_interpretations, list):
            interpretations = [
                item for item in raw_interpretations if isinstance(item, dict)
            ]
        selected_mentions = [
            item
            for item in resolution.get("selected_mentions", [])
            if isinstance(item, dict) and _normalize_qid(item.get("qid"))
        ]
        selected_qids = list(
            dict.fromkeys(
                _normalize_qid(item.get("qid")) for item in selected_mentions
            )
        )
        if not selected_qids:
            selected_qid = _normalize_qid(resolution.get("selected_qid"))
            if selected_qid:
                selected_qids = [selected_qid]
                selected_mentions = [
                    {
                        "text": query,
                        "qid": selected_qid,
                        "score": resolution.get("selected_score", 0.0),
                        "strategy": "complete_query",
                    }
                ]
        if selected_qids:
            entities = _wikidata_entities(
                http,
                selected_qids,
                include_claims=True,
                include_sitelinks=True,
            )
            mention_by_qid = {
                _normalize_qid(item.get("qid")): item for item in selected_mentions
            }
            for resolved_qid in selected_qids:
                entity = entities.get(resolved_qid, {})
                if not entity:
                    continue
                entity_profile, entity_relationships = _profile_from_wikidata_entity(
                    http,
                    resolved_qid,
                    entity,
                )
                mention = mention_by_qid.get(resolved_qid, {})
                entity_profile["matched_mention"] = _clean_text(
                    mention.get("text")
                ) or _clean_text(entity_profile.get("label"))
                entity_profile["resolution_score"] = _safe_float(
                    mention.get("score")
                )
                resolved_entities.append(entity_profile)
                relationships.extend(entity_relationships)
            if resolved_entities:
                relationships.extend(
                    _cross_entity_bridge_relationships(
                        resolved_entities=resolved_entities,
                        relationships=relationships,
                    )
                )
                direct_relationship_count = len(relationships)
                incoming_relationships, incoming_diagnostics = (
                    _wikidata_incoming_relationships(http, resolved_entities)
                )
                relationships.extend(incoming_relationships)
                path_relationships, path_diagnostics = _wikidata_path_relationships(
                    http,
                    resolved_entities=resolved_entities,
                    root_relationships=relationships,
                )
                relationships.extend(path_relationships)
                wikidata_graph_expansion = {
                    "direct_relationship_count": direct_relationship_count,
                    "incoming": incoming_diagnostics,
                    "paths": path_diagnostics,
                    "added_relationship_count": (
                        len(incoming_relationships) + len(path_relationships)
                    ),
                }
                resolved_entity = resolved_entities[0]
                if qid and not interpretations:
                    interpretations = [
                        {
                            "qid": resolved_entity.get("qid", ""),
                            "label": _clean_text(resolved_entity.get("label")),
                            "description": _clean_text(
                                resolved_entity.get("description")
                            ),
                            "query_mention": query,
                            "selected": True,
                        }
                    ]
                source_status["Wikidata"] = {
                    "status": "used",
                    "detail": (
                        f"Resolved {len(resolved_entities)} query mention(s) and "
                        f"extracted {direct_relationship_count} direct, "
                        f"{len(incoming_relationships)} inverse, and "
                        f"{len(path_relationships)} bounded path relationships."
                    ),
                }
            else:
                source_status["Wikidata"] = {
                    "status": "no_evidence",
                    "detail": "Selected Wikidata identifiers returned no entity data.",
                }
        else:
            source_status["Wikidata"] = {
                "status": "no_evidence",
                "detail": _clean_text(resolution.get("detail"))
                or "No candidate passed the entity-resolution confidence gate.",
            }
    except Exception as exc:
        source_status["Wikidata"] = {
            "status": "unavailable",
            "detail": str(exc),
        }
        warnings.append(f"Wikidata was unavailable: {exc}")

    wikipedia_entities = [
        entity
        for entity in resolved_entities
        if _clean_text(entity.get("wikipedia_title"))
    ]
    if include_wikipedia and wikipedia_entities:
        try:
            redirect_count = 0
            for entity in wikipedia_entities:
                wikipedia_title = _clean_text(entity.get("wikipedia_title"))
                redirects = _wikipedia_redirects(http, wikipedia_title)
                redirect_count += len(redirects)
                entity["aliases"] = _deduplicate_text(
                    [*list(entity.get("aliases", [])), *redirects]
                )[:60]
                entity["wikipedia_url"] = (
                    "https://en.wikipedia.org/wiki/"
                    + quote(wikipedia_title.replace(" ", "_"), safe="()_-")
                )
            resolved_entity = resolved_entities[0]
            source_status["Wikipedia"] = {
                "status": "used",
                "detail": (
                    f"Loaded {redirect_count} redirects for "
                    f"{len(wikipedia_entities)} resolved page(s). "
                    f"Full-text fallback considered "
                    f"{int(resolution.get('wikipedia_candidate_count', 0))} candidates."
                ),
            }
        except Exception as exc:
            source_status["Wikipedia"] = {
                "status": "unavailable",
                "detail": str(exc),
            }
            warnings.append(f"Wikipedia redirects were unavailable: {exc}")
    elif include_wikipedia:
        source_status["Wikipedia"] = {
            "status": (
                "no_match" if resolution.get("wikipedia_search_used") else "skipped"
            ),
            "detail": (
                f"Full-text fallback considered "
                f"{int(resolution.get('wikipedia_candidate_count', 0))} candidates, "
                "but none passed the resolution confidence gate."
                if resolution.get("wikipedia_search_used")
                else "The resolved entity has no English Wikipedia page."
            ),
        }
    else:
        source_status["Wikipedia"] = {
            "status": "not_applicable",
            "detail": (
                "English Wikipedia was not selected because Latin-script input alone "
                "does not establish an English-language route."
            ),
        }

    descriptions = " ".join(
        _clean_text(entity.get("description")).casefold()
        for entity in resolved_entities
    )
    resolved_named_entity = bool(resolved_entities) and any(
        marker in descriptions for marker in _NAMED_ENTITY_DESCRIPTION_MARKERS
    )
    should_use_wordnet = include_wordnet is not False and not resolved_named_entity
    wordnet_relationships: list[dict[str, object]] = []
    if should_use_wordnet:
        wordnet_relationships, wordnet_detail = _wordnet_relationships(query)
        relationships.extend(wordnet_relationships)
        source_status["WordNet"] = {
            "status": "used" if wordnet_relationships else "no_evidence",
            "detail": wordnet_detail,
        }
        if wordnet_detail.startswith("Unavailable:"):
            source_status["WordNet"]["status"] = "unavailable"
            warnings.append(wordnet_detail)
    elif include_wordnet is True:
        source_status["WordNet"] = {
            "status": "not_applicable",
            "detail": "Skipped because Wikidata resolved the query as a named entity.",
        }
    elif include_wordnet is None:
        source_status["WordNet"] = {
            "status": "not_applicable",
            "detail": "Automatic routing did not classify this as an English lexical query.",
        }
    else:
        source_status["WordNet"] = {
            "status": "explicitly_disabled",
            "detail": "Explicitly disabled for this run.",
        }

    use_conceptnet = include_conceptnet is True or (
        include_conceptnet is None and bool(wordnet_relationships)
    )
    if use_conceptnet:
        try:
            conceptnet_relationships, conceptnet_diagnostics = (
                fetch_conceptnet_relationships(
                    query=query,
                    fetch_json=lambda url, params: _http_get_json(
                        http,
                        url,
                        params=params,
                        max_attempts=2 if include_conceptnet is True else 1,
                    ),
                    fetch_text=lambda url, params: _http_get_text(
                        http,
                        url,
                        params=params,
                        max_attempts=2 if include_conceptnet is True else 1,
                    ),
                )
            )
            relationships.extend(conceptnet_relationships)
            conceptnet_route = str(
                conceptnet_diagnostics.get("retrieval_mode", "api")
            )
            route_detail = (
                " The JSON API was unavailable, so the official ConceptNet web "
                "representation was used."
                if conceptnet_route == "official_web_fallback"
                else ""
            )
            source_status["ConceptNet"] = {
                "status": "used" if conceptnet_relationships else "no_evidence",
                "detail": (
                    f"Considered {int(conceptnet_diagnostics.get('edge_count', 0))} "
                    "edges from the explicit /c/en namespace and accepted "
                    f"{len(conceptnet_relationships)} bounded lexical/context edges. "
                    "Latin script was not treated as proof of English language."
                    f"{route_detail}"
                ),
            }
        except Exception as exc:
            source_status["ConceptNet"] = {
                "status": "unavailable",
                "detail": (
                    "The optional public ConceptNet service did not respond. "
                    "WordNet and local-corpus evidence remained available."
                ),
            }
            if include_conceptnet is True:
                warnings.append(f"ConceptNet was unavailable: {exc}")
    elif include_conceptnet is None:
        source_status["ConceptNet"] = {
            "status": "not_applicable",
            "detail": (
                "Automatic routing found no English lexical evidence for the current "
                "ConceptNet /c/en adapter."
            ),
        }
    else:
        source_status["ConceptNet"] = {
            "status": "explicitly_disabled",
            "detail": "Explicitly disabled for this run.",
        }

    use_gdelt = include_gdelt is not False
    if use_gdelt:
        try:
            gdelt_query_cues = _deduplicate_text(
                [
                    entity.get("matched_mention", "")
                    for entity in resolved_entities
                ]
            )
            gdelt_relationships, gdelt_diagnostics = fetch_gdelt_relationships(
                query=query,
                query_cues=gdelt_query_cues or [query],
                fetch_json=lambda url, params: _http_get_json(
                    http,
                    url,
                    params=params,
                    max_attempts=2 if include_gdelt is True else 1,
                ),
            )
            relationships.extend(gdelt_relationships)
            source_status["GDELT"] = {
                "status": "used" if gdelt_relationships else "no_evidence",
                "detail": (
                    f"Reviewed {int(gdelt_diagnostics.get('article_count', 0))} "
                    "recent matching articles and accepted "
                    f"{len(gdelt_relationships)} phrases repeated across at least "
                    "two publisher domains."
                ),
            }
        except Exception as exc:
            source_status["GDELT"] = {
                "status": "unavailable",
                "detail": str(exc),
            }
            warnings.append(f"GDELT was unavailable: {exc}")
    else:
        source_status["GDELT"] = {
            "status": "explicitly_disabled",
            "detail": "Explicitly disabled for this run.",
        }

    source_status["DBpedia"] = {
        "status": "not_integrated",
        "detail": "Provider adapter is not integrated in this increment.",
    }
    source_status["YAGO"] = {
        "status": "not_integrated",
        "detail": "Provider adapter is not integrated in this increment.",
    }
    source_status["IndoWordNet"] = {
        "status": "not_integrated",
        "detail": (
            "Provider adapter and evidence-based language/lemma routing are not yet "
            "integrated."
        ),
    }

    relationships = _deduplicate_relationships(relationships)
    source_relationship_count = sum(
        _relationship_counts_toward_density(item) for item in relationships
    )
    coverage = _estimate_relationship_coverage(
        query=query,
        resolved_entity=resolved_entity,
        resolved_entities=resolved_entities,
        relationships=relationships,
        title_summary=title_summary,
    )
    coverage_sparse = (
        int(coverage.get("matched_non_primary_title_count", 0))
        < DEFAULT_EFFECTIVE_COVERAGE_THRESHOLD
    )
    relationship_sparse = (
        source_relationship_count < DEFAULT_RELATIONSHIP_DENSITY_THRESHOLD
    )
    density_fallback: dict[str, object] = {
        "threshold": DEFAULT_RELATIONSHIP_DENSITY_THRESHOLD,
        "effective_coverage_threshold": DEFAULT_EFFECTIVE_COVERAGE_THRESHOLD,
        "source_relationship_count": source_relationship_count,
        "sparse": relationship_sparse,
        "coverage_sparse": coverage_sparse,
        "activation_reason": "",
        "source_coverage": coverage,
        "activated": False,
        "added_relationship_count": 0,
        "corpus_fingerprint": corpus_fingerprint,
    }
    should_use_corpus = relationship_sparse or coverage_sparse
    if should_use_corpus:
        activation_reasons = []
        if relationship_sparse:
            activation_reasons.append("relationship density")
        if coverage_sparse:
            activation_reasons.append("effective corpus coverage")
        density_fallback["activation_reason"] = " and ".join(activation_reasons)
        if isinstance(title_summary, pd.DataFrame) and not title_summary.empty:
            corpus_relationships, corpus_diagnostics = (
                _corpus_cooccurrence_relationships(
                    query=query,
                    resolved_entity=_aggregate_resolved_entity(resolved_entities),
                    resolved_entities=resolved_entities,
                    title_summary=title_summary,
                )
            )
            relationships = _merge_relationship_sets(
                relationships,
                corpus_relationships,
            )
            density_fallback.update(corpus_diagnostics)
            density_fallback["activated"] = True
            density_fallback["corpus_relationship_count"] = len(
                corpus_relationships
            )
            density_fallback["added_relationship_count"] = max(
                0,
                len(relationships) - source_relationship_count,
            )
            source_status["Local corpus"] = {
                "status": "complete" if corpus_relationships else "no_match",
                "detail": (
                    f"Profile needed coverage enrichment ({density_fallback['activation_reason']}; "
                    f"{source_relationship_count} source relationships and "
                    f"{int(coverage.get('matched_non_primary_title_count', 0))} "
                    "effective non-primary title matches); "
                    f"mined {len(corpus_relationships)} recency-weighted "
                    "co-occurrence relationships from the local title corpus."
                ),
            }
        else:
            source_status["Local corpus"] = {
                "status": "skipped",
                "detail": (
                    f"Profile needed coverage enrichment ({density_fallback['activation_reason']}), "
                    "but no local title corpus was supplied for the density fallback."
                ),
            }
    else:
        source_status["Local corpus"] = {
            "status": "skipped",
            "detail": (
                f"Dense profile ({source_relationship_count} source relationships); "
                "the corpus fallback was not needed."
            ),
        }
    relationships, relationship_selection = _select_relationships_for_retrieval(
        query=query,
        resolved_entities=resolved_entities,
        relationships=relationships,
        title_summary=title_summary,
        limit=DEFAULT_MAX_RELATIONSHIPS,
    )
    relationships = [
        _complete_relationship_contract(item, query=query)
        for item in relationships
    ]
    relationship_coverage_audit = _research_taxonomy_coverage_audit(
        resolved_entities=resolved_entities,
        relationships=relationships,
        graph_expansion=wikidata_graph_expansion,
    )
    seed_document_count = int(density_fallback.get("query_seed_document_count", 0))
    if resolved_entities:
        resolution_status = "resolved"
    elif relationships:
        resolution_status = "corpus_or_lexical_only"
    elif seed_document_count:
        resolution_status = "primary_corpus_only"
    else:
        resolution_status = "unresolved"

    profile: dict[str, object] = {
        "schema_version": SOURCE_PROFILE_SCHEMA_VERSION,
        "query": query,
        "query_input": {
            "script_constraint": "Latin",
            "language_assumption": None,
            "transliteration_applied": False,
            "phonetic_expansion_applied": False,
        },
        "resolved_entity": resolved_entity,
        "resolved_entities": resolved_entities,
        "query_structure": query_structure,
        "query_graph": _build_query_graph(
            query_structure=query_structure,
            resolved_entities=resolved_entities,
            relationships=relationships,
        ),
        "interpretations": interpretations,
        "entity_resolution": resolution,
        "resolution_status": resolution_status,
        "relationships": relationships,
        "relationship_selection": relationship_selection,
        "relationship_coverage_audit": relationship_coverage_audit,
        "source_status": source_status,
        "warnings": warnings,
        "density_fallback": density_fallback,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
    }
    # Optional public-source outages must not force every normal run to repeat
    # the full resolution path. A refresh explicitly retries unavailable sources.
    cacheable = bool(relationships or resolved_entities or seed_document_count)
    if cacheable and (relationships or resolved_entity or seed_document_count):
        try:
            _save_cached_profile(selected_cache_path, cache_key, query, profile)
        except sqlite3.Error:
            profile["warnings"] = [
                *warnings,
                "The source profile could not be saved to the local SQLite cache.",
            ]
    return profile


def _resolved_entities_from_profile(
    profile: dict[str, object],
) -> list[dict[str, object]]:
    values = profile.get("resolved_entities", [])
    if isinstance(values, list):
        entities = [item for item in values if isinstance(item, dict)]
        if entities:
            return entities
    legacy = profile.get("resolved_entity", {})
    return [legacy] if isinstance(legacy, dict) and legacy else []


def _entity_description_is_named(entity: dict[str, object]) -> bool:
    """Distinguish named entities from Wikidata concepts without language guessing."""

    description = _clean_text(entity.get("description")).casefold()
    label = _clean_text(entity.get("label"))
    stable_markers = tuple(
        marker for marker in _NAMED_ENTITY_DESCRIPTION_MARKERS if marker != "state"
    )
    if any(marker in description for marker in stable_markers):
        return True
    # Geographic state descriptions are commonly "state of <country>".  The
    # label casing separates them from abstract concepts such as weather being
    # described as a "state of the atmosphere".
    return (
        "state" in description
        and bool(label)
        and label[0].isupper()
    )


def _query_identity_groups(
    query: object,
    resolved_entities: list[dict[str, object]],
) -> list[list[str]]:
    groups: list[list[str]] = []
    for entity in resolved_entities:
        forms = _safe_match_phrases(
            [
                entity.get("matched_mention", ""),
                entity.get("label", ""),
                *list(entity.get("aliases", [])),
            ],
            allow_single_word=True,
        )
        if forms:
            groups.append(forms)
    if not groups:
        fallback = _safe_match_phrases([query], allow_single_word=True)
        if fallback:
            groups.append(fallback)
    return groups


def _is_direct_query_title(
    title: object,
    *,
    query: object,
    identity_groups: list[list[str]],
) -> bool:
    if _query_title_match_strength(title, [query]) >= 0.70:
        return True
    if not identity_groups:
        return False
    matched_groups = sum(
        _query_title_match_strength(title, group) >= 0.70
        for group in identity_groups
    )
    required_groups = 1 if len(identity_groups) == 1 else min(2, len(identity_groups))
    return matched_groups >= required_groups


_METADATA_ONLY_PREDICATES = frozenset(
    {
        "p31",   # instance of
        "p279",  # subclass of
        "p106",  # occupation
        "p101",  # field of work
        "p452",  # industry
        "p921",  # main subject
    }
)
_METADATA_ONLY_ROLES = frozenset({"GENERIC_TOPIC"})


def _corpus_relationship_can_generate(
    relationship: dict[str, object],
) -> bool:
    """Return whether a corpus association is strong enough to propose titles.

    Corpus co-occurrence is not a factual knowledge-graph edge, so it receives a
    lower result tier.  It may nevertheless generate discovery candidates when
    the association is specific, repeated, and concentrated in the query seed
    set.  Requiring the complete evidence contract prevents a caller from
    promoting a relationship by setting a single boolean flag.
    """

    predicate = _clean_text(relationship.get("predicate_id")).casefold()
    if not predicate.startswith("corpus:"):
        return False
    if relationship.get("corpus_high_specificity") is not True:
        return False
    support = int(_safe_float(relationship.get("corpus_support")))
    weighted_support = _safe_float(
        relationship.get("recency_weighted_support")
    )
    pmi = _safe_float(relationship.get("corpus_pmi"))
    precision = _safe_float(
        relationship.get("corpus_association_precision")
    )
    return (
        support >= 2
        and weighted_support >= 1.0
        and pmi >= 2.5
        and precision >= 0.20
    )


def _corpus_relationship_can_compose(
    relationship: dict[str, object],
) -> bool:
    """Allow a corpus cue to complete, but not replace, a multi-part query."""

    predicate = _clean_text(relationship.get("predicate_id")).casefold()
    if not predicate.startswith("corpus:"):
        return False
    support = int(_safe_float(relationship.get("corpus_support")))
    weighted_support = _safe_float(
        relationship.get("recency_weighted_support")
    )
    pmi = _safe_float(relationship.get("corpus_pmi"))
    precision = _safe_float(
        relationship.get("corpus_association_precision")
    )
    subject = _normalize_for_match(relationship.get("related_subject", ""))
    tokens = subject.split()
    if len(tokens) == 1:
        phrase_is_specific = (
            len(subject) >= 7 or subject in _CORPUS_ACRONYMS
        ) and precision >= 0.005
    else:
        phrase_is_specific = (
            relationship.get("corpus_high_specificity") is True
            or precision >= 0.05
        )
    return (
        phrase_is_specific
        and support >= 2
        and weighted_support >= 1.0
        and pmi >= 3.0
    )


def _relationship_retrieval_policy(
    relationship: dict[str, object],
) -> str:
    """Return the role a relationship may play during title retrieval.

    Truth in a knowledge graph is not the same as retrieval usefulness. Broad
    classifications are profile metadata, while volatile title co-occurrences
    can only corroborate a candidate grounded by a durable relationship. This
    policy is predicate/source based and therefore applies to future entities
    without maintaining keyword-specific exceptions.
    """

    predicate = _clean_text(relationship.get("predicate_id")).casefold()
    role = _clean_text(relationship.get("relationship_role")).upper()
    evidence_kind = _clean_text(relationship.get("evidence_kind")).casefold()
    source_family = _clean_text(relationship.get("source_family")).casefold()
    sources = {
        _clean_text(source).casefold()
        for source in relationship.get("source_names", [])
        if _clean_text(source)
    }

    if predicate in _METADATA_ONLY_PREDICATES or role in _METADATA_ONLY_ROLES:
        return "metadata_only"
    if _corpus_relationship_can_generate(relationship):
        return "corpus_supported"
    if (
        predicate.startswith(("corpus:", "gdelt:"))
        or evidence_kind == "current_events"
        or source_family == "current_events"
        or (
            bool(sources)
            and sources.issubset({"gdelt", "local corpus"})
        )
    ):
        return "corroboration_only"
    if relationship.get("can_retrieve_standalone") is True:
        return "standalone"
    return "context_required"


def _relationship_requires_query_anchor(
    relationship: dict[str, object],
) -> bool:
    predicate = _clean_text(relationship.get("predicate_id")).casefold()
    sources = {
        _clean_text(source).casefold()
        for source in relationship.get("source_names", [])
    }
    if predicate.startswith(("corpus:", "gdelt:")):
        return relationship.get("corpus_high_specificity") is not True
    return bool(sources.intersection({"gdelt", "conceptnet"}))


def _title_subject_centrality(
    *,
    normalized_title: str,
    matched_phrase: str,
    canonical_subject: object,
) -> float:
    """Estimate whether an exact subject mention is central in a title.

    This is intentionally source- and domain-neutral. It rewards canonical,
    specific phrases that occupy a meaningful share of the title and appear
    earlier, while long headlines containing a late incidental mention must
    supply a second structured relationship to pass the candidate gate.
    """

    normalized_phrase = _normalize_for_match(matched_phrase)
    title_tokens = normalized_title.split()
    phrase_tokens = normalized_phrase.split()
    if not title_tokens or not phrase_tokens:
        return 0.0
    phrase_width = len(phrase_tokens)
    start_index = -1
    for index in range(0, len(title_tokens) - phrase_width + 1):
        if title_tokens[index : index + phrase_width] == phrase_tokens:
            start_index = index
            break
    if start_index < 0:
        return 0.0

    available_starts = max(1, len(title_tokens) - phrase_width + 1)
    position_score = 0.24 * (
        1.0 - min(1.0, start_index / available_starts)
    )
    coverage_score = min(
        0.32,
        1.20 * phrase_width / max(1, len(title_tokens)),
    )
    canonical_bonus = (
        0.10
        if normalized_phrase == _normalize_for_match(canonical_subject)
        else 0.04
    )
    specificity_bonus = 0.08 if phrase_width >= 2 else 0.03
    return round(
        min(
            1.0,
            0.22
            + position_score
            + coverage_score
            + canonical_bonus
            + specificity_bonus,
        ),
        3,
    )


def _minimum_centrality_for_evidence(evidence: dict[str, object]) -> float:
    """Return an auditable, phrase-aware centrality threshold.

    A fixed threshold systematically penalizes concise one-token entities in
    ordinary news headlines.  Ambiguous aliases and distant graph paths retain
    stricter thresholds, while exact canonical names receive a bounded
    adjustment.  Semantic scoring remains a separate check.
    """

    matched = _normalize_for_match(evidence.get("matched_subject", ""))
    canonical = _normalize_for_match(evidence.get("related_subject", ""))
    token_count = len(matched.split())
    policy = _clean_text(evidence.get("retrieval_policy")).casefold()
    hops = max(1, int(_safe_float(evidence.get("relationship_hops", 1))))
    threshold = DEFAULT_MIN_TITLE_SUBJECT_CENTRALITY
    if token_count == 1 and matched == canonical and len(matched) >= 5:
        threshold = 0.60
    elif token_count >= 2 and matched == canonical:
        threshold = 0.68
    if policy == "corpus_supported":
        threshold = max(threshold, 0.64)
    if matched != canonical:
        threshold += 0.04
    if hops > 1:
        threshold += min(0.10, 0.04 * (hops - 1))
    return round(min(0.82, threshold), 3)


def _select_relationships_for_retrieval(
    *,
    query: str,
    resolved_entities: list[dict[str, object]],
    relationships: list[dict[str, object]],
    title_summary: pd.DataFrame | None,
    limit: int = DEFAULT_MAX_RELATIONSHIPS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select a diverse, corpus-aware relationship set before truncation.

    Confidence alone is a poor top-k selector for retrieval: a high-confidence
    endpoint absent from the corpus cannot produce a result.  This selector
    prioritizes usable one-hop evidence, caps repeated predicates and distant
    paths, and still retains a small amount of metadata for auditability.
    """

    deduplicated = _deduplicate_relationships(relationships)
    normalized_titles: list[str] = []
    if isinstance(title_summary, pd.DataFrame) and not title_summary.empty:
        identity_groups = _query_identity_groups(query, resolved_entities)
        for raw_title in title_summary.get(
            "page_title", pd.Series(dtype=str)
        ).tolist():
            title = _normalize_for_match(raw_title)
            if title and not _is_direct_query_title(
                title,
                query=query,
                identity_groups=identity_groups,
            ):
                normalized_titles.append(title)

    scored: list[tuple[tuple[float, ...], dict[str, object], int]] = []
    for relationship in deduplicated:
        phrases = _relationship_match_phrases(relationship)
        coverage = 0
        if phrases and normalized_titles:
            for title in normalized_titles:
                if any(_contains_phrase(title, phrase) for phrase in phrases):
                    coverage += 1
                    if coverage >= 100:
                        break
        policy = _relationship_retrieval_policy(relationship)
        predicate = _clean_text(relationship.get("predicate_id")).casefold()
        hops = max(1, int(_safe_float(relationship.get("hop_count", 1))))
        is_path = predicate.startswith("wikidata:path:")
        policy_rank = {
            "standalone": 4.0,
            "corpus_supported": 3.5,
            "context_required": 3.0,
            "corroboration_only": 1.0,
            "metadata_only": 0.0,
        }.get(policy, 0.0)
        score = (
            1.0 if coverage else 0.0,
            min(3.0, math.log1p(coverage)),
            policy_rank,
            1.0 if hops == 1 else 0.0,
            0.0 if is_path else 1.0,
            _safe_float(relationship.get("confidence")),
            -float(hops),
        )
        scored.append((score, relationship, coverage))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[dict[str, object]] = []
    selected_ids: set[int] = set()
    predicate_counts: Counter[str] = Counter()
    path_count = 0
    metadata_count = 0
    for _, relationship, coverage in scored:
        policy = _relationship_retrieval_policy(relationship)
        predicate = _clean_text(relationship.get("predicate_id")).casefold()
        is_path = predicate.startswith("wikidata:path:")
        predicate_bucket = "wikidata:path" if is_path else predicate
        if predicate_counts[predicate_bucket] >= 8:
            continue
        if is_path and path_count >= 8:
            continue
        if policy == "metadata_only" and metadata_count >= 6:
            continue
        item = dict(relationship)
        item["local_title_coverage"] = coverage
        selected.append(item)
        selected_ids.add(id(relationship))
        predicate_counts[predicate_bucket] += 1
        path_count += int(is_path)
        metadata_count += int(policy == "metadata_only")
        if len(selected) >= max(1, int(limit)):
            break

    # Sparse graphs should not lose evidence merely because a diversity cap was
    # reached. Fill remaining slots in score order without relaxing the global
    # limit; selected rows remain ordered by retrieval usefulness.
    if len(selected) < max(1, int(limit)):
        for _, relationship, coverage in scored:
            if id(relationship) in selected_ids:
                continue
            item = dict(relationship)
            item["local_title_coverage"] = coverage
            selected.append(item)
            selected_ids.add(id(relationship))
            if len(selected) >= max(1, int(limit)):
                break

    diagnostics = {
        "input_relationship_count": len(deduplicated),
        "selected_relationship_count": len(selected),
        "selected_with_local_coverage": sum(
            int(_safe_float(item.get("local_title_coverage"))) > 0
            for item in selected
        ),
        "selected_path_count": sum(
            _clean_text(item.get("predicate_id"))
            .casefold()
            .startswith("wikidata:path:")
            for item in selected
        ),
        "selection_method": "coverage_policy_diversity",
    }
    return selected, diagnostics


def _estimate_relationship_coverage(
    *,
    query: str,
    resolved_entity: dict[str, object],
    resolved_entities: list[dict[str, object]] | None = None,
    relationships: list[dict[str, object]],
    title_summary: pd.DataFrame | None,
) -> dict[str, object]:
    """Estimate usable retrieval coverage before deciding on corpus enrichment.

    Relationship count is not a coverage metric: a profile can have many broad
    endpoints and still retrieve no titles.  This bounded dry run applies the
    same exact-phrase standalone/context rule used by final retrieval.
    """

    diagnostics: dict[str, object] = {
        "eligible_title_count": 0,
        "matched_non_primary_title_count": 0,
        "exact_match_count": 0,
    }
    if not isinstance(title_summary, pd.DataFrame) or title_summary.empty:
        return diagnostics

    entities = resolved_entities or ([resolved_entity] if resolved_entity else [])
    identity_groups = _query_identity_groups(query, entities)
    query_context_phrases = list(
        dict.fromkeys(phrase for group in identity_groups for phrase in group)
    )
    anchor_phrases: list[str] = []
    prepared: list[tuple[dict[str, object], list[str], str]] = []
    for relationship in relationships:
        retrieval_policy = _relationship_retrieval_policy(relationship)
        phrases = _relationship_match_phrases(relationship)
        if phrases and retrieval_policy in {"standalone", "context_required"}:
            prepared.append((relationship, phrases, retrieval_policy))
        if retrieval_policy == "standalone":
            anchor_phrases.extend(phrases)
    anchor_phrases = list(dict.fromkeys(anchor_phrases))[:120]

    exact_count = 0
    matched_count = 0
    eligible_count = 0
    for row in title_summary.to_dict("records"):
        title = _clean_text(row.get("page_title"))
        if not title or _is_direct_query_title(
            title,
            query=query,
            identity_groups=identity_groups,
        ):
            continue
        eligible_count += 1
        generating_matches: list[dict[str, object]] = []
        normalized_title = _normalize_for_match(title)
        surface_title = _normalize_surface(title)
        for relationship, phrases, retrieval_policy in prepared:
            subject_match = next(
                (
                    phrase
                    for phrase in phrases
                    if _contains_phrase(normalized_title, phrase)
                ),
                "",
            )
            if not subject_match:
                continue
            if relationship.get("requires_exact_surface") is True:
                surface_forms = _safe_surface_phrases(
                    [relationship.get("related_subject", "")]
                )
                if not any(
                    _contains_phrase(surface_title, form) for form in surface_forms
                ):
                    continue
            if retrieval_policy == "context_required":
                contextual_anchors = (
                    query_context_phrases
                    if _relationship_requires_query_anchor(relationship)
                    else anchor_phrases
                )
                required = _safe_match_phrases(
                    [
                        *list(relationship.get("required_title_cues", [])),
                        *contextual_anchors,
                    ],
                    allow_single_word=True,
                )
                context_match = next(
                    (
                        cue
                        for cue in required
                        if _is_distinct_context_cue(cue, phrases)
                        and _contains_phrase(normalized_title, cue)
                    ),
                    "",
                )
                if not context_match:
                    continue
            centrality = _title_subject_centrality(
                normalized_title=normalized_title,
                matched_phrase=subject_match,
                canonical_subject=relationship.get("related_subject", ""),
            )
            interpretation_id = _clean_text(
                relationship.get("interpretation_id")
            )
            generating_matches.append(
                {
                    "retrieval_policy": retrieval_policy,
                    "title_subject_centrality": centrality,
                    "matched_subject": subject_match,
                    "related_subject": relationship.get("related_subject", ""),
                    "relationship_hops": max(
                        1,
                        int(_safe_float(relationship.get("hop_count", 1))),
                    ),
                    "relationship_origin": interpretation_id,
                    "source_names": list(relationship.get("source_names", [])),
                }
            )
        has_central_standalone = any(
            item.get("retrieval_policy") == "standalone"
            and int(item.get("relationship_hops", 1)) == 1
            and _safe_float(item.get("title_subject_centrality"))
            >= _minimum_centrality_for_evidence(item)
            for item in generating_matches
        )
        independent_origins = {
            (
                _clean_text(item.get("relationship_origin")),
                tuple(
                    sorted(
                        _clean_text(source).casefold()
                        for source in item.get("source_names", [])
                        if _clean_text(source)
                    )
                ),
            )
            for item in generating_matches
        }
        if not has_central_standalone and len(independent_origins) < 2:
            continue
        matched_count += 1
        exact_count += 1

    diagnostics.update(
        {
            "eligible_title_count": eligible_count,
            "matched_non_primary_title_count": matched_count,
            "exact_match_count": exact_count,
        }
    )
    return diagnostics


def retrieve_source_related_stories(
    *,
    profile: dict[str, object],
    title_summary: pd.DataFrame,
    excluded_story_ids: Collection[str] = (),
    max_results: int = DEFAULT_MAX_RESULTS,
    use_semantic_centrality: bool = False,
    semantic_model_name: str = DEFAULT_SOURCE_SEMANTIC_MODEL,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_columns = [
        "story_id",
        "page_title",
        "related_subject",
        "relationship",
        "relationship_family",
        "relationship_role",
        "relationship_hops",
        "relationship_path",
        "supporting_related_subjects",
        "supporting_relationship_count",
        "matched_title_evidence",
        "source_names",
        "confidence",
        "result_tier",
        "result_scope",
        "matched_query_components",
        "query_component_coverage",
        "total_views",
        "active_months",
        "first_month",
        "last_month",
        "retrieval_methods",
        "title_subject_centrality",
        "semantic_subject_score",
        "evidence_score",
        "semantic_recovery",
        "lexical_score",
        "rrf_score",
    ]
    if title_summary.empty:
        return pd.DataFrame(columns=output_columns), {
            "candidate_count": 0,
            "excluded_primary_count": 0,
            "reason": "The local title corpus is empty.",
        }
    relationships = [
        item
        for item in profile.get("relationships", [])
        if isinstance(item, dict)
    ]
    if not relationships:
        return pd.DataFrame(columns=output_columns), {
            "candidate_count": 0,
            "excluded_primary_count": 0,
            "reason": "No approved source relationships were available.",
        }

    resolved_entities = _resolved_entities_from_profile(profile)
    primary_entity_type = _source_entity_type(resolved_entities, relationships)
    identity_groups = _query_identity_groups(
        profile.get("query", ""),
        resolved_entities,
    )
    query_context_phrases = list(
        dict.fromkeys(phrase for group in identity_groups for phrase in group)
    )
    qid_to_component_index = {
        _normalize_qid(entity.get("qid")): index
        for index, entity in enumerate(resolved_entities)
        if _normalize_qid(entity.get("qid"))
    }
    named_entity_component_indices = {
        index
        for index, entity in enumerate(resolved_entities)
        if _entity_description_is_named(entity)
    }
    query_specific_anchor_phrases: list[str] = []
    for relationship in relationships:
        # Only a durable standalone relationship may validate a contextual
        # relationship. Corpus/GDELT associations and broad classifications
        # are never promoted to anchors, regardless of phrase specificity.
        if _relationship_retrieval_policy(relationship) != "standalone":
            continue
        query_specific_anchor_phrases.extend(
            _relationship_match_phrases(relationship)
        )
    query_specific_anchor_phrases = list(
        dict.fromkeys(query_specific_anchor_phrases)
    )[:120]
    prepared_relationships: list[
        tuple[dict[str, object], list[str], list[str], str]
    ] = []
    for relationship in relationships:
        retrieval_policy = _relationship_retrieval_policy(relationship)
        if retrieval_policy == "metadata_only":
            continue
        relation_phrases = _relationship_match_phrases(relationship)
        if not relation_phrases:
            continue
        contextual_anchors = query_specific_anchor_phrases
        if retrieval_policy == "context_required" and not contextual_anchors:
            contextual_anchors = query_context_phrases
        required_cues = _safe_match_phrases(
            [
                *list(relationship.get("required_title_cues", [])),
                *contextual_anchors,
            ],
            allow_single_word=True,
        )
        prepared_relationships.append(
            (relationship, relation_phrases, required_cues, retrieval_policy)
        )

    excluded_ids = {
        _clean_text(story_id)
        for story_id in excluded_story_ids
        if _clean_text(story_id)
    }
    best_by_story_id: dict[str, dict[str, object]] = {}
    evidence_by_story_id: dict[str, list[dict[str, object]]] = {}
    primary_overlap_count = 0
    centrality_rejection_count = 0
    subject_match_story_ids: set[str] = set()
    context_rejection_story_ids: set[str] = set()
    policy_rejection_story_ids: set[str] = set()

    for row in title_summary.to_dict("records"):
        story_id = _clean_text(row.get("story_id"))
        title = _clean_text(row.get("page_title"))
        if not story_id or not title:
            continue
        normalized_title = _normalize_for_match(title)
        surface_title = _normalize_surface(title)
        is_direct_match = story_id in excluded_ids or _is_direct_query_title(
            normalized_title,
            query=profile.get("query", ""),
            identity_groups=identity_groups,
        )
        if is_direct_match:
            primary_overlap_count += 1
            # Direct/refined matches may inform relationship discovery upstream,
            # but this tab must never emit stories already served by direct search.
            continue
        matched_query_component_indices = {
            index
            for index, group in enumerate(identity_groups)
            if _query_title_match_strength(normalized_title, group) >= 0.70
        }
        matched_query_components = len(matched_query_component_indices)
        component_coverage = (
            matched_query_components / len(identity_groups)
            if identity_groups
            else 0.0
        )

        for (
            relationship,
            relation_phrases,
            required_cues,
            retrieval_policy,
        ) in prepared_relationships:
            subject_match = next(
                (
                    phrase
                    for phrase in relation_phrases
                    if _contains_phrase(normalized_title, phrase)
                ),
                "",
            )
            if not subject_match:
                continue
            subject_match_story_ids.add(story_id)
            matched_subject = subject_match
            if relationship.get("requires_exact_surface") is True:
                raw_surface_forms = [relationship.get("related_subject", "")]
                surface_match = next(
                    (
                        _normalize_surface(value)
                        for value in raw_surface_forms
                        if _normalize_surface(value)
                        and _contains_phrase(
                            surface_title,
                            _normalize_surface(value),
                        )
                    ),
                    "",
                )
                if not surface_match:
                    continue
                matched_subject = surface_match

            matched_context_cue = ""
            standalone = retrieval_policy == "standalone"
            if retrieval_policy == "context_required":
                context_match = next(
                    (
                        cue
                        for cue in required_cues
                        if _is_distinct_context_cue(cue, relation_phrases)
                        and _contains_phrase(normalized_title, cue)
                    ),
                    "",
                )
                if not context_match:
                    context_rejection_story_ids.add(story_id)
                    continue
                matched_context_cue = context_match

            interpretation_id = _clean_text(
                relationship.get("interpretation_id")
            )
            relationship_qid = _normalize_qid(
                interpretation_id.removeprefix("wikidata:")
            )
            relationship_component_index = qid_to_component_index.get(
                relationship_qid
            )
            context_component_indices = {
                index
                for index, group in enumerate(identity_groups)
                if matched_context_cue and matched_context_cue in group
            }
            context_from_query = bool(context_component_indices)
            cross_component_context = context_from_query and (
                len(identity_groups) <= 1
                or relationship_component_index is None
                or any(
                    index != relationship_component_index
                    for index in context_component_indices
                )
            )
            corpus_composed = (
                retrieval_policy == "corroboration_only"
                and len(identity_groups) > 1
                and bool(matched_query_component_indices)
                and (
                    not named_entity_component_indices
                    or bool(
                        matched_query_component_indices.intersection(
                            named_entity_component_indices
                        )
                    )
                )
                and _corpus_relationship_can_compose(relationship)
            )

            title_subject_centrality = _title_subject_centrality(
                normalized_title=normalized_title,
                matched_phrase=matched_subject,
                canonical_subject=relationship.get("related_subject", ""),
            )

            try:
                source_confidence = float(relationship.get("confidence", 0.0))
            except (TypeError, ValueError):
                source_confidence = 0.0
            canonical_subject = _normalize_for_match(
                relationship.get("related_subject", "")
            )
            canonical_bonus = 0.04 if matched_subject == canonical_subject else 0.0
            context_bonus = 0.03 if matched_context_cue else 0.0
            match_method = "exact"
            confidence = min(
                0.99,
                max(
                    0.0,
                    source_confidence
                    + canonical_bonus
                    + context_bonus,
                ),
            )
            role = _clean_text(relationship.get("relationship_role")).upper()
            if corpus_composed:
                tier = "Corpus-supported discovery"
                tier_rank = 1
            elif retrieval_policy == "corroboration_only":
                tier = "Corroborating evidence"
                tier_rank = 0
            elif retrieval_policy == "corpus_supported":
                tier = "Corpus-supported discovery"
                tier_rank = 1
            elif role == "LEXICAL_SYNONYM":
                tier = "Lexical review"
                tier_rank = 1
            elif relationship.get("evidence_kind") == "current_events":
                tier = "Current-events evidence"
                tier_rank = 2
            elif standalone and confidence >= 0.86:
                tier = "High confidence"
                tier_rank = 3
            else:
                tier = "Context-supported"
                tier_rank = 2
            evidence = {
                "related_subject": _clean_text(
                    relationship.get("related_subject")
                ),
                "relationship": _clean_text(relationship.get("factual_bridge")),
                "relationship_family": _clean_text(
                    relationship.get("relationship_family")
                ),
                "relationship_role": _clean_text(
                    relationship.get("relationship_role")
                ),
                "related_subject_type": _clean_text(
                    relationship.get("related_subject_type")
                ),
                "predicate_label": _clean_text(
                    relationship.get("predicate_label")
                ),
                "relationship_hops": max(
                    1,
                    int(_safe_float(relationship.get("hop_count", 1))),
                ),
                "requires_exact_surface": relationship.get(
                    "requires_exact_surface"
                ) is True,
                "relationship_path": list(
                    relationship.get("relationship_path", [])
                ) if isinstance(relationship.get("relationship_path"), list) else [],
                "matched_subject": matched_subject,
                "matched_context_cue": matched_context_cue,
                "context_from_query": context_from_query,
                "cross_component_context": cross_component_context,
                "corpus_composed": corpus_composed,
                "relationship_component_index": relationship_component_index,
                "query_entity_type": primary_entity_type,
                "match_method": match_method,
                "source_names": list(relationship.get("source_names", [])),
                "confidence": round(confidence, 3),
                "title_subject_centrality": title_subject_centrality,
                "retrieval_policy": retrieval_policy,
                "candidate_generating": (
                    retrieval_policy
                    in {"standalone", "context_required", "corpus_supported"}
                    or corpus_composed
                ),
                "tier": tier,
                "tier_rank": tier_rank,
                "evidence_urls": list(relationship.get("evidence_urls", [])),
            }
            evidence_by_story_id.setdefault(story_id, []).append(evidence)

        evidence_rows = evidence_by_story_id.get(story_id, [])
        if not evidence_rows:
            continue
        generating_evidence = [
            evidence
            for evidence in evidence_rows
            if evidence.get("candidate_generating") is True
        ]
        has_central_standalone = any(
            (
                evidence.get("retrieval_policy")
                in {"standalone", "corpus_supported"}
                or evidence.get("corpus_composed") is True
            )
            and (
                int(evidence.get("relationship_hops", 1)) == 1
                or evidence.get("requires_exact_surface") is True
                or _clean_text(evidence.get("relationship_family")).casefold()
                == "family-linked organization"
                or (
                    _clean_text(evidence.get("query_entity_type")).casefold()
                    in {"person", "organization"}
                    and _clean_text(
                        evidence.get("relationship_family")
                    ).casefold()
                    == "corporate and institutional network"
                    and _normalize_for_match(
                        evidence.get("matched_subject", "")
                    )
                    == _normalize_for_match(
                        evidence.get("related_subject", "")
                    )
                    and len(
                        _normalize_for_match(
                            evidence.get("matched_subject", "")
                        ).split()
                    )
                    >= 2
                )
            )
            and float(evidence.get("title_subject_centrality", 0.0))
            >= _minimum_centrality_for_evidence(evidence)
            for evidence in generating_evidence
        )
        has_query_grounded_context = any(
            evidence.get("retrieval_policy") == "context_required"
            and evidence.get("cross_component_context") is True
            for evidence in generating_evidence
        )
        independent_generating_keys = {
            (
                evidence.get("relationship_component_index"),
                tuple(
                    sorted(
                        _clean_text(source).casefold()
                        for source in evidence.get("source_names", [])
                        if _clean_text(source)
                    )
                ),
            )
            for evidence in generating_evidence
        }
        has_multiple_independent_evidence = (
            len(independent_generating_keys) >= 2
        )
        if (
            not has_central_standalone
            and not has_query_grounded_context
            and not has_multiple_independent_evidence
        ):
            centrality_rejection_count += 1
            if not generating_evidence:
                policy_rejection_story_ids.add(story_id)
            continue
        evidence_rows.sort(
            key=lambda item: (
                int(item["tier_rank"]),
                float(item["confidence"]),
            ),
            reverse=True,
        )
        primary = evidence_rows[0]
        source_names = sorted(
            {
                str(source)
                for evidence in evidence_rows
                for source in evidence.get("source_names", [])
                if str(source).strip()
            }
        )
        matched_evidence = str(primary["matched_subject"])
        if primary["matched_context_cue"]:
            matched_evidence += f" + {primary['matched_context_cue']}"
        evidence_score = _candidate_evidence_score(
            evidence_rows,
            component_coverage=component_coverage,
        )
        best_by_story_id[story_id] = {
            "story_id": story_id,
            "page_title": title,
            "related_subject": primary["related_subject"],
            "relationship": primary["relationship"],
            "relationship_family": primary["relationship_family"],
            "relationship_role": primary["relationship_role"],
            "relationship_hops": primary["relationship_hops"],
            "relationship_path": primary["relationship_path"],
            "supporting_related_subjects": ", ".join(
                dict.fromkeys(
                    str(evidence.get("related_subject", ""))
                    for evidence in evidence_rows
                    if str(evidence.get("related_subject", "")).strip()
                )
            ),
            "supporting_relationship_count": len(evidence_rows),
            "matched_title_evidence": matched_evidence,
            "source_names": ", ".join(source_names),
            "confidence": round(float(primary["confidence"]), 3),
            "title_subject_centrality": round(
                float(primary.get("title_subject_centrality", 0.0)),
                3,
            ),
            "evidence_score": evidence_score,
            "semantic_recovery": False,
            "result_tier": primary["tier"],
            "result_tier_rank": int(primary["tier_rank"]),
            "result_scope": (
                "Component-related"
                if matched_query_components
                else (
                    f"Relational expansion ({int(primary['relationship_hops'])}-hop)"
                    if int(primary["relationship_hops"]) > 1
                    else "Relational expansion"
                )
            ),
            "matched_query_components": matched_query_components,
            "query_component_coverage": round(component_coverage, 3),
            "relationship_evidence": evidence_rows[:5],
            "retrieval_methods": ", ".join(
                sorted(
                    {
                        str(evidence.get("match_method", ""))
                        for evidence in evidence_rows
                        if str(evidence.get("match_method", "")).strip()
                    }
                )
            ),
            "total_views": row.get("total_views", 0),
            "active_months": row.get("active_months", 0),
            "first_month": row.get("first_month", ""),
            "last_month": row.get("last_month", ""),
        }

    expanded_candidates = list(best_by_story_id.values())
    accepted_candidates = expanded_candidates
    semantic_centrality_diagnostics: dict[str, object] = {
        "enabled": bool(use_semantic_centrality),
        "applied": False,
        "processed": 0,
        "rejected": 0,
        "model": semantic_model_name,
    }
    if use_semantic_centrality and accepted_candidates:
        try:
            (
                accepted_candidates,
                semantic_centrality_diagnostics,
            ) = _apply_semantic_subject_centrality(
                accepted_candidates,
                model_name=semantic_model_name,
            )
        except RuntimeError as exc:
            semantic_centrality_diagnostics["reason"] = str(exc)
    evidence_ranked = sorted(
        accepted_candidates,
        key=lambda item: (
            float(item.get("query_component_coverage", 0.0)),
            int(item.get("result_tier_rank", 0)),
            float(item.get("evidence_score", 0.0)),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    evidence_ranks = {
        str(item["story_id"]): rank
        for rank, item in enumerate(evidence_ranked, start=1)
    }
    lexical_scores = _bm25_candidate_scores(
        title_summary=title_summary,
        candidates=evidence_ranked,
        query=profile.get("query", ""),
        relationships=relationships,
    )
    lexical_ranked_ids = sorted(
        lexical_scores,
        key=lambda story_id: lexical_scores[story_id],
        reverse=True,
    )
    lexical_ranks = {
        story_id: rank for rank, story_id in enumerate(lexical_ranked_ids, start=1)
    }
    popularity_ranked = sorted(
        evidence_ranked,
        key=lambda item: _safe_float(item.get("total_views", 0)),
        reverse=True,
    )
    popularity_ranks = {
        str(item["story_id"]): rank
        for rank, item in enumerate(popularity_ranked, start=1)
    }
    for item in evidence_ranked:
        story_id = str(item["story_id"])
        rrf_score = (
            1.0 / (DEFAULT_RRF_K + evidence_ranks[story_id])
            + 0.70 / (DEFAULT_RRF_K + lexical_ranks[story_id])
            + 0.15 / (DEFAULT_RRF_K + popularity_ranks[story_id])
        )
        item["lexical_score"] = round(lexical_scores[story_id], 4)
        item["rrf_score"] = round(rrf_score, 6)

    ranked = sorted(
        evidence_ranked,
        key=lambda item: (
            float(item.get("query_component_coverage", 0.0)),
            int(item.get("result_tier_rank", 0)),
            float(item.get("evidence_score", 0.0)),
            float(item.get("rrf_score", 0.0)),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )[: max(1, int(max_results))]
    frame = pd.DataFrame(ranked)
    if frame.empty:
        frame = pd.DataFrame(columns=output_columns)
    if not frame.empty:
        empty_reason = ""
    elif not subject_match_story_ids:
        empty_reason = (
            "No non-direct title contained an exact source-related subject. "
            + (
                "Confirmed direct matches from Search Results refined were excluded."
                if primary_overlap_count
                else ""
            )
        )
    elif expanded_candidates and semantic_centrality_diagnostics.get("applied"):
        empty_reason = (
            "All evidence-qualified titles fell below the semantic relevance "
            "threshold and the controlled recovery floor."
        )
    elif policy_rejection_story_ids:
        empty_reason = (
            "Related phrases were present, but their source evidence was "
            "corroborating or metadata-only and could not independently justify "
            "a result."
        )
    else:
        empty_reason = (
            "No non-direct title satisfied the relationship evidence and adaptive "
            "subject-centrality requirements."
        )
    diagnostics = {
        "candidate_count": len(frame),
        "relationship_count": len(relationships),
        "standalone_relationship_count": sum(
            _relationship_retrieval_policy(item) == "standalone"
            for item in relationships
        ),
        "contextual_relationship_count": sum(
            _relationship_retrieval_policy(item) == "context_required"
            for item in relationships
        ),
        "metadata_only_relationship_count": sum(
            _relationship_retrieval_policy(item) == "metadata_only"
            for item in relationships
        ),
        "corroboration_only_relationship_count": sum(
            _relationship_retrieval_policy(item) == "corroboration_only"
            for item in relationships
        ),
        "corpus_supported_relationship_count": sum(
            _relationship_retrieval_policy(item) == "corpus_supported"
            for item in relationships
        ),
        "centrality_rejection_count": centrality_rejection_count,
        "minimum_title_subject_centrality": DEFAULT_MIN_TITLE_SUBJECT_CENTRALITY,
        "semantic_centrality": semantic_centrality_diagnostics,
        "excluded_primary_count": primary_overlap_count,
        "primary_overlap_count": primary_overlap_count,
        "expanded_candidate_count": len(expanded_candidates),
        "stage_counts": {
            "direct_or_refined_excluded": primary_overlap_count,
            "titles_with_related_subject": len(subject_match_story_ids),
            "titles_missing_required_context": len(context_rejection_story_ids),
            "titles_with_non_generating_evidence_only": len(
                policy_rejection_story_ids
            ),
            "titles_passing_deterministic_evidence": len(expanded_candidates),
            "titles_returned": len(frame),
        },
        "direct_evidence_candidate_count": 0,
        "direct_evidence_included": False,
        "direct_fallback_used": False,
        "exact_match_count": sum(
            "exact" in str(item.get("retrieval_methods", "")).split(", ")
            for item in best_by_story_id.values()
        ),
        "retrieval_version": SOURCE_RETRIEVAL_VERSION,
        "retrieval_fusion": {
            "method": "tiered_rrf",
            "rrf_k": DEFAULT_RRF_K,
            "signals": [
                "relationship_evidence",
                "bm25_lexical",
                "traffic_tiebreak",
            ],
            "weights": [1.0, 0.70, 0.15],
        },
        "density_fallback": profile.get("density_fallback", {}),
        "relationship_selection": profile.get("relationship_selection", {}),
        "reason": empty_reason,
    }
    return frame, diagnostics


def _candidate_evidence_score(
    evidence_rows: list[dict[str, object]],
    *,
    component_coverage: float,
) -> float:
    """Combine transparent evidence signals without claiming factual certainty."""

    generating = [
        item for item in evidence_rows if item.get("candidate_generating") is True
    ]
    if not generating:
        return 0.0
    primary = max(
        generating,
        key=lambda item: (
            int(item.get("tier_rank", 0)),
            _safe_float(item.get("confidence")),
        ),
    )
    policy = _clean_text(primary.get("retrieval_policy")).casefold()
    policy_weight = {
        "standalone": 0.16,
        "context_required": 0.13,
        "corpus_supported": 0.10,
        "corroboration_only": 0.06,
    }.get(policy, 0.04)
    sources = {
        _clean_text(source).casefold()
        for item in evidence_rows
        for source in item.get("source_names", [])
        if _clean_text(source)
    }
    source_weight = min(0.10, 0.05 * len(sources))
    independent_weight = min(0.08, 0.04 * max(0, len(generating) - 1))
    hops = max(1, int(_safe_float(primary.get("relationship_hops", 1))))
    hop_penalty = min(0.12, 0.04 * (hops - 1))
    score = (
        0.42 * _safe_float(primary.get("confidence"))
        + 0.24 * _safe_float(primary.get("title_subject_centrality"))
        + 0.10 * max(0.0, min(1.0, component_coverage))
        + policy_weight
        + source_weight
        + independent_weight
        - hop_penalty
    )
    return round(max(0.0, min(1.0, score)), 4)


def _apply_semantic_subject_centrality(
    candidates: list[dict[str, object]],
    *,
    model_name: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Reject evidence-qualified titles semantically detached from their route.

    Exact phrase matching cannot distinguish a story centrally about an entity
    from a headline using that entity only as an affiliation modifier. The local
    multilingual encoder compares each title with the related entity's sourced
    name, type, predicate, relationship family, and factual bridge. A controlled
    recovery is available only when every normal candidate is rejected.
    """

    eligible: list[tuple[dict[str, object], dict[str, object]]] = []
    for candidate in candidates:
        evidence_rows = candidate.get("relationship_evidence", [])
        if not isinstance(evidence_rows, list):
            continue
        generating = [
            evidence
            for evidence in evidence_rows
            if isinstance(evidence, dict)
            and evidence.get("candidate_generating") is True
        ]
        if not generating:
            continue
        primary = max(
            generating,
            key=lambda item: (
                int(item.get("tier_rank", 0)),
                _safe_float(item.get("confidence")),
            ),
        )
        if not _clean_text(primary.get("related_subject_type")):
            continue
        eligible.append((candidate, primary))

    diagnostics: dict[str, object] = {
        "enabled": True,
        "applied": False,
        "processed": len(eligible),
        "rejected": 0,
        "model": model_name,
        "minimum_score": DEFAULT_MIN_SEMANTIC_SUBJECT_SCORE,
    }
    if not eligible:
        diagnostics["reason"] = (
            "No single-edge candidate had a sourced related-entity type."
        )
        return candidates, diagnostics

    route_texts = [
        ". ".join(
            value
            for value in (
                _clean_text(evidence.get("related_subject")),
                _clean_text(evidence.get("related_subject_type")),
                _clean_text(evidence.get("predicate_label")),
                _clean_text(evidence.get("relationship_family")),
                _clean_text(evidence.get("relationship")),
            )
            if value
        )
        for _, evidence in eligible
    ]
    title_texts = [
        _clean_text(candidate.get("page_title"))
        for candidate, _ in eligible
    ]
    try:
        from src.relationship_embeddings import encode_relationship_texts

        vectors = encode_relationship_texts(
            [*route_texts, *title_texts],
            model_name=model_name,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Local semantic subject-centrality scoring was unavailable: {exc}"
        ) from exc
    if len(vectors) != len(eligible) * 2:
        raise RuntimeError(
            "Local semantic subject-centrality scoring returned an invalid shape."
        )

    route_vectors = vectors[: len(eligible)]
    title_vectors = vectors[len(eligible) :]
    similarities = (route_vectors * title_vectors).sum(axis=1)
    rejected_ids: set[str] = set()
    for (candidate, _), raw_score in zip(eligible, similarities):
        score = max(-1.0, min(1.0, float(raw_score)))
        candidate["semantic_subject_score"] = round(score, 4)
        if score < DEFAULT_MIN_SEMANTIC_SUBJECT_SCORE:
            rejected_ids.add(_clean_text(candidate.get("story_id")))

    accepted = [
        candidate
        for candidate in candidates
        if _clean_text(candidate.get("story_id")) not in rejected_ids
    ]
    recovered: list[dict[str, object]] = []
    if not accepted and rejected_ids:
        def recovery_is_grounded(candidate: dict[str, object]) -> bool:
            evidence_rows = candidate.get("relationship_evidence", [])
            if not isinstance(evidence_rows, list):
                return False
            generating = [
                item
                for item in evidence_rows
                if isinstance(item, dict)
                and item.get("candidate_generating") is True
            ]
            if not generating:
                return False
            evidence_origins = {
                (
                    item.get("relationship_component_index"),
                    tuple(
                        sorted(
                            _clean_text(source).casefold()
                            for source in item.get("source_names", [])
                            if _clean_text(source)
                        )
                    ),
                )
                for item in generating
            }
            if len(evidence_origins) >= 2:
                return True
            evidence = generating[0]
            policy = _clean_text(evidence.get("retrieval_policy")).casefold()
            if policy == "corpus_supported" or evidence.get("corpus_composed") is True:
                return True
            return max(
                1,
                int(_safe_float(evidence.get("relationship_hops", 1))),
            ) == 1

        recovery_pool = [
            candidate
            for candidate in candidates
            if _safe_float(candidate.get("semantic_subject_score"))
            >= DEFAULT_SEMANTIC_RECOVERY_FLOOR
            and _safe_float(candidate.get("evidence_score")) >= 0.52
            and recovery_is_grounded(candidate)
        ]
        recovery_pool.sort(
            key=lambda item: (
                _safe_float(item.get("semantic_subject_score")),
                _safe_float(item.get("evidence_score")),
                _safe_float(item.get("query_component_coverage")),
            ),
            reverse=True,
        )
        recovered = recovery_pool[:DEFAULT_MAX_SEMANTIC_RECOVERY_RESULTS]
        for candidate in recovered:
            candidate["semantic_recovery"] = True
            candidate["result_tier"] = "Evidence-backed exploration"
            candidate["result_tier_rank"] = 1
        accepted = recovered
    diagnostics.update(
        {
            "applied": True,
            "rejected": len(rejected_ids),
            "accepted": len(accepted),
            "recovered": len(recovered),
            "recovery_floor": DEFAULT_SEMANTIC_RECOVERY_FLOOR,
        }
    )
    return accepted, diagnostics


def _bm25_candidate_scores(
    *,
    title_summary: pd.DataFrame,
    candidates: list[dict[str, object]],
    query: object,
    relationships: list[dict[str, object]],
) -> dict[str, float]:
    """Score accepted candidates with a small in-process BM25 signal.

    BM25 does not decide acceptance.  It only improves ordering after the
    source relationship gate has accepted a title, which prevents broad terms
    from becoming ungrounded related stories.
    """

    query_values = [
        query,
        *(relationship.get("related_subject", "") for relationship in relationships),
    ]
    query_terms = {
        token
        for value in query_values
        for token in _normalize_for_match(value).split()
        if len(token) >= 3 and token not in _CORPUS_STOPWORDS
    }
    candidate_ids = {str(item.get("story_id", "")) for item in candidates}
    if not query_terms or not candidate_ids:
        return {story_id: 0.0 for story_id in candidate_ids}

    document_frequency: Counter[str] = Counter()
    candidate_tokens: dict[str, list[str]] = {}
    total_length = 0
    document_count = 0
    for row in title_summary.to_dict("records"):
        tokens = _normalize_for_match(row.get("page_title", "")).split()
        if not tokens:
            continue
        document_count += 1
        total_length += len(tokens)
        document_frequency.update(set(tokens).intersection(query_terms))
        story_id = _clean_text(row.get("story_id"))
        if story_id in candidate_ids:
            candidate_tokens[story_id] = tokens
    if document_count <= 0:
        return {story_id: 0.0 for story_id in candidate_ids}

    average_length = max(1.0, total_length / document_count)
    k1 = 1.2
    b = 0.75
    scores: dict[str, float] = {}
    for story_id in candidate_ids:
        tokens = candidate_tokens.get(story_id, [])
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            frequency_in_documents = document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0
                + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(tokens) / average_length
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0) / denominator
            )
        scores[story_id] = score
    return scores


def _analyze_query_structure(
    query: str,
    *,
    title_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Create bounded, source-neutral spans for downstream entity linking.

    This is segmentation, not transliteration or language inference.  It uses
    token boundaries, numeric time expressions, and local phrase support only.
    """

    normalized = _normalize_surface(query)
    tokens = normalized.split()
    years = [
        token
        for token in tokens
        if token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100
    ]
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, token in enumerate(tokens):
        boundary = token in _CORPUS_STOPWORDS or token.isdigit()
        if boundary and run_start is not None:
            runs.append((run_start, index))
            run_start = None
        elif not boundary and run_start is None:
            run_start = index
    if run_start is not None:
        runs.append((run_start, len(tokens)))

    raw_spans: list[QuerySpan] = []
    for run_start, run_end in runs:
        width_limit = min(4, run_end - run_start)
        for width in range(width_limit, 0, -1):
            for start in range(run_start, run_end - width + 1):
                end = start + width
                text = " ".join(tokens[start:end])
                if text == normalized or (width == 1 and len(text) < 3):
                    continue
                raw_spans.append(QuerySpan(text, start, end, "content_span"))

    titles = []
    if isinstance(title_summary, pd.DataFrame) and not title_summary.empty:
        titles = [
            _normalize_for_match(value)
            for value in title_summary.get("page_title", pd.Series(dtype=str)).tolist()
        ]
    ranked: list[tuple[int, int, int, QuerySpan]] = []
    seen: set[str] = set()
    for span in raw_spans:
        key = _normalize_surface(span.text)
        if not key or key in seen:
            continue
        seen.add(key)
        support = sum(_contains_phrase(title, span.text) for title in titles)
        ranked.append((span.end - span.start, min(support, 100), -span.start, span))
    ranked.sort(reverse=True, key=lambda item: item[:3])
    mention_spans = [item[3] for item in ranked[:8]]
    return {
        "normalized_query": normalized,
        "tokens": tokens,
        "temporal_constraints": years,
        "mention_candidates": [span.__dict__ for span in mention_spans],
        "script_constraint": "Latin",
        "language_assumption": None,
    }


def _resolve_wikimedia_query(
    session: requests.Session,
    *,
    query: str,
    title_summary: pd.DataFrame | None,
    selected_wikidata_qid: str = "",
    include_wikipedia: bool = False,
) -> dict[str, object]:
    """Resolve a complete query, then independently resolve non-overlapping spans."""

    complete = _resolve_wikimedia_entity(
        session,
        query=query,
        title_summary=title_summary,
        selected_wikidata_qid=selected_wikidata_qid,
        include_wikipedia=include_wikipedia,
    )
    complete_qid = _normalize_qid(complete.get("selected_qid"))
    if complete_qid:
        complete["strategy"] = "complete_query"
        complete["selected_mentions"] = [
            {
                "text": query,
                "start": 0,
                "end": len(_normalize_for_match(query).split()),
                "qid": complete_qid,
                "score": complete.get("selected_score", 0.0),
                "strategy": "complete_query",
            }
        ]
        for candidate in complete.get("candidates", []):
            if isinstance(candidate, dict):
                candidate["query_mention"] = query
        return complete

    structure = _analyze_query_structure(query, title_summary=title_summary)
    occupied: set[int] = set()
    selected_mentions: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for candidate in complete.get("candidates", []):
        if isinstance(candidate, dict):
            item = dict(candidate)
            item["query_mention"] = query
            candidates.append(item)

    wikipedia_candidate_count = int(complete.get("wikipedia_candidate_count", 0))
    wikipedia_search_used = bool(complete.get("wikipedia_search_used"))
    for raw_span in structure.get("mention_candidates", []):
        if not isinstance(raw_span, dict):
            continue
        start = int(raw_span.get("start", 0))
        end = int(raw_span.get("end", start))
        if any(index in occupied for index in range(start, end)):
            continue
        mention_text = _clean_text(raw_span.get("text"))
        if not mention_text:
            continue
        mention_resolution = _resolve_wikimedia_entity(
            session,
            query=mention_text,
            title_summary=title_summary,
            include_wikipedia=include_wikipedia,
        )
        wikipedia_candidate_count += int(
            mention_resolution.get("wikipedia_candidate_count", 0)
        )
        wikipedia_search_used = wikipedia_search_used or bool(
            mention_resolution.get("wikipedia_search_used")
        )
        for candidate in mention_resolution.get("candidates", []):
            if isinstance(candidate, dict):
                item = dict(candidate)
                item["query_mention"] = mention_text
                candidates.append(item)
        mention_qid = _normalize_qid(mention_resolution.get("selected_qid"))
        contextual_acceptance = False
        if not mention_qid and selected_mentions:
            alternatives = [
                item
                for item in mention_resolution.get("candidates", [])
                if isinstance(item, dict)
            ]
            if alternatives:
                top = alternatives[0]
                joint_support = _joint_mention_corpus_support(
                    title_summary=title_summary,
                    candidate=top,
                    context_mentions=[
                        _clean_text(item.get("text")) for item in selected_mentions
                    ],
                )
                if (
                    _safe_float(top.get("score")) >= 0.74
                    and _safe_float(top.get("label_similarity")) >= 0.95
                    and joint_support >= 2
                    and not bool(top.get("is_disambiguation"))
                ):
                    mention_qid = _normalize_qid(top.get("qid"))
                    contextual_acceptance = bool(mention_qid)
        if not mention_qid:
            continue
        selected_mentions.append(
            {
                "text": mention_text,
                "start": start,
                "end": end,
                "qid": mention_qid,
                "score": mention_resolution.get("selected_score", 0.0),
                "strategy": raw_span.get("strategy", "content_span"),
                "contextual_acceptance": contextual_acceptance,
            }
        )
        occupied.update(range(start, end))
        if len(selected_mentions) >= 3:
            break

    deduplicated_mentions: list[dict[str, object]] = []
    seen_qids: set[str] = set()
    for mention in selected_mentions:
        qid = _normalize_qid(mention.get("qid"))
        if not qid or qid in seen_qids:
            continue
        seen_qids.add(qid)
        deduplicated_mentions.append(mention)
    selected_mentions = deduplicated_mentions
    selected_qids = [
        _normalize_qid(mention.get("qid")) for mention in selected_mentions
    ]
    for candidate in candidates:
        candidate_selected = _normalize_qid(candidate.get("qid")) in selected_qids
        candidate["selected"] = candidate_selected
        if candidate_selected:
            candidate["selection_reason"] = (
                "Selected for a complete query or a non-overlapping mention with "
                "local cross-mention support."
            )
    selected_score = (
        sum(_safe_float(item.get("score")) for item in selected_mentions)
        / len(selected_mentions)
        if selected_mentions
        else _safe_float(complete.get("selected_score"))
    )
    detail = _clean_text(complete.get("detail"))
    if selected_mentions:
        detail = (
            f"Resolved {len(selected_mentions)} non-overlapping query mention(s): "
            + ", ".join(
                f"{item['text']} ({item['qid']})" for item in selected_mentions
            )
            + "."
        )
    return {
        **complete,
        "strategy": "multi_mention" if selected_mentions else "unresolved",
        "selected_qid": selected_qids[0] if selected_qids else "",
        "selected_qids": selected_qids,
        "selected_score": round(selected_score, 3),
        "selected_mentions": selected_mentions,
        "candidates": candidates[:20],
        "wikipedia_search_used": wikipedia_search_used,
        "wikipedia_candidate_count": wikipedia_candidate_count,
        "query_structure": structure,
        "detail": detail,
    }


def _joint_mention_corpus_support(
    *,
    title_summary: pd.DataFrame | None,
    candidate: dict[str, object],
    context_mentions: list[str],
) -> int:
    if not isinstance(title_summary, pd.DataFrame) or title_summary.empty:
        return 0
    candidate_forms = _safe_match_phrases(
        [candidate.get("label", ""), *list(candidate.get("aliases", []))],
        allow_single_word=True,
    )
    context_forms = _safe_match_phrases(context_mentions, allow_single_word=True)
    if not candidate_forms or not context_forms:
        return 0
    support = 0
    for title in title_summary.get("page_title", pd.Series(dtype=str)).tolist():
        normalized_title = _normalize_for_match(title)
        if not any(_contains_phrase(normalized_title, form) for form in candidate_forms):
            continue
        if not any(_contains_phrase(normalized_title, form) for form in context_forms):
            continue
        support += 1
        if support >= 10:
            break
    return support


def _aggregate_resolved_entity(
    resolved_entities: list[dict[str, object]],
) -> dict[str, object]:
    if not resolved_entities:
        return {}
    primary = dict(resolved_entities[0])
    primary["aliases"] = _deduplicate_text(
        [
            *list(primary.get("aliases", [])),
            *(
                value
                for entity in resolved_entities
                for value in [
                    entity.get("matched_mention", ""),
                    entity.get("label", ""),
                    *list(entity.get("aliases", [])),
                ]
            ),
        ]
    )
    return primary


def _cross_entity_bridge_relationships(
    *,
    resolved_entities: list[dict[str, object]],
    relationships: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Derive bounded shared-neighbor bridges between resolved query mentions."""

    query_entities = {
        _normalize_qid(entity.get("qid")): entity
        for entity in resolved_entities
        if _normalize_qid(entity.get("qid"))
    }
    if len(query_entities) < 2:
        return []
    allowed_roles = {
        "DIRECT_PARTICIPANT",
        "INSTITUTIONAL_BRIDGE",
        "QUERY_SPECIFIC_MANIFESTATION",
    }
    by_target: dict[str, list[dict[str, object]]] = {}
    for relationship in relationships:
        source_qid = _normalize_qid(
            _clean_text(relationship.get("interpretation_id")).removeprefix(
                "wikidata:"
            )
        )
        target_qid = _normalize_qid(relationship.get("related_subject_id"))
        role = _clean_text(relationship.get("relationship_role")).upper()
        if source_qid in query_entities and target_qid and role in allowed_roles:
            by_target.setdefault(target_qid, []).append(relationship)

    bridges: list[dict[str, object]] = []
    for target_qid, edges in by_target.items():
        source_qids = list(
            dict.fromkeys(
                _normalize_qid(
                    _clean_text(edge.get("interpretation_id")).removeprefix(
                        "wikidata:"
                    )
                )
                for edge in edges
            )
        )
        source_qids = [qid for qid in source_qids if qid]
        if len(source_qids) < 2:
            continue
        selected_edges = [
            next(
                edge
                for edge in edges
                if _clean_text(edge.get("interpretation_id")) == f"wikidata:{qid}"
            )
            for qid in source_qids[:3]
        ]
        subject = _clean_text(selected_edges[0].get("related_subject"))
        if not subject:
            continue
        source_labels = [
            _clean_text(query_entities[qid].get("label")) for qid in source_qids[:3]
        ]
        predicate_labels = [
            _clean_text(edge.get("predicate_label")) for edge in selected_edges
        ]
        confidence = max(
            0.55,
            min(_safe_float(edge.get("confidence")) for edge in selected_edges) - 0.04,
        )
        bridge_key = "|".join([*source_qids[:3], target_qid])
        bridges.append(
            {
                "relationship_id": "wikidata-bridge:"
                + hashlib.sha1(bridge_key.encode("utf-8")).hexdigest()[:16],
                "interpretation_id": "query:multi-mention",
                "related_subject": subject,
                "aliases": list(selected_edges[0].get("aliases", [])),
                "related_subject_id": target_qid,
                "related_subject_type": _clean_text(
                    selected_edges[0].get("related_subject_type")
                ),
                "relationship_class": "CONTEXTUAL",
                "relationship_family": "shared structured neighbor",
                "relationship_role": "INSTITUTIONAL_BRIDGE",
                "property_specificity": "context_required",
                "predicate_id": "wikidata:shared-neighbor",
                "predicate_label": "shared structured connection",
                "factual_bridge": (
                    f"{subject} connects "
                    + " and ".join(source_labels)
                    + " through Wikidata properties "
                    + ", ".join(predicate_labels)
                    + "."
                ),
                "direction": "multi_mention_to_shared_neighbor",
                "temporal_scope": "current_or_durable",
                "can_retrieve_standalone": False,
                "required_title_cues": source_labels,
                "source_names": ["Wikidata"],
                "source_family": "wikimedia_structured",
                "evidence_urls": _deduplicate_text(
                    url
                    for edge in selected_edges
                    for url in edge.get("evidence_urls", [])
                ),
                "confidence": round(confidence, 3),
                "false_positive_risk": "medium",
                "explicit_or_derived": "DERIVED",
                "evidence_kind": "structured_graph",
                "bridge_source_qids": source_qids[:3],
                "bridge_property_ids": [
                    _clean_text(edge.get("predicate_id")) for edge in selected_edges
                ],
            }
        )
    return bridges[:8]


def _build_query_graph(
    *,
    query_structure: dict[str, object],
    resolved_entities: list[dict[str, object]],
    relationships: list[dict[str, object]],
) -> dict[str, object]:
    nodes: dict[str, dict[str, object]] = {}
    for entity in resolved_entities:
        qid = _normalize_qid(entity.get("qid"))
        if not qid:
            continue
        nodes[qid] = {
            "id": qid,
            "label": _clean_text(entity.get("label")),
            "node_kind": "query_entity",
            "matched_mention": _clean_text(entity.get("matched_mention")),
        }
    edges: list[dict[str, object]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for relationship in relationships:
        relationship_path = relationship.get("relationship_path", [])
        if isinstance(relationship_path, list) and relationship_path:
            for step in relationship_path:
                if not isinstance(step, dict):
                    continue
                source_id = _clean_text(step.get("source_id"))
                target_id = _clean_text(step.get("target_id"))
                predicate_id = _clean_text(step.get("predicate_id"))
                direction = _clean_text(step.get("statement_direction"))
                if not source_id or not target_id or not predicate_id:
                    continue
                nodes.setdefault(
                    source_id,
                    {
                        "id": source_id,
                        "label": _clean_text(step.get("source_label")),
                        "node_kind": "path_entity",
                    },
                )
                nodes.setdefault(
                    target_id,
                    {
                        "id": target_id,
                        "label": _clean_text(step.get("target_label")),
                        "node_kind": "related_entity",
                    },
                )
                edge_key = (source_id, predicate_id, target_id, direction)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "predicate_id": predicate_id,
                        "predicate_label": _clean_text(step.get("predicate_label")),
                        "statement_direction": direction,
                        "confidence": _safe_float(relationship.get("confidence")),
                        "sources": list(relationship.get("source_names", [])),
                    }
                )
            continue
        target_id = _normalize_qid(relationship.get("related_subject_id"))
        source_id = _clean_text(relationship.get("interpretation_id"))
        if source_id.startswith("wikidata:"):
            source_id = source_id.removeprefix("wikidata:")
        if target_id:
            nodes.setdefault(
                target_id,
                {
                    "id": target_id,
                    "label": _clean_text(relationship.get("related_subject")),
                    "node_kind": "related_entity",
                },
            )
        edge_key = (
            source_id,
            _clean_text(relationship.get("predicate_id")),
            target_id or _clean_text(relationship.get("related_subject")),
            _clean_text(relationship.get("direction")),
        )
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edges.append(
            {
                "source": source_id,
                "target": target_id or _clean_text(relationship.get("related_subject")),
                "predicate_id": _clean_text(relationship.get("predicate_id")),
                "predicate_label": _clean_text(relationship.get("predicate_label")),
                "confidence": _safe_float(relationship.get("confidence")),
                "sources": list(relationship.get("source_names", [])),
            }
        )
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "temporal_constraints": list(
            query_structure.get("temporal_constraints", [])
        ),
    }


def _resolve_wikimedia_entity(
    session: requests.Session,
    *,
    query: str,
    title_summary: pd.DataFrame | None,
    selected_wikidata_qid: str = "",
    include_wikipedia: bool = False,
) -> dict[str, object]:
    """Resolve a query from Wikidata and an optional Wikipedia fallback.

    Candidate generation and candidate acceptance are deliberately separate:
    search rank contributes evidence, but no source's first result is trusted by
    itself.  Low-confidence and close-scoring candidates remain alternatives.
    """
    override_qid = _normalize_qid(selected_wikidata_qid)
    variants = _build_query_variants(query)
    if override_qid:
        entities = _wikidata_entities(
            session,
            [override_qid],
            include_sitelinks=True,
        )
        entity = entities.get(override_qid, {})
        label = _localized_value(entity.get("labels"), "en") or override_qid
        description = _localized_value(entity.get("descriptions"), "en")
        candidate = {
            "qid": override_qid,
            "label": label,
            "description": description,
            "aliases": _localized_aliases(entity.get("aliases"), "en"),
            "sitelink_count": len(entity.get("sitelinks", {}))
            if isinstance(entity.get("sitelinks"), dict)
            else 0,
            "sources": ["QID override"],
            "matched_variants": [query],
            "score": 1.0,
            "selected": bool(entity),
            "selection_reason": "Explicit Wikidata QID override.",
        }
        return {
            "selected_qid": override_qid if entity else "",
            "selected_score": 1.0 if entity else 0.0,
            "acceptance_threshold": DEFAULT_ENTITY_ACCEPTANCE_SCORE,
            "acceptance_margin": DEFAULT_ENTITY_ACCEPTANCE_MARGIN,
            "expected_types": sorted(_expected_query_types(query)),
            "query_variants": [variant.__dict__ for variant in variants],
            "candidates": [candidate] if entity else [],
            "wikipedia_search_used": False,
            "wikipedia_candidate_count": 0,
            "detail": (
                f"Used explicit Wikidata override {override_qid}."
                if entity
                else f"No Wikidata entity was returned for {override_qid}."
            ),
        }

    candidates_by_qid: dict[str, dict[str, object]] = {}
    # The original query is the highest-value Wikidata request. Wikipedia
    # full-text search handles label mismatch; bounded variants are a final
    # backoff only, avoiding a slow serial fan-out across public APIs.
    for variant_index, variant in enumerate(variants[:1]):
        search_results = _wikidata_search(session, variant.text)
        for rank, item in enumerate(search_results[:5]):
            qid = _normalize_qid(item.get("id"))
            if not qid:
                continue
            candidate = candidates_by_qid.setdefault(
                qid,
                {
                    "qid": qid,
                    "label": _clean_text(item.get("label")),
                    "description": _clean_text(item.get("description")),
                    "aliases": [],
                    "sitelink_count": 0,
                    "sources": [],
                    "matched_variants": [],
                    "search_rank_score": 0.0,
                    "best_variant_penalty": 1.0,
                },
            )
            candidate["search_rank_score"] = max(
                _safe_float(candidate.get("search_rank_score")),
                max(0.20, 1.0 - 0.15 * rank - 0.05 * variant_index),
            )
            candidate["best_variant_penalty"] = min(
                _safe_float(candidate.get("best_variant_penalty")),
                variant.penalty,
            )
            candidate["sources"] = _deduplicate_text(
                [*list(candidate.get("sources", [])), "Wikidata search"]
            )
            candidate["matched_variants"] = _deduplicate_text(
                [*list(candidate.get("matched_variants", [])), variant.text]
            )

    if candidates_by_qid:
        enriched = _wikidata_entities(
            session,
            list(candidates_by_qid),
            include_sitelinks=True,
        )
        _enrich_resolution_candidates(candidates_by_qid, enriched)

    prelim = _score_resolution_candidates(
        candidates_by_qid,
        query=query,
        title_summary=title_summary,
    )
    preliminary_margin = (
        _safe_float(prelim[0].get("score")) - _safe_float(prelim[1].get("score"))
        if len(prelim) > 1
        else 1.0
    )
    needs_resolution_backoff = not prelim or (
        _safe_float(prelim[0].get("score")) < DEFAULT_ENTITY_ACCEPTANCE_SCORE + 0.10
        or preliminary_margin < DEFAULT_ENTITY_ACCEPTANCE_MARGIN
    )
    use_wikipedia = include_wikipedia and needs_resolution_backoff
    wikipedia_titles: list[str] = []
    wikipedia_title_metadata: dict[str, dict[str, object]] = {}
    if needs_resolution_backoff and not include_wikipedia:
        for variant in variants[1:3]:
            wikidata_results = _wikidata_search(session, variant.text)
            if not wikidata_results:
                continue
            for rank, item in enumerate(wikidata_results[:5]):
                qid = _normalize_qid(item.get("id"))
                if not qid:
                    continue
                candidate = candidates_by_qid.setdefault(
                    qid,
                    {
                        "qid": qid,
                        "label": _clean_text(item.get("label")),
                        "description": _clean_text(item.get("description")),
                        "aliases": [],
                        "sitelink_count": 0,
                        "sources": [],
                        "matched_variants": [],
                        "search_rank_score": 0.0,
                        "best_variant_penalty": variant.penalty,
                    },
                )
                candidate["sources"] = _deduplicate_text(
                    [*list(candidate.get("sources", [])), "Wikidata search"]
                )
                candidate["matched_variants"] = _deduplicate_text(
                    [*list(candidate.get("matched_variants", [])), variant.text]
                )
                candidate["search_rank_score"] = max(
                    _safe_float(candidate.get("search_rank_score")),
                    max(0.20, 1.0 - 0.15 * rank - variant.penalty),
                )
            break
    if use_wikipedia:
        wikipedia_variants = variants[:1]
        for variant in wikipedia_variants:
            for rank, result in enumerate(
                _wikipedia_search(session, variant.text)[:DEFAULT_MAX_WIKIPEDIA_CANDIDATES]
            ):
                title = _clean_text(result.get("title"))
                if not title:
                    continue
                wikipedia_titles.append(title)
                metadata = wikipedia_title_metadata.setdefault(title.casefold(), {})
                metadata["rank_score"] = max(
                    _safe_float(metadata.get("rank_score")),
                    max(0.20, 1.0 - 0.15 * rank - variant.penalty),
                )
                metadata["matched_variants"] = _deduplicate_text(
                    [*list(metadata.get("matched_variants", [])), variant.text]
                )
        if not wikipedia_titles:
            for variant in variants[1:3]:
                wikidata_results = _wikidata_search(session, variant.text)
                if wikidata_results:
                    for rank, item in enumerate(wikidata_results[:5]):
                        qid = _normalize_qid(item.get("id"))
                        if not qid:
                            continue
                        candidate = candidates_by_qid.setdefault(
                            qid,
                            {
                                "qid": qid,
                                "label": _clean_text(item.get("label")),
                                "description": _clean_text(item.get("description")),
                                "aliases": [],
                                "sitelink_count": 0,
                                "sources": [],
                                "matched_variants": [],
                                "search_rank_score": 0.0,
                                "best_variant_penalty": variant.penalty,
                            },
                        )
                        candidate["sources"] = _deduplicate_text(
                            [*list(candidate.get("sources", [])), "Wikidata search"]
                        )
                        candidate["matched_variants"] = _deduplicate_text(
                            [*list(candidate.get("matched_variants", [])), variant.text]
                        )
                        candidate["search_rank_score"] = max(
                            _safe_float(candidate.get("search_rank_score")),
                            max(0.20, 1.0 - 0.15 * rank - variant.penalty),
                        )
                    break
        wikipedia_titles = _deduplicate_text(wikipedia_titles)[:15]
        if wikipedia_titles:
            wikipedia_entities = _wikidata_entities_for_wikipedia_titles(
                session,
                wikipedia_titles,
            )
            for qid, entity in wikipedia_entities.items():
                sitelinks = entity.get("sitelinks", {})
                enwiki = sitelinks.get("enwiki", {}) if isinstance(sitelinks, dict) else {}
                title = _clean_text(enwiki.get("title")) if isinstance(enwiki, dict) else ""
                metadata = wikipedia_title_metadata.get(title.casefold(), {})
                candidate = candidates_by_qid.setdefault(
                    qid,
                    {
                        "qid": qid,
                        "label": "",
                        "description": "",
                        "aliases": [],
                        "sitelink_count": 0,
                        "sources": [],
                        "matched_variants": [],
                        "search_rank_score": 0.0,
                        "best_variant_penalty": 0.0,
                    },
                )
                candidate["sources"] = _deduplicate_text(
                    [*list(candidate.get("sources", [])), "Wikipedia full-text search"]
                )
                candidate["matched_variants"] = _deduplicate_text(
                    [
                        *list(candidate.get("matched_variants", [])),
                        *list(metadata.get("matched_variants", [])),
                    ]
                )
                candidate["search_rank_score"] = max(
                    _safe_float(candidate.get("search_rank_score")),
                    _safe_float(metadata.get("rank_score", 0.5)),
                )
            _enrich_resolution_candidates(candidates_by_qid, wikipedia_entities)

    unenriched_qids = [
        qid
        for qid, candidate in candidates_by_qid.items()
        if candidate.get("_enriched") is not True
    ]
    if unenriched_qids:
        _enrich_resolution_candidates(
            candidates_by_qid,
            _wikidata_entities(session, unenriched_qids, include_sitelinks=True),
        )

    scored = _score_resolution_candidates(
        candidates_by_qid,
        query=query,
        title_summary=title_summary,
    )
    top_score = _safe_float(scored[0].get("score")) if scored else 0.0
    margin = (
        top_score - _safe_float(scored[1].get("score"))
        if len(scored) > 1
        else 1.0
    )
    local_query_seed_count = _count_query_seed_titles(title_summary, query)
    local_evidence_conflict = bool(
        "event" in _expected_query_types(query)
        and local_query_seed_count >= 2
        and scored
        and _safe_float(scored[0].get("local_corpus_support")) < 0.15
        and sum(not bool(item.get("is_disambiguation")) for item in scored) > 1
    )
    selected_qid = ""
    if (
        scored
        and top_score >= DEFAULT_ENTITY_ACCEPTANCE_SCORE
        and margin >= DEFAULT_ENTITY_ACCEPTANCE_MARGIN
        and not local_evidence_conflict
    ):
        selected_qid = _normalize_qid(scored[0].get("qid"))
    interpretations: list[dict[str, object]] = []
    for candidate in scored[:5]:
        selected = _normalize_qid(candidate.get("qid")) == selected_qid
        candidate["selected"] = selected
        candidate["selection_reason"] = (
            "Highest candidate above the confidence and margin thresholds."
            if selected
            else "Alternative candidate retained for editorial review."
        )
        interpretations.append(candidate)

    if selected_qid:
        detail = (
            f"Selected {selected_qid} from {len(scored)} scored candidates "
            f"(score {top_score:.2f}; margin {margin:.2f})."
        )
    elif scored:
        detail = (
            f"Retained {min(5, len(scored))} candidate interpretations, but the best "
            f"score ({top_score:.2f}) or margin ({margin:.2f}) did not pass the "
            "automatic resolution gate."
        )
        if local_evidence_conflict:
            detail = (
                "External event candidates did not match the distinguishing language "
                f"in {local_query_seed_count} local query titles; retained them as "
                "alternatives and continued in corpus-only mode."
            )
    else:
        detail = (
            "Wikidata and Wikipedia returned no usable entity candidates."
            if include_wikipedia
            else "Wikidata returned no usable entity candidates."
        )
    return {
        "selected_qid": selected_qid,
        "selected_score": round(top_score, 3),
        "candidate_margin": round(margin, 3),
        "acceptance_threshold": DEFAULT_ENTITY_ACCEPTANCE_SCORE,
        "acceptance_margin": DEFAULT_ENTITY_ACCEPTANCE_MARGIN,
        "expected_types": sorted(_expected_query_types(query)),
        "local_query_seed_count": local_query_seed_count,
        "local_evidence_conflict": local_evidence_conflict,
        "query_variants": [variant.__dict__ for variant in variants],
        "candidates": interpretations,
        "wikipedia_search_used": use_wikipedia,
        "wikipedia_candidate_count": len(wikipedia_titles),
        "detail": detail,
    }


def _build_query_variants(query: str) -> list[QueryVariant]:
    original = _clean_text(query)
    normalized = _normalize_for_match(original)
    candidates = [QueryVariant(original, "original", 0.0)]
    if normalized and normalized.casefold() != original.casefold():
        candidates.append(QueryVariant(normalized, "normalized", 0.02))
    tokens = normalized.split()
    if tokens:
        for index, token in enumerate(tokens):
            if token not in _EVENT_QUERY_TERMS:
                continue
            plural = _pluralize_token(token)
            if plural != token:
                varied = [*tokens]
                varied[index] = plural
                candidates.append(
                    QueryVariant(" ".join(varied), "event_inflection", 0.03)
                )
        core_tokens = [token for token in tokens if token not in _EVENT_QUERY_TERMS]
        if core_tokens and len(core_tokens) < len(tokens):
            candidates.append(
                QueryVariant(" ".join(core_tokens), "context_entity_backoff", 0.22)
            )
    result: list[QueryVariant] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize_surface(candidate.text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) >= DEFAULT_MAX_QUERY_VARIANTS:
            break
    return result


def _pluralize_token(token: str) -> str:
    if token.endswith("y") and len(token) > 2 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    if token.endswith(("s", "x", "z", "ch", "sh")):
        return token + "es"
    return token + "s"


def _enrich_resolution_candidates(
    candidates_by_qid: dict[str, dict[str, object]],
    entities: dict[str, dict[str, object]],
) -> None:
    for qid, entity in entities.items():
        candidate = candidates_by_qid.get(qid)
        if candidate is None:
            continue
        label = _localized_value(entity.get("labels"), "en")
        description = _localized_value(entity.get("descriptions"), "en")
        aliases = _localized_aliases(entity.get("aliases"), "en")
        sitelinks = entity.get("sitelinks", {})
        if label:
            candidate["label"] = label
        if description:
            candidate["description"] = description
        candidate["aliases"] = aliases
        candidate["sitelink_count"] = (
            len(sitelinks) if isinstance(sitelinks, dict) else 0
        )
        enwiki = sitelinks.get("enwiki", {}) if isinstance(sitelinks, dict) else {}
        candidate["wikipedia_title"] = (
            _clean_text(enwiki.get("title")) if isinstance(enwiki, dict) else ""
        )
        candidate["_enriched"] = True


def _score_resolution_candidates(
    candidates_by_qid: dict[str, dict[str, object]],
    *,
    query: str,
    title_summary: pd.DataFrame | None,
) -> list[dict[str, object]]:
    expected_types = _expected_query_types(query)
    scored: list[dict[str, object]] = []
    for raw_candidate in candidates_by_qid.values():
        candidate = dict(raw_candidate)
        candidate.pop("_enriched", None)
        description = _clean_text(candidate.get("description")).casefold()
        is_disambiguation = any(
            marker in description for marker in _DISAMBIGUATION_MARKERS
        )
        forms = _deduplicate_text(
            [candidate.get("label", ""), *list(candidate.get("aliases", []))]
        )
        lexical_score = max(
            (_text_similarity(query, form) for form in forms),
            default=0.0,
        )
        type_score = _candidate_type_score(description, expected_types)
        corpus_score = _candidate_corpus_score(
            candidate,
            query=query,
            title_summary=title_summary,
        )
        rank_score = _safe_float(candidate.get("search_rank_score", 0.0))
        sitelinks = max(0, int(_safe_float(candidate.get("sitelink_count", 0))))
        notability_score = min(1.0, math.log1p(sitelinks) / math.log1p(300))
        sources = list(candidate.get("sources", []))
        source_agreement = 1.0 if len(set(sources)) >= 2 else 0.5
        penalty = min(0.25, _safe_float(candidate.get("best_variant_penalty", 0.0)))
        score = (
            lexical_score * 0.38
            + type_score * 0.16
            + rank_score * 0.14
            + corpus_score * 0.16
            + notability_score * 0.08
            + source_agreement * 0.08
            - penalty
        )
        if is_disambiguation:
            score = 0.0
        candidate.update(
            {
                "score": round(max(0.0, min(1.0, score)), 3),
                "label_similarity": round(lexical_score, 3),
                "type_compatibility": round(type_score, 3),
                "local_corpus_support": round(corpus_score, 3),
                "search_rank_score": round(rank_score, 3),
                "sitelink_count": sitelinks,
                "is_disambiguation": is_disambiguation,
            }
        )
        scored.append(candidate)
    scored.sort(
        key=lambda item: (
            _safe_float(item.get("score")),
            _safe_float(item.get("local_corpus_support")),
            _safe_float(item.get("search_rank_score")),
        ),
        reverse=True,
    )
    return scored


def _text_similarity(left: object, right: object) -> float:
    normalized_left = _normalize_for_match(left)
    normalized_right = _normalize_for_match(right)
    if not normalized_left or not normalized_right:
        return 0.0
    sequence = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    left_tokens = set(normalized_left.split())
    right_tokens = set(normalized_right.split())
    token_score = (
        len(left_tokens.intersection(right_tokens))
        / max(1, len(left_tokens.union(right_tokens)))
    )
    containment = (
        len(left_tokens.intersection(right_tokens))
        / max(1, min(len(left_tokens), len(right_tokens)))
    )
    return max(sequence, 0.55 * token_score + 0.45 * containment)


def _expected_query_types(query: str) -> set[str]:
    tokens = set(_normalize_for_match(query).split())
    expected: set[str] = set()
    if tokens.intersection(_EVENT_QUERY_TERMS):
        expected.add("event")
    if tokens.intersection({"law", "act", "bill", "policy", "regulation"}):
        expected.add("policy")
    if tokens.intersection({"company", "corporation", "organisation", "organization"}):
        expected.add("organization")
    if tokens.intersection({"film", "movie", "song", "album", "book", "series"}):
        expected.add("creative_work")
    return expected


def _candidate_type_score(description: str, expected_types: set[str]) -> float:
    if not expected_types:
        return 0.55
    type_markers = {
        "event": _EVENT_DESCRIPTION_MARKERS,
        "policy": ("law", "act", "bill", "policy", "regulation", "legislation"),
        "organization": ("company", "organization", "organisation", "business"),
        "creative_work": ("film", "song", "album", "book", "series", "novel"),
    }
    matches = [
        any(marker in description for marker in type_markers.get(expected, ()))
        for expected in expected_types
    ]
    if any(matches):
        return 1.0
    if any(marker in description for marker in ("country", "city", "person", "company")):
        return 0.10
    return 0.35


def _candidate_corpus_score(
    candidate: dict[str, object],
    *,
    query: str,
    title_summary: pd.DataFrame | None,
) -> float:
    if not isinstance(title_summary, pd.DataFrame) or title_summary.empty:
        return 0.50
    seed_titles = [
        _normalize_for_match(row.get("page_title"))
        for row in title_summary.to_dict("records")
        if _query_title_match_strength(row.get("page_title"), [query]) >= 0.70
    ]
    if not seed_titles:
        return 0.25
    forms = _safe_match_phrases(
        [candidate.get("label", ""), *list(candidate.get("aliases", []))],
        allow_single_word=True,
    )
    if not forms:
        return 0.0
    exact_support = max(
        (
            sum(_contains_phrase(title, form) for title in seed_titles)
            / len(seed_titles)
            for form in forms
        ),
        default=0.0,
    )
    query_tokens = set(_normalize_for_match(query).split())
    distinctive_support = 0.0
    for form in forms:
        distinctive = {
            token
            for token in form.split()
            if token not in query_tokens and not token.isdigit() and len(token) >= 2
        }
        if not distinctive:
            continue
        supported = sum(
            distinctive.issubset(set(title.split())) for title in seed_titles
        ) / len(seed_titles)
        distinctive_support = max(distinctive_support, supported)
    return min(1.0, max(exact_support, distinctive_support))


def _count_query_seed_titles(
    title_summary: pd.DataFrame | None,
    query: str,
) -> int:
    if not isinstance(title_summary, pd.DataFrame) or title_summary.empty:
        return 0
    return sum(
        _query_title_match_strength(row.get("page_title"), [query]) >= 0.70
        for row in title_summary.to_dict("records")
    )


def _wikidata_search(session: requests.Session, query: str) -> list[dict[str, object]]:
    payload = _http_get_json(
        session,
        WIKIDATA_API_URL,
        params={
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "uselang": "en",
            "type": "item",
            "limit": 5,
            "format": "json",
            "formatversion": 2,
        },
    )
    results = payload.get("search", [])
    return [item for item in results if isinstance(item, dict)]


def _wikipedia_search(
    session: requests.Session,
    query: str,
) -> list[dict[str, object]]:
    payload = _http_get_json(
        session,
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 0,
            "srlimit": DEFAULT_MAX_WIKIPEDIA_CANDIDATES,
            "srprop": "snippet|titlesnippet",
            "format": "json",
            "formatversion": 2,
        },
    )
    raw_results = payload.get("query", {})
    results = raw_results.get("search", []) if isinstance(raw_results, dict) else []
    return [item for item in results if isinstance(item, dict)]


def _wikidata_entities_for_wikipedia_titles(
    session: requests.Session,
    titles: list[str],
) -> dict[str, dict[str, object]]:
    clean_titles = _deduplicate_text(titles)
    if not clean_titles:
        return {}
    entities: dict[str, dict[str, object]] = {}
    for start in range(0, len(clean_titles), 20):
        payload = _http_get_json(
            session,
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "sites": "enwiki",
                "titles": "|".join(clean_titles[start : start + 20]),
                "props": "labels|aliases|descriptions|sitelinks",
                "languages": "en",
                "languagefallback": 1,
                "format": "json",
                "formatversion": 2,
            },
        )
        raw_entities = payload.get("entities", {})
        if isinstance(raw_entities, dict):
            entities.update(
                {
                    str(key): value
                    for key, value in raw_entities.items()
                    if isinstance(value, dict) and not value.get("missing")
                }
            )
    return entities


def _wikidata_entities(
    session: requests.Session,
    qids: list[str],
    *,
    include_claims: bool = False,
    include_sitelinks: bool = False,
) -> dict[str, dict[str, object]]:
    normalized_qids = list(dict.fromkeys(_normalize_qid(qid) for qid in qids))
    normalized_qids = [qid for qid in normalized_qids if qid]
    if not normalized_qids:
        return {}
    props = ["labels", "aliases", "descriptions"]
    if include_claims:
        props.append("claims")
    if include_sitelinks:
        props.append("sitelinks")
    entities: dict[str, dict[str, object]] = {}
    for start in range(0, len(normalized_qids), 40):
        batch = normalized_qids[start : start + 40]
        payload = _http_get_json(
            session,
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "|".join(props),
                "languages": "en",
                "languagefallback": 1,
                "format": "json",
                "formatversion": 2,
            },
        )
        raw_entities = payload.get("entities", {})
        if isinstance(raw_entities, dict):
            entities.update(
                {
                    str(key): value
                    for key, value in raw_entities.items()
                    if isinstance(value, dict) and not value.get("missing")
                }
            )
    return entities


def _profile_from_wikidata_entity(
    session: requests.Session,
    qid: str,
    entity: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    label = _localized_value(entity.get("labels"), "en") or qid
    aliases = _localized_aliases(entity.get("aliases"), "en")
    description = _localized_value(entity.get("descriptions"), "en")
    sitelinks = entity.get("sitelinks", {})
    enwiki = sitelinks.get("enwiki", {}) if isinstance(sitelinks, dict) else {}
    wikipedia_title = (
        _clean_text(enwiki.get("title")) if isinstance(enwiki, dict) else ""
    )
    claims = entity.get("claims", {})
    if not isinstance(claims, dict):
        claims = {}

    statement_rows: list[
        tuple[str, WikidataPropertyRule, str, dict[str, object], str]
    ] = []
    related_qids: list[str] = []
    for property_id, rule in WIKIDATA_PROPERTY_RULES.items():
        property_specificity = WIKIDATA_PROPERTY_SPECIFICITY.get(
            property_id,
            "standalone" if rule.can_retrieve_standalone else "context_required",
        )
        raw_statements = claims.get(property_id, [])
        if not isinstance(raw_statements, list):
            continue
        accepted_for_property = 0
        for statement in raw_statements:
            if not isinstance(statement, dict) or statement.get("rank") == "deprecated":
                continue
            related_qid = _statement_entity_id(statement)
            if not related_qid:
                continue
            statement_rows.append(
                (property_id, rule, property_specificity, statement, related_qid)
            )
            related_qids.append(related_qid)
            accepted_for_property += 1
            if accepted_for_property >= 6:
                break

    related_entities = _wikidata_entities(session, related_qids)
    query_cues = _deduplicate_text([label, *aliases])[:20]
    relationships: list[dict[str, object]] = []
    for property_id, rule, property_specificity, statement, related_qid in statement_rows:
        related_entity = related_entities.get(related_qid, {})
        related_label = _localized_value(related_entity.get("labels"), "en")
        if not related_label or related_qid == qid:
            continue
        related_aliases = _localized_aliases(related_entity.get("aliases"), "en")[:12]
        temporal_scope, ended = _statement_temporal_scope(statement)
        qualifiers = statement.get("qualifiers", {})
        if not isinstance(qualifiers, dict):
            qualifiers = {}
        valid_from_values = _qualifier_times(qualifiers.get("P580"))
        valid_until_values = _qualifier_times(qualifiers.get("P582"))
        confidence = max(0.55, rule.confidence - (0.12 if ended else 0.0))
        evidence_urls = [
            f"https://www.wikidata.org/wiki/{qid}",
            *_statement_reference_urls(statement),
        ]
        relationship_key = f"{qid}|{property_id}|{related_qid}"
        can_retrieve_standalone = property_specificity == "standalone" and not ended
        relationship_class = (
            "CORE_RELATED"
            if property_specificity == "standalone"
            else "CONTEXTUAL"
        )
        relationships.append(
            {
                "relationship_id": "source:" + hashlib.sha1(
                    relationship_key.encode("utf-8")
                ).hexdigest()[:16],
                "interpretation_id": f"wikidata:{qid}",
                "related_subject": related_label,
                "aliases": related_aliases,
                "related_subject_id": related_qid,
                "related_subject_type": _localized_value(
                    related_entity.get("descriptions"), "en"
                ),
                "relationship_class": relationship_class,
                "relationship_family": rule.family,
                "relationship_role": rule.role,
                "property_specificity": property_specificity,
                "predicate_id": property_id,
                "predicate_label": rule.label,
                "factual_bridge": rule.bridge_template.format(
                    query=label,
                    related=related_label,
                ),
                "direction": "query_to_related",
                "temporal_scope": temporal_scope,
                "valid_from": valid_from_values[0] if valid_from_values else "",
                "valid_until": valid_until_values[0] if valid_until_values else "",
                "fact_as_of": datetime.now(timezone.utc).date().isoformat(),
                "durable_or_current": "historical" if ended else "current_or_durable",
                "can_retrieve_standalone": can_retrieve_standalone,
                "required_title_cues": query_cues,
                "source_names": ["Wikidata"],
                "evidence_urls": list(dict.fromkeys(evidence_urls)),
                "confidence": round(confidence, 3),
                "false_positive_risk": (
                    "low" if can_retrieve_standalone and confidence >= 0.86
                    else "medium"
                ),
                "explicit_or_derived": "EXPLICIT",
                "factual_status": "FORMER" if ended else "CURRENT",
                "editorial_persistence": (
                    "PERIOD_SPECIFIC" if ended else "ENDURING"
                ),
                "evidence_summary": (
                    f"Wikidata records {property_id} from {qid} to {related_qid}."
                ),
                "hop_count": 1,
                "relationship_path": [
                    {
                        "source_id": qid,
                        "source_label": label,
                        "predicate_id": property_id,
                        "predicate_label": rule.label,
                        "target_id": related_qid,
                        "target_label": related_label,
                        "statement_direction": "outgoing",
                    }
                ],
            }
        )

    resolved_entity = {
        "qid": qid,
        "label": label,
        "aliases": aliases,
        "description": description,
        "wikipedia_title": wikipedia_title,
        "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
        "wikipedia_url": "",
    }
    return resolved_entity, relationships


def _wikidata_incoming_relationships(
    session: requests.Session,
    resolved_entities: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load bounded inverse edges that cannot be found on the query item."""

    entity_by_qid = {
        _normalize_qid(entity.get("qid")): entity
        for entity in resolved_entities
        if _normalize_qid(entity.get("qid"))
    }
    if not entity_by_qid:
        return [], {"status": "skipped", "accepted_relationship_count": 0}

    property_rows = "\n".join(
        f"(wd:{property_id} wdt:{property_id})"
        for property_id in sorted(_INCOMING_WIKIDATA_PROPERTIES)
    )
    query_entities = " ".join(f"wd:{qid}" for qid in sorted(entity_by_qid))
    sparql = f"""
SELECT ?queryEntity ?propertyEntity ?subject ?subjectLabel ?subjectDescription WHERE {{
  VALUES ?queryEntity {{ {query_entities} }}
  VALUES (?propertyEntity ?predicate) {{
    {property_rows}
  }}
  ?subject ?predicate ?queryEntity .
  FILTER(?subject != ?queryEntity)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 80
""".strip()
    try:
        payload = _http_get_json(
            session,
            WIKIDATA_SPARQL_URL,
            params={"query": sparql, "format": "json"},
            max_attempts=1,
        )
    except Exception as exc:
        return [], {
            "status": "unavailable",
            "accepted_relationship_count": 0,
            "detail": str(exc),
        }

    results = payload.get("results", {})
    bindings = results.get("bindings", []) if isinstance(results, dict) else []
    if not isinstance(bindings, list):
        bindings = []
    incoming_subject_qids = [
        _qid_from_binding(binding.get("subject"))
        for binding in bindings
        if isinstance(binding, dict)
    ]
    try:
        incoming_subject_entities = _wikidata_entities(
            session,
            incoming_subject_qids[:80],
        )
    except Exception:
        incoming_subject_entities = {}
    relationships: list[dict[str, object]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        query_qid = _qid_from_binding(binding.get("queryEntity"))
        property_id = _qid_from_binding(binding.get("propertyEntity"), prefix="P")
        subject_qid = _qid_from_binding(binding.get("subject"))
        query_entity = entity_by_qid.get(query_qid)
        rule = WIKIDATA_PROPERTY_RULES.get(property_id)
        if not query_entity or not rule or not subject_qid or subject_qid == query_qid:
            continue
        subject_entity = incoming_subject_entities.get(subject_qid, {})
        subject = (
            _localized_value(subject_entity.get("labels"), "en")
            or _binding_value(binding.get("subjectLabel"))
        )
        if not subject or subject == subject_qid:
            continue
        description = (
            _localized_value(subject_entity.get("descriptions"), "en")
            or _binding_value(binding.get("subjectDescription"))
        )
        query_label = _clean_text(query_entity.get("label")) or query_qid
        specificity = WIKIDATA_PROPERTY_SPECIFICITY.get(
            property_id,
            "standalone" if rule.can_retrieve_standalone else "context_required",
        )
        standalone = specificity == "standalone"
        confidence = max(0.55, rule.confidence - 0.02)
        bridge = _INVERSE_BRIDGE_TEMPLATES.get(
            property_id,
            "{query} has a structured Wikidata relationship to {related}.",
        ).format(query=query_label, related=subject)
        relationship_key = f"incoming|{query_qid}|{property_id}|{subject_qid}"
        relationships.append(
            {
                "relationship_id": "wikidata-incoming:"
                + hashlib.sha1(relationship_key.encode("utf-8")).hexdigest()[:16],
                "interpretation_id": f"wikidata:{query_qid}",
                "related_subject": subject,
                "aliases": _localized_aliases(subject_entity.get("aliases"), "en")[:12],
                "related_subject_id": subject_qid,
                "related_subject_type": description,
                "relationship_class": "CORE_RELATED" if standalone else "CONTEXTUAL",
                "relationship_family": rule.family,
                "relationship_role": rule.role,
                "property_specificity": specificity,
                "predicate_id": property_id,
                "predicate_label": rule.label,
                "factual_bridge": bridge,
                "direction": "incoming_statement_to_query",
                "temporal_scope": "current_or_durable",
                "can_retrieve_standalone": standalone,
                "required_title_cues": [] if standalone else [query_label],
                "source_names": ["Wikidata"],
                "evidence_urls": [
                    f"https://www.wikidata.org/wiki/{subject_qid}",
                    f"https://www.wikidata.org/wiki/{query_qid}",
                ],
                "confidence": round(confidence, 3),
                "false_positive_risk": "low" if standalone else "medium",
                "explicit_or_derived": "EXPLICIT",
                "factual_status": "CURRENT",
                "editorial_persistence": "ENDURING",
                "evidence_summary": (
                    f"Wikidata records incoming {property_id} from "
                    f"{subject_qid} to {query_qid}."
                ),
                "hop_count": 1,
                "relationship_path": [
                    {
                        "source_id": query_qid,
                        "source_label": query_label,
                        "predicate_id": property_id,
                        "predicate_label": rule.label,
                        "statement_direction": "incoming",
                        "target_id": subject_qid,
                        "target_label": subject,
                    }
                ],
            }
        )
    relationships = _deduplicate_relationships(relationships)
    return relationships, {
        "status": "used" if relationships else "no_evidence",
        "binding_count": len(bindings),
        "accepted_relationship_count": len(relationships),
    }


def _binding_value(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return _clean_text(value.get("value"))


def _qid_from_binding(value: object, *, prefix: str = "Q") -> str:
    raw = _binding_value(value)
    candidate = raw.rsplit("/", 1)[-1].upper()
    if prefix == "P":
        return candidate if re.fullmatch(r"P\d+", candidate) else ""
    return _normalize_qid(candidate)


def _wikidata_path_relationships(
    session: requests.Session,
    *,
    resolved_entities: list[dict[str, object]],
    root_relationships: list[dict[str, object]],
    max_hops: int = DEFAULT_MAX_WIKIDATA_GRAPH_HOPS,
    max_relationships: int = DEFAULT_MAX_WIKIDATA_PATH_RELATIONSHIPS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Traverse only research-taxonomy paths with bounded breadth and depth."""

    roots = {
        _normalize_qid(entity.get("qid")): entity
        for entity in resolved_entities
        if _normalize_qid(entity.get("qid"))
    }
    root_qids = set(roots)
    if not roots or max_hops < 2:
        return [], {"expanded_node_count": 0, "accepted_relationship_count": 0}

    frontier: list[dict[str, object]] = []
    for relationship in root_relationships:
        target_qid = _normalize_qid(relationship.get("related_subject_id"))
        source_qid = _normalize_qid(
            _clean_text(relationship.get("interpretation_id")).removeprefix("wikidata:")
        )
        if not target_qid or source_qid not in roots:
            continue
        role = _clean_text(relationship.get("relationship_role")).upper()
        predicate_id = _clean_text(relationship.get("predicate_id")).upper()
        if (
            role not in {"IMMEDIATE_FAMILY", "INSTITUTIONAL_BRIDGE"}
            and predicate_id not in _PERSON_ORGANIZATION_PROPERTIES
        ):
            continue
        path = relationship.get("relationship_path")
        if not isinstance(path, list) or not path:
            path = [
                {
                    "source_id": source_qid,
                    "source_label": _clean_text(roots[source_qid].get("label")),
                    "predicate_id": predicate_id,
                    "predicate_label": _clean_text(relationship.get("predicate_label")),
                    "statement_direction": "outgoing",
                    "target_id": target_qid,
                    "target_label": _clean_text(relationship.get("related_subject")),
                }
            ]
        frontier.append(
            {
                "root_qid": source_qid,
                "root_label": _clean_text(roots[source_qid].get("label")),
                "current_qid": target_qid,
                "current_label": _clean_text(relationship.get("related_subject")),
                "confidence": _safe_float(relationship.get("confidence")),
                "origin_role": role,
                "path": list(path),
                "visited_qids": {source_qid, target_qid},
            }
        )

    discovered: list[dict[str, object]] = []
    expanded_qids: set[str] = set()
    raw_relationship_limit = max_relationships * 3
    for hop_count in range(2, max(2, int(max_hops)) + 1):
        if not frontier or len(discovered) >= raw_relationship_limit:
            break
        current_qids = list(
            dict.fromkeys(
                _normalize_qid(item.get("current_qid")) for item in frontier
            )
        )
        current_qids = [qid for qid in current_qids if qid]
        try:
            current_entities = _wikidata_entities(
                session,
                current_qids[:20],
                include_claims=True,
            )
        except Exception:
            break
        expanded_qids.update(current_entities)
        pending: list[tuple[dict[str, object], str, WikidataPropertyRule, str]] = []
        target_qids: list[str] = []
        for state in frontier:
            current_qid = _normalize_qid(state.get("current_qid"))
            entity = current_entities.get(current_qid, {})
            claims = entity.get("claims", {}) if isinstance(entity, dict) else {}
            if not isinstance(claims, dict):
                continue
            for property_id in _allowed_wikidata_path_properties(
                state,
                hop_count=hop_count,
            ):
                rule = WIKIDATA_PROPERTY_RULES[property_id]
                statements = claims.get(property_id, [])
                if not isinstance(statements, list):
                    continue
                accepted_for_property = 0
                for statement in statements:
                    if not isinstance(statement, dict) or statement.get("rank") == "deprecated":
                        continue
                    target_qid = _statement_entity_id(statement)
                    visited = state.get("visited_qids", set())
                    if (
                        not target_qid
                        or target_qid in root_qids
                        or (isinstance(visited, set) and target_qid in visited)
                    ):
                        continue
                    pending.append((state, property_id, rule, target_qid))
                    target_qids.append(target_qid)
                    accepted_for_property += 1
                    if accepted_for_property >= 4:
                        break
        if not pending:
            break
        try:
            target_entities = _wikidata_entities(session, target_qids[:80])
        except Exception:
            break

        next_frontier: list[dict[str, object]] = []
        for state, property_id, rule, target_qid in pending:
            if len(discovered) >= raw_relationship_limit:
                break
            target_entity = target_entities.get(target_qid, {})
            target_label = _localized_value(target_entity.get("labels"), "en")
            if not target_label:
                continue
            if not _wikidata_path_target_is_material(
                state,
                target_entity=target_entity,
                hop_count=hop_count,
            ):
                continue
            relationship, next_state = _build_wikidata_path_relationship(
                state=state,
                property_id=property_id,
                rule=rule,
                target_qid=target_qid,
                target_entity=target_entity,
                target_label=target_label,
                hop_count=hop_count,
            )
            discovered.append(relationship)
            next_frontier.append(next_state)
        frontier = next_frontier[:20]

    discovered = _deduplicate_relationships(discovered)[:max_relationships]
    return discovered, {
        "expanded_node_count": len(expanded_qids),
        "accepted_relationship_count": len(discovered),
        "maximum_hops": max_hops,
    }


def _allowed_wikidata_path_properties(
    state: dict[str, object],
    *,
    hop_count: int,
) -> tuple[str, ...]:
    origin_role = _clean_text(state.get("origin_role")).upper()
    path = state.get("path", [])
    last_predicate = ""
    if isinstance(path, list) and path and isinstance(path[-1], dict):
        last_predicate = _clean_text(path[-1].get("predicate_id")).upper()
    if hop_count == 2 and origin_role == "IMMEDIATE_FAMILY":
        return tuple(sorted(_PERSON_ORGANIZATION_PROPERTIES))
    if hop_count == 2 and (
        origin_role == "INSTITUTIONAL_BRIDGE"
        or last_predicate in _PERSON_ORGANIZATION_PROPERTIES
    ):
        return tuple(sorted(_CORPORATE_STRUCTURE_PROPERTIES))
    if (
        hop_count >= 3
        and origin_role == "INSTITUTIONAL_BRIDGE"
        and last_predicate in (
        _PERSON_ORGANIZATION_PROPERTIES | _CORPORATE_STRUCTURE_PROPERTIES
        )
    ):
        # A third hop is reserved for explicit corporate composition. It must
        # not fan out again through people, locations, industries, or events.
        return ("P355", "P527", "P749")
    return ()


def _wikidata_path_can_retrieve_standalone(
    state: dict[str, object],
    *,
    property_id: str,
    hop_count: int,
) -> bool:
    origin_role = _clean_text(state.get("origin_role")).upper()
    if hop_count == 2 and origin_role == "IMMEDIATE_FAMILY":
        # A relative's company, foundation, or team is a genuine path, but an
        # arbitrary endpoint-only story is too broad.  The title must retain an
        # intermediate family or corporate cue.
        return False
    if hop_count == 2 and origin_role == "INSTITUTIONAL_BRIDGE":
        return property_id in {"P355", "P527", "P127", "P749"}
    if hop_count == 3:
        # Three-hop endpoints are useful discovery routes, but endpoint-only
        # titles are too far from the query to qualify without a path cue.
        return False
    return False


def _wikidata_path_target_is_material(
    state: dict[str, object],
    *,
    target_entity: dict[str, object],
    hop_count: int,
) -> bool:
    if hop_count != 2 or _clean_text(state.get("origin_role")).upper() != "IMMEDIATE_FAMILY":
        return True
    description = _clean_text(
        _localized_value(target_entity.get("descriptions"), "en")
    ).casefold()
    return any(
        marker in description
        for marker in (
            "business", "company", "conglomerate", "corporation", "enterprise",
            "foundation", "franchise", "group", "institution", "network",
            "nonprofit", "organisation", "organization", "team",
        )
    )


def _build_wikidata_path_relationship(
    *,
    state: dict[str, object],
    property_id: str,
    rule: WikidataPropertyRule,
    target_qid: str,
    target_entity: dict[str, object],
    target_label: str,
    hop_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    current_label = _clean_text(state.get("current_label"))
    root_label = _clean_text(state.get("root_label"))
    second_bridge = rule.bridge_template.format(
        query=current_label,
        related=target_label,
    )
    previous_path = state.get("path", [])
    path = list(previous_path) if isinstance(previous_path, list) else []
    path.append(
        {
            "source_id": _normalize_qid(state.get("current_qid")),
            "source_label": current_label,
            "predicate_id": property_id,
            "predicate_label": rule.label,
            "statement_direction": "outgoing",
            "target_id": target_qid,
            "target_label": target_label,
        }
    )
    confidence = max(
        0.55,
        min(_safe_float(state.get("confidence")), rule.confidence)
        - 0.08 * (hop_count - 1),
    )
    standalone = _wikidata_path_can_retrieve_standalone(
        state,
        property_id=property_id,
        hop_count=hop_count,
    )
    origin_role = _clean_text(state.get("origin_role")).upper()
    strict_surface_standalone = (
        hop_count == 3
        and origin_role == "INSTITUTIONAL_BRIDGE"
        and property_id in {"P355", "P527"}
        and len(_normalize_surface(target_label).split()) >= 2
    )
    standalone = standalone or strict_surface_standalone
    family = (
        "family-linked organization"
        if origin_role == "IMMEDIATE_FAMILY"
        else "corporate and institutional network"
    )
    path_labels = [
        root_label,
        *[
            _clean_text(step.get("target_label"))
            for step in path
            if isinstance(step, dict) and _clean_text(step.get("target_label"))
        ],
    ]
    path_text = " -> ".join(path_labels)
    relationship_key = "|".join(
        [
            _normalize_qid(state.get("root_qid")),
            *[
                _clean_text(step.get("predicate_id"))
                + ":"
                + _normalize_qid(step.get("target_id"))
                for step in path
                if isinstance(step, dict)
            ],
        ]
    )
    aliases = _localized_aliases(target_entity.get("aliases"), "en")[:12]
    intermediate_cues = [
        _clean_text(step.get("target_label"))
        for step in path[:-1]
        if isinstance(step, dict) and _clean_text(step.get("target_label"))
    ]
    relationship = {
        "relationship_id": "wikidata-path:"
        + hashlib.sha1(relationship_key.encode("utf-8")).hexdigest()[:16],
        "interpretation_id": f"wikidata:{state.get('root_qid', '')}",
        "related_subject": target_label,
        "aliases": aliases,
        "related_subject_id": target_qid,
        "related_subject_type": _localized_value(
            target_entity.get("descriptions"), "en"
        ),
        "relationship_class": "CORE_RELATED" if standalone else "CONTEXTUAL",
        "relationship_family": family,
        "relationship_role": rule.role,
        "property_specificity": "standalone" if standalone else "context_required",
        "predicate_id": "wikidata:path:"
        + ">".join(
            _clean_text(step.get("predicate_id"))
            for step in path
            if isinstance(step, dict)
        ),
        "predicate_label": f"{hop_count}-hop {family}",
        "factual_bridge": f"{path_text}. {second_bridge}",
        "direction": "bounded_outgoing_path",
        "temporal_scope": "current_or_durable",
        "can_retrieve_standalone": standalone,
        "requires_exact_surface": strict_surface_standalone,
        "required_title_cues": [] if standalone else intermediate_cues[-2:],
        "source_names": ["Wikidata"],
        "evidence_urls": _deduplicate_text(
            f"https://www.wikidata.org/wiki/{step.get('source_id')}"
            for step in path
            if isinstance(step, dict) and step.get("source_id")
        ),
        "confidence": round(confidence, 3),
        "false_positive_risk": "medium" if standalone else "high",
        "explicit_or_derived": "DERIVED",
        "factual_status": "CURRENT",
        "editorial_persistence": "ENDURING",
        "evidence_summary": f"Bounded {hop_count}-hop Wikidata path: {path_text}.",
        "hop_count": hop_count,
        "relationship_path": path,
    }
    visited = state.get("visited_qids", set())
    next_state = {
        **state,
        "current_qid": target_qid,
        "current_label": target_label,
        "confidence": confidence,
        "path": path,
        "visited_qids": {
            *(visited if isinstance(visited, set) else set()),
            target_qid,
        },
    }
    return relationship, next_state


def _wikipedia_redirects(session: requests.Session, title: str) -> list[str]:
    payload = _http_get_json(
        session,
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "titles": title,
            "prop": "redirects",
            "rdnamespace": 0,
            "rdlimit": 100,
            "format": "json",
            "formatversion": 2,
        },
    )
    query = payload.get("query", {})
    pages = query.get("pages", []) if isinstance(query, dict) else []
    if not isinstance(pages, list) or not pages:
        return []
    redirects = pages[0].get("redirects", []) if isinstance(pages[0], dict) else []
    return _deduplicate_text(
        item.get("title", "") for item in redirects if isinstance(item, dict)
    )


def _wordnet_relationships(query: str) -> tuple[list[dict[str, object]], str]:
    try:
        import nltk
        from nltk.corpus import wordnet as wordnet
    except ImportError:
        return [], (
            "Unavailable: install the nltk package and download the WordNet corpus "
            "to enable local lexical relationships."
        )
    bundled_data_path = Path(__file__).resolve().parents[1] / "data" / "nltk_data"
    if bundled_data_path.exists() and str(bundled_data_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(bundled_data_path))
    normalized_query = _normalize_for_match(query)
    lookup = normalized_query.replace(" ", "_")
    try:
        synsets = wordnet.synsets(lookup)
    except LookupError:
        return [], (
            "Unavailable: the NLTK WordNet corpus is not installed. Run "
            "python -m nltk.downloader wordnet omw-1.4 once in the application environment."
        )
    lookup_groups: list[tuple[str, list[object], bool]] = [
        (normalized_query, list(synsets[:2]), True)
    ]
    if not synsets and len(normalized_query.split()) > 1:
        for token in normalized_query.split()[:4]:
            if token in _CORPUS_STOPWORDS or len(token) < 4:
                continue
            lookup_groups.append((token, list(wordnet.synsets(token)[:1]), False))

    relationships: list[dict[str, object]] = []
    seen: set[str] = set()
    component_routes = 0
    for component, component_synsets, exact_query_lookup in lookup_groups:
        for synset in component_synsets:
            definition = _clean_text(synset.definition())
            for lemma in synset.lemmas():
                subject = _clean_text(lemma.name().replace("_", " "))
                normalized = _normalize_for_match(subject)
                if not subject or normalized == component or normalized in seen:
                    continue
                seen.add(normalized)
                standalone = exact_query_lookup
                required_cues = [
                    token
                    for token in normalized_query.split()
                    if token != component
                ]
                confidence = 0.72 if standalone else 0.64
                relationships.append(
                    {
                        "relationship_id": "wordnet:" + hashlib.sha1(
                            f"{synset.name()}|{normalized}".encode("utf-8")
                        ).hexdigest()[:16],
                        "related_subject": subject,
                        "aliases": [],
                        "related_subject_id": synset.name(),
                        "related_subject_type": synset.lexname(),
                        "relationship_class": (
                            "CORE_RELATED" if standalone else "CONTEXTUAL"
                        ),
                        "relationship_family": "lexical synonym",
                        "relationship_role": "LEXICAL_SYNONYM",
                        "predicate_id": "wordnet:synonym",
                        "predicate_label": "synonym",
                        "factual_bridge": (
                            f"{subject} is a WordNet synonym of "
                            f"{query if standalone else component} in the sense: "
                            f"{definition}"
                        ),
                        "direction": "equivalent",
                        "temporal_scope": "durable",
                        "can_retrieve_standalone": standalone,
                        "required_title_cues": required_cues,
                        "source_names": ["WordNet"],
                        "evidence_urls": ["https://wordnet.princeton.edu/"],
                        "confidence": confidence,
                        "false_positive_risk": "medium" if standalone else "high",
                        "evidence_kind": "lexical",
                        "explicit_or_derived": "EXPLICIT",
                    }
                )
                if not standalone:
                    component_routes += 1
                if len(relationships) >= 10:
                    break
            if len(relationships) >= 10:
                break
        if len(relationships) >= 10:
            break
    detail = f"Loaded {len(relationships)} synonym relationships locally."
    if component_routes:
        detail += f" {component_routes} were context-bound compound-query expansions."
    return relationships, detail


def _http_get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object],
    max_attempts: int = 3,
) -> dict[str, object]:
    last_error: Exception | None = None
    headers = {
        "User-Agent": get_knowledge_source_user_agent(),
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    for attempt in range(max(1, int(max_attempts))):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=(3.05, 8.0),
            )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt + 1 < max_attempts:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        delay = min(8.0, max(0.0, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = 0.35 * (attempt + 1)
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("The source returned a non-object JSON response.")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(0.35 * (attempt + 1))
    raise KnowledgeSourceError(str(last_error or "The source request failed."))


def _http_get_text(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object],
    max_attempts: int = 3,
) -> str:
    last_error: Exception | None = None
    headers = {
        "User-Agent": get_knowledge_source_user_agent(),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Encoding": "gzip, deflate",
    }
    for attempt in range(max(1, int(max_attempts))):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=(3.05, 8.0),
            )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt + 1 < max_attempts:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        delay = min(8.0, max(0.0, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = 0.35 * (attempt + 1)
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(0.35 * (attempt + 1))
    raise KnowledgeSourceError(str(last_error or "The source request failed."))


def _statement_entity_id(statement: dict[str, object]) -> str:
    mainsnak = statement.get("mainsnak", {})
    if not isinstance(mainsnak, dict):
        return ""
    datavalue = mainsnak.get("datavalue", {})
    if not isinstance(datavalue, dict):
        return ""
    value = datavalue.get("value", {})
    if not isinstance(value, dict):
        return ""
    entity_id = value.get("id")
    if not entity_id and value.get("numeric-id") is not None:
        entity_id = f"Q{value['numeric-id']}"
    return _normalize_qid(entity_id)


def _statement_reference_urls(statement: dict[str, object]) -> list[str]:
    urls: list[str] = []
    references = statement.get("references", [])
    if not isinstance(references, list):
        return urls
    for reference in references:
        snaks = reference.get("snaks", {}) if isinstance(reference, dict) else {}
        url_snaks = snaks.get("P854", []) if isinstance(snaks, dict) else []
        if not isinstance(url_snaks, list):
            continue
        for snak in url_snaks:
            datavalue = snak.get("datavalue", {}) if isinstance(snak, dict) else {}
            value = datavalue.get("value") if isinstance(datavalue, dict) else None
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                urls.append(value.strip())
    return list(dict.fromkeys(urls))[:3]


def _statement_temporal_scope(statement: dict[str, object]) -> tuple[str, bool]:
    qualifiers = statement.get("qualifiers", {})
    if not isinstance(qualifiers, dict):
        return "unspecified", False
    starts = _qualifier_times(qualifiers.get("P580"))
    ends = _qualifier_times(qualifiers.get("P582"))
    points = _qualifier_times(qualifiers.get("P585"))
    parts = []
    if starts:
        parts.append(f"start {starts[0]}")
    if ends:
        parts.append(f"end {ends[0]}")
    if points:
        parts.append(f"point in time {points[0]}")
    ended = False
    if ends:
        try:
            ended = int(ends[0][:4]) < datetime.now(timezone.utc).year
        except (TypeError, ValueError):
            ended = False
    return ", ".join(parts) or "unspecified", ended


def _qualifier_times(raw_snaks: object) -> list[str]:
    if not isinstance(raw_snaks, list):
        return []
    values = []
    for snak in raw_snaks:
        datavalue = snak.get("datavalue", {}) if isinstance(snak, dict) else {}
        value = datavalue.get("value", {}) if isinstance(datavalue, dict) else {}
        raw_time = value.get("time", "") if isinstance(value, dict) else ""
        match = re.search(r"([+-]?\d{4})-(\d{2})-(\d{2})", str(raw_time))
        if match:
            values.append(f"{match.group(1).lstrip('+')}-{match.group(2)}-{match.group(3)}")
    return values


def _localized_value(raw: object, language: str) -> str:
    if not isinstance(raw, dict):
        return ""
    selected = raw.get(language, {})
    if isinstance(selected, dict):
        return _clean_text(selected.get("value"))
    return ""


def _localized_aliases(raw: object, language: str) -> list[str]:
    if not isinstance(raw, dict):
        return []
    selected = raw.get(language, [])
    if not isinstance(selected, list):
        return []
    return _deduplicate_text(
        item.get("value", "") for item in selected if isinstance(item, dict)
    )


def _title_corpus_fingerprint(title_summary: pd.DataFrame | None) -> str:
    if not isinstance(title_summary, pd.DataFrame) or title_summary.empty:
        return ""
    columns = [
        column
        for column in ("story_id", "page_title", "last_month")
        if column in title_summary.columns
    ]
    if not columns:
        return ""
    rows = title_summary[columns].fillna("").astype(str)
    if "story_id" in rows.columns:
        rows = rows.sort_values("story_id", kind="stable")
    hashed = pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()[:20]


def _corpus_cooccurrence_relationships(
    *,
    query: str,
    resolved_entity: dict[str, object],
    resolved_entities: list[dict[str, object]] | None = None,
    title_summary: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Mine bounded, recency-weighted title associations for a sparse profile."""
    aliases = resolved_entity.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    identity_phrases = _safe_match_phrases(
        [query, resolved_entity.get("label", ""), *aliases],
        allow_single_word=True,
    )
    identity_groups = _query_identity_groups(
        query,
        resolved_entities or ([resolved_entity] if resolved_entity else []),
    )
    identity_tokens = {
        token
        for phrase in identity_phrases
        for token in phrase.split()
        if token
    }
    rows: list[dict[str, object]] = []
    latest_month: pd.Timestamp | None = None
    for raw_row in title_summary.to_dict("records"):
        title = _clean_text(raw_row.get("page_title"))
        normalized_title = _normalize_for_match(title)
        if not title or not normalized_title:
            continue
        parsed_month = pd.to_datetime(raw_row.get("last_month"), errors="coerce")
        month = None if pd.isna(parsed_month) else pd.Timestamp(parsed_month)
        if month is not None and (latest_month is None or month > latest_month):
            latest_month = month
        rows.append(
            {
                "story_id": _clean_text(raw_row.get("story_id")),
                "title": title,
                "normalized_title": normalized_title,
                "month": month,
            }
        )
    seed_rows = [
        row
        for row in rows
        if _is_direct_query_title(
            row["normalized_title"],
            query=query,
            identity_groups=identity_groups,
        )
    ]
    exact_seed_count = sum(
        any(
            _contains_phrase(str(row["normalized_title"]), phrase)
            for phrase in identity_phrases
        )
        for row in seed_rows
    )
    diagnostics: dict[str, object] = {
        "corpus_document_count": len(rows),
        "query_seed_document_count": len(seed_rows),
        "exact_query_seed_document_count": exact_seed_count,
        "normalized_query_seed_document_count": len(seed_rows) - exact_seed_count,
        "candidate_phrase_count": 0,
        "selected_phrase_count": 0,
        "recency_reference_month": (
            latest_month.strftime("%Y-%m") if latest_month is not None else ""
        ),
    }
    if len(rows) < 2 or len(seed_rows) < 2 or not identity_phrases:
        return [], diagnostics

    seed_counts: Counter[str] = Counter()
    seed_weighted_support: Counter[str] = Counter()
    evidence_by_phrase: dict[str, list[dict[str, str]]] = {}
    for row in seed_rows:
        phrases = _corpus_title_candidate_phrases(
            str(row["normalized_title"]),
            identity_tokens=identity_tokens,
        )
        weight = _corpus_recency_weight(
            row.get("month"),
            latest_month,
        )
        for phrase in phrases:
            seed_counts[phrase] += 1
            seed_weighted_support[phrase] += weight
            evidence = evidence_by_phrase.setdefault(phrase, [])
            if len(evidence) < 5:
                evidence.append(
                    {
                        "story_id": str(row.get("story_id", "")),
                        "page_title": str(row.get("title", "")),
                    }
                )

    minimum_support = max(2, math.ceil(len(seed_rows) * 0.015))
    viable_phrases = {
        phrase
        for phrase, support in seed_counts.items()
        if support >= minimum_support
    }
    diagnostics["candidate_phrase_count"] = len(viable_phrases)
    if not viable_phrases:
        return [], diagnostics

    document_frequency: Counter[str] = Counter()
    for row in rows:
        row_phrases = _corpus_title_candidate_phrases(
            str(row["normalized_title"]),
            identity_tokens=set(),
        )
        for phrase in row_phrases.intersection(viable_phrases):
            document_frequency[phrase] += 1

    scored: list[dict[str, object]] = []
    corpus_size = len(rows)
    seed_size = len(seed_rows)
    for phrase in viable_phrases:
        support = int(seed_counts[phrase])
        corpus_frequency = int(document_frequency[phrase])
        if not corpus_frequency:
            continue
        frequency_ratio = corpus_frequency / corpus_size
        if frequency_ratio > 0.05:
            continue
        pmi = math.log2(
            (support * corpus_size) / (seed_size * corpus_frequency)
        )
        if pmi < 1.75:
            continue
        weighted_support = float(seed_weighted_support[phrase])
        recency_ratio = weighted_support / support
        association_precision = support / corpus_frequency
        token_count = len(phrase.split())
        high_specificity = (
            token_count >= 2
            and frequency_ratio <= 0.05
            and pmi >= 2.5
            and support >= minimum_support
            and association_precision >= 0.20
        ) or (
            token_count == 1
            and (len(phrase) >= 7 or phrase in _CORPUS_ACRONYMS)
            and frequency_ratio <= (
                0.015 if phrase in _CORPUS_ACRONYMS else 0.005
            )
            and pmi >= (2.5 if phrase in _CORPUS_ACRONYMS else 4.0)
            and support >= max(3, minimum_support)
            and association_precision >= 0.25
        )
        score = (
            pmi
            + 1.60 * math.log1p(support)
            + 0.60 * recency_ratio
            + 0.12 * max(0, token_count - 1)
        )
        scored.append(
            {
                "phrase": phrase,
                "support": support,
                "corpus_frequency": corpus_frequency,
                "pmi": pmi,
                "weighted_support": weighted_support,
                "recency_ratio": recency_ratio,
                "association_precision": association_precision,
                "high_specificity": high_specificity,
                "score": score,
            }
        )
    scored.sort(
        key=lambda item: (
            float(item["score"]),
            int(item["support"]),
            len(str(item["phrase"]).split()),
        ),
        reverse=True,
    )

    selected: list[dict[str, object]] = []
    for candidate in scored:
        phrase = str(candidate["phrase"])
        phrase_tokens = set(phrase.split())
        if any(
            phrase == str(existing["phrase"])
            or (
                phrase_tokens.issubset(set(str(existing["phrase"]).split()))
                or set(str(existing["phrase"]).split()).issubset(phrase_tokens)
            )
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= DEFAULT_MAX_CORPUS_RELATIONSHIPS:
            break

    query_label = (
        query
        if resolved_entities and len(resolved_entities) > 1
        else _clean_text(resolved_entity.get("label")) or query
    )
    relationships: list[dict[str, object]] = []
    for candidate in selected:
        phrase = str(candidate["phrase"])
        support = int(candidate["support"])
        corpus_frequency = int(candidate["corpus_frequency"])
        pmi = float(candidate["pmi"])
        weighted_support = float(candidate["weighted_support"])
        recency_ratio = float(candidate["recency_ratio"])
        confidence = min(
            0.86,
            0.60
            + 0.035 * min(6.0, pmi)
            + 0.018 * min(6.0, math.log1p(support))
            + 0.06 * recency_ratio,
        )
        relationships.append(
            {
                "relationship_id": "corpus:" + hashlib.sha1(
                    f"{_normalize_for_match(query_label)}|{phrase}".encode("utf-8")
                ).hexdigest()[:16],
                "related_subject": _display_corpus_phrase(phrase),
                "aliases": [],
                "related_subject_id": "",
                "related_subject_type": "local title-corpus association",
                "relationship_class": "CONTEXTUAL",
                "relationship_family": "current-events co-occurrence",
                "relationship_role": "CURRENT_EVENTS_ASSOCIATION",
                "predicate_id": "corpus:recency_cooccurrence",
                "predicate_label": "recency-weighted co-occurrence",
                "property_specificity": "context_required",
                "factual_bridge": (
                    f"{phrase} co-occurred with {query_label} in {support} local "
                    f"titles (PMI {pmi:.2f}; recency-weighted support "
                    f"{weighted_support:.2f})."
                ),
                "direction": "corpus_association",
                "temporal_scope": (
                    f"current through {latest_month.strftime('%Y-%m')}"
                    if latest_month is not None
                    else "current corpus snapshot"
                ),
                "can_retrieve_standalone": False,
                "required_title_cues": _deduplicate_text(
                    [query_label, query, *aliases]
                )[:20],
                "source_names": ["Local corpus"],
                "evidence_urls": [],
                "evidence_story_rows": evidence_by_phrase.get(phrase, []),
                "confidence": round(confidence, 3),
                "false_positive_risk": "medium",
                "evidence_kind": "current_events",
                "corpus_high_specificity": bool(
                    candidate["high_specificity"]
                ),
                "corpus_support": support,
                "corpus_document_frequency": corpus_frequency,
                "corpus_pmi": round(pmi, 3),
                "recency_weighted_support": round(weighted_support, 3),
                "corpus_association_precision": round(
                    float(candidate["association_precision"]), 3
                ),
            }
        )
    diagnostics["selected_phrase_count"] = len(relationships)
    diagnostics["minimum_seed_support"] = minimum_support
    return relationships, diagnostics


def _corpus_title_candidate_phrases(
    normalized_title: str,
    *,
    identity_tokens: set[str],
) -> set[str]:
    tokens = normalized_title.split()
    phrases: set[str] = set()
    for size in (1, 2, 3):
        for start in range(0, len(tokens) - size + 1):
            selected = tokens[start : start + size]
            if any(token in identity_tokens for token in selected):
                continue
            if any(token in _CORPUS_STOPWORDS for token in selected):
                continue
            if (
                selected[0] in _CORPUS_GENERIC_TERMS
                or selected[-1] in _CORPUS_GENERIC_TERMS
            ):
                continue
            if any(token.isdigit() for token in selected):
                continue
            if any(len(token) < 2 for token in selected):
                continue
            phrase = " ".join(selected)
            if phrase in _CORPUS_GENERIC_TERMS:
                continue
            if all(token in _CORPUS_GENERIC_TERMS for token in selected):
                continue
            if size == 1:
                if (
                    len(phrase) < 4
                    and phrase not in _CORPUS_ACRONYMS
                ) or phrase in _CORPUS_STOPWORDS:
                    continue
            elif all(
                token in _CORPUS_GENERIC_TERMS or token in _CORPUS_STOPWORDS
                for token in selected
            ):
                continue
            phrases.add(phrase)
    return phrases


def _corpus_recency_weight(
    month: object,
    latest_month: pd.Timestamp | None,
) -> float:
    if not isinstance(month, pd.Timestamp) or latest_month is None:
        return 0.5
    age_months = max(
        0,
        (latest_month.year - month.year) * 12 + latest_month.month - month.month,
    )
    return 0.5 ** (age_months / CORPUS_RECENCY_HALF_LIFE_MONTHS)


def _display_corpus_phrase(phrase: str) -> str:
    return " ".join(
        token.upper() if token in _CORPUS_ACRONYMS else token.title()
        for token in phrase.split()
    )


def _merge_relationship_sets(
    relationships: list[dict[str, object]],
    additions: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged = [dict(item) for item in relationships]
    subject_index: dict[str, int] = {}
    for index, relationship in enumerate(merged):
        values = [relationship.get("related_subject", "")]
        aliases = relationship.get("aliases", [])
        if isinstance(aliases, list):
            values.extend(aliases)
        for value in values:
            normalized = _normalize_for_match(value)
            if normalized:
                subject_index.setdefault(normalized, index)

    for addition in additions:
        normalized = _normalize_for_match(addition.get("related_subject", ""))
        target_index = subject_index.get(normalized)
        if target_index is None:
            subject_index[normalized] = len(merged)
            merged.append(dict(addition))
            continue
        current = merged[target_index]
        current["source_names"] = _deduplicate_text(
            [
                *list(current.get("source_names", [])),
                *list(addition.get("source_names", [])),
            ]
        )
        current["confidence"] = round(
            max(
                _safe_float(current.get("confidence")),
                _safe_float(addition.get("confidence")),
            ),
            3,
        )
        current["has_current_events_evidence"] = True
        current["corpus_high_specificity"] = bool(
            addition.get("corpus_high_specificity")
        )
        for field in (
            "corpus_support",
            "corpus_document_frequency",
            "corpus_pmi",
            "recency_weighted_support",
            "evidence_story_rows",
        ):
            current[field] = addition.get(field)
    return merged


def _deduplicate_relationships(
    relationships: list[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    index_by_key: dict[tuple[str, str], int] = {}
    for relationship in relationships:
        subject_key = _normalize_for_match(relationship.get("related_subject", ""))
        role = _clean_text(relationship.get("relationship_role")).upper()
        predicate_key = _clean_text(relationship.get("predicate_id")).casefold()
        if role == "IMMEDIATE_FAMILY":
            # The same family edge may be recorded twice in opposite Wikidata
            # directions (for example mother and inverse child).
            predicate_key = "immediate_family"
        elif predicate_key.startswith("wikidata:path:"):
            # Keep the best bounded path to an endpoint instead of presenting
            # the same organization repeatedly through several relatives.
            predicate_key = "wikidata_path"
        key = (predicate_key, subject_key)
        if not key[1]:
            continue
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(result)
            result.append(relationship)
            continue
        existing = result[existing_index]
        existing_rank = (
            existing.get("can_retrieve_standalone") is True,
            -max(1, int(_safe_float(existing.get("hop_count", 1)))),
            _safe_float(existing.get("confidence", 0)),
        )
        incoming_rank = (
            relationship.get("can_retrieve_standalone") is True,
            -max(1, int(_safe_float(relationship.get("hop_count", 1)))),
            _safe_float(relationship.get("confidence", 0)),
        )
        if incoming_rank > existing_rank:
            result[existing_index] = relationship
    result.sort(
        key=lambda item: (
            item.get("can_retrieve_standalone") is True,
            _safe_float(item.get("confidence", 0)),
            _clean_text(item.get("related_subject")),
        ),
        reverse=True,
    )
    return result


def _relationship_counts_toward_density(relationship: dict[str, object]) -> int:
    role = _clean_text(relationship.get("relationship_role")).upper()
    if role in {"GEOGRAPHIC_CONTEXT", "GENERIC_TOPIC", "LEXICAL_SYNONYM"}:
        return 0
    if relationship.get("evidence_kind") == "lexical":
        return 0
    return 1


def _research_taxonomy_coverage_audit(
    *,
    resolved_entities: list[dict[str, object]],
    relationships: list[dict[str, object]],
    graph_expansion: dict[str, object],
) -> dict[str, object]:
    """Report which research-prompt branches structured sources could support."""

    primary_type = _source_entity_type(resolved_entities, relationships)
    branches = _RESEARCH_TAXONOMY_BRANCHES.get(primary_type, {})
    covered_properties: set[str] = set()
    for relationship in relationships:
        predicate = _clean_text(relationship.get("predicate_id")).upper()
        covered_properties.update(re.findall(r"P\d+", predicate))
    branch_rows: list[dict[str, object]] = []
    for branch, properties in branches.items():
        matched = sorted(properties.intersection(covered_properties))
        branch_rows.append(
            {
                "relationship_family": branch,
                "status": "covered" if matched else "no_structured_evidence",
                "matched_property_ids": matched,
            }
        )
    current_event_sources = {
        source
        for relationship in relationships
        for source in relationship.get("source_names", [])
        if source in {"GDELT", "Local corpus"}
    }
    if primary_type == "person":
        branch_rows.append(
            {
                "relationship_family": "current developments",
                "status": "covered" if current_event_sources else "no_structured_evidence",
                "matched_sources": sorted(current_event_sources),
            }
        )
    return {
        "policy_source": "grounded research prompt relationship taxonomy",
        "primary_type": primary_type,
        "applicable_branch_count": len(branch_rows),
        "covered_branch_count": sum(
            item.get("status") == "covered" for item in branch_rows
        ),
        "branches": branch_rows,
        "wikidata_graph_expansion": graph_expansion,
    }


def _source_entity_type(
    resolved_entities: list[dict[str, object]],
    relationships: list[dict[str, object]],
) -> str:
    descriptions = " ".join(
        _clean_text(entity.get("description")).casefold()
        for entity in resolved_entities
    )
    predicates = {
        _clean_text(item.get("predicate_id")).upper(): _normalize_for_match(
            item.get("related_subject")
        )
        for item in relationships
    }
    if predicates.get("P31") == "human" or any(
        marker in descriptions
        for marker in (
            "person", "businessperson", "politician", "actor", "actress",
            "singer", "heiress", "entrepreneur", "executive", "athlete",
        )
    ):
        return "person"
    if any(
        marker in descriptions
        for marker in ("company", "corporation", "organization", "organisation", "business")
    ):
        return "organization"
    if any(marker in descriptions for marker in _EVENT_DESCRIPTION_MARKERS):
        return "event"
    if any(
        marker in descriptions
        for marker in ("city", "country", "state", "district", "place", "region")
    ):
        return "place"
    if any(
        marker in descriptions
        for marker in ("film", "song", "album", "book", "series", "novel")
    ):
        return "creative_work"
    return "other"


def _complete_relationship_contract(
    relationship: dict[str, object],
    *,
    query: str,
) -> dict[str, object]:
    """Apply the source-neutral relationship and retrieval contract.

    This mirrors the evidence/scope fields used by the grounded research flow,
    but every value here is derived from deterministic source metadata.
    """
    completed = dict(relationship)
    retrieval_policy = _relationship_retrieval_policy(completed)
    completed["retrieval_policy"] = retrieval_policy
    if retrieval_policy in {"metadata_only", "corroboration_only"}:
        completed["can_retrieve_standalone"] = False
    subject = _clean_text(completed.get("related_subject"))
    predicate = _clean_text(completed.get("predicate_label")) or "related subject"
    bridge = _clean_text(completed.get("factual_bridge"))
    standalone = retrieval_policy == "standalone"
    aliases = completed.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    cues = _deduplicate_text([subject, *aliases])[:12]
    required = completed.get("required_title_cues", [])
    if not isinstance(required, list):
        required = []
    completed.setdefault(
        "interpretation_id",
        "query:" + hashlib.sha1(_normalize_for_match(query).encode("utf-8")).hexdigest()[:12],
    )
    completed.setdefault("explicit_or_derived", "DERIVED")
    completed.setdefault("factual_status", "CURRENT")
    completed.setdefault(
        "editorial_persistence",
        "EVENT_TRIGGERED"
        if completed.get("evidence_kind") == "current_events"
        else "ENDURING",
    )
    completed.setdefault("durable_or_current", "current_or_durable")
    completed.setdefault("hop_count", 1)
    completed.setdefault(
        "evidence_summary",
        bridge or f"The configured source records {predicate} for {query}.",
    )
    confidence = max(0.0, min(1.0, _safe_float(completed.get("confidence"))))
    completed.setdefault("factual_confidence", round(confidence, 3))
    completed.setdefault("bridge_confidence", round(confidence, 3))
    completed.setdefault(
        "editorial_confidence",
        round(confidence if standalone else max(0.0, confidence - 0.08), 3),
    )
    completed.setdefault("positive_title_cues", cues)
    completed.setdefault("negative_title_cues", [])
    completed.setdefault(
        "acceptance_condition",
        (
            f"The title centrally identifies {subject}."
            if standalone
            else f"The title identifies {subject} and a separate cue establishing its {predicate} relationship to {query}."
        ),
    )
    completed.setdefault(
        "allowed_story_angles",
        [f"Coverage centrally expressing the {predicate} relationship."],
    )
    completed.setdefault(
        "excluded_story_angles",
        [f"Endpoint-only coverage of {subject} outside the relationship scope."],
    )
    completed.setdefault(
        "rejection_rule",
        (
            "Reject ambiguous aliases, namesakes, and titles where the related subject is incidental."
            if standalone
            else "Reject endpoint-only mentions and titles lacking a distinct query-specific context cue."
        ),
    )
    if not standalone and not required:
        completed["required_title_cues"] = _deduplicate_text([query])
    completed.setdefault("editorial_manifestations", [subject] if subject else [])
    surface_values = _deduplicate_text(
        [subject, *list(completed.get("aliases", []))]
    )
    completed["surface_forms"] = [
        {
            "text": value,
            "form_type": "canonical" if value == subject else "alias",
        }
        for value in surface_values
    ]
    return completed


def _safe_match_phrases(
    values: Collection[object],
    *,
    allow_single_word: bool,
) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = _normalize_for_match(value)
        compact = phrase.replace(" ", "")
        if not phrase or phrase in seen:
            continue
        if len(compact) < 3:
            continue
        if not allow_single_word and len(phrase.split()) < 2:
            continue
        seen.add(phrase)
        phrases.append(phrase)
    return phrases


def _safe_surface_phrases(values: Collection[object]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = _normalize_surface(value)
        if len(phrase.replace(" ", "")) < 3 or phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
    return phrases


def _relationship_match_phrases(relationship: dict[str, object]) -> list[str]:
    """Keep the canonical subject and only structurally specific aliases.

    A source alias can lose its distinguishing symbol during title
    normalization (for example, `` Watch`` becoming ``watch``). Single-word
    aliases are therefore retained only when they look like an acronym,
    camel-cased brand, or alphanumeric product name. The canonical subject is
    always eligible when it has at least three normalized characters.
    """
    phrases = _safe_match_phrases(
        [relationship.get("related_subject", "")],
        allow_single_word=True,
    )
    seen = set(phrases)
    aliases = relationship.get("aliases", [])
    if not isinstance(aliases, list):
        return phrases
    for raw_alias in aliases:
        cleaned = _clean_text(raw_alias)
        normalized = _normalize_for_match(cleaned)
        if not normalized or normalized in seen:
            continue
        if len(normalized.replace(" ", "")) < 3:
            continue
        if len(normalized.split()) == 1:
            ascii_compact = re.sub(r"[^A-Za-z0-9]", "", cleaned)
            has_digit = any(character.isdigit() for character in ascii_compact)
            is_acronym = (
                2 <= len(ascii_compact) <= 10 and ascii_compact.isupper()
            )
            has_internal_capital = any(
                character.isupper() for character in ascii_compact[1:]
            )
            if not (has_digit or is_acronym or has_internal_capital):
                continue
            if is_acronym and relationship.get("can_retrieve_standalone") is True:
                # Acronyms are highly polysemous in headlines (for example CWC
                # can denote a treaty, a cricket competition, or a water agency).
                # They may support a context-anchored route but cannot qualify a
                # standalone related story by themselves.
                continue
        seen.add(normalized)
        phrases.append(normalized)
    return phrases


def _contains_phrase(normalized_title: str, normalized_phrase: str) -> bool:
    return f" {normalized_phrase} " in f" {normalized_title} "


def _is_distinct_context_cue(cue: str, subject_phrases: list[str]) -> bool:
    """Require context evidence beyond a longer/shorter form of the subject."""
    cue_tokens = set(cue.split())
    if not cue_tokens:
        return False
    for subject in subject_phrases:
        subject_tokens = set(subject.split())
        if not subject_tokens:
            continue
        if cue_tokens.issubset(subject_tokens) or subject_tokens.issubset(cue_tokens):
            return False
    return True


def _query_title_match_strength(
    title: object,
    identity_values: Collection[object],
) -> float:
    normalized_title = _normalize_for_match(title)
    if not normalized_title:
        return 0.0
    title_tokens = normalized_title.split()
    best = 0.0
    for value in identity_values:
        phrase = _normalize_for_match(value)
        if not phrase:
            continue
        if _contains_phrase(normalized_title, phrase):
            best = max(best, 1.0)
            continue
        query_tokens = phrase.split()
        if len(query_tokens) < 2 or not set(query_tokens).issubset(set(title_tokens)):
            continue
        cursor = -1
        ordered_positions: list[int] = []
        for token in query_tokens:
            try:
                cursor = title_tokens.index(token, cursor + 1)
            except ValueError:
                ordered_positions = []
                break
            ordered_positions.append(cursor)
        if ordered_positions:
            span = ordered_positions[-1] - ordered_positions[0] + 1
            best = max(best, 0.88 if span <= len(query_tokens) + 4 else 0.78)
        else:
            best = max(best, 0.72)
    return best


def _normalize_surface(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _simple_match_lemma(token: str) -> str:
    if token in {"analysis", "business", "crisis", "news", "series", "species", "status"}:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4 and token[:-2].endswith(
        ("s", "x", "z", "ch", "sh")
    ):
        return token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "sis")):
        return token[:-1]
    return token


def _normalize_for_match(value: object) -> str:
    return " ".join(
        _simple_match_lemma(token) for token in _normalize_surface(value).split()
    )


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _deduplicate_text(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        normalized = cleaned.casefold()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
    return result


def _normalize_qid(value: object) -> str:
    candidate = _clean_text(value).upper()
    return candidate if re.fullmatch(r"Q[1-9]\d*", candidate) else ""


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_cached_profile(cache_path: Path, cache_key: str) -> dict[str, object] | None:
    if not cache_path.exists():
        return None
    connection = sqlite3.connect(cache_path, timeout=5)
    try:
        try:
            row = connection.execute(
                """
                SELECT profile_json, fetched_at
                FROM source_profiles
                WHERE cache_key = ? AND schema_version = ?
                """,
                (cache_key, SOURCE_PROFILE_SCHEMA_VERSION),
            ).fetchone()
        except sqlite3.Error:
            return None
    finally:
        connection.close()
    if row is None:
        return None
    try:
        fetched_at = datetime.fromisoformat(str(row[1]))
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_hours = (
            datetime.now(timezone.utc) - fetched_at.astimezone(timezone.utc)
        ).total_seconds() / 3600
        if age_hours > DEFAULT_SOURCE_PROFILE_MAX_AGE_HOURS:
            return None
        profile = json.loads(str(row[0]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(profile, dict):
        return None
    if profile.get("schema_version") != SOURCE_PROFILE_SCHEMA_VERSION:
        return None
    return profile


def _save_cached_profile(
    cache_path: Path,
    cache_key: str,
    query: str,
    profile: dict[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_at = _clean_text(profile.get("fetched_at")) or datetime.now(
        timezone.utc
    ).isoformat()
    connection = sqlite3.connect(cache_path, timeout=5)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS source_profiles (
                cache_key TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                query TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO source_profiles (
                cache_key, schema_version, query, profile_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                schema_version = excluded.schema_version,
                query = excluded.query,
                profile_json = excluded.profile_json,
                fetched_at = excluded.fetched_at
            """,
            (
                cache_key,
                SOURCE_PROFILE_SCHEMA_VERSION,
                query,
                json.dumps(profile, ensure_ascii=False, sort_keys=True),
                fetched_at,
            ),
        )
        connection.commit()
    finally:
        connection.close()
