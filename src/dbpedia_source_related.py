"""Source-backed, non-LLM related-title retrieval using DBpedia and DBpedia Spotlight.

External evidence defines the permitted relationship subjects. The local title
corpus is then matched deterministically against those subjects after all
refined-search story IDs have been removed.
"""

from __future__ import annotations

import os
import json
import re
from html import unescape
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Sequence
from urllib.parse import unquote

import pandas as pd
import requests

from src.data_processing import simple_query_tokens, simple_title_tokens


SOURCE_RELATED_PIPELINE_VERSION = "2026-09-07-dbpedia-context-v3"
DEFAULT_DBPEDIA_SPOTLIGHT_URL = "https://api.dbpedia-spotlight.org/en/annotate"
DEFAULT_DBPEDIA_SPARQL_URL = "https://dbpedia.org/sparql"
DEFAULT_DBPEDIA_LOOKUP_URL = "https://lookup.dbpedia.org/api/search"
DEFAULT_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

DBPEDIA = "http://dbpedia.org/resource/"
DBO = "http://dbpedia.org/ontology/"

# Concrete relationships only: broad types and categories cannot expand the graph.
ALLOWED_PREDICATES = {
    DBO + name: label for name, label in {
        "starring": "cast member", "director": "director", "producer": "producer",
        "writer": "writer", "author": "author", "musicComposer": "music by",
        "musicalArtist": "performer", "musicalBand": "performer", "artist": "performer",
        "lyrics": "lyricist", "album": "album", "soundtrack": "soundtrack",
        "partOf": "part of", "series": "series", "previousWork": "previous work",
        "subsequentWork": "subsequent work", "recordLabel": "record label",
        "distributor": "distributor", "publisher": "publisher", "basedOn": "based on",
        "spouse": "spouse", "child": "child", "parent": "parent", "relative": "relative",
        "birthPlace": "birthplace", "deathPlace": "place of death", "almaMater": "alumni of",
        "employer": "works for", "knownFor": "known for", "award": "award",
        "influencedBy": "influenced by", "influenced": "influenced",
        "associatedBand": "associated band", "associatedMusicalArtist": "associated artist",
        "formerBandMember": "former band member", "bandMember": "band member",
        "founder": "founder", "parentCompany": "parent company", "subsidiary": "subsidiary",
        "owner": "owned by", "location": "location", "headquarter": "headquarters",
        "country": "country", "capital": "capital", "leader": "leader",
        "team": "team", "manager": "manager", "coach": "coach",
    }.items()
}
INCOMING_LABELS = {
    DBO + name: label for name, label in {
        "starring": "work featuring", "director": "work directed by",
        "author": "work authored by", "writer": "work written by",
        "musicComposer": "work with music by", "lyrics": "work with lyrics by",
        "artist": "work performed by", "musicalArtist": "work performed by",
        "musicalBand": "work performed by",
    }.items()
}
ALLOWED_INCOMING_PREDICATES = frozenset(INCOMING_LABELS)
BRIDGE_PREDICATES = frozenset(DBO + name for name in ("album", "soundtrack", "partOf", "basedOn"))
SECOND_HOP_PREDICATES = frozenset(INCOMING_LABELS) | {DBO + "producer"}
HIGH_CONFIDENCE_PREDICATES = SECOND_HOP_PREDICATES | BRIDGE_PREDICATES


GENERIC_NEWS_WORDS = frozenset(
    {
        "amid",
        "breaking",
        "career",
        "celebrity",
        "could",
        "daily",
        "explained",
        "film",
        "internet",
        "latest",
        "live",
        "movie",
        "news",
        "report",
        "reports",
        "said",
        "says",
        "song",
        "special",
        "today",
        "update",
        "updates",
        "video",
        "viral",
        "watch",
        "what",
        "when",
        "where",
        "why",
    }
)

CURRENT_EVENT_TOPIC_TERMS = frozenset(
    {
        "ceasefire",
        "conflict",
        "crisis",
        "cyclone",
        "drought",
        "earthquake",
        "election",
        "flood",
        "gas",
        "inflation",
        "oil",
        "pandemic",
        "recession",
        "sanction",
        "tariff",
        "war",
    }
)

PERSON_RELATION_MARKERS = (
    "actor",
    "author",
    "cast member",
    "director",
    "lyricist",
    "music by",
    "performer",
)
PERSON_ROLE_TOKENS = frozenset(
    {"actor", "actress", "author", "composer", "director", "lyricist", "poet", "singer", "writer"}
)


class SourceRelatedError(RuntimeError):
    """Raised when the source pipeline cannot produce a trustworthy profile."""


@dataclass(frozen=True)
class LinkedEntity:
    uri: str
    anchor: str
    confidence: float
    begin_index: int = 0
    end_index: int = 0
    resolution_source: str = "DBpedia Spotlight"


@dataclass(frozen=True)
class EvidenceRoute:
    related_subject: str
    aliases: tuple[str, ...]
    relationship_type: str
    why_related: str
    source_name: str
    evidence_url: str
    confidence: int
    source_kind: str
    supporting_mentions: int = 1


@dataclass
class SourceProfile:
    query: str
    linked_entities: list[LinkedEntity] = field(default_factory=list)
    routes: list[EvidenceRoute] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSettings:
    spotlight_url: str
    dbpedia_sparql_url: str
    google_news_rss_url: str
    timeout_seconds: float
    user_agent: str
    lookup_url: str = DEFAULT_DBPEDIA_LOOKUP_URL


def load_source_settings() -> SourceSettings:
    return SourceSettings(
        spotlight_url=os.getenv(
            "DBPEDIA_SPOTLIGHT_URL", DEFAULT_DBPEDIA_SPOTLIGHT_URL
        ).strip(),
        dbpedia_sparql_url=os.getenv(
            "DBPEDIA_SPARQL_URL", DEFAULT_DBPEDIA_SPARQL_URL
        ).strip(),
        lookup_url=os.getenv("DBPEDIA_LOOKUP_URL", DEFAULT_DBPEDIA_LOOKUP_URL).strip(),
        google_news_rss_url=os.getenv(
            "GOOGLE_NEWS_RSS_URL", DEFAULT_GOOGLE_NEWS_RSS_URL
        ).strip(),
        timeout_seconds=max(
            2.0, float(os.getenv("SOURCE_RELATED_TIMEOUT_SECONDS", "15"))
        ),
        user_agent=os.getenv(
            "SOURCE_RELATED_USER_AGENT",
            "TrafficPredictionSourceResearch/1.0 (editorial research tool)",
        ).strip(),
    )


class DBpediaSpotlightClient:
    def __init__(
        self,
        settings: SourceSettings | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or load_source_settings()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept-Language": "en",
            }
        )

    def build_profile(self, query: str, *, max_news_items: int = 40) -> SourceProfile:
        normalized_query = " ".join(str(query).split())
        if not normalized_query:
            raise ValueError("A query is required.")
        profile = SourceProfile(query=normalized_query)
        spotlight_available = True

        try:
            profile.linked_entities = self.link_entities(normalized_query)
            profile.source_status["DBpedia Spotlight"] = (
                f"Available ({len(profile.linked_entities):,} linked entities)"
            )
        except (requests.RequestException, ValueError) as exc:
            spotlight_available = False
            profile.source_status["DBpedia Spotlight"] = "Unavailable"
            profile.diagnostics["DBpedia Spotlight"] = str(exc)
            profile.warnings.append(
                "DBpedia Spotlight is unavailable. Trying DBpedia Lookup for the query; "
                "news evidence will use headline keywords without Spotlight entity linking."
            )

        if not profile.linked_entities:
            try:
                profile.linked_entities = self.lookup_entities(normalized_query)
                profile.source_status["DBpedia Lookup"] = (
                    f"Available ({len(profile.linked_entities):,} matched entities)"
                )
            except (requests.RequestException, ValueError) as exc:
                profile.source_status["DBpedia Lookup"] = "Unavailable"
                profile.diagnostics["DBpedia Lookup"] = str(exc)
                profile.warnings.append("DBpedia Lookup is unavailable; trying DBpedia's exact English names.")

        dbpedia_query_entities = list(profile.linked_entities)
        if not dbpedia_query_entities:
            dbpedia_query_entities = [
                LinkedEntity(
                    uri="",
                    anchor=normalized_query,
                    confidence=0.70,
                )
            ]
            profile.warnings.append(
                "No unambiguous entity was resolved. DBpedia will try exact English names; "
                "a specific person, place, organization, or work title may give better results."
            )
        if dbpedia_query_entities:
            try:
                profile.routes.extend(
                    self._build_dbpedia_routes(dbpedia_query_entities)
                )
                profile.source_status["DBpedia"] = (
                    f"Available ({sum(route.source_kind.startswith('dbpedia') for route in profile.routes):,} routes)"
                )
                if not profile.routes:
                    profile.warnings.append("DBpedia returned no usable relationships for this query.")
            except (requests.RequestException, ValueError, KeyError) as exc:
                profile.source_status["DBpedia"] = "Unavailable"
                profile.warnings.append(f"DBpedia could not be used: {exc}")
        try:
            news_items = self.load_google_news(normalized_query, max_items=max_news_items)
            news_text, spans = join_news_headlines(news_items)
            news_entities: list[LinkedEntity] = []
            if news_text and spotlight_available:
                try:
                    news_entities = self.link_entities(news_text)
                except (requests.RequestException, ValueError) as exc:
                    profile.diagnostics["DBpedia Spotlight news"] = str(exc)
                    profile.warnings.append(
                        "Spotlight could not link the news headlines. News evidence uses headline keywords."
                    )
            profile.routes.extend(
                build_news_routes(
                    query=normalized_query,
                    query_entities=profile.linked_entities,
                    news_items=news_items,
                    headline_spans=spans,
                    news_entities=news_entities,
                )
            )
            profile.source_status["Google News"] = (
                f"Available ({len(news_items):,} sourced headlines)"
            )
        except (requests.RequestException, ValueError, ET.ParseError) as exc:
            profile.source_status["Google News"] = "Unavailable"
            profile.warnings.append(f"Google News could not be used: {exc}")

        profile.routes = deduplicate_routes(profile.routes)
        if not profile.routes:
            raise SourceRelatedError(
                "The configured sources returned no usable relationship evidence."
            )
        return profile

    def link_entities(self, text: str) -> list[LinkedEntity]:
        response = self.session.post(
            self.settings.spotlight_url,
            data={
                "text": text,
                "confidence": "0.35",
                "support": "0",
            },
            headers={"Accept": "application/json"},
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        return parse_spotlight_json(response.json())

    def lookup_entities(self, query: str) -> list[LinkedEntity]:
        response = self.session.get(
            self.settings.lookup_url,
            params={"query": query, "format": "JSON", "maxResults": 20},
            headers={"Accept": "application/json"},
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        return parse_lookup_json(response.json(), query)

    def load_google_news(self, query: str, *, max_items: int) -> list[dict[str, Any]]:
        response = self.session.get(
            self.settings.google_news_rss_url,
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items: list[dict[str, Any]] = []
        for item in root.findall(".//item")[: max(1, int(max_items))]:
            raw_title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            source_node = item.find("source")
            publisher = (
                (source_node.text or "").strip() if source_node is not None else ""
            )
            title = strip_publisher_suffix(raw_title, publisher)
            if title and link:
                items.append(
                    {
                        "title": title,
                        "url": link,
                        "publisher": publisher or "Publisher",
                        "published_at": parse_news_date(item.findtext("pubDate") or ""),
                    }
                )
        return items

    def _sparql(self, query: str) -> list[dict[str, Any]]:
        response = self.session.post(
            self.settings.dbpedia_sparql_url,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=max(self.settings.timeout_seconds, 30.0),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
            raise ValueError("DBpedia returned an unexpected SPARQL response.")
        bindings = payload["results"].get("bindings")
        if not isinstance(bindings, list):
            raise ValueError("DBpedia returned an unexpected SPARQL response.")
        return [binding for binding in bindings if isinstance(binding, dict)]

    def _build_dbpedia_routes(
        self, query_entities: Sequence[LinkedEntity]
    ) -> list[EvidenceRoute]:
        unresolved = [entity.anchor for entity in query_entities if not valid_dbpedia_uri(entity.uri)]
        label_mappings = self._map_labels_to_dbpedia(unresolved)
        routes: list[EvidenceRoute] = []
        bridge_routes: list[tuple[str, str, str, str]] = []
        seen: set[str] = set()
        for query_entity in query_entities:
            entities = (
                [(query_entity.uri, dbpedia_uri_label(query_entity.uri))]
                if valid_dbpedia_uri(query_entity.uri)
                else label_mappings.get(canonical_text(query_entity.anchor), [])
            )
            for dbpedia_uri, root_label in entities[:5]:
                if dbpedia_uri in seen:
                    continue
                seen.add(dbpedia_uri)
                facts = self._load_dbpedia_facts(dbpedia_uri)
                if not any(fact.get("predicate") in ALLOWED_PREDICATES for fact in facts):
                    routes.extend(self._build_context_routes(dbpedia_uri, root_label))
                for fact in facts:
                    predicate = fact.get("predicate", "")
                    if predicate not in ALLOWED_PREDICATES:
                        continue
                    related_uri = fact.get("related", "")
                    related_label = fact.get("related_label", "") or dbpedia_uri_label(
                        related_uri
                    )
                    if not useful_subject(related_label, query_entity.anchor):
                        continue
                    direction = fact.get("direction", "outgoing")
                    relation = relation_label(predicate, direction)
                    confidence = dbpedia_route_confidence(
                        predicate, direction, query_entity.confidence
                    )
                    subject_label = root_label or query_entity.anchor
                    why = dbpedia_reason(subject_label, related_label, relation, direction)
                    routes.append(
                        EvidenceRoute(
                            related_subject=related_label,
                            aliases=(),
                            relationship_type=f"DBpedia: {relation}",
                            why_related=why,
                            source_name="DBpedia",
                            evidence_url=dbpedia_uri,
                            confidence=confidence,
                            source_kind="dbpedia_direct",
                        )
                    )
                    if predicate in BRIDGE_PREDICATES and direction == "outgoing":
                        bridge_routes.append(
                            (subject_label, relation, related_uri, related_label)
                        )

        for root_label, bridge_relation, bridge_uri, bridge_label in bridge_routes[:12]:
            for fact in self._load_dbpedia_facts(bridge_uri):
                predicate = fact.get("predicate", "")
                if predicate not in SECOND_HOP_PREDICATES:
                    continue
                related_uri = fact.get("related", "")
                related_label = fact.get("related_label", "") or dbpedia_uri_label(
                    related_uri
                )
                if not useful_subject(related_label, root_label):
                    continue
                direction = fact.get("direction", "outgoing")
                relation = relation_label(predicate, direction)
                routes.append(
                    EvidenceRoute(
                        related_subject=related_label,
                        aliases=(),
                        relationship_type=(
                            f"DBpedia two-hop: {bridge_relation} → {relation}"
                        ),
                        why_related=(
                            f"DBpedia connects {root_label} to {bridge_label} through "
                            f"'{bridge_relation}', and connects {bridge_label} to "
                            f"{related_label} through '{relation}'."
                        ),
                        source_name="DBpedia",
                        evidence_url=bridge_uri,
                        confidence=max(
                            68,
                            dbpedia_route_confidence(predicate, direction, 0.9) - 10,
                        ),
                        source_kind="dbpedia_two_hop",
                    )
                )
        return routes

    def _build_context_routes(self, entity_uri: str, root_label: str) -> list[EvidenceRoute]:
        """Sparse entities can use reciprocal page references, never category expansion."""
        if not valid_dbpedia_uri(entity_uri):
            return []
        query = f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?related ?label WHERE {{
  <{entity_uri}> dbo:wikiPageWikiLink ?related .
  ?related dbo:wikiPageWikiLink <{entity_uri}> .
  ?related rdfs:label ?label .
  FILTER(LANG(?label) = "en")
  FILTER(?related != <{entity_uri}>)
}}
ORDER BY ?related ?label
LIMIT 100
"""
        names_by_uri: dict[str, list[str]] = defaultdict(list)
        for binding in self._sparql(query):
            related_uri = binding_value(binding, "related")
            label = binding_value(binding, "label")
            if valid_dbpedia_uri(related_uri) and useful_subject(label, root_label):
                names_by_uri[related_uri].append(label)
        return [
            EvidenceRoute(
                related_subject=labels[0],
                aliases=tuple(dict.fromkeys(labels[1:])),
                relationship_type="DBpedia: reciprocal page reference",
                why_related=(
                    f"DBpedia records links from the {root_label} page to {labels[0]} "
                    f"and back ({related_uri}). This is contextual association, "
                    "not a typed factual relationship."
                ),
                source_name="DBpedia",
                evidence_url=entity_uri,
                confidence=62,
                source_kind="dbpedia_context",
            )
            for related_uri, labels in names_by_uri.items()
        ]

    def _map_labels_to_dbpedia(
        self, labels: Sequence[str]
    ) -> dict[str, list[tuple[str, str]]]:
        clean_labels = [
            " ".join(str(label).split())
            for label in dict.fromkeys(labels)
            if str(label).strip()
        ][:20]
        if not clean_labels:
            return {}
        label_values = " ".join(
            f"{json.dumps(label, ensure_ascii=False)}@en" for label in clean_labels
        )
        sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?label WHERE {{
  VALUES ?label {{ {label_values} }}
  ?entity rdfs:label ?label .
}}
LIMIT 100
"""
        mapped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for binding in self._sparql(sparql):
            entity_uri = binding_value(binding, "entity")
            label = binding_value(binding, "label")
            if valid_dbpedia_uri(entity_uri) and label:
                mapped[canonical_text(label)].append((entity_uri, label))
        return {
            label: list(dict.fromkeys(matches))
            for label, matches in mapped.items()
            if len({uri for uri, _ in matches}) == 1
        }

    def _load_dbpedia_facts(self, entity_uri: str) -> list[dict[str, str]]:
        if not valid_dbpedia_uri(entity_uri):
            return []
        outgoing_values = " ".join(
            f"<{predicate}>" for predicate in ALLOWED_PREDICATES
        )
        incoming_values = " ".join(
            f"<{predicate}>" for predicate in ALLOWED_INCOMING_PREDICATES
        )
        sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?predicate ?related ?relatedLabel ?direction WHERE {{
  {{
    VALUES ?predicate {{ {outgoing_values} }}
    <{entity_uri}> ?predicate ?related .
    BIND("outgoing" AS ?direction)
  }} UNION {{
    VALUES ?predicate {{ {incoming_values} }}
    ?related ?predicate <{entity_uri}> .
    BIND("incoming" AS ?direction)
  }}
  FILTER(ISIRI(?related) && STRSTARTS(STR(?related), "http://dbpedia.org/resource/"))
  OPTIONAL {{ ?related rdfs:label ?relatedLabel . FILTER(LANG(?relatedLabel) = "en") }}
}}
LIMIT 750
"""
        return [
            {
                "predicate": binding_value(binding, "predicate"),
                "related": binding_value(binding, "related"),
                "related_label": binding_value(binding, "relatedLabel"),
                "direction": binding_value(binding, "direction") or "outgoing",
            }
            for binding in self._sparql(sparql)
        ]


def parse_lookup_json(payload: Any, query: str) -> list[LinkedEntity]:
    """Accept a unique name/alias match, not the search service's fuzzy top hit."""
    if not isinstance(payload, dict) or not isinstance(payload.get("docs"), list):
        raise ValueError("DBpedia Lookup returned an unexpected JSON response.")

    def normalized_name(value: str) -> str:
        plain = unescape(re.sub(r"</?b>", "", value, flags=re.I))
        return " ".join(re.findall(r"\w+", plain.casefold()))

    query_name = normalized_name(query)
    if not query_name:
        return []
    matched: dict[str, LinkedEntity] = {}
    for doc in payload["docs"]:
        if not isinstance(doc, dict):
            continue
        resources = doc.get("resource", [])
        if not isinstance(resources, list) or not resources or not isinstance(resources[0], str):
            continue
        uri = resources[0].replace("https://dbpedia.org/resource/", DBPEDIA, 1)
        if not valid_dbpedia_uri(uri):
            continue
        names = [dbpedia_uri_label(uri)]
        for field_name in ("label", "redirectlabel"):
            values = doc.get(field_name, [])
            if isinstance(values, list):
                names.extend(value for value in values if isinstance(value, str))
        if any(normalized_name(name) == query_name for name in names):
            matched[uri] = LinkedEntity(
                uri, query, 0.70, 0, len(query), resolution_source="DBpedia Lookup (name/alias match)"
            )
    # Search rank is not a disambiguation confidence. Do not expand homonyms.
    return list(matched.values()) if len(matched) == 1 else []


def valid_dbpedia_uri(uri: str) -> bool:
    if not re.fullmatch(r"http://dbpedia\.org/resource/[^<>\s\x00-\x20\"{}|^`\\]+", uri):
        return False
    local = unquote(uri[len(DBPEDIA):])
    return not local.startswith(("Category:", "Template:", "File:", "Help:", "Wikipedia:"))


def parse_spotlight_json(payload: Any) -> list[LinkedEntity]:
    """Read Spotlight annotations, retaining offsets for headline attribution."""
    if not isinstance(payload, dict):
        raise ValueError("DBpedia Spotlight returned an unexpected JSON response.")
    resources = payload.get("Resources", [])
    if not isinstance(resources, list):
        raise ValueError("DBpedia Spotlight returned invalid Resources.")
    unique: dict[tuple[str, int, int], LinkedEntity] = {}
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        uri = str(resource.get("@URI", "")).replace("https://dbpedia.org/resource/", DBPEDIA, 1)
        anchor = str(resource.get("@surfaceForm", ""))
        if not valid_dbpedia_uri(uri) or not anchor.strip():
            continue
        try:
            start = int(resource["@offset"])
            score = float(resource["@similarityScore"])
        except (KeyError, ValueError, TypeError):
            continue
        if start < 0 or not 0 <= score <= 1:
            continue
        entity = LinkedEntity(uri, anchor, score, start, start + len(anchor))
        key = (uri, entity.begin_index, entity.end_index)
        if key not in unique or score > unique[key].confidence:
            unique[key] = entity
    return sorted(unique.values(), key=lambda item: (item.begin_index, -item.confidence))


def join_news_headlines(
    items: Sequence[dict[str, Any]],
) -> tuple[str, list[tuple[int, int]]]:
    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for item in items:
        title = str(item.get("title", "")).strip()
        if not title:
            spans.append((cursor, cursor))
            continue
        if chunks:
            cursor += 1
        start = cursor
        chunks.append(title)
        cursor += len(title)
        spans.append((start, cursor))
    return "\n".join(chunks), spans


def build_news_routes(
    *,
    query: str,
    query_entities: Sequence[LinkedEntity],
    news_items: Sequence[dict[str, Any]],
    headline_spans: Sequence[tuple[int, int]],
    news_entities: Sequence[LinkedEntity],
) -> list[EvidenceRoute]:
    query_uris = {entity.uri for entity in query_entities}
    query_tokens = set(simple_query_tokens(query))
    mentions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for index, item in enumerate(news_items):
        headline = str(item.get("title", ""))
        span = headline_spans[index] if index < len(headline_spans) else (0, 0)
        headline_entities = [
            entity
            for entity in news_entities
            if span[0] <= entity.begin_index < span[1]
        ]
        headline_uris = {entity.uri for entity in headline_entities}
        lexical_query_match = bool(query_tokens) and query_tokens.issubset(
            set(simple_title_tokens(headline))
        )
        linked_query_match = bool(query_uris.intersection(headline_uris))
        if not lexical_query_match and not linked_query_match:
            continue

        for entity in headline_entities:
            if entity.uri in query_uris or canonical_text(entity.anchor) == canonical_text(query):
                continue
            mentions[(f"uri:{entity.uri}", entity.anchor)].append(item)

        for phrase in extract_news_topic_phrases(headline, query_tokens):
            mentions[(f"topic:{canonical_text(phrase)}", phrase)].append(item)

    routes: list[EvidenceRoute] = []
    for (subject_key, subject), supporting_items in mentions.items():
        distinct_publishers = {
            str(item.get("publisher", "")).strip().lower()
            for item in supporting_items
            if str(item.get("publisher", "")).strip()
        }
        is_linked_entity = subject_key.startswith("uri:")
        subject_tokens = simple_title_tokens(subject)
        if not subject_tokens:
            continue
        if not is_linked_entity:
            if len(subject_tokens) == 1 and len(distinct_publishers) < 2:
                continue
            if len(subject_tokens) == 2 and len(distinct_publishers) < 2:
                continue
        first = supporting_items[0]
        publisher = str(first.get("publisher", "Publisher"))
        mention_count = len({str(item.get("url", "")) for item in supporting_items})
        confidence = 64
        confidence += 10 if is_linked_entity else 0
        confidence += min(12, max(0, len(distinct_publishers) - 1) * 4)
        confidence += 6 if len(subject_tokens) >= 2 else 0
        confidence += news_recency_bonus(first.get("published_at"))
        confidence = min(92, confidence)
        entity_note = "DBpedia Spotlight-linked entity" if is_linked_entity else "news topic"
        routes.append(
            EvidenceRoute(
                related_subject=subject,
                aliases=(),
                relationship_type=f"Current news: {entity_note}",
                why_related=(
                    f"A current sourced headline about {query} also identifies {subject}: "
                    f"{first.get('title', '')}"
                ),
                source_name=f"Google News / {publisher}",
                evidence_url=str(first.get("url", "")),
                confidence=confidence,
                source_kind="news_entity" if is_linked_entity else "news_topic",
                supporting_mentions=max(1, mention_count),
            )
        )
    return routes


def extract_news_topic_phrases(headline: str, query_tokens: set[str]) -> set[str]:
    tokens = [
        token
        for token in simple_title_tokens(headline)
        if (
            token not in query_tokens
            and token not in GENERIC_NEWS_WORDS
            and len(token) >= 3
        )
    ]
    phrases: set[str] = {
        token for token in tokens if token in CURRENT_EVENT_TOPIC_TERMS
    }
    for width in (3, 2):
        for start in range(0, len(tokens) - width + 1):
            window = tokens[start : start + width]
            if not any(token.isdigit() for token in window):
                phrases.add(" ".join(window))
    return phrases


def build_source_related_results(
    *,
    title_summary: pd.DataFrame,
    story_months: pd.DataFrame,
    excluded_story_ids: Iterable[str],
    routes: Sequence[EvidenceRoute],
    identity_validator=None,
    relationship_validator=None,
    title_tokenizer=simple_title_tokens,
) -> pd.DataFrame:
    required = {"story_id", "page_title", "total_views"}
    missing = required.difference(title_summary.columns)
    if missing:
        raise ValueError(f"Title data is missing columns: {', '.join(sorted(missing))}")

    excluded = {str(story_id).strip() for story_id in excluded_story_ids}
    corpus = title_summary.copy()
    corpus["story_id"] = corpus["story_id"].astype(str)
    corpus = corpus.loc[~corpus["story_id"].isin(excluded)].copy()
    corpus["source_match_tokens"] = corpus["page_title"].apply(title_tokenizer)

    corpus_records = corpus.to_dict("records")
    token_postings: dict[str, set[int]] = defaultdict(set)
    for position, row in enumerate(corpus_records):
        for token in set(row["source_match_tokens"]):
            token_postings[token].add(position)

    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    name_match_ids: set[str] = set()
    for route in routes:
        route_terms = [route.related_subject, *route.aliases]
        normalized_terms = [
            (term, title_tokenizer(term))
            for term in route_terms
            if title_tokenizer(term)
        ]
        candidate_positions: set[int] = set()
        for _, term_tokens in normalized_terms:
            posting_sets = [token_postings.get(token, set()) for token in set(term_tokens)]
            if posting_sets and all(posting_sets):
                candidate_positions.update(set.intersection(*posting_sets))
        for position in candidate_positions:
            row = corpus_records[position]
            matched = best_term_match(
                row["source_match_tokens"],
                normalized_terms,
                primary_subject_required=route_requires_primary_subject(route),
            )
            if matched is None:
                continue
            name_match_ids.add(str(row["story_id"]))
            term, lexical_score = matched
            relationship = None
            if relationship_validator is not None:
                relationship = relationship_validator(route, row)
                if relationship is None:
                    continue
            identity = None
            if identity_validator is not None:
                identity = identity_validator(route, row)
                if identity is None:
                    continue
                term = identity["matched_name"]
            score = min(100, round(route.confidence * 0.74 + lexical_score * 0.26))
            if identity is not None:
                # Relationship confidence must never compensate for uncertain identity.
                score = identity["identity_score"]
            matches[str(row["story_id"])].append(
                {"route": route, "term": term, "score": score, "identity": identity,
                 "relationship": relationship}
            )

    peak_views = build_peak_view_lookup(story_months)
    corpus_by_id = corpus.set_index("story_id", drop=False)
    rows: list[dict[str, Any]] = []
    for story_id, story_matches in matches.items():
        ranked = sorted(
            story_matches,
            key=lambda item: (
                int(item["score"]),
                -getattr(item["route"], "hop_count", 1),
                int(item["route"].supporting_mentions),
                len(title_tokenizer(item["term"])),
            ),
            reverse=True,
        )
        best = ranked[0]
        best_route: EvidenceRoute = best["route"]
        sources = list(dict.fromkeys(item["route"].source_name for item in ranked))
        subjects = list(dict.fromkeys(item["route"].related_subject for item in ranked))
        source_kinds = {item["route"].source_kind for item in ranked}
        score = min(100, int(best["score"]) + min(6, 2 * (len(source_kinds) - 1)))
        title_row = corpus_by_id.loc[story_id]
        if isinstance(title_row, pd.DataFrame):
            title_row = title_row.iloc[0]
        peak = peak_views.get(story_id, {})
        rows.append(
            {
                "story_id": story_id,
                "page_title": str(title_row.get("page_title", "")),
                "total_views": safe_int(title_row.get("total_views", 0)),
                "highest_view_month": peak.get("highest_view_month", ""),
                "highest_views": peak.get("highest_views", 0),
                "relationship_type": best_route.relationship_type,
                "why_related": best_route.why_related,
                "evidence_source": "; ".join(sources[:3]),
                "evidence_url": best_route.evidence_url,
                "related_subject": "; ".join(subjects[:5]),
                "relevance_score": score,
                "supporting_evidence_count": len(ranked),
            }
        )
        if best.get("identity") is not None:
            rows[-1].update(best["identity"])
        if best.get("relationship") is not None:
            rows[-1].update(best["relationship"])
        origin_matches = [item for item in ranked if getattr(item["route"], "query_entity_qid", "")]
        if origin_matches:
            origins = {}
            connections = []
            for item in origin_matches:
                route = item["route"]
                origin = {"qid": route.query_entity_qid, "label": route.query_entity_label,
                          "description": route.query_entity_description}
                origins[origin["qid"]] = origin
                connections.append({
                    "query_entity": origin, "related_subject": route.related_subject,
                    "related_entity_qid": getattr(route, "target_qid", ""),
                    "relationship_path": getattr(route, "relationship_path", ""),
                    "path_entity_ids": list(getattr(route, "path_qids", ())),
                    "source_urls": list(getattr(route, "path_evidence_urls", ()) or (route.evidence_url,)),
                    "title_evidence": (item.get("relationship") or {}).get("relationship_title_evidence", ""),
                })
            rows[-1]["matched_query_entities"] = "; ".join(
                origin["label"] + (f" ({origin['description']})" if origin["description"] else "")
                for origin in origins.values())
            rows[-1]["matched_query_entity_ids"] = "; ".join(origins)
            rows[-1]["supporting_relationships"] = json.dumps(connections, ensure_ascii=False)

    columns = [
        "story_id",
        "page_title",
        "total_views",
        "highest_view_month",
        "highest_views",
        "relationship_type",
        "why_related",
        "evidence_source",
        "evidence_url",
        "related_subject",
        "relevance_score",
        "supporting_evidence_count",
    ]
    if identity_validator is not None:
        columns.extend(["matched_name", "related_entity_qid", "identity_reason", "identity_score"])
        columns.extend(["hop_count", "relationship_path", "path_entity_ids", "path_evidence_urls"])
    if identity_validator is not None or relationship_validator is not None:
        # Optional Wikidata article-policy evidence; other consumers retain
        # their existing schema when their validator does not supply it.
        for column in ("relationship_family", "can_retrieve_standalone", "relationship_title_evidence",
                       "acceptance_condition", "rejection_rule"):
            if any(column in row for row in rows):
                columns.append(column)
    for column in ("matched_query_entities", "matched_query_entity_ids", "supporting_relationships"):
        if any(column in row for row in rows):
            columns.append(column)
    results = pd.DataFrame(rows, columns=columns)
    results.attrs["name_match_candidate_count"] = len(name_match_ids)
    if results.empty:
        return results
    sort_columns = ["relevance_score", "supporting_evidence_count", "total_views", "story_id"]
    ascending = [False, False, False, True]
    if identity_validator is not None:
        sort_columns.insert(1, "hop_count")
        ascending.insert(1, True)
    return results.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def best_term_match(
    title_tokens: Sequence[str],
    terms: Sequence[tuple[str, list[str]]],
    *,
    primary_subject_required: bool = False,
) -> tuple[str, int] | None:
    title_set = set(title_tokens)
    found: list[tuple[str, int]] = []
    for term, tokens in terms:
        if not tokens or not set(tokens).issubset(title_set):
            continue
        if len(tokens) == 1:
            if len(tokens[0]) < 4 or tokens[0] in GENERIC_NEWS_WORDS:
                continue
            if primary_subject_required and not single_person_is_title_subject(
                title_tokens, tokens[0]
            ):
                continue
            score = 84
        else:
            score = 100 if contains_phrase(title_tokens, tokens) else 92
        found.append((term, score))
    return max(found, key=lambda item: (item[1], len(item[0]))) if found else None


def route_requires_primary_subject(route: EvidenceRoute) -> bool:
    relationship = route.relationship_type.lower()
    return any(marker in relationship for marker in PERSON_RELATION_MARKERS)


def single_person_is_title_subject(title_tokens: Sequence[str], person_token: str) -> bool:
    positions = [
        index for index, token in enumerate(title_tokens) if token == person_token
    ]
    for index in positions:
        if index == 0:
            return True
        if any(
            token in PERSON_ROLE_TOKENS for token in title_tokens[max(0, index - 2) : index]
        ):
            return True
    return False


def contains_phrase(title_tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    width = len(phrase_tokens)
    return any(
        list(title_tokens[start : start + width]) == list(phrase_tokens)
        for start in range(0, len(title_tokens) - width + 1)
    )


def build_peak_view_lookup(story_months: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {"story_id", "month", "views"}
    if story_months.empty or not required.issubset(story_months.columns):
        return {}
    monthly = story_months.copy()
    monthly["story_id"] = monthly["story_id"].astype(str)
    monthly["views"] = pd.to_numeric(monthly["views"], errors="coerce").fillna(0)
    peak_indices = monthly.groupby("story_id")["views"].idxmax()
    peaks = monthly.loc[peak_indices, ["story_id", "month", "views"]]
    result: dict[str, dict[str, Any]] = {}
    for row in peaks.itertuples(index=False):
        month = pd.to_datetime(row.month, errors="coerce")
        result[str(row.story_id)] = {
            "highest_view_month": month.strftime("%b %Y") if not pd.isna(month) else "",
            "highest_views": safe_int(row.views),
        }
    return result


def deduplicate_routes(routes: Sequence[EvidenceRoute]) -> list[EvidenceRoute]:
    best: dict[tuple[str, str, str], EvidenceRoute] = {}
    for route in routes:
        key = (
            canonical_text(route.related_subject),
            route.relationship_type,
            route.evidence_url,
        )
        existing = best.get(key)
        if existing is None or route.confidence > existing.confidence:
            best[key] = route
    return sorted(
        best.values(),
        key=lambda route: (route.confidence, route.supporting_mentions),
        reverse=True,
    )


def relation_label(predicate: str, direction: str) -> str:
    if direction == "incoming" and predicate in INCOMING_LABELS:
        return INCOMING_LABELS[predicate]
    label = ALLOWED_PREDICATES.get(predicate, uri_local_name(predicate))
    return f"incoming {label}" if direction == "incoming" else label


def dbpedia_reason(root: str, related: str, relation: str, direction: str) -> str:
    if direction == "incoming":
        return f"DBpedia connects {related} to {root} through '{relation}'."
    return f"DBpedia connects {root} to {related} through '{relation}'."


def dbpedia_route_confidence(predicate: str, direction: str, linker_score: float) -> int:
    base = 94 if predicate in HIGH_CONFIDENCE_PREDICATES else 84
    if direction == "incoming":
        base -= 3
    bounded_linker = max(0.0, min(1.0, float(linker_score)))
    return round(base * 0.8 + bounded_linker * 20)


def binding_value(binding: dict[str, Any], name: str) -> str:
    value = binding.get(name, {})
    return str(value.get("value", "")).strip() if isinstance(value, dict) else ""


def dbpedia_uri_label(uri: str) -> str:
    local = unquote(uri_local_name(uri))
    return local.replace("_", " ").strip()


def uri_local_name(uri: str) -> str:
    return str(uri).rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def useful_subject(label: str, root_label: str) -> bool:
    tokens = simple_title_tokens(label)
    return bool(tokens) and canonical_text(label) != canonical_text(root_label)


def canonical_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def strip_publisher_suffix(title: str, publisher: str) -> str:
    suffix = f" - {publisher}"
    return title[: -len(suffix)].strip() if publisher and title.endswith(suffix) else title


def parse_news_date(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def news_recency_bonus(value: object) -> int:
    if not isinstance(value, datetime):
        return 0
    published = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    days = max(0, (datetime.now(timezone.utc) - published).days)
    if days <= 7:
        return 8
    if days <= 30:
        return 4
    return 0


def safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0
