from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import quote, unquote

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
CONCEPTNET_API_URL = "https://api.conceptnet.io/query"
CONCEPTNET_WEB_URL = "https://conceptnet.io"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC_TERMS = frozenset(
    {
        "about", "after", "again", "against", "amid", "among", "before",
        "could", "during", "from", "have", "into", "latest", "more", "news",
        "over", "report", "says", "story", "their", "there", "these", "this",
        "today", "under", "update", "what", "when", "where", "which", "with",
        "would", "year", "years",
    }
)

_CONCEPTNET_RULES: dict[str, tuple[str, bool, float]] = {
    "/r/Synonym": ("synonym", True, 0.72),
    "/r/RelatedTo": ("related concept", False, 0.60),
    "/r/IsA": ("type relationship", False, 0.62),
    "/r/HasContext": ("usage context", False, 0.60),
    "/r/PartOf": ("part relationship", False, 0.62),
    "/r/UsedFor": ("usage", False, 0.60),
    "/r/SymbolOf": ("symbolic relationship", False, 0.62),
    "/r/DefinedAs": ("definition", False, 0.64),
}


def fetch_conceptnet_relationships(
    *,
    query: str,
    fetch_json: Callable[[str, dict[str, object]], dict[str, object]],
    fetch_text: Callable[[str, dict[str, object]], str] | None = None,
    limit: int = 50,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load conservative ConceptNet edges from its English concept namespace.

    The namespace is explicit in diagnostics.  It is a lexical/context source,
    not proof that the user's Latin-script query is English.
    """

    normalized_query = _concept_slug(query)
    diagnostics: dict[str, object] = {
        "language_namespace": "en",
        "edge_count": 0,
        "accepted_relationship_count": 0,
        "retrieval_mode": "api",
        "canonical_form_count": 0,
    }
    if not normalized_query:
        return [], diagnostics
    query_uri = f"/c/en/{normalized_query}"
    bounded_limit = max(1, min(100, int(limit)))
    accepted_query_uris = {query_uri}
    try:
        edges = _conceptnet_api_edges(
            query_uri=query_uri,
            fetch_json=fetch_json,
            limit=bounded_limit,
        )
        canonical_uris = _conceptnet_form_roots(edges, query_uri)
        accepted_query_uris.update(canonical_uris)
        for canonical_uri in canonical_uris:
            edges.extend(
                _conceptnet_api_edges(
                    query_uri=canonical_uri,
                    fetch_json=fetch_json,
                    limit=bounded_limit,
                )
            )
    except Exception as exc:
        if fetch_text is None:
            raise
        diagnostics["retrieval_mode"] = "official_web_fallback"
        diagnostics["api_error"] = str(exc)
        edges = _conceptnet_web_edges(
            query_uri=query_uri,
            fetch_text=fetch_text,
            limit=bounded_limit,
        )
        canonical_uris = _conceptnet_form_roots(edges, query_uri)
        accepted_query_uris.update(canonical_uris)
        for canonical_uri in canonical_uris:
            edges.extend(
                _conceptnet_web_edges(
                    query_uri=canonical_uri,
                    fetch_text=fetch_text,
                    limit=bounded_limit,
                )
            )
    edges = _deduplicate_conceptnet_edges(edges)
    diagnostics["canonical_form_count"] = len(accepted_query_uris) - 1
    diagnostics["edge_count"] = len(edges)

    relationships: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for edge in sorted(
        (item for item in edges if isinstance(item, dict)),
        key=lambda item: float(item.get("weight", 0.0) or 0.0),
        reverse=True,
    ):
        relation = edge.get("rel", {})
        relation_id = str(relation.get("@id", "")) if isinstance(relation, dict) else ""
        rule = _CONCEPTNET_RULES.get(relation_id)
        weight = float(edge.get("weight", 0.0) or 0.0)
        if rule is None or weight < 1.0:
            continue
        start = edge.get("start", {})
        end = edge.get("end", {})
        if not isinstance(start, dict) or not isinstance(end, dict):
            continue
        start_id = str(start.get("@id", ""))
        end_id = str(end.get("@id", ""))
        matching_start_uri = next(
            (
                uri
                for uri in accepted_query_uris
                if start_id == uri or start_id.startswith(uri + "/")
            ),
            "",
        )
        matching_end_uri = next(
            (
                uri
                for uri in accepted_query_uris
                if end_id == uri or end_id.startswith(uri + "/")
            ),
            "",
        )
        if matching_start_uri:
            other = end
            direction = "query_to_related"
        elif matching_end_uri:
            other = start
            direction = "related_to_query"
        else:
            continue
        if str(other.get("language", "")) != "en":
            continue
        subject = _clean_label(other.get("label", ""))
        subject_key = " ".join(_TOKEN_RE.findall(subject.casefold()))
        if (
            not subject_key
            or subject_key == normalized_query.replace("_", " ")
            or subject_key in _GENERIC_TERMS
            or len(subject_key.replace(" ", "")) < 4
        ):
            continue
        dedupe_key = (relation_id, subject_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        family, standalone, base_confidence = rule
        confidence = min(0.78, base_confidence + 0.02 * math.log1p(weight))
        edge_id = str(edge.get("@id", ""))
        relationship_id = hashlib.sha1(
            f"{query_uri}|{relation_id}|{subject_key}".encode("utf-8")
        ).hexdigest()[:16]
        relationships.append(
            {
                "relationship_id": f"conceptnet:{relationship_id}",
                "related_subject": subject,
                "aliases": [],
                "related_subject_id": str(other.get("@id", "")),
                "related_subject_type": "ConceptNet concept",
                "relationship_class": "CORE_RELATED" if standalone else "CONTEXTUAL",
                "relationship_family": family,
                "relationship_role": "LEXICAL_SYNONYM" if standalone else "GENERIC_TOPIC",
                "predicate_id": relation_id,
                "predicate_label": family,
                "factual_bridge": _conceptnet_bridge(query, subject, family, direction),
                "direction": direction,
                "temporal_scope": "durable",
                "can_retrieve_standalone": standalone,
                "required_title_cues": [] if standalone else [query],
                "source_names": ["ConceptNet"],
                "source_family": "commonsense_lexical",
                "evidence_urls": [
                    f"https://conceptnet.io{quote(edge_id, safe='/[],:')}"
                ] if edge_id else ["https://conceptnet.io/"],
                "confidence": round(confidence, 3),
                "false_positive_risk": "medium" if standalone else "high",
                "evidence_kind": "lexical" if standalone else "commonsense",
                "explicit_or_derived": "EXPLICIT",
                "conceptnet_weight": weight,
            }
        )
        if len(relationships) >= 10:
            break
    diagnostics["accepted_relationship_count"] = len(relationships)
    return relationships, diagnostics


def _conceptnet_api_edges(
    *,
    query_uri: str,
    fetch_json: Callable[[str, dict[str, object]], dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    payload = fetch_json(
        CONCEPTNET_API_URL,
        {"node": query_uri, "limit": limit},
    )
    edges = payload.get("edges", [])
    if not isinstance(edges, list):
        return []
    return [item for item in edges if isinstance(item, dict)]


class _ConceptNetEdgeLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.edge_uris: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "a":
            return
        href = next((value for key, value in attrs if key == "href"), None)
        if href and href.startswith("/a/["):
            self.edge_uris.append(unquote(href))


def _conceptnet_web_edges(
    *,
    query_uri: str,
    fetch_text: Callable[[str, dict[str, object]], str],
    limit: int,
) -> list[dict[str, object]]:
    html = fetch_text(
        f"{CONCEPTNET_WEB_URL}{quote(query_uri, safe='/')}",
        {"limit": limit},
    )
    parser = _ConceptNetEdgeLinkParser()
    parser.feed(html)
    edges: list[dict[str, object]] = []
    for edge_uri in parser.edge_uris:
        edge = _conceptnet_edge_from_uri(edge_uri)
        if edge is not None:
            edges.append(edge)
    return edges


def _conceptnet_edge_from_uri(edge_uri: str) -> dict[str, object] | None:
    if not edge_uri.startswith("/a/[") or not edge_uri.endswith("]"):
        return None
    components = edge_uri[len("/a/[") : -1].split(",")
    if len(components) != 3:
        return None
    relation_id, start_id, end_id = (
        component.rstrip("/") for component in components
    )
    if not relation_id.startswith("/r/"):
        return None
    start = _conceptnet_node_from_uri(start_id)
    end = _conceptnet_node_from_uri(end_id)
    if start is None or end is None:
        return None
    return {
        "@id": edge_uri,
        "rel": {"@id": relation_id},
        "start": start,
        "end": end,
        # The HTML representation does not expose assertion weights. Its
        # displayed assertions have already passed ConceptNet's own ranking.
        "weight": 1.0,
    }


def _conceptnet_node_from_uri(node_uri: str) -> dict[str, str] | None:
    components = node_uri.split("/")
    if len(components) < 4 or components[1] != "c":
        return None
    language = components[2]
    label = unquote(components[3]).replace("_", " ").strip()
    if not language or not label:
        return None
    return {"@id": node_uri, "language": language, "label": label}


def _conceptnet_form_roots(
    edges: list[dict[str, object]],
    query_uri: str,
) -> list[str]:
    roots: list[str] = []
    for edge in edges:
        relation = edge.get("rel", {})
        if not isinstance(relation, dict) or relation.get("@id") != "/r/FormOf":
            continue
        start = edge.get("start", {})
        end = edge.get("end", {})
        if not isinstance(start, dict) or not isinstance(end, dict):
            continue
        start_id = str(start.get("@id", ""))
        end_id = str(end.get("@id", ""))
        if start_id == query_uri or start_id.startswith(query_uri + "/"):
            root = end_id
        elif end_id == query_uri or end_id.startswith(query_uri + "/"):
            root = start_id
        else:
            continue
        if root.startswith("/c/en/") and root != query_uri and root not in roots:
            roots.append(root)
        if len(roots) >= 2:
            break
    return roots


def _deduplicate_conceptnet_edges(
    edges: list[dict[str, object]],
) -> list[dict[str, object]]:
    deduplicated: list[dict[str, object]] = []
    seen: set[str] = set()
    for edge in edges:
        edge_id = str(edge.get("@id", ""))
        key = edge_id or repr(edge)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(edge)
    return deduplicated


def fetch_gdelt_relationships(
    *,
    query: str,
    fetch_json: Callable[[str, dict[str, object]], dict[str, object]],
    query_cues: list[str] | None = None,
    max_records: int = 100,
    timespan: str = "3months",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Mine source-diverse, repeated phrases from recent GDELT article titles."""

    payload = fetch_json(
        GDELT_DOC_API_URL,
        {
            "query": query,
            "mode": "artlist",
            "maxrecords": max(10, min(250, int(max_records))),
            "timespan": timespan,
            "sort": "datedesc",
            "format": "json",
        },
    )
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        articles = []
    query_tokens = set(_tokens(query))
    phrase_documents: Counter[str] = Counter()
    phrase_domains: dict[str, set[str]] = defaultdict(set)
    phrase_urls: dict[str, list[str]] = defaultdict(list)
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title", "")).strip()
        domain = str(article.get("domain", "")).strip().casefold()
        url = str(article.get("url", "")).strip()
        phrases = set(_article_phrases(title, excluded_tokens=query_tokens))
        for phrase in phrases:
            phrase_documents[phrase] += 1
            if domain:
                phrase_domains[phrase].add(domain)
            if url and len(phrase_urls[phrase]) < 3:
                phrase_urls[phrase].append(url)

    ranked_phrases = sorted(
        (
            phrase
            for phrase, count in phrase_documents.items()
            if count >= 2 and len(phrase_domains[phrase]) >= 2
        ),
        key=lambda phrase: (
            phrase_documents[phrase] * math.log1p(len(phrase_domains[phrase])),
            len(phrase.split()),
            len(phrase),
        ),
        reverse=True,
    )
    selected: list[str] = []
    for phrase in ranked_phrases:
        phrase_token_set = set(phrase.split())
        if any(
            phrase_token_set.issubset(set(existing.split()))
            or set(existing.split()).issubset(phrase_token_set)
            for existing in selected
        ):
            continue
        selected.append(phrase)
        if len(selected) >= 8:
            break

    now = datetime.now(timezone.utc)
    relationships: list[dict[str, object]] = []
    for phrase in selected:
        support = phrase_documents[phrase]
        domain_count = len(phrase_domains[phrase])
        confidence = min(0.76, 0.54 + 0.025 * support + 0.02 * domain_count)
        relationship_id = hashlib.sha1(
            f"{query.casefold()}|{phrase}|{now.date().isoformat()}".encode("utf-8")
        ).hexdigest()[:16]
        relationships.append(
            {
                "relationship_id": f"gdelt:{relationship_id}",
                "related_subject": " ".join(word.capitalize() for word in phrase.split()),
                "aliases": [],
                "related_subject_id": "",
                "related_subject_type": "recent cross-publisher title association",
                "relationship_class": "CONTEXTUAL",
                "relationship_family": "current-events co-occurrence",
                "relationship_role": "CURRENT_EVENTS_ASSOCIATION",
                "predicate_id": "gdelt:recent-title-cooccurrence",
                "predicate_label": "recently reported with",
                "factual_bridge": (
                    f"{phrase} repeatedly appeared in recent, source-diverse GDELT "
                    f"coverage matching {query}."
                ),
                "direction": "query_to_related",
                "temporal_scope": f"rolling {timespan}",
                "fact_as_of": now.date().isoformat(),
                "durable_or_current": "current",
                "can_retrieve_standalone": False,
                "required_title_cues": [
                    cue for cue in (query_cues or [query]) if str(cue).strip()
                ],
                "source_names": ["GDELT"],
                "source_family": "current_events",
                "evidence_urls": phrase_urls[phrase],
                "confidence": round(confidence, 3),
                "false_positive_risk": "medium",
                "evidence_kind": "current_events",
                "explicit_or_derived": "DERIVED",
                "factual_status": "CURRENT",
                "editorial_persistence": "VOLATILE",
                "gdelt_document_support": support,
                "gdelt_source_domain_count": domain_count,
            }
        )
    diagnostics = {
        "article_count": len(articles),
        "repeated_phrase_count": len(ranked_phrases),
        "accepted_relationship_count": len(relationships),
        "timespan": timespan,
        "minimum_document_support": 2,
        "minimum_source_domains": 2,
    }
    return relationships, diagnostics


def _article_phrases(title: str, *, excluded_tokens: set[str]) -> list[str]:
    tokens = [
        token
        for token in _tokens(title)
        if token not in _GENERIC_TERMS
        and token not in excluded_tokens
    ]
    phrases: list[str] = []
    for width in (3, 2, 1):
        for start in range(len(tokens) - width + 1):
            phrase_tokens = tokens[start : start + width]
            if any(token.isdigit() for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            if width == 1 and len(phrase) < 5:
                continue
            phrases.append(phrase)
    return phrases


def _tokens(value: object) -> list[str]:
    return _TOKEN_RE.findall(str(value or "").casefold())


def _concept_slug(value: object) -> str:
    return "_".join(_tokens(value))


def _clean_label(value: object) -> str:
    return " ".join(str(value or "").split())


def _conceptnet_bridge(query: str, subject: str, family: str, direction: str) -> str:
    if direction == "query_to_related":
        return f"ConceptNet records {subject} as a {family} of {query}."
    return f"ConceptNet records {query} as a {family} of {subject}."
