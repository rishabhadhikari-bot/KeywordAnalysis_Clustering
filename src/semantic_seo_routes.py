"""Typed, isolated Gemini route discovery for the Semantic SEO lab.

This module deliberately does not reuse the legacy free-form relationship profile.
Grounded research and controlled JSON generation are separate API calls because
Vertex does not support response schemas and Google Search in the same request.
Only schema-valid, independently corroborated profiles are returned or persisted.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from src.vertex_related import (
    VERTEX_SCOPE,
    VERTEX_SYSTEM_INSTRUCTION,
    _build_vertex_endpoint,
    _extract_grounding_search_queries,
    _extract_grounding_sources,
    _extract_vertex_text,
    _generation_config_for_model,
    _raise_for_vertex_error,
    _without_proxy_environment,
)


SEMANTIC_SEO_ROUTE_SCHEMA_VERSION = "2026-08-25-compact-grounded-routes-v7"
# Four complete, corroborated routes fit the controlled output budget reliably.
# The serving schema intentionally has no maxItems because Vertex expands nested
# array limits into too many decoder states; the application enforces this bound.
SEMANTIC_SEO_MAX_ROUTES = 4
SEMANTIC_SEO_MAX_OUTPUT_TOKENS = 8192
SEMANTIC_SEO_MIN_GROUNDING_SOURCES = 1


class SemanticSEORouteError(RuntimeError):
    """Raised when no complete, verified route profile can be produced."""


@dataclass(frozen=True)
class SemanticSEORouteProfile:
    keyword_query: str
    canonical_query: str
    primary_type: str
    research: dict[str, object]
    sources: list[dict[str, str]]
    web_search_queries: list[str]
    model_name: str
    created_at: str

    @property
    def relationship_count(self) -> int:
        profile = _profile_object(self.research)
        relationships = profile.get("relationship_map", [])
        return len(relationships) if isinstance(relationships, list) else 0


def discover_verified_semantic_routes(
    keyword_query: str,
    matched_titles: list[str],
    service_account_path: Path,
    *,
    location: str,
    model_name: str,
    max_routes: int = SEMANTIC_SEO_MAX_ROUTES,
) -> SemanticSEORouteProfile:
    """Discover a compact route set from two independently grounded memos."""
    query = " ".join(str(keyword_query).split())
    if not query:
        raise ValueError("A query is required for Semantic SEO route discovery.")

    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=[VERTEX_SCOPE],
    )
    project_id = str(credentials.project_id or "").strip()
    if not project_id:
        raise SemanticSEORouteError(
            "The service account file does not contain a project_id."
        )

    with _without_proxy_environment():
        refresh_session = requests.Session()
        refresh_session.trust_env = False
        credentials.refresh(Request(session=refresh_session))
        endpoint = _build_vertex_endpoint(project_id, location, model_name)
        token = str(credentials.token)

        with ThreadPoolExecutor(max_workers=2) as executor:
            discovery_future = executor.submit(
                _run_grounded_evidence_call,
                endpoint=endpoint,
                access_token=token,
                model_name=model_name,
                prompt=_discovery_evidence_prompt(query, matched_titles, max_routes),
                stage="discovery evidence",
            )
            verification_future = executor.submit(
                _run_grounded_evidence_call,
                endpoint=endpoint,
                access_token=token,
                model_name=model_name,
                prompt=_verification_evidence_prompt(query, matched_titles, max_routes),
                stage="verification evidence",
            )
            discovery = discovery_future.result()
            verification = verification_future.result()

        combined_sources = _deduplicate_sources(
            list(discovery["sources"]) + list(verification["sources"])
        )
        discovery_catalog, verification_catalog = _controlled_source_catalogs(
            discovery_sources=list(discovery["sources"]),
            verification_sources=list(verification["sources"]),
        )
        controlled_profile = _run_controlled_route_call(
            endpoint=endpoint,
            access_token=token,
            model_name=model_name,
            prompt=_controlled_route_prompt(
                query=query,
                matched_titles=matched_titles,
                discovery=discovery,
                verification=verification,
                max_routes=max_routes,
            ),
            max_routes=max_routes,
            discovery_source_ids=[
                item["source_id"] for item in discovery_catalog
            ],
            verification_source_ids=[
                item["source_id"] for item in verification_catalog
            ],
        )
        _resolve_controlled_source_ids(
            controlled_profile,
            discovery_sources=list(discovery["sources"]),
            verification_sources=list(verification["sources"]),
        )
        verified_profile = _validate_stage_payload(
            {"profile": controlled_profile, "sources": combined_sources},
            stage="controlled route synthesis",
            max_routes=max_routes,
        )
        _assign_stable_route_ids(verified_profile)
        verified_profile["relationship_map"] = _retain_dual_grounded_routes(
            verified_profile["relationship_map"],
            discovery_sources=list(discovery["sources"]),
            verification_sources=list(verification["sources"]),
        )

    relationships = verified_profile["relationship_map"]
    if not relationships:
        raise SemanticSEORouteError(
            "No route was supported by distinct sources from both grounded branches."
        )

    combined_queries = list(
        dict.fromkeys(
            list(discovery["web_search_queries"])
            + list(verification["web_search_queries"])
        )
    )
    now = datetime.now(timezone.utc).isoformat()
    research = {
        "research_text": json.dumps(verified_profile, ensure_ascii=False),
        "sources": combined_sources,
        "web_search_queries": combined_queries,
        "quality_warnings": [],
        "schema_version": SEMANTIC_SEO_ROUTE_SCHEMA_VERSION,
        "research_branch_count": 2,
    }
    resolution = verified_profile["query_resolution"]
    return SemanticSEORouteProfile(
        keyword_query=query,
        canonical_query=str(resolution["canonical_query"]),
        primary_type=str(resolution["primary_type"]),
        research=research,
        sources=combined_sources,
        web_search_queries=combined_queries,
        model_name=model_name,
        created_at=now,
    )


def save_semantic_route_profile(
    db_path: Path,
    profile: SemanticSEORouteProfile,
) -> str:
    """Atomically persist a profile only after full local validation."""
    profile_key = semantic_route_profile_key(
        profile.keyword_query,
        profile.model_name,
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_seo_route_profiles (
                profile_key TEXT PRIMARY KEY,
                normalized_query TEXT NOT NULL,
                canonical_query TEXT NOT NULL,
                primary_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        serialized = json.dumps(profile.research, ensure_ascii=False)
        connection.execute(
            """
            INSERT INTO semantic_seo_route_profiles (
                profile_key, normalized_query, canonical_query, primary_type,
                schema_version, model_name, profile_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_key) DO UPDATE SET
                canonical_query = excluded.canonical_query,
                primary_type = excluded.primary_type,
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (
                profile_key,
                _normalize_query(profile.keyword_query),
                profile.canonical_query,
                profile.primary_type,
                SEMANTIC_SEO_ROUTE_SCHEMA_VERSION,
                profile.model_name,
                serialized,
                profile.created_at,
                profile.created_at,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return profile_key


def load_semantic_route_profile(
    db_path: Path,
    keyword_query: str,
    model_name: str,
) -> SemanticSEORouteProfile | None:
    if not db_path.exists():
        return None
    profile_key = semantic_route_profile_key(keyword_query, model_name)
    try:
        connection = sqlite3.connect(db_path, timeout=5)
        try:
            row = connection.execute(
                """
                SELECT normalized_query, canonical_query, primary_type, model_name,
                       profile_json, updated_at
                FROM semantic_seo_route_profiles
                WHERE profile_key = ? AND schema_version = ?
                """,
                (profile_key, SEMANTIC_SEO_ROUTE_SCHEMA_VERSION),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        research = json.loads(row[4])
        _validate_cached_research(research)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return SemanticSEORouteProfile(
        keyword_query=str(row[0]),
        canonical_query=str(row[1]),
        primary_type=str(row[2]),
        research=research,
        sources=list(research.get("sources", [])),
        web_search_queries=list(research.get("web_search_queries", [])),
        model_name=str(row[3]),
        created_at=str(row[5]),
    )


def semantic_route_profile_key(keyword_query: str, model_name: str) -> str:
    value = (
        f"{SEMANTIC_SEO_ROUTE_SCHEMA_VERSION}|{_normalize_query(keyword_query)}|"
        f"{str(model_name).strip()}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_grounded_evidence_call(
    *,
    endpoint: str,
    access_token: str,
    model_name: str,
    prompt: str,
    stage: str,
    retry_on_missing_grounding: bool = True,
) -> dict[str, object]:
    session = requests.Session()
    session.trust_env = False
    response: requests.Response | None = None
    for attempt in range(2):
        try:
            response = session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "systemInstruction": {
                        "parts": [{"text": VERTEX_SYSTEM_INSTRUCTION}]
                    },
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "tools": [{"googleSearch": {}}],
                    "generationConfig": {
                        **_generation_config_for_model(
                            model_name=model_name,
                            max_output_tokens=4096,
                            legacy_temperature=0.0,
                            legacy_top_p=0.8,
                            thinking_budget=1024,
                            thinking_level="medium",
                        ),
                    },
                },
                timeout=120,
            )
        except requests.RequestException as exc:
            if attempt:
                raise SemanticSEORouteError(
                    f"Semantic SEO {stage} request failed: {exc}"
                ) from exc
            time.sleep(1)
            continue
        if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
            time.sleep(1)
            continue
        break
    if response is None:
        raise SemanticSEORouteError(
            f"Semantic SEO {stage} returned no HTTP response."
        )
    _raise_for_vertex_error(response)
    payload = response.json()
    candidate = (payload.get("candidates") or [{}])[0]
    finish_reason = str(candidate.get("finishReason", "")).strip().upper()
    if finish_reason in {"MAX_TOKENS", "SAFETY", "RECITATION", "BLOCKLIST"}:
        raise SemanticSEORouteError(
            f"Semantic SEO {stage} stopped before completing the evidence memo: "
            f"{finish_reason}."
        )
    text = _extract_vertex_text(payload)
    sources = _extract_grounding_sources(payload)
    queries = _extract_grounding_search_queries(payload)
    if (
        not text.strip()
        or len(sources) < SEMANTIC_SEO_MIN_GROUNDING_SOURCES
        or not queries
    ):
        if retry_on_missing_grounding:
            return _run_grounded_evidence_call(
                endpoint=endpoint,
                access_token=access_token,
                model_name=model_name,
                prompt=(
                    prompt
                    + "\n\nCRITICAL RETRY: The prior response did not return complete "
                    "grounding metadata. Invoke Google Search, open multiple source "
                    "pages, and base every factual claim on those pages."
                ),
                stage=stage,
                retry_on_missing_grounding=False,
            )
        metadata = candidate.get("groundingMetadata", {})
        metadata_keys = (
            sorted(str(key) for key in metadata)
            if isinstance(metadata, dict)
            else []
        )
        raise SemanticSEORouteError(
            f"Semantic SEO {stage} returned incomplete Google grounding metadata "
            f"(text={bool(text.strip())}, sources={len(sources)}, "
            f"queries={len(queries)}, metadata_keys={metadata_keys})."
        )
    return {
        "research_text": text,
        "sources": sources,
        "web_search_queries": queries,
    }


def _run_controlled_route_call(
    *,
    endpoint: str,
    access_token: str,
    model_name: str,
    prompt: str,
    max_routes: int,
    discovery_source_ids: list[str],
    verification_source_ids: list[str],
) -> dict[str, Any]:
    """Convert grounded memos into strict JSON without enabling Search."""
    session = requests.Session()
    session.trust_env = False
    response: requests.Response | None = None
    for attempt in range(2):
        try:
            response = session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "systemInstruction": {
                        "parts": [{"text": VERTEX_SYSTEM_INSTRUCTION}]
                    },
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        **_generation_config_for_model(
                            model_name=model_name,
                            max_output_tokens=SEMANTIC_SEO_MAX_OUTPUT_TOKENS,
                            legacy_temperature=0.0,
                            legacy_top_p=0.8,
                            thinking_budget=1024,
                            thinking_level="medium",
                        ),
                        "responseMimeType": "application/json",
                        "responseSchema": _route_response_schema(
                            max_routes,
                            discovery_source_ids=discovery_source_ids,
                            verification_source_ids=verification_source_ids,
                        ),
                    },
                },
                timeout=120,
            )
        except requests.RequestException as exc:
            if attempt:
                raise SemanticSEORouteError(
                    f"Semantic SEO controlled synthesis request failed: {exc}"
                ) from exc
            time.sleep(1)
            continue
        if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
            time.sleep(1)
            continue
        break
    if response is None:
        raise SemanticSEORouteError(
            "Semantic SEO controlled synthesis returned no HTTP response."
        )
    _raise_for_vertex_error(response)
    payload = response.json()
    candidate = (payload.get("candidates") or [{}])[0]
    finish_reason = str(candidate.get("finishReason", "")).strip().upper()
    if finish_reason in {"MAX_TOKENS", "SAFETY", "RECITATION", "BLOCKLIST"}:
        raise SemanticSEORouteError(
            "Semantic SEO controlled synthesis stopped before a complete profile: "
            f"{finish_reason}."
        )
    try:
        parsed = json.loads(_extract_vertex_text(payload))
    except json.JSONDecodeError as exc:
        raise SemanticSEORouteError(
            "Semantic SEO controlled synthesis violated its JSON contract."
        ) from exc
    if not isinstance(parsed, dict):
        raise SemanticSEORouteError(
            "Semantic SEO controlled synthesis returned a non-object response."
        )
    return parsed


def _validate_stage_payload(
    result: dict[str, object],
    *,
    stage: str,
    max_routes: int,
) -> dict[str, Any]:
    profile = result.get("profile")
    if not isinstance(profile, dict):
        raise SemanticSEORouteError(f"Semantic SEO {stage} profile is missing.")
    resolution = profile.get("query_resolution")
    if not isinstance(resolution, dict):
        raise SemanticSEORouteError(
            f"Semantic SEO {stage} query resolution is missing."
        )
    for field in ("canonical_query", "primary_type"):
        if not _clean_text(resolution.get(field)):
            raise SemanticSEORouteError(
                f"Semantic SEO {stage} query resolution lacks {field}."
            )

    relationships = profile.get("relationship_map")
    if not isinstance(relationships, list) or not relationships:
        raise SemanticSEORouteError(
            f"Semantic SEO {stage} returned no relationship routes."
        )
    if len(relationships) > max(1, int(max_routes)):
        raise SemanticSEORouteError(
            f"Semantic SEO {stage} exceeded the configured route limit."
        )
    source_urls = {
        _normalize_url(item.get("uri", ""))
        for item in result.get("sources", [])
        if isinstance(item, dict)
    }
    valid_relationships = []
    for relationship in relationships:
        validated = _validate_relationship(relationship, source_urls)
        if validated is not None:
            valid_relationships.append(validated)
    if not valid_relationships:
        raise SemanticSEORouteError(
            f"Semantic SEO {stage} produced no source-backed valid routes."
        )
    return {
        "query_resolution": {
            "canonical_query": _clean_text(resolution["canonical_query"]),
            "primary_type": _clean_text(resolution["primary_type"]),
        },
        "relationship_map": valid_relationships,
    }


def _validate_relationship(
    relationship: object,
    grounded_source_urls: set[str],
) -> dict[str, Any] | None:
    if not isinstance(relationship, dict):
        return None
    required_text = (
        "related_subject",
        "related_subject_type",
        "relationship_class",
        "relationship_family",
        "relationship_role",
        "factual_bridge",
        "direction",
        "durable_or_current",
        "evidence_summary",
        "acceptance_condition",
        "rejection_rule",
        "false_positive_risk",
    )
    if any(not _clean_text(relationship.get(field)) for field in required_text):
        return None
    relationship_class = _clean_text(relationship["relationship_class"]).upper()
    if relationship_class not in {"DIRECT", "CORE_RELATED", "CONTEXTUAL"}:
        return None
    if not isinstance(relationship.get("can_retrieve_standalone"), bool):
        return None
    required_lists = (
        "allowed_story_angles",
        "excluded_story_angles",
        "required_title_cues",
        "editorial_manifestations",
    )
    if any(
        not isinstance(relationship.get(field), list)
        or not any(_clean_text(item) for item in relationship[field])
        for field in required_lists
    ):
        return None
    try:
        confidence = float(relationship.get("confidence"))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    evidence_urls = relationship.get("evidence_source_urls", [])
    matched_urls = [
        _clean_text(url)
        for url in evidence_urls
        if _normalize_url(url) in grounded_source_urls
    ] if isinstance(evidence_urls, list) else []
    if not matched_urls:
        return None
    validated = dict(relationship)
    validated["relationship_class"] = relationship_class
    validated["confidence"] = confidence
    validated["evidence_source_urls"] = list(dict.fromkeys(matched_urls))
    for field in required_lists:
        validated[field] = [
            _clean_text(item) for item in relationship[field] if _clean_text(item)
        ]
    return validated


def _assign_stable_route_ids(profile: dict[str, Any]) -> None:
    for index, relationship in enumerate(profile["relationship_map"], start=1):
        relationship["relationship_id"] = f"route-{index:02d}"
        relationship["interpretation_id"] = "primary"


def _source_catalog(
    sources: list[dict[str, str]],
    branch_prefix: str,
    max_sources: int = 10,
) -> list[dict[str, str]]:
    return [
        {
            "source_id": f"{branch_prefix}{index}",
            "uri": str(source.get("uri", "")).strip(),
            "title": str(source.get("title", "")).strip(),
            "domain": str(source.get("domain", "")).strip(),
        }
        for index, source in enumerate(sources[: max(1, int(max_sources))], start=1)
        if isinstance(source, dict) and str(source.get("uri", "")).strip()
    ]


def _controlled_source_catalogs(
    *,
    discovery_sources: list[dict[str, str]],
    verification_sources: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    discovery_catalog = _source_catalog(discovery_sources, "D")
    discovery_urls = {
        _normalize_url(item["uri"]) for item in discovery_catalog
    }
    unique_verification_sources = [
        source
        for source in verification_sources
        if _normalize_url(source.get("uri", "")) not in discovery_urls
    ]
    verification_catalog = _source_catalog(
        unique_verification_sources or verification_sources,
        "V",
    )
    return discovery_catalog, verification_catalog


def _resolve_controlled_source_ids(
    profile: dict[str, Any],
    *,
    discovery_sources: list[dict[str, str]],
    verification_sources: list[dict[str, str]],
) -> None:
    discovery_catalog, verification_catalog = _controlled_source_catalogs(
        discovery_sources=discovery_sources,
        verification_sources=verification_sources,
    )
    catalogs = discovery_catalog + verification_catalog
    url_by_id = {
        item["source_id"].casefold(): item["uri"]
        for item in catalogs
    }
    relationships = profile.get("relationship_map", [])
    if not isinstance(relationships, list):
        return
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        source_ids = [
            relationship.get("discovery_source_id", ""),
            relationship.get("verification_source_id", ""),
        ]
        relationship["evidence_source_urls"] = list(
            dict.fromkeys(
                url_by_id[source_id]
                for raw_id in source_ids
                if (source_id := _clean_text(raw_id).casefold()) in url_by_id
            )
        )


def _retain_dual_grounded_routes(
    relationships: list[dict[str, Any]],
    *,
    discovery_sources: list[dict[str, str]],
    verification_sources: list[dict[str, str]],
) -> list[dict[str, Any]]:
    discovery_urls = {
        _normalize_url(item.get("uri", ""))
        for item in discovery_sources
        if isinstance(item, dict)
    }
    verification_urls = {
        _normalize_url(item.get("uri", ""))
        for item in verification_sources
        if isinstance(item, dict)
    }
    accepted = []
    for relationship in relationships:
        cited_urls = {
            _normalize_url(url): str(url)
            for url in relationship.get("evidence_source_urls", [])
            if _normalize_url(url)
        }
        discovery_matches = set(cited_urls) & discovery_urls
        verification_matches = set(cited_urls) & verification_urls
        if not discovery_matches or not verification_matches:
            continue
        corroborating_urls = discovery_matches | verification_matches
        if len(corroborating_urls) < 2:
            continue
        row = dict(relationship)
        row["evidence_source_urls"] = [
            cited_urls[url] for url in corroborating_urls
        ]
        row["verification_branch_count"] = 2
        row["verification_branches"] = [
            "grounded_discovery",
            "grounded_verification",
        ]
        accepted.append(row)
    return accepted


def _merge_verified_routes(
    discovery_profile: dict[str, Any],
    verification_profile: dict[str, Any],
    *,
    discovery_sources: list[dict[str, str]],
    verification_sources: list[dict[str, str]],
) -> dict[str, Any]:
    discovery_by_id = {
        str(item.get("relationship_id", "")): item
        for item in discovery_profile["relationship_map"]
    }
    verification_by_id = {
        str(item.get("relationship_id", "")): item
        for item in verification_profile["relationship_map"]
    }
    discovery_source_urls = {
        _normalize_url(item.get("uri", "")) for item in discovery_sources
    }
    verification_source_urls = {
        _normalize_url(item.get("uri", "")) for item in verification_sources
    }
    merged = []
    for relationship_id, discovered in discovery_by_id.items():
        verified = verification_by_id.get(relationship_id)
        if verified is None:
            continue
        if _route_signature(discovered) != _route_signature(verified):
            continue
        discovery_urls = {
            _normalize_url(url): str(url)
            for url in discovered.get("evidence_source_urls", [])
            if _normalize_url(url) in discovery_source_urls
        }
        verification_urls = {
            _normalize_url(url): str(url)
            for url in verified.get("evidence_source_urls", [])
            if _normalize_url(url) in verification_source_urls
        }
        if not discovery_urls or not verification_urls:
            continue
        accepted = dict(discovered)
        accepted["evidence_source_urls"] = list(
            dict.fromkeys(list(discovery_urls.values()) + list(verification_urls.values()))
        )
        accepted["verification_branch_count"] = 2
        accepted["verification_branches"] = ["discovery", "verification"]
        accepted["confidence"] = min(
            float(discovered["confidence"]),
            float(verified["confidence"]),
        )
        merged.append(accepted)
    return {
        "query_resolution": discovery_profile["query_resolution"],
        "relationship_map": merged,
    }


def _route_signature(relationship: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _clean_text(relationship.get(field)).casefold()
        for field in (
            "relationship_id",
            "relationship_class",
            "relationship_role",
            "direction",
        )
    )


def _validate_cached_research(research: object) -> None:
    if not isinstance(research, dict):
        raise ValueError("Cached Semantic SEO research is not an object.")
    if research.get("schema_version") != SEMANTIC_SEO_ROUTE_SCHEMA_VERSION:
        raise ValueError("Cached Semantic SEO schema version is stale.")
    profile = _profile_object(research)
    relationships = profile.get("relationship_map")
    if not isinstance(relationships, list) or not relationships:
        raise ValueError("Cached Semantic SEO profile has no relationships.")
    for relationship in relationships:
        if not isinstance(relationship, dict):
            raise ValueError("Cached Semantic SEO relationship is invalid.")
        if int(relationship.get("verification_branch_count", 0)) < 2:
            raise ValueError("Cached Semantic SEO relationship is not verified.")


def _profile_object(research: dict[str, object]) -> dict[str, Any]:
    raw = research.get("research_text", "")
    profile = json.loads(str(raw))
    if not isinstance(profile, dict):
        raise ValueError("Semantic SEO profile is not an object.")
    return profile


def _discovery_evidence_prompt(
    query: str,
    matched_titles: list[str],
    max_routes: int,
) -> str:
    title_context = [
        " ".join(str(title).split())[:240]
        for title in matched_titles
        if str(title).strip()
    ][:30]
    return f"""
Research source-backed indirect editorial relationships for the query below.
You MUST invoke Google Search before answering, even if the query seems familiar.
Verify current facts and open source pages; do not answer from model memory.
Return a concise evidence memo in prose; do not return JSON.

QUERY: {json.dumps(query, ensure_ascii=False)}
DIRECT TITLE CONTEXT: {json.dumps(title_context, ensure_ascii=False)}

Investigate at most {max(1, int(max_routes))} high-value routes that could retrieve
an article whose title does not contain the original query. Explain the specific
factual bridge, precise title cues, scope, and false-positive risks. Exclude broad
category overlap, endpoint-only associations, and unsupported claims.
""".strip()


def _verification_evidence_prompt(
    query: str,
    matched_titles: list[str],
    max_routes: int,
) -> str:
    title_context = [
        " ".join(str(title).split())[:240]
        for title in matched_titles
        if str(title).strip()
    ][:30]
    return f"""
Independently research whether strong indirect editorial routes exist for this
query. You MUST invoke Google Search before answering, open source pages, and use
fresh evidence rather than model memory. Be adversarial about weak, generic,
outdated, or merely co-occurring relationships. Return an evidence memo in prose;
do not return JSON.

QUERY: {json.dumps(query, ensure_ascii=False)}
DIRECT TITLE CONTEXT: {json.dumps(title_context, ensure_ascii=False)}

Assess no more than {max(1, int(max_routes))} candidate relationship families.
Focus on specific causal consequences, participants, named events, and durable
institutional bridges that could support a standalone title without the query.
Document counter-evidence and rejection conditions.
""".strip()


def _controlled_route_prompt(
    *,
    query: str,
    matched_titles: list[str],
    discovery: dict[str, object],
    verification: dict[str, object],
    max_routes: int,
) -> str:
    title_context = [
        " ".join(str(title).split())[:240]
        for title in matched_titles
        if str(title).strip()
    ][:30]
    discovery_sources, verification_sources = _controlled_source_catalogs(
        discovery_sources=list(discovery.get("sources", [])),
        verification_sources=list(verification.get("sources", [])),
    )
    return f"""
Create the controlled route profile using only the two independently grounded
evidence branches below. Do not use outside knowledge and do not invent URLs.

QUERY: {json.dumps(query, ensure_ascii=False)}
DIRECT TITLE CONTEXT: {json.dumps(title_context, ensure_ascii=False)}

DISCOVERY MEMO:
{str(discovery.get('research_text', '')).strip()}
DISCOVERY SOURCE CATALOG:
{json.dumps(discovery_sources, ensure_ascii=False, separators=(',', ':'))}

VERIFICATION MEMO:
{str(verification.get('research_text', '')).strip()}
VERIFICATION SOURCE CATALOG:
{json.dumps(verification_sources, ensure_ascii=False, separators=(',', ':'))}

Return at most {max(1, int(max_routes))} precise indirect routes. A route is
eligible only if it is supported by one source from each source catalog. Copy one
D source_id into discovery_source_id and one V source_id into
verification_source_id; do not copy or generate URLs. All required cue/angle
arrays must contain specific non-empty values, and confidence must be between 0
and 1. State exact acceptance/rejection conditions, direction, scope, and
false-positive risk. Omit generic topic overlap and weak or disputed routes. The
application resolves IDs to grounded URLs and rejects every route that fails.
""".strip()


def _route_response_schema(
    _max_routes: int,
    *,
    discovery_source_ids: list[str] | None = None,
    verification_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    string_array = {
        "type": "ARRAY",
        "minItems": 1,
        "items": {"type": "STRING"},
    }
    relationship_properties = {
        "relationship_id": {"type": "STRING"},
        "related_subject": {"type": "STRING"},
        "related_subject_type": {"type": "STRING"},
        "relationship_class": {
            "type": "STRING",
            "enum": ["DIRECT", "CORE_RELATED", "CONTEXTUAL"],
        },
        "relationship_family": {"type": "STRING"},
        "relationship_role": {
            "type": "STRING",
            "enum": [
                "IDENTITY_EQUIVALENT",
                "QUERY_SPECIFIC_MANIFESTATION",
                "NAMED_EVENT",
                "DIRECT_PARTICIPANT",
                "SPECIFIC_CONSEQUENCE",
                "INSTITUTIONAL_BRIDGE",
                "GEOGRAPHIC_CONTEXT",
                "GENERIC_TOPIC",
            ],
        },
        "factual_bridge": {"type": "STRING"},
        "direction": {"type": "STRING"},
        "durable_or_current": {"type": "STRING"},
        "evidence_summary": {"type": "STRING"},
        "discovery_source_id": {
            "type": "STRING",
            **(
                {"enum": list(dict.fromkeys(discovery_source_ids))}
                if discovery_source_ids
                else {}
            ),
        },
        "verification_source_id": {
            "type": "STRING",
            **(
                {"enum": list(dict.fromkeys(verification_source_ids))}
                if verification_source_ids
                else {}
            ),
        },
        "confidence": {"type": "NUMBER"},
        "false_positive_risk": {
            "type": "STRING",
            "enum": ["low", "medium", "high"],
        },
        "can_retrieve_standalone": {"type": "BOOLEAN"},
        "acceptance_condition": {"type": "STRING"},
        "allowed_story_angles": string_array,
        "excluded_story_angles": string_array,
        "required_title_cues": string_array,
        "rejection_rule": {"type": "STRING"},
        "editorial_manifestations": string_array,
    }
    return {
        "type": "OBJECT",
        "properties": {
            "query_resolution": {
                "type": "OBJECT",
                "properties": {
                    "canonical_query": {"type": "STRING"},
                    "primary_type": {"type": "STRING"},
                },
                "required": ["canonical_query", "primary_type"],
            },
            "relationship_map": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": relationship_properties,
                    "required": list(relationship_properties),
                },
            },
        },
        "required": ["query_resolution", "relationship_map"],
    }


def _deduplicate_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    seen = set()
    for source in sources:
        uri = _normalize_url(source.get("uri", ""))
        if not uri or uri in seen:
            continue
        seen.add(uri)
        result.append(dict(source))
    return result


def _normalize_query(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _normalize_url(value: object) -> str:
    return str(value or "").strip().rstrip("/").casefold()


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())
