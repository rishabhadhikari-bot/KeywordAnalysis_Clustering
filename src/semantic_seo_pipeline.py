"""Pure orchestration helpers for the isolated Semantic SEO experiment."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.opensearch_refined import OpenSearchSettings, load_opensearch_settings
from src.opensearch_semantic import (
    OpenSearchSemanticClient,
    OpenSearchSemanticSettings,
)
from src.relationship_embeddings import RelationshipRoute
from src.semantic_seo_models import (
    SemanticSEOModelSettings,
    add_indexed_entity_compatibility,
    rerank_relationship_candidates,
    score_deberta_title_relevance,
)
SEMANTIC_SEO_PIPELINE_VERSION = "2026-08-25-title-ingestion-fast-query-v2"
DEFAULT_SEMANTIC_SEO_INDEX_ALIAS = "traffic-semantic-seo-lab"
DEFAULT_ROUTE_MINIMUM_SCORE = 0.68
DEFAULT_TOP_K_PER_ROUTE = 30
DEFAULT_MAX_RETRIEVAL_ROUTES = 30
DEFAULT_MAX_RERANK_CANDIDATES = 20
DEFAULT_MAX_NLI_CANDIDATES = 10
DEFAULT_MAX_GEMINI_CANDIDATES = 20


@dataclass(frozen=True)
class SemanticSEORunOptions:
    use_indexed_entities: bool = True
    use_reranker: bool = True
    use_deberta_title_relevance: bool = False
    maximum_rerank_candidates: int = DEFAULT_MAX_RERANK_CANDIDATES
    maximum_nli_candidates: int = DEFAULT_MAX_NLI_CANDIDATES
    maximum_gemini_candidates: int = DEFAULT_MAX_GEMINI_CANDIDATES


def build_semantic_seo_client(
    model_settings: SemanticSEOModelSettings,
) -> OpenSearchSemanticClient:
    base = load_opensearch_settings()
    connection = OpenSearchSettings(
        url=base.url,
        index_alias=_safe_index_alias(
            os.getenv("SEMANTIC_SEO_OPENSEARCH_INDEX", DEFAULT_SEMANTIC_SEO_INDEX_ALIAS)
        ),
        username=base.username,
        password=base.password,
        verify_certs=base.verify_certs,
        timeout_seconds=base.timeout_seconds,
        auto_start=base.auto_start,
        home=base.home,
        startup_timeout_seconds=base.startup_timeout_seconds,
    )
    settings = OpenSearchSemanticSettings(
        connection=connection,
        embedding_model=model_settings.embedding_model,
        minimum_score=DEFAULT_ROUTE_MINIMUM_SCORE,
        max_results=10_000,
        entity_model=model_settings.gliner_model,
        enable_title_entities=True,
    )
    return OpenSearchSemanticClient(settings)


def run_local_semantic_seo_stages(
    *,
    keyword_query: str,
    routes: list[RelationshipRoute],
    client: OpenSearchSemanticClient,
    excluded_story_ids: set[str],
    title_summary: pd.DataFrame,
    model_settings: SemanticSEOModelSettings,
    options: SemanticSEORunOptions,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    stage_started = time.perf_counter()
    candidates, retrieval = client.search_relationship_routes(
        routes,
        excluded_story_ids=excluded_story_ids,
        minimum_score=DEFAULT_ROUTE_MINIMUM_SCORE,
        top_k_per_route=DEFAULT_TOP_K_PER_ROUTE,
        max_routes=DEFAULT_MAX_RETRIEVAL_ROUTES,
    )
    candidates = attach_available_article_evidence(candidates, title_summary)
    diagnostics: list[dict[str, object]] = [{
        "stage": "bge_m3_retrieval",
        "applied": True,
        "duration_ms": round((time.perf_counter() - stage_started) * 1000, 1),
        **retrieval,
    }]

    if options.use_indexed_entities:
        stage_started = time.perf_counter()
        candidates, diagnostic = add_indexed_entity_compatibility(
            candidates,
            max_candidates=options.maximum_rerank_candidates,
        )
        diagnostic["duration_ms"] = round(
            (time.perf_counter() - stage_started) * 1000,
            1,
        )
        diagnostics.append(diagnostic)

    if options.use_reranker:
        stage_started = time.perf_counter()
        try:
            candidates, diagnostic = rerank_relationship_candidates(
                keyword_query,
                candidates,
                model_name=model_settings.reranker_model,
                max_candidates=options.maximum_rerank_candidates,
            )
        except RuntimeError as exc:
            diagnostic = {
                "stage": "bge_reranker",
                "applied": False,
                "processed": 0,
                "reason": str(exc),
            }
        diagnostic["duration_ms"] = round(
            (time.perf_counter() - stage_started) * 1000,
            1,
        )
        diagnostics.append(diagnostic)

    if options.use_deberta_title_relevance:
        stage_started = time.perf_counter()
        try:
            candidates, diagnostic = score_deberta_title_relevance(
                keyword_query,
                candidates,
                model_name=model_settings.nli_model,
                max_candidates=options.maximum_nli_candidates,
            )
        except RuntimeError as exc:
            diagnostic = {
                "stage": "deberta_title_relevance",
                "applied": False,
                "processed": 0,
                "reason": str(exc),
            }
        diagnostic["duration_ms"] = round(
            (time.perf_counter() - stage_started) * 1000,
            1,
        )
        diagnostics.append(diagnostic)

    return candidates, diagnostics


def build_direct_query_route(keyword_query: str) -> RelationshipRoute:
    """Create a no-Gemini route so OpenSearch can always return a fast baseline."""
    query = " ".join(str(keyword_query).split())
    if not query:
        raise ValueError("A query is required for Semantic SEO retrieval.")
    return RelationshipRoute(
        relationship_id="local-direct-query",
        related_subject=query,
        relationship_class="DIRECT_SEMANTIC",
        relationship_family="title-only semantic retrieval",
        factual_bridge=(
            "The candidate title is semantically compatible with the supplied query."
        ),
        acceptance_condition="Retain semantic title compatibility.",
        rejection_rule="Reject weak vector matches and direct refined-search duplicates.",
        allowed_story_angles=(),
        excluded_story_angles=(),
        retrieval_text=query,
        quality_score=1.0,
        relationship_role="QUERY_BASELINE",
        can_retrieve_standalone=True,
    )


def attach_available_article_evidence(
    candidates: list[dict[str, Any]],
    title_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    evidence_columns = [
        column
        for column in ("summary", "description", "article_text")
        if column in title_summary.columns
    ]
    if not evidence_columns or "story_id" not in title_summary.columns:
        return [dict(item) for item in candidates]
    evidence_by_id = {
        str(row["story_id"]): {
            column: row.get(column, "") for column in evidence_columns
        }
        for row in title_summary.to_dict("records")
    }
    enriched = []
    for candidate in candidates:
        row = dict(candidate)
        row.update(evidence_by_id.get(str(row.get("story_id", "")), {}))
        enriched.append(row)
    return enriched


def merge_gemini_validation(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, object]],
) -> list[dict[str, Any]]:
    selected_by_id = {
        str(item.get("story_id", "")).strip(): item
        for item in selected
        if str(item.get("story_id", "")).strip()
    }
    result = []
    for candidate in candidates:
        story_id = str(candidate.get("story_id", "")).strip()
        validation = selected_by_id.get(story_id)
        if validation is None:
            continue
        row = dict(candidate)
        row["gemini_relationship"] = str(validation.get("ai_relationship", ""))
        row["gemini_audit_reason"] = str(validation.get("ai_audit_reason", ""))
        row["gemini_confidence"] = float(validation.get("ai_confidence", 0.0))
        row["gemini_relevance_level"] = int(
            validation.get("ai_relevance_level", 0)
        )
        row["gemini_relevance_type"] = str(
            validation.get("ai_relevance_type", "")
        )
        result.append(row)
    result.sort(
        key=lambda item: (
            int(item.get("gemini_relevance_level", 0)),
            float(item.get("gemini_confidence", 0.0)),
            float(item.get("semantic_seo_local_score", 0.0)),
            int(item.get("total_views", 0) or 0),
        ),
        reverse=True,
    )
    return result


def semantic_seo_result_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "semantic_seo_rank",
        "story_id",
        "page_title",
        "related_subject",
        "factual_bridge",
        "route_semantic_score",
        "bge_reranker_score",
        "indexed_entity_score",
        "matched_indexed_entities",
        "matched_canonical_entity_ids",
        "deberta_title_relevance_score",
        "gemini_confidence",
        "gemini_relationship",
        "gemini_audit_reason",
        "total_views",
        "active_months",
        "first_month",
        "last_month",
    ]
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        primary = _primary_evidence(candidate)
        rows.append(
            {
                "semantic_seo_rank": rank,
                "story_id": str(candidate.get("story_id", "")),
                "page_title": str(candidate.get("page_title", "")),
                "related_subject": str(primary.get("related_subject", "")),
                "factual_bridge": str(primary.get("factual_bridge", "")),
                "route_semantic_score": float(primary.get("semantic_score", 0.0)),
                "bge_reranker_score": candidate.get("bge_reranker_score"),
                "indexed_entity_score": candidate.get("indexed_entity_score"),
                "matched_indexed_entities": candidate.get(
                    "matched_indexed_entities", []
                ),
                "matched_canonical_entity_ids": candidate.get(
                    "matched_canonical_entity_ids", []
                ),
                "deberta_title_relevance_score": candidate.get(
                    "deberta_title_relevance_score"
                ),
                "gemini_confidence": candidate.get("gemini_confidence"),
                "gemini_relationship": str(candidate.get("gemini_relationship", "")),
                "gemini_audit_reason": str(candidate.get("gemini_audit_reason", "")),
                "total_views": candidate.get("total_views", 0),
                "active_months": candidate.get("active_months", 0),
                "first_month": candidate.get("first_month"),
                "last_month": candidate.get("last_month"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _primary_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("retrieval_evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                return item
    return {}


def _safe_index_alias(value: str) -> str:
    cleaned = str(value).strip().casefold().replace("_", "-")
    cleaned = "".join(character for character in cleaned if character.isalnum() or character == "-")
    return cleaned.strip("-") or DEFAULT_SEMANTIC_SEO_INDEX_ALIAS
