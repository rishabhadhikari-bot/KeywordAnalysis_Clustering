"""Isolated OpenSearch vector index and retrieval for semantic title search."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
import pandas as pd
import requests

from src.opensearch_refined import (
    OpenSearchRefinedClient,
    OpenSearchSettings,
    load_opensearch_settings,
    title_corpus_fingerprint,
)
from src.relationship_embeddings import (
    RelationshipRoute,
    encode_relationship_texts,
    get_relationship_embedding_model_name,
    load_or_build_title_embedding_index,
)
from src.search_aliases import SEARCH_ALIAS_ENTITIES
from src.data_processing import simple_title_tokens


SEMANTIC_SEARCH_PIPELINE_VERSION = "2026-08-21-opensearch-semantic-v1"
SEMANTIC_TITLE_ENTITY_PIPELINE_VERSION = "2026-08-25-title-gliner-linking-v1"
SEMANTIC_INDEX_META_DOCUMENT_ID = "__semantic_search_index_meta__"
DEFAULT_SEMANTIC_INDEX_ALIAS = "traffic-title-semantic"
DEFAULT_SEMANTIC_MIN_SCORE = 0.80
DEFAULT_SEMANTIC_MAX_RESULTS = 10_000
DEFAULT_RELATIONSHIP_ROUTE_MIN_SCORE = 0.72
DEFAULT_RELATIONSHIP_TOP_K_PER_ROUTE = 20
DEFAULT_RELATIONSHIP_MAX_ROUTES = 30


class OpenSearchSemanticError(RuntimeError):
    """Raised when semantic indexing or retrieval cannot be completed."""


@dataclass(frozen=True)
class OpenSearchSemanticSettings:
    connection: OpenSearchSettings
    embedding_model: str
    minimum_score: float = DEFAULT_SEMANTIC_MIN_SCORE
    max_results: int = DEFAULT_SEMANTIC_MAX_RESULTS
    entity_model: str = ""
    enable_title_entities: bool = False

    @property
    def configured(self) -> bool:
        return self.connection.configured


def load_opensearch_semantic_settings() -> OpenSearchSemanticSettings:
    base = load_opensearch_settings()
    semantic_connection = OpenSearchSettings(
        url=base.url,
        index_alias=_safe_index_name(
            os.getenv("OPENSEARCH_SEMANTIC_INDEX", DEFAULT_SEMANTIC_INDEX_ALIAS)
            or DEFAULT_SEMANTIC_INDEX_ALIAS
        ),
        username=base.username,
        password=base.password,
        verify_certs=base.verify_certs,
        timeout_seconds=base.timeout_seconds,
        auto_start=base.auto_start,
        home=base.home,
        startup_timeout_seconds=base.startup_timeout_seconds,
    )
    return OpenSearchSemanticSettings(
        connection=semantic_connection,
        embedding_model=get_relationship_embedding_model_name(),
        minimum_score=min(
            1.0,
            max(
                0.0,
                float(
                    os.getenv(
                        "OPENSEARCH_SEMANTIC_MIN_SCORE",
                        str(DEFAULT_SEMANTIC_MIN_SCORE),
                    )
                ),
            ),
        ),
        max_results=max(
            1,
            min(
                10_000,
                int(
                    os.getenv(
                        "OPENSEARCH_SEMANTIC_MAX_RESULTS",
                        str(DEFAULT_SEMANTIC_MAX_RESULTS),
                    )
                ),
            ),
        ),
    )


def semantic_corpus_fingerprint(
    title_summary: pd.DataFrame,
    embedding_model: str,
    entity_model: str = "",
) -> str:
    lexical_fingerprint = title_corpus_fingerprint(title_summary)
    entity_component = (
        f"|{SEMANTIC_TITLE_ENTITY_PIPELINE_VERSION}|{entity_model}"
        if str(entity_model).strip()
        else ""
    )
    value = (
        f"{SEMANTIC_SEARCH_PIPELINE_VERSION}|{embedding_model}{entity_component}|"
        f"{lexical_fingerprint}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def semantic_entity_anchor_phrases(normalized_query: str) -> tuple[str, ...]:
    """Return safe entity phrases that semantic candidates must preserve."""
    query_tokens = simple_title_tokens(normalized_query)
    alias_owners: dict[str, set[str]] = {}
    for entity in SEARCH_ALIAS_ENTITIES:
        for alias in entity.aliases:
            normalized_alias = " ".join(simple_title_tokens(alias))
            if normalized_alias:
                alias_owners.setdefault(normalized_alias, set()).add(entity.entity_id)

    anchors: list[str] = []
    for entity in SEARCH_ALIAS_ENTITIES:
        canonical = " ".join(simple_title_tokens(entity.canonical))
        normalized_aliases = [
            " ".join(simple_title_tokens(alias)) for alias in entity.aliases
        ]
        matched_canonical = _contains_token_phrase(query_tokens, canonical.split())
        matched_safe_alias = any(
            alias
            and len(alias_owners.get(alias, set())) == 1
            and _contains_token_phrase(query_tokens, alias.split())
            for alias in normalized_aliases
        )
        if not matched_canonical and not matched_safe_alias:
            continue
        anchors.append(canonical)
        anchors.extend(
            alias
            for alias in normalized_aliases
            if alias and len(alias_owners.get(alias, set())) == 1
        )
    return tuple(dict.fromkeys(anchor for anchor in anchors if anchor))


def collect_all_refined_story_ids(
    client: OpenSearchRefinedClient,
    normalized_query: str,
    match_mode: str,
    *,
    page_size: int = 1000,
) -> set[str]:
    """Collect the complete refined hit set so semantic results cannot overlap it."""
    excluded: set[str] = set()
    search_after: list[Any] | None = None
    total_hits: int | None = None
    previous_cursor: list[Any] | None = None

    while total_hits is None or len(excluded) < total_hits:
        hits, total_hits, next_cursor = client.search_page(
            normalized_query=normalized_query,
            match_mode=match_mode,
            size=max(1, min(int(page_size), 10_000)),
            search_after=search_after,
        )
        excluded.update(str(hit.get("story_id", "")) for hit in hits)
        excluded.discard("")
        if not hits or next_cursor is None or next_cursor == previous_cursor:
            break
        previous_cursor = next_cursor
        search_after = next_cursor

    return excluded


def build_semantic_performance_result(
    semantic_hits: list[dict[str, Any]],
    excluded_story_ids: set[str],
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
    *,
    total_semantic_hits: int,
) -> dict[str, Any]:
    """Remove refined matches and join semantic-only hits to local traffic data."""
    excluded = {str(story_id) for story_id in excluded_story_ids}
    unique_hits: list[dict[str, Any]] = []
    seen_story_ids: set[str] = set()
    for retrieval_rank, hit in enumerate(semantic_hits, start=1):
        story_id = str(hit.get("story_id", ""))
        if not story_id or story_id in excluded or story_id in seen_story_ids:
            continue
        seen_story_ids.add(story_id)
        row = dict(hit)
        row["story_id"] = story_id
        row["semantic_retrieval_rank"] = retrieval_rank
        row["semantic_rank"] = len(unique_hits) + 1
        unique_hits.append(row)

    hit_columns = [
        "story_id",
        "semantic_rank",
        "semantic_retrieval_rank",
        "semantic_score",
    ]
    hit_df = pd.DataFrame(unique_hits)
    if hit_df.empty:
        hit_df = pd.DataFrame(columns=hit_columns)
    else:
        hit_df = hit_df[[column for column in hit_columns if column in hit_df.columns]]

    local_titles = title_summary.copy()
    local_titles["story_id"] = local_titles["story_id"].astype(str)
    matched_titles = hit_df.merge(local_titles, on="story_id", how="inner")
    if not matched_titles.empty:
        matched_titles = matched_titles.sort_values("semantic_rank")

    local_months = story_months.copy()
    local_months["story_id"] = local_months["story_id"].astype(str)
    matched_story_months = local_months.merge(
        hit_df[["story_id", "semantic_score"]],
        on="story_id",
        how="inner",
    )
    if matched_story_months.empty:
        monthly_summary = pd.DataFrame(
            columns=["month", "related_titles_without_query", "views"]
        )
    else:
        monthly_summary = (
            matched_story_months.groupby("month", as_index=False)
            .agg(
                related_titles_without_query=("story_id", "nunique"),
                views=("views", "sum"),
            )
            .sort_values("month")
        )

    return {
        "matched_titles": matched_titles,
        "matched_story_months": matched_story_months,
        "monthly_summary": monthly_summary,
        "total_semantic_hits": int(total_semantic_hits),
        "retrieved_semantic_hits": len(semantic_hits),
        "excluded_refined_titles": len(
            {str(hit.get("story_id", "")) for hit in semantic_hits} & excluded
        ),
        "refined_story_ids": excluded,
    }


class OpenSearchSemanticClient:
    def __init__(self, settings: OpenSearchSemanticSettings):
        if not settings.configured:
            raise ValueError("OPENSEARCH_URL is not configured.")
        self.settings = settings
        self.session = requests.Session()
        if settings.connection.username:
            self.session.auth = (
                settings.connection.username,
                settings.connection.password,
            )

    def connection_info(self) -> dict[str, Any]:
        payload = self._request("GET", "/")
        version = payload.get("version", {}) if isinstance(payload, dict) else {}
        return {
            "cluster_name": str(payload.get("cluster_name", "OpenSearch")),
            "version": str(version.get("number", "unknown")),
        }

    def index_state(self) -> dict[str, Any]:
        response = self._request(
            "GET",
            (
                f"/{self.settings.connection.index_alias}/_doc/"
                f"{SEMANTIC_INDEX_META_DOCUMENT_ID}"
            ),
            allowed_statuses={404},
        )
        if response.get("status") == 404 or response.get("found") is False:
            return {
                "ready": False,
                "fingerprint": "",
                "document_count": 0,
                "embedding_model": "",
                "embedding_dimension": 0,
                "entity_model": "",
                "title_entities_indexed": False,
            }
        source = response.get("_source", {})
        return {
            "ready": True,
            "fingerprint": str(source.get("corpus_fingerprint", "")),
            "document_count": int(source.get("document_count", 0)),
            "pipeline_version": str(source.get("pipeline_version", "")),
            "physical_index": str(response.get("_index", "")),
            "embedding_model": str(source.get("embedding_model", "")),
            "embedding_dimension": int(source.get("embedding_dimension", 0)),
            "entity_model": str(source.get("entity_model", "")),
            "title_entities_indexed": bool(source.get("title_entities_indexed", False)),
        }

    def sync_title_index(
        self,
        title_summary: pd.DataFrame,
        *,
        batch_size: int = 250,
    ) -> dict[str, Any]:
        fingerprint = semantic_corpus_fingerprint(
            title_summary,
            self.settings.embedding_model,
            self.settings.entity_model if self.settings.enable_title_entities else "",
        )
        current = self.index_state()
        if (
            current.get("fingerprint") == fingerprint
            and current.get("pipeline_version") == SEMANTIC_SEARCH_PIPELINE_VERSION
            and current.get("embedding_model") == self.settings.embedding_model
            and (
                not self.settings.enable_title_entities
                or (
                    current.get("title_entities_indexed") is True
                    and current.get("entity_model") == self.settings.entity_model
                )
            )
        ):
            return {**current, "created": False}

        candidate_titles = title_summary.to_dict("records")
        embeddings, story_ids, manifest = load_or_build_title_embedding_index(
            candidate_titles,
            model_name=self.settings.embedding_model,
        )
        dimension = int(manifest.get("embedding_dimension", embeddings.shape[1]))
        physical_index = f"{self.settings.connection.index_alias}-{fingerprint[:12]}"
        exists = self._request(
            "HEAD",
            f"/{physical_index}",
            allowed_statuses={404},
        )
        if exists.get("status") == 404:
            self._request(
                "PUT",
                f"/{physical_index}",
                json_body=_semantic_index_definition(dimension),
            )

        title_by_story_id = {
            str(row.get("story_id", "")): row for row in candidate_titles
        }
        title_entities_by_story_id: dict[str, list[dict[str, object]]] = {}
        if self.settings.enable_title_entities:
            from src.semantic_seo_models import extract_and_link_title_entities

            title_entities_by_story_id = extract_and_link_title_entities(
                {
                    story_id: str(row.get("page_title", ""))
                    for story_id, row in title_by_story_id.items()
                },
                model_name=self.settings.entity_model,
            )
        safe_batch_size = max(1, int(batch_size))
        for start in range(0, len(story_ids), safe_batch_size):
            end = start + safe_batch_size
            bulk_body = _semantic_bulk_index_body(
                physical_index,
                story_ids[start:end],
                np.asarray(embeddings[start:end], dtype=np.float32),
                title_by_story_id,
                title_entities_by_story_id,
            )
            bulk_result = self._request(
                "POST",
                "/_bulk",
                data=bulk_body,
                headers={"Content-Type": "application/x-ndjson"},
            )
            if bulk_result.get("errors"):
                failures = _bulk_failures(bulk_result)
                raise OpenSearchSemanticError(
                    f"Semantic bulk indexing failed for {len(failures)} document(s): "
                    f"{failures[:3]}"
                )

        meta_document = {
            "document_type": "meta",
            "corpus_fingerprint": fingerprint,
            "document_count": len(story_ids),
            "pipeline_version": SEMANTIC_SEARCH_PIPELINE_VERSION,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": dimension,
            "entity_model": (
                self.settings.entity_model if self.settings.enable_title_entities else ""
            ),
            "title_entities_indexed": bool(self.settings.enable_title_entities),
            "title_entity_pipeline_version": (
                SEMANTIC_TITLE_ENTITY_PIPELINE_VERSION
                if self.settings.enable_title_entities
                else ""
            ),
        }
        self._request(
            "PUT",
            (
                f"/{physical_index}/_doc/"
                f"{SEMANTIC_INDEX_META_DOCUMENT_ID}"
            ),
            params={"refresh": "true"},
            json_body=meta_document,
        )
        self._switch_alias(physical_index)
        return {
            "ready": True,
            "fingerprint": fingerprint,
            "document_count": len(story_ids),
            "pipeline_version": SEMANTIC_SEARCH_PIPELINE_VERSION,
            "physical_index": physical_index,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": dimension,
            "entity_model": meta_document["entity_model"],
            "title_entities_indexed": meta_document["title_entities_indexed"],
            "created": True,
        }

    def search(
        self,
        normalized_query: str,
        *,
        minimum_score: float | None = None,
        size: int | None = None,
        entity_anchor_phrases: tuple[str, ...] = (),
    ) -> tuple[list[dict[str, Any]], int]:
        query = " ".join(str(normalized_query).split())
        if not query:
            raise ValueError("A normalized semantic query is required.")
        query_embeddings = encode_relationship_texts(
            [query],
            model_name=self.settings.embedding_model,
        )
        if query_embeddings.shape[0] != 1:
            raise OpenSearchSemanticError("The semantic query embedding was not generated.")
        query_vector = query_embeddings[0].astype(float).tolist()
        result_size = max(
            1,
            min(int(size or self.settings.max_results), self.settings.max_results),
        )
        score_threshold = (
            self.settings.minimum_score
            if minimum_score is None
            else min(1.0, max(0.0, float(minimum_score)))
        )
        vector_filter: dict[str, Any] = {"term": {"document_type": "story"}}
        if entity_anchor_phrases:
            vector_filter = {
                "bool": {
                    "must": [{"term": {"document_type": "story"}}],
                    "should": [
                        {"match_phrase": {"page_title": phrase}}
                        for phrase in entity_anchor_phrases
                    ],
                    "minimum_should_match": 1,
                }
            }
        payload = self._request(
            "POST",
            f"/{self.settings.connection.index_alias}/_search",
            json_body={
                "size": result_size,
                "track_total_hits": True,
                "_source": {"excludes": ["title_embedding"]},
                "query": {
                    "knn": {
                        "title_embedding": {
                            "vector": query_vector,
                            "min_score": score_threshold,
                            "filter": vector_filter,
                        }
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"total_views": {"order": "desc"}},
                    {"story_id": {"order": "asc"}},
                ],
            },
        )
        return _parse_semantic_hits(payload)

    def search_relationship_routes(
        self,
        routes: list[RelationshipRoute],
        *,
        excluded_story_ids: Collection[str] = (),
        minimum_score: float = DEFAULT_RELATIONSHIP_ROUTE_MIN_SCORE,
        top_k_per_route: int = DEFAULT_RELATIONSHIP_TOP_K_PER_ROUTE,
        max_routes: int = DEFAULT_RELATIONSHIP_MAX_ROUTES,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve multiple relationship routes in one OpenSearch request."""
        selected_routes = routes[: max(1, int(max_routes))]
        if not selected_routes:
            return [], {
                "mode": "opensearch_relationship_semantic",
                "route_count": 0,
                "candidate_count": 0,
                "failed_route_count": 0,
            }

        route_embeddings = encode_relationship_texts(
            [route.retrieval_text for route in selected_routes],
            model_name=self.settings.embedding_model,
        )
        if route_embeddings.shape[0] != len(selected_routes):
            raise OpenSearchSemanticError(
                "Relationship route embeddings were not generated completely."
            )

        excluded = sorted(
            {
                str(story_id).strip()
                for story_id in excluded_story_ids
                if str(story_id).strip()
            }
        )
        vector_filter: dict[str, Any] = {"term": {"document_type": "story"}}
        if excluded:
            vector_filter = {
                "bool": {
                    "must": [{"term": {"document_type": "story"}}],
                    "must_not": [{"terms": {"story_id": excluded}}],
                }
            }

        safe_top_k = max(1, min(int(top_k_per_route), 100))
        score_threshold = min(1.0, max(0.0, float(minimum_score)))
        lines: list[str] = []
        for route_embedding in route_embeddings:
            lines.append("{}")
            lines.append(
                json.dumps(
                    {
                        "size": safe_top_k,
                        "track_total_hits": False,
                        "_source": {"excludes": ["title_embedding"]},
                        "query": {
                            "knn": {
                                "title_embedding": {
                                    "vector": route_embedding.astype(float).tolist(),
                                    "min_score": score_threshold,
                                    "filter": vector_filter,
                                }
                            }
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

        payload = self._request(
            "POST",
            f"/{self.settings.connection.index_alias}/_msearch",
            data="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        responses = payload.get("responses", [])
        evidence_by_story_id: dict[str, list[dict[str, Any]]] = {}
        source_by_story_id: dict[str, dict[str, Any]] = {}
        failed_route_count = 0
        for route, response in zip(selected_routes, responses):
            if not isinstance(response, dict) or response.get("error"):
                failed_route_count += 1
                continue
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                story_id = str(source.get("story_id", hit.get("_id", ""))).strip()
                if not story_id or story_id in excluded:
                    continue
                source_by_story_id[story_id] = {
                    "story_id": story_id,
                    "page_title": str(source.get("page_title", "")),
                    "total_views": _safe_int(source.get("total_views")),
                    "active_months": _safe_int(source.get("active_months")),
                    "first_month": source.get("first_month"),
                    "last_month": source.get("last_month"),
                    "title_entity_ids": list(source.get("title_entity_ids", [])),
                    "title_entity_names": list(source.get("title_entity_names", [])),
                    "title_entity_labels": list(source.get("title_entity_labels", [])),
                }
                evidence_by_story_id.setdefault(story_id, []).append(
                    {
                        "relationship_id": route.relationship_id,
                        "related_subject": route.related_subject,
                        "relationship_class": route.relationship_class,
                        "relationship_family": route.relationship_family,
                        "factual_bridge": route.factual_bridge,
                        "acceptance_condition": route.acceptance_condition,
                        "rejection_rule": route.rejection_rule,
                        "allowed_story_angles": list(route.allowed_story_angles),
                        "excluded_story_angles": list(route.excluded_story_angles),
                        "retrieval_text": route.retrieval_text,
                        "semantic_score": round(float(hit.get("_score") or 0.0), 6),
                        "relationship_quality": route.quality_score,
                        "relationship_role": route.relationship_role,
                        "can_retrieve_standalone": route.can_retrieve_standalone,
                        "required_title_cues": list(route.required_title_cues),
                    }
                )

        candidates = []
        for story_id, source in source_by_story_id.items():
            evidence = sorted(
                evidence_by_story_id.get(story_id, []),
                key=lambda item: (
                    float(item.get("semantic_score", 0.0)),
                    float(item.get("relationship_quality", 0.0)),
                ),
                reverse=True,
            )
            candidates.append({**source, "retrieval_evidence": evidence[:5]})
        candidates.sort(
            key=lambda row: (
                float(row.get("retrieval_evidence", [{}])[0].get("semantic_score", 0.0)),
                int(row.get("total_views", 0)),
            ),
            reverse=True,
        )
        return candidates, {
            "mode": "opensearch_relationship_semantic",
            "route_count": len(selected_routes),
            "candidate_count": len(candidates),
            "failed_route_count": failed_route_count,
            "excluded_story_count": len(excluded),
            "top_k_per_route": safe_top_k,
            "minimum_score": score_threshold,
        }

    def search_relationship_routes_lexically(
        self,
        routes: list[RelationshipRoute],
        *,
        excluded_story_ids: Collection[str] = (),
        top_k_per_relationship: int = DEFAULT_RELATIONSHIP_TOP_K_PER_ROUTE,
        max_relationships: int = 15,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search cached relationship subjects without loading an embedding model."""
        selected_routes: list[RelationshipRoute] = []
        seen_relationship_ids: set[str] = set()
        for route in routes:
            if route.relationship_id in seen_relationship_ids:
                continue
            if not _relationship_route_title_phrases(route):
                continue
            seen_relationship_ids.add(route.relationship_id)
            selected_routes.append(route)
            if len(selected_routes) >= max(1, int(max_relationships)):
                break
        if not selected_routes:
            return [], {
                "mode": "opensearch_relationship_lexical",
                "route_count": 0,
                "candidate_count": 0,
                "failed_route_count": 0,
            }

        excluded = sorted(
            {
                str(story_id).strip()
                for story_id in excluded_story_ids
                if str(story_id).strip()
            }
        )
        base_filter: list[dict[str, Any]] = [
            {"term": {"document_type": "story"}}
        ]
        must_not = [{"terms": {"story_id": excluded}}] if excluded else []
        safe_top_k = max(1, min(int(top_k_per_relationship), 100))
        lines: list[str] = []
        for route in selected_routes:
            clauses: list[dict[str, Any]] = []
            for phrase_index, phrase in enumerate(
                _relationship_route_title_phrases(route)
            ):
                clauses.append(
                    {
                        "match_phrase": {
                            "page_title": {
                                "query": phrase,
                                "boost": 6.0 if phrase_index == 0 else 5.0,
                                "_name": f"relationship_phrase_{phrase_index}",
                            }
                        }
                    }
                )
            lines.append("{}")
            lines.append(
                json.dumps(
                    {
                        "size": safe_top_k,
                        "track_total_hits": False,
                        "query": {
                            "bool": {
                                "should": clauses,
                                "minimum_should_match": 1,
                                "filter": base_filter,
                                "must_not": must_not,
                            }
                        },
                        "sort": [
                            {"_score": {"order": "desc"}},
                            {"total_views": {"order": "desc"}},
                            {"story_id": {"order": "asc"}},
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

        payload = self._request(
            "POST",
            f"/{self.settings.connection.index_alias}/_msearch",
            data="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        responses = payload.get("responses", [])
        evidence_by_story_id: dict[str, list[dict[str, Any]]] = {}
        source_by_story_id: dict[str, dict[str, Any]] = {}
        failed_route_count = 0
        for route, response in zip(selected_routes, responses):
            if not isinstance(response, dict) or response.get("error"):
                failed_route_count += 1
                continue
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                story_id = str(source.get("story_id", hit.get("_id", ""))).strip()
                if not story_id or story_id in excluded:
                    continue
                raw_score = float(hit.get("_score") or 0.0)
                normalized_score = raw_score / (raw_score + 1.0) if raw_score else 0.0
                source_by_story_id[story_id] = {
                    "story_id": story_id,
                    "page_title": str(source.get("page_title", "")),
                    "total_views": _safe_int(source.get("total_views")),
                    "active_months": _safe_int(source.get("active_months")),
                    "first_month": source.get("first_month"),
                    "last_month": source.get("last_month"),
                }
                evidence_by_story_id.setdefault(story_id, []).append(
                    {
                        "relationship_id": route.relationship_id,
                        "related_subject": route.related_subject,
                        "relationship_class": route.relationship_class,
                        "relationship_family": route.relationship_family,
                        "factual_bridge": route.factual_bridge,
                        "acceptance_condition": route.acceptance_condition,
                        "rejection_rule": route.rejection_rule,
                        "allowed_story_angles": list(route.allowed_story_angles),
                        "excluded_story_angles": list(route.excluded_story_angles),
                        "retrieval_text": route.retrieval_text,
                        "semantic_score": round(normalized_score, 6),
                        "opensearch_score": round(raw_score, 6),
                        "relationship_quality": route.quality_score,
                        "relationship_role": route.relationship_role,
                        "can_retrieve_standalone": route.can_retrieve_standalone,
                        "required_title_cues": list(route.required_title_cues),
                        "match_method": "opensearch_relationship_phrase",
                    }
                )

        candidates = []
        for story_id, source in source_by_story_id.items():
            evidence = sorted(
                evidence_by_story_id.get(story_id, []),
                key=lambda item: (
                    float(item.get("semantic_score", 0.0)),
                    float(item.get("relationship_quality", 0.0)),
                ),
                reverse=True,
            )
            candidates.append({**source, "retrieval_evidence": evidence[:5]})
        candidates.sort(
            key=lambda row: (
                float(row.get("retrieval_evidence", [{}])[0].get("semantic_score", 0.0)),
                int(row.get("total_views", 0)),
            ),
            reverse=True,
        )
        return candidates, {
            "mode": "opensearch_relationship_lexical",
            "route_count": len(selected_routes),
            "candidate_count": len(candidates),
            "failed_route_count": failed_route_count,
            "excluded_story_count": len(excluded),
            "top_k_per_relationship": safe_top_k,
        }

    def _switch_alias(self, physical_index: str) -> None:
        alias = self.settings.connection.index_alias
        aliases = self._request(
            "GET",
            f"/_alias/{alias}",
            allowed_statuses={404},
        )
        actions: list[dict[str, Any]] = []
        if aliases.get("status") != 404:
            actions.extend(
                {"remove": {"index": index_name, "alias": alias}}
                for index_name in aliases
            )
        actions.append({"add": {"index": physical_index, "alias": alias}})
        self._request("POST", "/_aliases", json_body={"actions": actions})

    def _request(
        self,
        method: str,
        path: str,
        *,
        allowed_statuses: set[int] | None = None,
        json_body: dict[str, Any] | None = None,
        data: str | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        connection = self.settings.connection
        try:
            response = self.session.request(
                method,
                f"{connection.url}{path}",
                json=json_body,
                data=data,
                headers=headers,
                params=params,
                timeout=connection.timeout_seconds,
                verify=connection.verify_certs,
            )
        except requests.RequestException as exc:
            raise OpenSearchSemanticError(
                f"Unable to reach OpenSearch at {connection.url}: {exc}"
            ) from exc

        if allowed_statuses and response.status_code in allowed_statuses:
            payload = _response_json(response)
            payload["status"] = response.status_code
            return payload
        if not response.ok:
            detail = response.text.strip()[:1000]
            raise OpenSearchSemanticError(
                f"OpenSearch {method} {path} returned HTTP {response.status_code}: {detail}"
            )
        if method == "HEAD" or not response.content:
            return {"status": response.status_code}
        return _response_json(response)


def _semantic_index_definition(dimension: int) -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.knn": True,
        },
        "mappings": {
            "dynamic": "strict",
            "_meta": {"pipeline_version": SEMANTIC_SEARCH_PIPELINE_VERSION},
            "properties": {
                "document_type": {"type": "keyword"},
                "story_id": {"type": "keyword"},
                "page_title": {"type": "text", "analyzer": "standard"},
                "title_entity_ids": {"type": "keyword"},
                "title_entity_names": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "title_entity_labels": {"type": "keyword"},
                "title_entity_count": {"type": "integer"},
                "title_embedding": {
                    "type": "knn_vector",
                    "dimension": int(dimension),
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24,
                        },
                    },
                },
                "total_views": {"type": "long"},
                "active_months": {"type": "integer"},
                "first_month": {"type": "date"},
                "last_month": {"type": "date"},
                "corpus_fingerprint": {"type": "keyword"},
                "document_count": {"type": "integer"},
                "pipeline_version": {"type": "keyword"},
                "embedding_model": {"type": "keyword"},
                "embedding_dimension": {"type": "integer"},
                "entity_model": {"type": "keyword"},
                "title_entities_indexed": {"type": "boolean"},
                "title_entity_pipeline_version": {"type": "keyword"},
            },
        },
    }


def _semantic_bulk_index_body(
    index_name: str,
    story_ids: list[str],
    embeddings: np.ndarray,
    title_by_story_id: dict[str, dict[str, Any]],
    title_entities_by_story_id: dict[str, list[dict[str, object]]] | None = None,
) -> str:
    lines: list[str] = []
    for story_id, embedding in zip(story_ids, embeddings):
        row = title_by_story_id.get(str(story_id), {})
        entities = (title_entities_by_story_id or {}).get(str(story_id), [])
        document = {
            "document_type": "story",
            "story_id": str(story_id),
            "page_title": str(row.get("page_title", "")),
            "title_embedding": np.asarray(embedding, dtype=np.float32).tolist(),
            "total_views": _safe_int(row.get("total_views")),
            "active_months": _safe_int(row.get("active_months")),
            "first_month": _safe_date(row.get("first_month")),
            "last_month": _safe_date(row.get("last_month")),
            "title_entity_ids": [
                str(entity.get("entity_id", "")) for entity in entities
                if str(entity.get("entity_id", ""))
            ],
            "title_entity_names": [
                str(entity.get("canonical_name", "")) for entity in entities
                if str(entity.get("canonical_name", ""))
            ],
            "title_entity_labels": [
                str(entity.get("label", "")) for entity in entities
                if str(entity.get("label", ""))
            ],
            "title_entity_count": len(entities),
        }
        lines.append(
            json.dumps(
                {"index": {"_index": index_name, "_id": str(story_id)}},
                separators=(",", ":"),
            )
        )
        lines.append(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def _parse_semantic_hits(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw_hits = payload.get("hits", {}).get("hits", [])
    total_value = payload.get("hits", {}).get("total", 0)
    if isinstance(total_value, dict):
        total = int(total_value.get("value", len(raw_hits)))
    else:
        total = int(total_value or len(raw_hits))
    hits = []
    for hit in raw_hits:
        source = hit.get("_source", {})
        hits.append(
            {
                "story_id": str(source.get("story_id", hit.get("_id", ""))),
                "page_title": str(source.get("page_title", "")),
                "semantic_score": round(float(hit.get("_score") or 0.0), 6),
                "title_entity_ids": list(source.get("title_entity_ids", [])),
                "title_entity_names": list(source.get("title_entity_names", [])),
                "title_entity_labels": list(source.get("title_entity_labels", [])),
            }
        )
    return hits, total


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {"value": payload}


def _bulk_failures(payload: dict[str, Any]) -> list[str]:
    failures = []
    for item in payload.get("items", []):
        operation = item.get("index", {})
        if int(operation.get("status", 500)) >= 300:
            failures.append(str(operation.get("error", operation)))
    return failures


def _safe_index_name(value: str) -> str:
    cleaned = str(value).strip().lower()
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in cleaned
    )
    return cleaned.strip("-_") or DEFAULT_SEMANTIC_INDEX_ALIAS


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_date(value: Any) -> str | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _contains_token_phrase(tokens: list[str], phrase: list[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    return any(
        tokens[index : index + width] == phrase
        for index in range(len(tokens) - width + 1)
    )


def _relationship_route_title_phrases(route: RelationshipRoute) -> list[str]:
    subject = str(route.related_subject).strip()
    compact_subject = re.sub(r"[^A-Za-z0-9]", "", subject)
    ambiguous_short_intelligence_name = (
        "INTELLIGENCE" in str(route.relationship_role).upper()
        and len(simple_title_tokens(subject)) == 1
        and len(compact_subject) <= 4
    )
    phrases = [] if ambiguous_short_intelligence_name else [subject]
    if subject and not ambiguous_short_intelligence_name:
        phrases.extend(re.findall(r"\(([^)]+)\)", subject))
        phrases.append(re.sub(r"\s*\([^)]*\)\s*", " ", subject).strip())
    phrases.extend(str(cue).strip() for cue in route.required_title_cues)
    normalized: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        compact_original = re.sub(r"[^A-Za-z0-9]", "", phrase)
        if (
            len(simple_title_tokens(phrase)) == 1
            and len(compact_original) <= 3
            and compact_original.isupper()
        ):
            # Short all-caps forms such as IOC, MI, or AI are commonly
            # ambiguous in headlines. The expanded subject remains searchable.
            continue
        clean_phrase = " ".join(simple_title_tokens(phrase))
        if not clean_phrase or clean_phrase in seen:
            continue
        seen.add(clean_phrase)
        normalized.append(clean_phrase)
    return normalized
