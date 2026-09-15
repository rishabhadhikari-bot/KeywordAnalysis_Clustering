from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from src.data_processing import simple_query_tokens, simple_title_tokens
from src.relationship_embeddings import (
    build_title_corpus_fingerprint,
    encode_relationship_texts,
    get_relationship_embedding_model_name,
    load_or_build_title_embedding_index,
)


BERTOPIC_INDEX_VERSION = "2026-09-07-bertopic-quality-v2"
BERTOPIC_RETRIEVAL_VERSION = "2026-09-14-prototype-relevance-gate-v5"
DEFAULT_MIN_CORPUS_SIZE = 100
DEFAULT_MIN_CLUSTER_SIZE = 20
DEFAULT_MAX_QUERY_TOPICS = 8
DEFAULT_MAX_SEMANTIC_ONLY_TOPICS = 2
DEFAULT_MIN_TOPIC_SIMILARITY = 0.50
DEFAULT_MIN_TOPIC_LABEL_SIMILARITY = 0.40
DEFAULT_MIN_DIRECT_TOPIC_SIMILARITY = 0.35
DEFAULT_MIN_DIRECT_TOPIC_MATCHES = 3
DEFAULT_MAX_DIRECT_TOPICS = 4
DEFAULT_DIRECT_CONTEXT_WEIGHT = 0.70
DEFAULT_MIN_CONTEXT_TITLES = 3
DEFAULT_MIN_DIRECT_TOPIC_TITLE_SIMILARITY = 0.25
DEFAULT_MIN_SEMANTIC_TOPIC_TITLE_SIMILARITY = 0.62
DEFAULT_MAX_CANDIDATES = 900
DEFAULT_MAX_CANDIDATES_PER_TOPIC = 180
DEFAULT_MAX_QUERY_PROTOTYPES = 6
DEFAULT_PROTOTYPE_MERGE_SIMILARITY = 0.72
DEFAULT_MIN_PROTOTYPE_TITLE_SIMILARITY = 0.50
DEFAULT_MIN_PROTOTYPE_RAW_TITLE_SIMILARITY = 0.60
DEFAULT_MIN_PROTOTYPE_NO_COVERAGE_RAW_TITLE_SIMILARITY = 0.65
DEFAULT_MIN_PROTOTYPE_QUERY_COVERAGE = 0.50
DEFAULT_MIN_PROTOTYPE_RAW_LABEL_SIMILARITY = 0.55
DEFAULT_TOP_K_PER_PROTOTYPE = 150

_BUILD_LOCK = threading.RLock()


def get_bertopic_training_config() -> dict[str, Any]:
    """All settings affecting assignments or representations belong in the cache key."""
    return {
        "min_cluster_size": max(5, int(os.getenv("BERTOPIC_MIN_CLUSTER_SIZE", "20"))),
        "min_samples": max(1, int(os.getenv("BERTOPIC_MIN_SAMPLES", "20"))),
        "n_neighbors": max(2, int(os.getenv("BERTOPIC_N_NEIGHBORS", "15"))),
        "n_components": max(2, int(os.getenv("BERTOPIC_N_COMPONENTS", "5"))),
        "min_dist": 0.0, "random_state": 42, "selection_method": "eom",
        "ngram_range": [1, 2], "min_df": 1, "stop_words": "english",
        "embedding_model": get_relationship_embedding_model_name(),
        "versions": {name: version(name) for name in
                     ("bertopic", "hdbscan", "umap-learn", "scikit-learn", "sentence-transformers")},
    }


def _serialized_build(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        with _BUILD_LOCK:
            return operation(*args, **kwargs)
    return wrapped


def get_bertopic_index_dir() -> Path:
    configured = os.getenv("BERTOPIC_INDEX_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "bertopic"


def retrieve_bertopic_candidates(
    keyword_query: str,
    candidate_titles: list[dict[str, object]],
    direct_story_ids: tuple[str, ...] = (),
    index_dir: Path | None = None,
    max_topics: int = DEFAULT_MAX_QUERY_TOPICS,
    min_topic_similarity: float = DEFAULT_MIN_TOPIC_SIMILARITY,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_candidates_per_topic: int = DEFAULT_MAX_CANDIDATES_PER_TOPIC,
    excluded_story_ids: tuple[str, ...] = (),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    diagnostics: dict[str, object] = {
        "mode": "bertopic",
        "topic_count": 0,
        "selected_topic_count": 0,
        "candidate_count": 0,
        "fallback_reason": "",
        "index_version": BERTOPIC_INDEX_VERSION,
        "retrieval_version": BERTOPIC_RETRIEVAL_VERSION,
        "prototype_count": 0,
        "prototype_candidate_count": 0,
        "outlier_rescue_count": 0,
    }
    valid_rows = _valid_sorted_rows(candidate_titles)
    if not keyword_query.strip():
        diagnostics["fallback_reason"] = "No query was supplied for BERTopic retrieval."
        return [], diagnostics
    if len(valid_rows) < DEFAULT_MIN_CORPUS_SIZE:
        diagnostics["fallback_reason"] = (
            f"BERTopic requires at least {DEFAULT_MIN_CORPUS_SIZE} corpus titles."
        )
        return [], diagnostics

    try:
        artifacts = load_or_build_bertopic_index(
            candidate_titles=valid_rows,
            index_dir=index_dir,
        )
    except Exception as exc:
        diagnostics["fallback_reason"] = str(exc)
        return [], diagnostics

    topic_ids = artifacts["topic_ids"]
    topic_probabilities = artifacts["topic_probabilities"]
    unique_topic_ids = artifacts["unique_topic_ids"]
    topic_embeddings = artifacts["topic_embeddings"]
    topic_labels = artifacts["topic_labels"]
    story_ids = artifacts["story_ids"]
    diagnostics["topic_count"] = len(unique_topic_ids)
    diagnostics["outlier_count"] = int(np.count_nonzero(topic_ids == -1))
    diagnostics["title_gate_rejected_count"] = 0
    diagnostics["prototype_gate_rejected_count"] = 0
    diagnostics["prototype_gate_rejection_reasons"] = {}
    diagnostics["embedding_model"] = artifacts["manifest"].get("embedding_model", "")

    embedding_model = str(artifacts["manifest"]["embedding_model"])
    topic_label_texts = [
        str(topic_labels.get(str(int(topic_id)), f"Topic {int(topic_id)}"))
        for topic_id in unique_topic_ids
    ]
    query_and_label_embeddings = encode_relationship_texts(
        [keyword_query, *topic_label_texts],
        model_name=embedding_model,
    )
    raw_query_embedding = query_and_label_embeddings[0]
    topic_label_embeddings = query_and_label_embeddings[1:]
    query_tokens = set(simple_query_tokens(keyword_query))

    direct_story_id_set = {str(item) for item in direct_story_ids}
    excluded_story_id_set = direct_story_id_set | {str(item) for item in excluded_story_ids}
    topic_embeddings = np.asarray(topic_embeddings).reshape(-1, len(raw_query_embedding))
    title_embeddings = None
    title_embedding_story_ids: list[str] = []
    contextual_query_embedding = raw_query_embedding
    direct_context_count = 0
    query_prototypes: list[np.ndarray] = []
    prototype_member_story_ids: list[list[str]] = []
    try:
        title_embeddings, title_embedding_story_ids, _ = load_or_build_title_embedding_index(
            candidate_titles=valid_rows,
            model_name=embedding_model,
        )
        title_embedding_index_by_story_id = {
            story_id: index for index, story_id in enumerate(title_embedding_story_ids)
        }
        direct_embedding_indices = [
            index
            for index, story_id in enumerate(title_embedding_story_ids)
            if story_id in direct_story_id_set
        ]
        direct_context_count = len(direct_embedding_indices)
        if direct_embedding_indices:
            (
                query_prototypes,
                prototype_member_indices,
            ) = _build_query_prototypes(
                np.asarray(title_embeddings[direct_embedding_indices], dtype=np.float32),
                max_prototypes=DEFAULT_MAX_QUERY_PROTOTYPES,
                merge_similarity=DEFAULT_PROTOTYPE_MERGE_SIMILARITY,
            )
            direct_embedding_story_ids = [
                title_embedding_story_ids[index] for index in direct_embedding_indices
            ]
            prototype_member_story_ids = [
                [direct_embedding_story_ids[index] for index in member_indices]
                for member_indices in prototype_member_indices
            ]
        if direct_context_count >= DEFAULT_MIN_CONTEXT_TITLES:
            direct_centroid = np.asarray(
                title_embeddings[direct_embedding_indices], dtype=np.float32
            ).mean(axis=0)
            direct_centroid_norm = float(np.linalg.norm(direct_centroid))
            if direct_centroid_norm:
                direct_centroid /= direct_centroid_norm
                contextual_query_embedding = (
                    (1.0 - DEFAULT_DIRECT_CONTEXT_WEIGHT) * raw_query_embedding
                    + DEFAULT_DIRECT_CONTEXT_WEIGHT * direct_centroid
                )
                contextual_norm = float(np.linalg.norm(contextual_query_embedding))
                if contextual_norm:
                    contextual_query_embedding /= contextual_norm
    except Exception as exc:
        diagnostics["contextual_query_fallback_reason"] = str(exc)
        title_embeddings = None
        title_embedding_story_ids = []

    diagnostics["direct_context_title_count"] = direct_context_count
    diagnostics["used_direct_title_context"] = (
        direct_context_count >= DEFAULT_MIN_CONTEXT_TITLES
    )
    diagnostics["prototype_count"] = len(query_prototypes)
    topic_similarities = np.asarray(
        topic_embeddings @ contextual_query_embedding, dtype=np.float32
    )
    raw_topic_similarities = np.asarray(
        topic_embeddings @ raw_query_embedding, dtype=np.float32
    )
    topic_label_similarities = np.asarray(
        topic_label_embeddings @ contextual_query_embedding, dtype=np.float32
    )
    raw_topic_label_similarities = np.asarray(
        topic_label_embeddings @ raw_query_embedding, dtype=np.float32
    )
    similarity_by_topic = {
        int(topic_id): float(similarity)
        for topic_id, similarity in zip(unique_topic_ids, topic_similarities)
    }
    raw_similarity_by_topic = {
        int(topic_id): float(similarity)
        for topic_id, similarity in zip(unique_topic_ids, raw_topic_similarities)
    }
    label_similarity_by_topic = {
        int(topic_id): float(similarity)
        for topic_id, similarity in zip(unique_topic_ids, topic_label_similarities)
    }
    raw_label_similarity_by_topic = {
        int(topic_id): float(similarity)
        for topic_id, similarity in zip(unique_topic_ids, raw_topic_label_similarities)
    }

    semantic_topic_ids = {
        topic_id
        for topic_id, _ in [
            item
            for item in sorted(
                similarity_by_topic.items(), key=lambda item: item[1], reverse=True
            )
            if item[1] >= float(min_topic_similarity)
            and label_similarity_by_topic.get(item[0], 0.0)
            >= DEFAULT_MIN_TOPIC_LABEL_SIMILARITY
        ][: min(max(1, int(max_topics)), DEFAULT_MAX_SEMANTIC_ONLY_TOPICS)]
    }
    direct_topic_counts: dict[int, int] = {}
    for index, story_id in enumerate(story_ids):
        topic_id = int(topic_ids[index])
        if story_id not in direct_story_id_set or topic_id == -1:
            continue
        direct_topic_counts[topic_id] = direct_topic_counts.get(topic_id, 0) + 1
    direct_topic_ids = {
        topic_id
        for topic_id, count in sorted(
            direct_topic_counts.items(),
            key=lambda item: (item[1], similarity_by_topic.get(item[0], 0.0)),
            reverse=True,
        )[:DEFAULT_MAX_DIRECT_TOPICS]
        if count >= DEFAULT_MIN_DIRECT_TOPIC_MATCHES
        and similarity_by_topic.get(topic_id, 0.0) >= DEFAULT_MIN_DIRECT_TOPIC_SIMILARITY
    }
    semantic_topic_ids.difference_update(direct_topic_ids)
    selected_topic_ids = set(semantic_topic_ids)
    selected_topic_ids.update(direct_topic_ids)
    diagnostics["selected_topic_count"] = len(selected_topic_ids)
    diagnostics["direct_topic_count"] = len(direct_topic_ids)
    diagnostics["direct_topic_matches"] = {
        str(topic_id): direct_topic_counts[topic_id]
        for topic_id in direct_topic_ids
    }
    diagnostics["semantic_topic_count"] = len(semantic_topic_ids)
    prototype_match_by_story_id: dict[str, dict[str, object]] = {}
    if title_embeddings is not None and query_prototypes:
        for prototype_index, prototype in enumerate(query_prototypes):
            similarities = np.asarray(
                np.asarray(title_embeddings) @ prototype,
                dtype=np.float32,
            )
            eligible_indices = np.asarray([
                i for i, story_id in enumerate(title_embedding_story_ids)
                if story_id not in excluded_story_id_set
            ], dtype=int)
            safe_top_k = min(DEFAULT_TOP_K_PER_PROTOTYPE, len(eligible_indices))
            if safe_top_k <= 0:
                continue
            top_indices = eligible_indices[np.argpartition(
                similarities[eligible_indices], -safe_top_k
            )[-safe_top_k:]]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            for embedding_index in top_indices:
                similarity = float(similarities[int(embedding_index)])
                if similarity < DEFAULT_MIN_PROTOTYPE_TITLE_SIMILARITY:
                    continue
                story_id = title_embedding_story_ids[int(embedding_index)]
                if story_id in excluded_story_id_set:
                    continue
                previous = prototype_match_by_story_id.get(story_id)
                if previous is None or similarity > float(previous["similarity"]):
                    prototype_match_by_story_id[story_id] = {
                        "prototype_index": prototype_index,
                        "similarity": similarity,
                    }
    diagnostics["prototype_candidate_count"] = len(prototype_match_by_story_id)
    rows_by_story_id = {
        str(row["story_id"]): row
        for row in valid_rows
    }
    diagnostics["query_prototypes"] = [
        {
            "prototype_id": prototype_index + 1,
            "anchor_count": len(member_story_ids),
            "representative_titles": " | ".join(
                str(rows_by_story_id[story_id].get("page_title", ""))
                for story_id in member_story_ids[:3]
                if story_id in rows_by_story_id
            ),
        }
        for prototype_index, member_story_ids in enumerate(prototype_member_story_ids)
    ]

    if not selected_topic_ids and not prototype_match_by_story_id:
        diagnostics["fallback_reason"] = "No BERTopic cluster was sufficiently similar to the query."
        return [], diagnostics

    title_embedding_index_by_story_id = {
        story_id: index for index, story_id in enumerate(title_embedding_story_ids)
    }
    ranked_by_topic: dict[int, list[dict[str, object]]] = {}
    for index, story_id in enumerate(story_ids):
        topic_id = int(topic_ids[index])
        prototype_match = prototype_match_by_story_id.get(story_id)
        selected_by_topic = topic_id in selected_topic_ids
        if story_id in excluded_story_id_set or not (selected_by_topic or prototype_match):
            continue
        title_similarity = 0.0
        raw_title_similarity = 0.0
        embedding_index = title_embedding_index_by_story_id.get(story_id)
        if title_embeddings is not None and embedding_index is not None:
            title_similarity = float(
                np.asarray(title_embeddings[embedding_index]) @ contextual_query_embedding
            )
            raw_title_similarity = float(
                np.asarray(title_embeddings[embedding_index]) @ raw_query_embedding
            )
        topic_similarity = similarity_by_topic.get(topic_id, 0.0)
        label_similarity = label_similarity_by_topic.get(topic_id, 0.0)
        raw_label_similarity = raw_label_similarity_by_topic.get(topic_id, 0.0)
        prototype_similarity = (
            float(prototype_match["similarity"]) if prototype_match else 0.0
        )
        coverage = 0.0
        if query_tokens:
            title_tokens = set(
                simple_title_tokens(str(rows_by_story_id[story_id].get("page_title", "")))
            )
            coverage = len(query_tokens.intersection(title_tokens)) / len(query_tokens)
        minimum_title_similarity = (
            DEFAULT_MIN_DIRECT_TOPIC_TITLE_SIMILARITY
            if topic_id in direct_topic_ids
            else DEFAULT_MIN_SEMANTIC_TOPIC_TITLE_SIMILARITY
        )
        if (
            title_embeddings is not None
            and prototype_match is None
            and raw_title_similarity < minimum_title_similarity
            and coverage < DEFAULT_MIN_PROTOTYPE_QUERY_COVERAGE
        ):
            diagnostics["title_gate_rejected_count"] = int(
                diagnostics.get("title_gate_rejected_count", 0)
            ) + 1
            continue

        # A prototype is useful for recall, but by itself it is not evidence
        # that a candidate is about the query. Require an independent signal
        # before admitting prototype-only candidates: either the title has
        # meaningful query-token coverage or its assigned topic label agrees
        # with the query. This protects every query type, while preserving
        # paraphrased candidates when their topic representation is relevant.
        if prototype_match is not None:
            minimum_prototype_title_similarity = (
                DEFAULT_MIN_PROTOTYPE_RAW_TITLE_SIMILARITY
                if coverage >= DEFAULT_MIN_PROTOTYPE_QUERY_COVERAGE
                else DEFAULT_MIN_PROTOTYPE_NO_COVERAGE_RAW_TITLE_SIMILARITY
            )
            if raw_title_similarity < minimum_prototype_title_similarity:
                diagnostics["prototype_gate_rejected_count"] += 1
                reason = "raw_title_similarity_below_threshold"
                reasons = diagnostics["prototype_gate_rejection_reasons"]
                reasons[reason] = int(reasons.get(reason, 0)) + 1
                continue
            if (
                coverage < DEFAULT_MIN_PROTOTYPE_QUERY_COVERAGE
                and raw_label_similarity < DEFAULT_MIN_PROTOTYPE_RAW_LABEL_SIMILARITY
            ):
                diagnostics["prototype_gate_rejected_count"] += 1
                reason = "no_query_coverage_or_topic_label_agreement"
                reasons = diagnostics["prototype_gate_rejection_reasons"]
                reasons[reason] = int(reasons.get(reason, 0)) + 1
                continue
        direct_support = min(
            1.0,
            direct_topic_counts.get(topic_id, 0) / float(DEFAULT_MIN_DIRECT_TOPIC_MATCHES),
        )
        if prototype_match is not None:
            candidate_relevance = (
                prototype_similarity * 0.45
                + max(0.0, raw_title_similarity) * 0.20
                + max(0.0, topic_similarity) * 0.15
                + max(0.0, label_similarity) * 0.10
                + max(0.0, float(topic_probabilities[index])) * 0.10
            )
        else:
            candidate_relevance = (
                title_similarity * 0.45
                + max(0.0, topic_similarity) * 0.20
                + max(0.0, label_similarity) * 0.10
                + direct_support * 0.15
                + max(0.0, float(topic_probabilities[index])) * 0.10
            )
        if prototype_match is not None and selected_by_topic:
            selection_route = "prototype_and_topic"
        elif prototype_match is not None and topic_id == -1:
            selection_route = "prototype_outlier_rescue"
        elif prototype_match is not None:
            selection_route = "query_prototype"
        elif topic_id in direct_topic_ids:
            selection_route = "direct_title_support"
        else:
            selection_route = "semantic_centroid_and_label"
        ranked_by_topic.setdefault(topic_id, []).append(
            {
                "candidate_relevance": candidate_relevance,
                "title_similarity": title_similarity,
                "raw_title_similarity": raw_title_similarity,
                "prototype_similarity": prototype_similarity,
                "prototype_index": (
                    int(prototype_match["prototype_index"])
                    if prototype_match is not None
                    else None
                ),
                "selection_route": selection_route,
                "probability": float(topic_probabilities[index]),
                "views": float(rows_by_story_id[story_id].get("total_views", 0) or 0),
                "story_id": story_id,
            }
        )

    ranked_candidates: list[dict[str, object]] = []
    for topic_id, topic_rows in ranked_by_topic.items():
        topic_rows.sort(
            key=lambda item: (
                float(item["candidate_relevance"]),
                float(item["probability"]),
                float(item["views"]),
            ),
            reverse=True,
        )
        for item in topic_rows[: max(1, int(max_candidates_per_topic))]:
            item["topic_id"] = topic_id
            ranked_candidates.append(item)
    ranked_candidates.sort(
        key=lambda item: (
            float(item["candidate_relevance"]),
            float(item["probability"]),
            float(item["views"]),
        ),
        reverse=True,
    )

    selected_rows: list[dict[str, object]] = []
    for candidate in ranked_candidates[: max(1, int(max_candidates))]:
        topic_id = int(candidate["topic_id"])
        story_id = str(candidate["story_id"])
        probability = float(candidate["probability"])
        topic_similarity = similarity_by_topic.get(topic_id, 0.0)
        source = dict(rows_by_story_id[story_id])
        topic_label = (
            "BERTopic outlier"
            if topic_id == -1
            else str(topic_labels.get(str(topic_id), f"Topic {topic_id}"))
        )
        prototype_index = candidate.get("prototype_index")
        anchor_titles: list[str] = []
        if prototype_index is not None:
            anchor_titles = [
                str(rows_by_story_id[anchor_story_id].get("page_title", ""))
                for anchor_story_id in prototype_member_story_ids[int(prototype_index)][:3]
                if anchor_story_id in rows_by_story_id
            ]
        selection_route = str(candidate["selection_route"])
        if topic_id == -1:
            factual_bridge = (
                "BERTopic marked this title as an outlier, but its embedding is close "
                "to a query-specific group of direct-match titles."
            )
        elif prototype_index is not None:
            factual_bridge = (
                f"The title is close to a query-specific direct-title group and belongs "
                f"to the corpus topic '{topic_label}'. Both are discovery signals, not "
                "proof of a factual relationship."
            )
        else:
            factual_bridge = (
                f"The title belongs to the corpus topic '{topic_label}', which was "
                "retrieved as semantically similar to the query. Topic membership is "
                "candidate evidence, not proof of a factual relationship."
            )
        source["retrieval_evidence"] = [
            {
                "relationship_id": f"bertopic:{topic_id}",
                "related_subject": topic_label,
                "relationship_class": "TOPIC_RELATED",
                "relationship_family": "BERTopic corpus cluster",
                "factual_bridge": factual_bridge,
                "acceptance_condition": (
                    "Accept only when the title itself has a concrete editorial relationship "
                    "to the original query."
                ),
                "rejection_rule": (
                    "Reject titles connected only by a broad topic with no specific factual bridge."
                ),
                "allowed_story_angles": ["Concrete stories within the retrieved corpus topic"],
                "excluded_story_angles": ["Broad thematic overlap without a factual relationship"],
                "retrieval_text": topic_label,
                "similarity": round(float(topic_similarity), 4),
                "raw_query_similarity": round(
                    raw_similarity_by_topic.get(topic_id, 0.0), 4
                ),
                "topic_label_similarity": round(
                    label_similarity_by_topic.get(topic_id, 0.0), 4
                ),
                "raw_topic_label_similarity": round(
                    raw_label_similarity_by_topic.get(topic_id, 0.0), 4
                ),
                "title_query_similarity": round(
                    float(candidate["title_similarity"]), 4
                ),
                "raw_title_query_similarity": round(
                    float(candidate["raw_title_similarity"]), 4
                ),
                "prototype_similarity": round(
                    float(candidate["prototype_similarity"]), 4
                ),
                "prototype_id": (
                    int(prototype_index) + 1 if prototype_index is not None else None
                ),
                "prototype_anchor_titles": anchor_titles,
                "topic_probability": round(float(probability), 4),
                "candidate_relevance_score": round(
                    float(candidate["candidate_relevance"]), 4
                ),
                "selection_route": selection_route,
                "relationship_quality": round(
                    float(candidate["candidate_relevance"]), 4
                ),
                "match_method": "bertopic_topic",
            }
        ]
        selected_rows.append(source)

    diagnostics["candidate_count"] = len(selected_rows)
    diagnostics["outlier_rescue_count"] = sum(
        1
        for row in selected_rows
        if row.get("retrieval_evidence", [{}])[0].get("selection_route")
        == "prototype_outlier_rescue"
    )
    diagnostics["selected_topics"] = [
        {
            "topic_id": topic_id,
            "label": str(topic_labels.get(str(topic_id), f"Topic {topic_id}")),
            "similarity": round(similarity_by_topic.get(topic_id, 0.0), 4),
            "raw_query_similarity": round(
                raw_similarity_by_topic.get(topic_id, 0.0), 4
            ),
            "label_similarity": round(
                label_similarity_by_topic.get(topic_id, 0.0), 4
            ),
            "from_direct_matches": topic_id in direct_topic_ids,
            "selection_route": (
                "Direct-title support"
                if topic_id in direct_topic_ids
                else "Semantic centroid + label"
            ),
        }
        for topic_id in sorted(
            selected_topic_ids,
            key=lambda item: similarity_by_topic.get(item, 0.0),
            reverse=True,
        )
    ]
    return selected_rows, diagnostics


def _build_query_prototypes(
    direct_embeddings: np.ndarray,
    *,
    max_prototypes: int,
    merge_similarity: float,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Build a small, deterministic set of semantic groups from direct matches."""
    embeddings = np.asarray(direct_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not len(embeddings):
        return [], []

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms != 0,
    )
    safe_max_prototypes = max(1, int(max_prototypes))
    used_minibatch_clustering = len(embeddings) > 96
    if used_minibatch_clustering:
        from sklearn.cluster import MiniBatchKMeans

        cluster_count = min(safe_max_prototypes, len(embeddings))
        labels = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=42,
            batch_size=min(256, len(embeddings)),
            n_init=3,
        ).fit_predict(embeddings)
        groups = [
            np.flatnonzero(labels == cluster_index).astype(int).tolist()
            for cluster_index in range(cluster_count)
        ]
        groups = [group for group in groups if group]
    else:
        groups = [[index] for index in range(len(embeddings))]

    while not used_minibatch_clustering and len(groups) > 1:
        centroids = [_normalized_centroid(embeddings[group]) for group in groups]
        best_pair: tuple[int, int] | None = None
        best_similarity = -1.0
        for left_index in range(len(centroids) - 1):
            for right_index in range(left_index + 1, len(centroids)):
                similarity = float(centroids[left_index] @ centroids[right_index])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (left_index, right_index)

        if best_pair is None:
            break
        if len(groups) <= safe_max_prototypes and best_similarity < merge_similarity:
            break

        left_index, right_index = best_pair
        groups[left_index] = sorted(groups[left_index] + groups[right_index])
        del groups[right_index]

    groups.sort(key=lambda group: (-len(group), group[0]))
    prototypes = [_normalized_centroid(embeddings[group]) for group in groups]
    return prototypes, groups


def _normalized_centroid(embeddings: np.ndarray) -> np.ndarray:
    centroid = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return centroid / norm if norm else centroid


def get_bertopic_topic_catalog(
    candidate_titles: list[dict[str, object]],
    index_dir: Path | None = None,
    excluded_story_ids: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Return every learned topic with corpus size, strength, and sample titles."""
    valid_rows = _valid_sorted_rows(candidate_titles)
    artifacts = load_or_build_bertopic_index(
        candidate_titles=valid_rows,
        index_dir=index_dir,
    )
    rows_by_story_id = {str(row["story_id"]): row for row in valid_rows}
    excluded = set(excluded_story_ids)
    members_by_topic: dict[int, list[tuple[float, float, str]]] = {}
    for story_id, topic_id, probability in zip(
        artifacts["story_ids"],
        artifacts["topic_ids"],
        artifacts["topic_probabilities"],
    ):
        numeric_topic_id = int(topic_id)
        if str(story_id) in excluded:
            continue
        source = rows_by_story_id[str(story_id)]
        members_by_topic.setdefault(numeric_topic_id, []).append(
            (
                float(probability),
                float(source.get("total_views", 0) or 0),
                str(source.get("page_title", "")),
            )
        )

    catalog = []
    for topic_id in sorted(members_by_topic):
        numeric_topic_id = int(topic_id)
        members = sorted(members_by_topic.get(numeric_topic_id, []), reverse=True)
        catalog.append(
            {
                "topic_id": numeric_topic_id,
                "topic_label": "Unassigned" if numeric_topic_id == -1 else str(
                    artifacts["topic_labels"].get(
                        str(numeric_topic_id), f"Topic {numeric_topic_id}"
                    )
                ),
                "title_count": len(members),
                "average_membership_probability": round(
                    (
                        sum(item[0] for item in members) / len(members)
                        if members
                        else 0.0
                    ),
                    4,
                ),
                "representative_titles": " | ".join(item[2] for item in members[:3]),
            }
        )
    return sorted(
        catalog,
        key=lambda item: (-int(item["title_count"]), int(item["topic_id"])),
    )


@_serialized_build
def load_or_build_bertopic_index(
    candidate_titles: list[dict[str, object]],
    index_dir: Path | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    valid_rows = _valid_sorted_rows(candidate_titles)
    if len(valid_rows) < DEFAULT_MIN_CORPUS_SIZE:
        raise ValueError(f"BERTopic requires at least {DEFAULT_MIN_CORPUS_SIZE} corpus titles.")

    selected_index_dir = index_dir or get_bertopic_index_dir()
    selected_index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = selected_index_dir / "manifest.json"
    assignments_path = selected_index_dir / "assignments.npz"
    topics_path = selected_index_dir / "topics.json"
    fingerprint = build_title_corpus_fingerprint(valid_rows)
    embedding_model = get_relationship_embedding_model_name()
    training_config = get_bertopic_training_config()

    manifest = _load_json(manifest_path)
    assignments_path = selected_index_dir / manifest.get("assignments_file", "assignments.npz")
    topics_path = selected_index_dir / manifest.get("topics_file", "topics.json")
    if (
        not force_rebuild
        and manifest.get("index_version") == BERTOPIC_INDEX_VERSION
        and manifest.get("corpus_fingerprint") == fingerprint
        and manifest.get("embedding_model") == embedding_model
        and manifest.get("training_config") == training_config
        and assignments_path.exists()
        and topics_path.exists()
    ):
        return _load_artifacts(manifest, selected_index_dir / manifest.get("assignments_file", assignments_path.name),
                               selected_index_dir / manifest.get("topics_file", topics_path.name))

    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except Exception as exc:
        raise RuntimeError(
            "BERTopic dependencies are unavailable. Install the project requirements first."
        ) from exc

    embeddings, story_ids, embedding_manifest = load_or_build_title_embedding_index(
        candidate_titles=valid_rows,
        model_name=embedding_model,
    )
    dense_embeddings = np.asarray(embeddings, dtype=np.float32)
    titles = [str(row["page_title"]) for row in valid_rows]
    min_cluster_size = training_config["min_cluster_size"]
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(
            n_neighbors=training_config["n_neighbors"],
            n_components=training_config["n_components"],
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            n_jobs=1,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=max(5, min_cluster_size),
            min_samples=training_config["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        ),
        calculate_probabilities=False,
        verbose=False,
    )
    topic_ids, probabilities = topic_model.fit_transform(titles, dense_embeddings)
    topic_ids_array = np.asarray(topic_ids, dtype=np.int32)
    probability_array = np.asarray(probabilities, dtype=np.float32)
    if (probability_array.ndim != 1 or len(probability_array) != len(topic_ids_array)
            or not np.isfinite(probability_array).all()
            or np.any((probability_array < 0) | (probability_array > 1))):
        raise ValueError("BERTopic returned invalid membership strengths; rebuild was not published.")

    unique_topic_ids = sorted({int(item) for item in topic_ids_array if int(item) != -1})
    topic_embeddings = []
    topic_labels: dict[str, str] = {}
    topic_info = topic_model.get_topic_info().set_index("Topic")
    for topic_id in unique_topic_ids:
        member_embeddings = dense_embeddings[topic_ids_array == topic_id]
        centroid = member_embeddings.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        topic_embeddings.append(centroid / norm if norm else centroid)
        words = [str(word) for word, _ in (topic_model.get_topic(topic_id) or [])[:5]]
        fallback_label = str(topic_info.loc[topic_id, "Name"]) if topic_id in topic_info.index else ""
        topic_labels[str(topic_id)] = ", ".join(words) or fallback_label or f"Topic {topic_id}"

    topic_embedding_array = np.asarray(topic_embeddings, dtype=np.float32).reshape(-1, dense_embeddings.shape[1])
    manifest = {
        "index_version": BERTOPIC_INDEX_VERSION,
        "corpus_fingerprint": fingerprint,
        "embedding_model": embedding_model,
        "embedding_dimension": int(embedding_manifest["embedding_dimension"]),
        "title_count": len(story_ids),
        "topic_count": len(unique_topic_ids),
        "outlier_count": int(np.count_nonzero(topic_ids_array == -1)),
        "min_cluster_size": max(5, min_cluster_size),
        "training_config": training_config,
    }
    _save_artifacts(
        manifest=manifest,
        assignments_path=assignments_path,
        topics_path=topics_path,
        manifest_path=manifest_path,
        story_ids=story_ids,
        topic_ids=topic_ids_array,
        topic_probabilities=probability_array,
        unique_topic_ids=np.asarray(unique_topic_ids, dtype=np.int32),
        topic_embeddings=topic_embedding_array,
        topic_labels=topic_labels,
    )
    return _load_artifacts(manifest, assignments_path, topics_path)


def _save_artifacts(
    *,
    manifest: dict[str, object],
    assignments_path: Path,
    topics_path: Path,
    manifest_path: Path,
    story_ids: list[str],
    topic_ids: np.ndarray,
    topic_probabilities: np.ndarray,
    unique_topic_ids: np.ndarray,
    topic_embeddings: np.ndarray,
    topic_labels: dict[str, str],
) -> None:
    # Publish immutable files first, then atomically switch a single manifest.
    # Concurrent processes may finish independently without mixing generations.
    generation = uuid.uuid4().hex
    temporary_assignments = manifest_path.parent / f"assignments-{generation}.npz"
    temporary_topics = manifest_path.parent / f"topics-{generation}.json"
    temporary_manifest = manifest_path.with_name(f"manifest-{generation}.tmp.json")
    manifest["assignments_file"] = temporary_assignments.name
    manifest["topics_file"] = temporary_topics.name
    np.savez_compressed(
        temporary_assignments,
        story_ids=np.asarray(story_ids, dtype=str),
        topic_ids=topic_ids,
        topic_probabilities=topic_probabilities,
        unique_topic_ids=unique_topic_ids,
        topic_embeddings=topic_embeddings,
    )
    temporary_topics.write_text(json.dumps(topic_labels, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)


def _load_artifacts(
    manifest: dict[str, Any],
    assignments_path: Path,
    topics_path: Path,
) -> dict[str, Any]:
    assignments_path = assignments_path.parent / manifest.get("assignments_file", assignments_path.name)
    topics_path = topics_path.parent / manifest.get("topics_file", topics_path.name)
    with np.load(assignments_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    count = len(arrays["story_ids"])
    if (len(arrays["topic_ids"]) != count or len(arrays["topic_probabilities"]) != count
            or count != manifest.get("title_count")
            or len(set(arrays["story_ids"].tolist())) != count):
        raise ValueError("BERTopic artifact assignments are inconsistent.")
    return {
        "manifest": manifest,
        "story_ids": [str(item) for item in arrays["story_ids"].tolist()],
        "topic_ids": arrays["topic_ids"],
        "topic_probabilities": arrays["topic_probabilities"],
        "unique_topic_ids": arrays["unique_topic_ids"],
        "topic_embeddings": arrays["topic_embeddings"],
        "topic_labels": _load_json(topics_path),
    }


def _valid_sorted_rows(candidate_titles: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        dict(row)
        for row in candidate_titles
        if str(row.get("story_id", "")).strip()
        and str(row.get("page_title", "")).strip()
    ]
    for row in rows:
        row["story_id"] = str(row["story_id"]).strip()
        row["page_title"] = " ".join(str(row["page_title"]).split())
    rows.sort(key=lambda row: str(row["story_id"]))
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
