from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection

import numpy as np
import pandas as pd


DEFAULT_RELATIONSHIP_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_MAX_RELATIONSHIPS: int | None = None
DEFAULT_MAX_TEXTS_PER_RELATIONSHIP = 3
DEFAULT_TOP_K_PER_ROUTE = 50
DEFAULT_MAX_CANDIDATES = 500
DEFAULT_MIN_SIMILARITY = 0.45
DEFAULT_EMBEDDING_ONLY_ACCEPTANCE_SIMILARITY = 0.68
DEFAULT_EMBEDDING_ONLY_ACCEPTANCE_CONFIDENCE = 0.75
DEFAULT_BERTOPIC_ACCEPTANCE_SIMILARITY = 0.50
DEFAULT_BERTOPIC_ACCEPTANCE_PROBABILITY = 0.20
RELATIONSHIP_RETRIEVAL_VERSION = "2026-08-18-non-direct-candidate-budget-v4"
DEFAULT_FAISS_HNSW_CONNECTIONS = 32
DEFAULT_FAISS_EF_SEARCH = 96


@dataclass(frozen=True)
class RelationshipRoute:
    relationship_id: str
    related_subject: str
    relationship_class: str
    relationship_family: str
    factual_bridge: str
    acceptance_condition: str
    rejection_rule: str
    allowed_story_angles: tuple[str, ...]
    excluded_story_angles: tuple[str, ...]
    retrieval_text: str
    quality_score: float
    relationship_role: str = ""
    can_retrieve_standalone: bool = True
    required_title_cues: tuple[str, ...] = ()


def get_relationship_embedding_model_name() -> str:
    return os.getenv(
        "RELATIONSHIP_EMBEDDING_MODEL",
        DEFAULT_RELATIONSHIP_EMBEDDING_MODEL,
    ).strip() or DEFAULT_RELATIONSHIP_EMBEDDING_MODEL


def get_relationship_embedding_index_dir() -> Path:
    configured = os.getenv("RELATIONSHIP_EMBEDDING_INDEX_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "embeddings"


def build_title_corpus_fingerprint(candidate_titles: list[dict[str, object]]) -> str:
    payload = "\n".join(
        f"{str(row.get('story_id', '')).strip()}\t{_clean_text(row.get('page_title', ''))}"
        for row in candidate_titles
        if str(row.get("story_id", "")).strip()
        and _clean_text(row.get("page_title", ""))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_relationship_routes(
    keyword_query: str,
    grounded_research: dict[str, object],
    max_relationships: int | None = DEFAULT_MAX_RELATIONSHIPS,
    max_texts_per_relationship: int = DEFAULT_MAX_TEXTS_PER_RELATIONSHIP,
    query_central_policy: bool = False,
) -> list[RelationshipRoute]:
    research_text = _clean_text(grounded_research.get("research_text", ""))
    if not keyword_query.strip() or not research_text:
        return []

    profile = _parse_profile(research_text)
    raw_relationships = profile.get("relationship_map", []) if profile else []
    if not isinstance(raw_relationships, list):
        raw_relationships = []
    if not raw_relationships:
        raw_relationships = _recover_relationship_objects(research_text)

    accepted: list[tuple[float, int, dict[str, Any]]] = []
    for source_index, relationship in enumerate(raw_relationships):
        if not isinstance(relationship, dict):
            continue
        quality_score = _relationship_quality_score(relationship)
        if quality_score < 0.60:
            continue
        accepted.append((quality_score, source_index, relationship))

    accepted.sort(
        key=lambda item: (
            _relationship_class_priority(item[2].get("relationship_class")),
            item[0],
            -item[1],
        ),
        reverse=True,
    )
    relationship_limit = (
        len(accepted) if max_relationships is None else max(1, max_relationships)
    )
    diverse_relationships: list[tuple[float, int, dict[str, Any]]] = []
    family_counts: dict[str, int] = {}
    deferred: list[tuple[float, int, dict[str, Any]]] = []
    for item in accepted:
        family = _clean_text(item[2].get("relationship_family")).casefold() or "other"
        if family_counts.get(family, 0) >= 5:
            deferred.append(item)
            continue
        family_counts[family] = family_counts.get(family, 0) + 1
        diverse_relationships.append(item)
        if len(diverse_relationships) >= relationship_limit:
            break
    if len(diverse_relationships) < relationship_limit:
        selected_ids = {
            _clean_text(item[2].get("relationship_id"))
            for item in diverse_relationships
        }
        for item in deferred:
            relationship_id = _clean_text(item[2].get("relationship_id"))
            if relationship_id in selected_ids:
                continue
            diverse_relationships.append(item)
            selected_ids.add(relationship_id)
            if len(diverse_relationships) >= relationship_limit:
                break
    accepted = diverse_relationships

    routes: list[RelationshipRoute] = []
    seen_route_texts: set[str] = set()
    for quality_score, _, relationship in accepted:
        relationship_id = _clean_text(relationship.get("relationship_id"))
        subject = _clean_text(relationship.get("related_subject"))
        relationship_class = _clean_text(relationship.get("relationship_class")).upper()
        family = _clean_text(relationship.get("relationship_family"))
        bridge = _clean_text(relationship.get("factual_bridge"))
        acceptance_condition = _clean_text(relationship.get("acceptance_condition"))
        rejection_rule = _clean_text(relationship.get("rejection_rule"))
        allowed_story_angles = tuple(
            _clean_text(item)
            for item in relationship.get("allowed_story_angles", [])
            if _clean_text(item)
        ) if isinstance(relationship.get("allowed_story_angles"), list) else ()
        excluded_story_angles = tuple(
            _clean_text(item)
            for item in relationship.get("excluded_story_angles", [])
            if _clean_text(item)
        ) if isinstance(relationship.get("excluded_story_angles"), list) else ()
        relationship_role = _clean_text(relationship.get("relationship_role")).upper()
        profile_allows_standalone = relationship.get("can_retrieve_standalone") is True
        context_only = query_central_policy and _is_context_only_relationship(
            relationship
        )
        can_retrieve_standalone = profile_allows_standalone and not context_only
        required_title_cues = tuple(
            _clean_text(item)
            for item in relationship.get("required_title_cues", [])
            if _clean_text(item)
        ) if isinstance(relationship.get("required_title_cues"), list) else ()
        route_texts = _relationship_retrieval_texts(
            keyword_query=keyword_query,
            relationship=relationship,
            max_texts=max_texts_per_relationship,
            query_central_policy=query_central_policy,
            can_retrieve_standalone=can_retrieve_standalone,
        )
        for route_text in route_texts:
            normalized = route_text.casefold()
            if normalized in seen_route_texts:
                continue
            seen_route_texts.add(normalized)
            routes.append(
                RelationshipRoute(
                    relationship_id=relationship_id,
                    related_subject=subject,
                    relationship_class=relationship_class,
                    relationship_family=family,
                    factual_bridge=bridge,
                    acceptance_condition=acceptance_condition,
                    rejection_rule=rejection_rule,
                    allowed_story_angles=allowed_story_angles,
                    excluded_story_angles=excluded_story_angles,
                    retrieval_text=route_text,
                    quality_score=round(quality_score, 3),
                    relationship_role=relationship_role,
                    can_retrieve_standalone=can_retrieve_standalone,
                    required_title_cues=required_title_cues,
                )
            )
    return routes


def retrieve_relationship_candidates(
    keyword_query: str,
    candidate_titles: list[dict[str, object]],
    grounded_research: dict[str, object],
    index_dir: Path | None = None,
    model_name: str | None = None,
    max_relationships: int | None = DEFAULT_MAX_RELATIONSHIPS,
    max_texts_per_relationship: int = DEFAULT_MAX_TEXTS_PER_RELATIONSHIP,
    top_k_per_route: int = DEFAULT_TOP_K_PER_ROUTE,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    query_central_policy: bool = False,
    excluded_story_ids: Collection[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    routes = extract_relationship_routes(
        keyword_query=keyword_query,
        grounded_research=grounded_research,
        max_relationships=max_relationships,
        max_texts_per_relationship=max_texts_per_relationship,
        query_central_policy=query_central_policy,
    )
    diagnostics: dict[str, object] = {
        "mode": "relationship_embeddings",
        "relationship_count": len({route.relationship_id for route in routes}),
        "route_count": len(routes),
        "candidate_count": 0,
        "fallback_reason": "",
    }
    if not routes:
        diagnostics["fallback_reason"] = (
            "No independently corroborated, source-backed relationships were available."
        )
        diagnostics["mode"] = "bounded_retrieval_no_candidates"
        return [], diagnostics

    valid_rows = [
        row
        for row in candidate_titles
        if str(row.get("story_id", "")).strip()
        and _clean_text(row.get("page_title", ""))
    ]
    valid_rows.sort(key=lambda row: str(row.get("story_id", "")).strip())
    if not valid_rows:
        diagnostics["fallback_reason"] = "No corpus titles were available for embedding retrieval."
        return [], diagnostics

    selected_model_name = model_name or get_relationship_embedding_model_name()
    selected_index_dir = index_dir or get_relationship_embedding_index_dir()
    embeddings, story_ids, manifest = _load_or_build_title_index(
        candidate_titles=valid_rows,
        index_dir=selected_index_dir,
        model_name=selected_model_name,
    )
    excluded_id_set = {
        str(story_id).strip()
        for story_id in (excluded_story_ids or ())
        if str(story_id).strip()
    }
    excluded_index_story_ids = excluded_id_set.intersection(story_ids)
    eligible_corpus_count = len(story_ids) - len(excluded_index_story_ids)
    diagnostics["excluded_story_count"] = len(excluded_index_story_ids)
    diagnostics["eligible_corpus_count"] = eligible_corpus_count
    if eligible_corpus_count <= 0:
        diagnostics["fallback_reason"] = (
            "No non-direct corpus titles remained after applying story exclusions."
        )
        return [], diagnostics

    model = _load_sentence_transformer(selected_model_name)
    route_embeddings = model.encode(
        [route.retrieval_text for route in routes],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)

    evidence_by_story_id: dict[str, list[dict[str, object]]] = {}
    safe_top_k = min(max(1, int(top_k_per_route)), eligible_corpus_count)
    # Search deeply enough that excluded direct matches cannot consume a route's
    # non-direct allowance. The persistent index still covers the complete corpus;
    # only result collection is query-specific.
    embedding_search_depth = min(
        len(story_ids),
        safe_top_k + len(excluded_index_story_ids),
    )
    top_scores, top_indices, search_backend = _search_title_embeddings(
        embeddings=embeddings,
        query_embeddings=route_embeddings,
        top_k=embedding_search_depth,
        index_dir=selected_index_dir,
        manifest=manifest,
        model_name=selected_model_name,
    )
    for route, route_scores, route_indices in zip(routes, top_scores, top_indices):
        collected_for_route = 0
        for similarity, index in zip(route_scores, route_indices):
            if int(index) < 0:
                continue
            similarity = float(similarity)
            if similarity < min_similarity:
                break
            story_id = story_ids[int(index)]
            if story_id in excluded_index_story_ids:
                continue
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
                    "similarity": round(similarity, 4),
                    "relationship_quality": route.quality_score,
                    "match_method": "embedding",
                    "relationship_role": route.relationship_role,
                    "can_retrieve_standalone": route.can_retrieve_standalone,
                    "required_title_cues": list(route.required_title_cues),
                }
            )
            collected_for_route += 1
            if collected_for_route >= safe_top_k:
                break

    normalized_titles = [
        _normalize_for_lexical_match(row.get("page_title", ""))
        for row in valid_rows
    ]
    seen_relationship_ids: set[str] = set()
    for route in routes:
        if not route.can_retrieve_standalone:
            continue
        if route.relationship_id in seen_relationship_ids:
            continue
        seen_relationship_ids.add(route.relationship_id)
        lexical_phrases = _lexical_subject_phrases(route.related_subject)
        if not lexical_phrases:
            continue
        for index, normalized_title in enumerate(normalized_titles):
            matched_phrase = next(
                (
                    phrase
                    for phrase in lexical_phrases
                    if f" {phrase} " in f" {normalized_title} "
                ),
                "",
            )
            if not matched_phrase:
                continue
            story_id = story_ids[index]
            if story_id in excluded_index_story_ids:
                continue
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
                    "retrieval_text": matched_phrase,
                    "similarity": 1.0,
                    "relationship_quality": route.quality_score,
                    "match_method": "lexical_relationship_subject",
                    "relationship_role": route.relationship_role,
                    "can_retrieve_standalone": route.can_retrieve_standalone,
                    "required_title_cues": list(route.required_title_cues),
                }
            )

    rows_by_story_id = {
        str(row.get("story_id", "")).strip(): row
        for row in valid_rows
    }
    ranked: list[tuple[float, float, int, str]] = []
    for story_id, evidence in evidence_by_story_id.items():
        evidence.sort(
            key=lambda item: (
                float(item.get("similarity", 0.0)),
                float(item.get("relationship_quality", 0.0)),
            ),
            reverse=True,
        )
        unique_relationships = len(
            {str(item.get("relationship_id", "")) for item in evidence}
        )
        ranked.append(
            (
                float(evidence[0].get("similarity", 0.0)),
                float(evidence[0].get("relationship_quality", 0.0)),
                unique_relationships,
                story_id,
            )
        )
    ranked.sort(reverse=True)

    selected_rows: list[dict[str, object]] = []
    for _, _, _, story_id in ranked[: max(1, int(max_candidates))]:
        source = dict(rows_by_story_id[story_id])
        source["retrieval_evidence"] = evidence_by_story_id[story_id][:5]
        selected_rows.append(source)

    diagnostics.update(
        {
            "candidate_count": len(selected_rows),
            "embedding_model": selected_model_name,
            "embedding_dimension": int(manifest["embedding_dimension"]),
            "index_corpus_fingerprint": str(manifest["corpus_fingerprint"]),
            "retrieval_version": RELATIONSHIP_RETRIEVAL_VERSION,
            "search_backend": search_backend,
            "embedding_search_depth": embedding_search_depth,
        }
    )
    return selected_rows, diagnostics


def _search_title_embeddings(
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
    index_dir: Path,
    manifest: dict[str, object],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load a prebuilt FAISS index, otherwise use safe exact NumPy search.

    FAISS index construction is deliberately excluded from interactive requests.
    On Windows, building HNSW after PyTorch inference can stall a Streamlit worker.
    """
    try:
        import faiss

        model_key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        index_path = index_dir / f"title_hnsw_{model_key}.faiss"
        metadata_path = index_dir / f"title_hnsw_{model_key}.json"
        expected_fingerprint = str(manifest.get("corpus_fingerprint", ""))
        metadata = _load_manifest(metadata_path)
        if not (
            index_path.exists()
            and metadata.get("corpus_fingerprint") == expected_fingerprint
            and int(metadata.get("title_count", -1)) == len(embeddings)
        ):
            raise FileNotFoundError("A current prebuilt FAISS index is unavailable.")
        index = faiss.read_index(str(index_path))
        index.hnsw.efSearch = max(DEFAULT_FAISS_EF_SEARCH, int(top_k))
        scores, indices = index.search(
            np.ascontiguousarray(query_embeddings, dtype=np.float32), int(top_k)
        )
        return scores, indices, "faiss_hnsw"
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError):
        scores, indices = _exact_embedding_search(embeddings, query_embeddings, top_k)
        return scores, indices, "numpy_exact"


def _exact_embedding_search(
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = np.asarray(query_embeddings @ embeddings.T, dtype=np.float32)
    indices = np.argpartition(similarities, -top_k, axis=1)[:, -top_k:]
    scores = np.take_along_axis(similarities, indices, axis=1)
    order = np.argsort(scores, axis=1)[:, ::-1]
    return (
        np.take_along_axis(scores, order, axis=1),
        np.take_along_axis(indices, order, axis=1),
    )


def build_faiss_title_index(
    embeddings: np.ndarray,
    index_dir: Path,
    manifest: dict[str, object],
    model_name: str,
) -> Path:
    """Build the persistent HNSW index in an explicit offline process."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required to build the title index.") from exc

    index_dir.mkdir(parents=True, exist_ok=True)
    model_key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    index_path = index_dir / f"title_hnsw_{model_key}.faiss"
    metadata_path = index_dir / f"title_hnsw_{model_key}.json"
    dimension = int(embeddings.shape[1])
    index = faiss.IndexHNSWFlat(
        dimension,
        DEFAULT_FAISS_HNSW_CONNECTIONS,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efConstruction = 160
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    temporary_path = index_path.with_suffix(".tmp.faiss")
    faiss.write_index(index, str(temporary_path))
    temporary_path.replace(index_path)
    metadata_path.write_text(
        json.dumps(
            {
                "corpus_fingerprint": str(manifest.get("corpus_fingerprint", "")),
                "title_count": len(embeddings),
                "dimension": dimension,
                "index_type": "IndexHNSWFlat",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return index_path


def passes_relationship_evidence_gate(selected_title: dict[str, object]) -> bool:
    if (
        str(selected_title.get("ai_evidence_scope", "")).strip().upper()
        == "DIRECT_QUERY"
        and str(
            selected_title.get("ai_query_specific_title_evidence", "")
        ).strip()
        and selected_title.get("ai_query_is_non_replaceable") is True
    ):
        # The generative normalizer already verified this value as an exact
        # contiguous quote from the supplied candidate headline.
        return True
    evidence = selected_title.get("retrieval_evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return False
    valid_evidence = [item for item in evidence if isinstance(item, dict)]
    if any(
        item.get("match_method") == "lexical_relationship_subject"
        for item in valid_evidence
    ):
        return True

    maximum_similarity = max(
        (
            float(item.get("similarity", 0.0))
            for item in valid_evidence
            if item.get("match_method") == "embedding"
        ),
        default=0.0,
    )
    minimum_similarity = float(
        os.getenv(
            "RELATIONSHIP_EMBEDDING_ACCEPTANCE_SIMILARITY",
            str(DEFAULT_EMBEDDING_ONLY_ACCEPTANCE_SIMILARITY),
        )
    )
    minimum_confidence = float(
        os.getenv(
            "RELATIONSHIP_EMBEDDING_ACCEPTANCE_CONFIDENCE",
            str(DEFAULT_EMBEDDING_ONLY_ACCEPTANCE_CONFIDENCE),
        )
    )
    relationship_embedding_passes = (
        maximum_similarity >= minimum_similarity
        and int(selected_title.get("ai_relevance_level", 0)) >= 2
        and float(selected_title.get("ai_confidence", 0.0)) >= minimum_confidence
    )
    topic_evidence = [
        item for item in valid_evidence if item.get("match_method") == "bertopic_topic"
    ]
    maximum_topic_similarity = max(
        (float(item.get("similarity", 0.0)) for item in topic_evidence),
        default=0.0,
    )
    maximum_topic_probability = max(
        (float(item.get("topic_probability", 0.0)) for item in topic_evidence),
        default=0.0,
    )
    minimum_topic_similarity = float(
        os.getenv(
            "BERTOPIC_ACCEPTANCE_SIMILARITY",
            str(DEFAULT_BERTOPIC_ACCEPTANCE_SIMILARITY),
        )
    )
    minimum_topic_probability = float(
        os.getenv(
            "BERTOPIC_ACCEPTANCE_PROBABILITY",
            str(DEFAULT_BERTOPIC_ACCEPTANCE_PROBABILITY),
        )
    )
    topic_evidence_passes = (
        maximum_topic_similarity >= minimum_topic_similarity
        and maximum_topic_probability >= minimum_topic_probability
        and int(selected_title.get("ai_relevance_level", 0)) >= 2
        and float(selected_title.get("ai_confidence", 0.0)) >= minimum_confidence
    )
    return relationship_embedding_passes or topic_evidence_passes


def load_or_build_title_embedding_index(
    candidate_titles: list[dict[str, object]],
    index_dir: Path | None = None,
    model_name: str | None = None,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Public access to the persistent multilingual title-embedding index."""
    valid_rows = [
        dict(row)
        for row in candidate_titles
        if str(row.get("story_id", "")).strip()
        and _clean_text(row.get("page_title", ""))
    ]
    valid_rows.sort(key=lambda row: str(row.get("story_id", "")).strip())
    return _load_or_build_title_index(
        candidate_titles=valid_rows,
        index_dir=index_dir or get_relationship_embedding_index_dir(),
        model_name=model_name or get_relationship_embedding_model_name(),
    )


def encode_relationship_texts(
    texts: list[str],
    model_name: str | None = None,
) -> np.ndarray:
    """Encode normalized retrieval text with the configured multilingual model."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = _load_sentence_transformer(
        model_name or get_relationship_embedding_model_name()
    )
    return model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)


def _load_or_build_title_index(
    candidate_titles: list[dict[str, object]],
    index_dir: Path,
    model_name: str,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    index_dir.mkdir(parents=True, exist_ok=True)
    model_key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    embedding_path = index_dir / f"title_embeddings_{model_key}.npy"
    story_ids_path = index_dir / f"story_ids_{model_key}.json"
    manifest_path = index_dir / f"manifest_{model_key}.json"
    fingerprint = build_title_corpus_fingerprint(candidate_titles)

    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("model_name") == model_name
        and manifest.get("corpus_fingerprint") == fingerprint
        and embedding_path.exists()
        and story_ids_path.exists()
    ):
        embeddings = np.load(embedding_path, mmap_mode="r")
        story_ids = json.loads(story_ids_path.read_text(encoding="utf-8"))
        if len(embeddings) == len(story_ids) == len(candidate_titles):
            return embeddings, [str(item) for item in story_ids], manifest

    model = _load_sentence_transformer(model_name)
    titles = [_clean_text(row.get("page_title", "")) for row in candidate_titles]
    story_ids = [str(row.get("story_id", "")).strip() for row in candidate_titles]
    embeddings = model.encode(
        titles,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    manifest = {
        "embedding_index_version": "2026-08-13-title-embeddings-v1",
        "retrieval_version": RELATIONSHIP_RETRIEVAL_VERSION,
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "title_count": len(story_ids),
        "corpus_fingerprint": fingerprint,
    }

    temporary_embedding_path = embedding_path.with_suffix(".tmp.npy")
    temporary_story_ids_path = story_ids_path.with_suffix(".tmp.json")
    temporary_manifest_path = manifest_path.with_suffix(".tmp.json")
    np.save(temporary_embedding_path, embeddings)
    temporary_story_ids_path.write_text(
        json.dumps(story_ids, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_embedding_path.replace(embedding_path)
    temporary_story_ids_path.replace(story_ids_path)
    temporary_manifest_path.replace(manifest_path)
    return np.load(embedding_path, mmap_mode="r"), story_ids, manifest


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers is unavailable.") from exc
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model '{model_name}' is not available in the local model cache."
        ) from exc


def _parse_profile(research_text: str) -> dict[str, Any]:
    cleaned = research_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _recover_relationship_objects(research_text: str) -> list[dict[str, Any]]:
    match = re.search(r'"relationship_map"\s*:\s*\[', research_text)
    if not match:
        return []
    array_start = research_text.find("[", match.start())
    objects: list[dict[str, Any]] = []
    depth = 0
    object_start: int | None = None
    in_string = False
    escaped = False
    for index in range(array_start + 1, len(research_text)):
        character = research_text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and object_start is not None:
                try:
                    parsed = json.loads(research_text[object_start : index + 1])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    objects.append(parsed)
                object_start = None
        elif character == "]" and depth == 0:
            break
    recovered_ids = {
        _clean_text(item.get("relationship_id"))
        for item in objects
        if _clean_text(item.get("relationship_id"))
    }
    relationship_markers = list(
        re.finditer(r'"relationship_id"\s*:\s*"', research_text[array_start:])
    )
    for marker_index, marker in enumerate(relationship_markers):
        segment_start = array_start + marker.start()
        segment_end = (
            array_start + relationship_markers[marker_index + 1].start()
            if marker_index + 1 < len(relationship_markers)
            else min(len(research_text), segment_start + 12000)
        )
        segment = research_text[segment_start:segment_end]
        partial = {
            field: _extract_json_string_field(segment, field)
            for field in (
                "relationship_id",
                "related_subject",
                "related_subject_type",
                "relationship_class",
                "relationship_family",
                "relationship_role",
                "factual_bridge",
                "evidence_summary",
                "confidence",
                "false_positive_risk",
                "acceptance_condition",
                "rejection_rule",
            )
        }
        standalone_match = re.search(
            r'"can_retrieve_standalone"\s*:\s*(true|false)',
            segment,
            flags=re.IGNORECASE,
        )
        if standalone_match:
            partial["can_retrieve_standalone"] = (
                standalone_match.group(1).casefold() == "true"
            )
        relationship_id = _clean_text(partial.get("relationship_id"))
        if not relationship_id or relationship_id in recovered_ids:
            continue
        manifestation_match = re.search(
            r'"editorial_manifestations"\s*:\s*\[(.*?)\]',
            segment,
            flags=re.DOTALL,
        )
        if manifestation_match:
            partial["editorial_manifestations"] = [
                bytes(value, "utf-8").decode("unicode_escape")
                for value in re.findall(
                    r'"((?:\\.|[^"\\])*)"',
                    manifestation_match.group(1),
                )
            ]
        for array_field in (
            "required_title_cues",
            "allowed_story_angles",
            "excluded_story_angles",
        ):
            array_match = re.search(
                rf'"{array_field}"\s*:\s*\[(.*?)\]',
                segment,
                flags=re.DOTALL,
            )
            if array_match:
                partial[array_field] = [
                    bytes(value, "utf-8").decode("unicode_escape")
                    for value in re.findall(
                        r'"((?:\\.|[^"\\])*)"',
                        array_match.group(1),
                    )
                ]
        if partial.get("related_subject") and partial.get("factual_bridge"):
            objects.append(partial)
            recovered_ids.add(relationship_id)
    return objects


def _extract_json_string_field(text: str, field: str) -> str:
    match = re.search(
        rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def _relationship_quality_score(relationship: dict[str, Any]) -> float:
    subject = _clean_text(relationship.get("related_subject"))
    bridge = _clean_text(relationship.get("factual_bridge"))
    relationship_class = _clean_text(relationship.get("relationship_class")).upper()
    if not subject or not bridge or relationship_class not in {
        "DIRECT",
        "CORE_RELATED",
        "CONTEXTUAL",
    }:
        return 0.0
    if int(relationship.get("verification_branch_count", 0)) < 2:
        return 0.0
    evidence_urls = relationship.get("evidence_source_urls")
    if not isinstance(evidence_urls, list) or not any(
        _clean_text(url) for url in evidence_urls
    ):
        return 0.0
    if not _clean_text(relationship.get("interpretation_id")):
        return 0.0
    if not isinstance(relationship.get("can_retrieve_standalone"), bool):
        return 0.0
    if not _clean_text(relationship.get("acceptance_condition")):
        return 0.0
    if not _clean_text(relationship.get("rejection_rule")):
        return 0.0

    confidence = _confidence_score(relationship.get("confidence"))
    class_score = {
        "DIRECT": 1.0,
        "CORE_RELATED": 0.95,
        "CONTEXTUAL": 0.70,
    }[relationship_class]
    evidence_score = 1.0 if _clean_text(relationship.get("evidence_summary")) else 0.5
    specificity_score = 1.0 if len(subject.split()) >= 2 else 0.75
    manifestations = relationship.get("editorial_manifestations")
    manifestation_score = 1.0 if isinstance(manifestations, list) and manifestations else 0.5
    policy_score = 1.0 if (
        _clean_text(relationship.get("acceptance_condition"))
        and isinstance(relationship.get("excluded_story_angles"), list)
        and relationship.get("excluded_story_angles")
    ) else 0.5
    risk = _clean_text(relationship.get("false_positive_risk")).casefold()
    risk_score = {"low": 1.0, "medium": 0.65, "high": 0.2}.get(risk, 0.5)
    return (
        confidence * 0.30
        + class_score * 0.20
        + evidence_score * 0.15
        + specificity_score * 0.10
        + manifestation_score * 0.10
        + risk_score * 0.10
        + policy_score * 0.05
    )


def _relationship_retrieval_texts(
    keyword_query: str,
    relationship: dict[str, Any],
    max_texts: int,
    query_central_policy: bool = False,
    can_retrieve_standalone: bool = True,
) -> list[str]:
    subject = _clean_text(relationship.get("related_subject"))
    bridge = _clean_text(relationship.get("factual_bridge"))
    family = _clean_text(relationship.get("relationship_family"))
    candidates = [subject] if can_retrieve_standalone else []
    manifestations = relationship.get("editorial_manifestations")
    if isinstance(manifestations, list):
        if not can_retrieve_standalone:
            candidates.extend(
                ". ".join(part for part in (keyword_query, _clean_text(item)) if part)
                for item in manifestations
            )
        else:
            candidates.extend(_clean_text(item) for item in manifestations)
    combined_parts = (subject, family, bridge)
    if not can_retrieve_standalone:
        combined_parts = (keyword_query, subject, family, bridge)
    candidates.append(". ".join(part for part in combined_parts if part))

    original_query = " ".join(keyword_query.casefold().split())
    selected: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.casefold().split())
        if not normalized or normalized == original_query or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(candidate)
        if len(selected) >= max(1, max_texts):
            break
    return selected


def _is_context_only_relationship(relationship: dict[str, Any]) -> bool:
    if relationship.get("can_retrieve_standalone") is False:
        return True
    role = _clean_text(relationship.get("relationship_role")).upper()
    if role in {"GEOGRAPHIC_CONTEXT", "GENERIC_TOPIC"}:
        return True
    subject = _clean_text(relationship.get("related_subject"))
    compact_subject = re.sub(r"[^A-Za-z0-9]", "", subject)
    if (
        "INTELLIGENCE" in role
        and len(subject.split()) == 1
        and len(compact_subject) <= 4
    ):
        # Names such as Israel's military-intelligence directorate "Aman"
        # collide heavily with ordinary personal names.  They require an
        # explicitly query-anchored route instead of standalone retrieval.
        return True
    subject_type = _clean_text(relationship.get("related_subject_type")).upper()
    if subject_type in {
        "LOCATION",
        "PLACE",
        "COUNTRY",
        "REGION",
        "CITY",
        "STATE",
        # A religion associated with a country is contextual; a generic story
        # about that religion is not automatically about the country.
        "RELIGION",
    }:
        return True
    family = _clean_text(relationship.get("relationship_family")).casefold()
    return any(
        marker in family
        for marker in ("geograph", "location", "broad topic", "generic topic")
    )


def _lexical_subject_phrases(subject: str) -> list[str]:
    candidates = [subject]
    candidates.extend(re.findall(r"\(([^)]+)\)", subject))
    if "(" in subject:
        candidates.append(subject.split("(", 1)[0])
    phrases: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        original_compact = re.sub(r"[^\w]", "", str(candidate), flags=re.UNICODE)
        normalized = _normalize_for_lexical_match(candidate)
        token_count = len(normalized.split())
        if not normalized or normalized in seen:
            continue
        if token_count < 2 and not (
            3 <= len(original_compact) <= 10 and original_compact.isupper()
        ):
            continue
        seen.add(normalized)
        phrases.append(normalized)
    return phrases


def _normalize_for_lexical_match(value: object) -> str:
    return " ".join(re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE))


def _confidence_score(value: object) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    normalized = _clean_text(value).casefold()
    return {
        "very high": 1.0,
        "high": 0.9,
        "medium": 0.65,
        "moderate": 0.65,
        "low": 0.3,
    }.get(normalized, 0.5)


def _relationship_class_priority(value: object) -> int:
    return {
        "DIRECT": 3,
        "CORE_RELATED": 2,
        "CONTEXTUAL": 1,
    }.get(_clean_text(value).upper(), 0)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())
