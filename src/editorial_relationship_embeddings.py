from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EDITORIAL_EMBEDDINGS_RUNTIME_VERSION = "2026-08-06-truncated-graph-recovery-v1"

DEFAULT_RELATIONSHIP_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_MAX_RELATIONSHIPS = 18
DEFAULT_MAX_TEXTS_PER_RELATIONSHIP = 3
DEFAULT_TOP_K_PER_ROUTE = 150
DEFAULT_MAX_CANDIDATES = 3000
DEFAULT_MIN_SIMILARITY = 0.35
DEFAULT_EMBEDDING_ONLY_ACCEPTANCE_SIMILARITY = 0.68
DEFAULT_EMBEDDING_ONLY_ACCEPTANCE_CONFIDENCE = 0.75
RELATIONSHIP_RETRIEVAL_VERSION = "2026-08-05-editorial-v2-relationship-embeddings-v1"


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


def get_relationship_embedding_model_name() -> str:
    return os.getenv(
        "EDITORIAL_RELATIONSHIP_EMBEDDING_MODEL",
        DEFAULT_RELATIONSHIP_EMBEDDING_MODEL,
    ).strip() or DEFAULT_RELATIONSHIP_EMBEDDING_MODEL


def get_relationship_embedding_index_dir() -> Path:
    configured = os.getenv("EDITORIAL_RELATIONSHIP_EMBEDDING_INDEX_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "editorial_embeddings"


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
    max_relationships: int = DEFAULT_MAX_RELATIONSHIPS,
    max_texts_per_relationship: int = DEFAULT_MAX_TEXTS_PER_RELATIONSHIP,
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
    if not raw_relationships:
        raw_relationships = _recover_knowledge_graph_relationships(research_text)
    manifestation_lookup = _build_editorial_manifestation_lookup(profile)
    raw_relationships = [
        _normalize_editorial_v2_relationship(relationship, manifestation_lookup)
        for relationship in raw_relationships
        if isinstance(relationship, dict)
    ]

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
    relationship_limit = max(1, max_relationships)
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
        route_texts = _relationship_retrieval_texts(
            keyword_query=keyword_query,
            relationship=relationship,
            max_texts=max_texts_per_relationship,
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
                )
            )
    return routes


def retrieve_relationship_candidates(
    keyword_query: str,
    candidate_titles: list[dict[str, object]],
    grounded_research: dict[str, object],
    index_dir: Path | None = None,
    model_name: str | None = None,
    max_relationships: int = DEFAULT_MAX_RELATIONSHIPS,
    max_texts_per_relationship: int = DEFAULT_MAX_TEXTS_PER_RELATIONSHIP,
    top_k_per_route: int = DEFAULT_TOP_K_PER_ROUTE,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    routes = extract_relationship_routes(
        keyword_query=keyword_query,
        grounded_research=grounded_research,
        max_relationships=max_relationships,
        max_texts_per_relationship=max_texts_per_relationship,
    )
    diagnostics: dict[str, object] = {
        "mode": "relationship_embeddings",
        "relationship_count": len({route.relationship_id for route in routes}),
        "route_count": len(routes),
        "candidate_count": 0,
        "fallback_reason": "",
    }
    if not routes:
        diagnostics["fallback_reason"] = "No validated structured relationships were available."
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
    model = _load_sentence_transformer(selected_model_name)
    route_embeddings = model.encode(
        [route.retrieval_text for route in routes],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)

    evidence_by_story_id: dict[str, list[dict[str, object]]] = {}
    safe_top_k = min(max(1, int(top_k_per_route)), len(story_ids))
    for route, route_embedding in zip(routes, route_embeddings):
        similarities = np.asarray(embeddings @ route_embedding, dtype=np.float32)
        top_indices = np.argpartition(similarities, -safe_top_k)[-safe_top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        for index in top_indices:
            similarity = float(similarities[index])
            if similarity < min_similarity:
                continue
            story_id = story_ids[int(index)]
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
                }
            )

    normalized_titles = [
        _normalize_for_lexical_match(row.get("page_title", ""))
        for row in valid_rows
    ]
    seen_relationship_ids: set[str] = set()
    for route in routes:
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
        }
    )
    return selected_rows, diagnostics


def passes_relationship_evidence_gate(selected_title: dict[str, object]) -> bool:
    evidence = selected_title.get("retrieval_evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return True
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
    return (
        maximum_similarity >= minimum_similarity
        and int(selected_title.get("ai_relevance_level", 0)) >= 2
        and float(selected_title.get("ai_confidence", 0.0)) >= minimum_confidence
    )


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
        manifest.get("retrieval_version") == RELATIONSHIP_RETRIEVAL_VERSION
        and manifest.get("model_name") == model_name
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
                "factual_bridge",
                "evidence_summary",
                "confidence",
                "false_positive_risk",
                "rejection_rule",
            )
        }
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
        if partial.get("related_subject") and partial.get("factual_bridge"):
            objects.append(partial)
            recovered_ids.add(relationship_id)
    return objects


def _recover_knowledge_graph_relationships(
    research_text: str,
) -> list[dict[str, Any]]:
    """Recover retrieval routes from completed graph edges in truncated JSON."""
    nodes = _recover_named_array_objects(research_text, "nodes")
    edges = _recover_named_array_objects(research_text, "edges")
    labels_by_id = {
        _clean_text(node.get("node_id")): _clean_text(
            node.get("canonical_label", node.get("label", ""))
        )
        for node in nodes
        if _clean_text(node.get("node_id"))
    }
    relationships: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        target_id = _clean_text(edge.get("target_node_id"))
        subject = labels_by_id.get(target_id) or target_id.replace("_", " ").strip()
        bridge = _clean_text(edge.get("factual_bridge"))
        if not subject or not bridge:
            continue
        importance = _confidence_score(edge.get("editorial_importance"))
        relationship_class = "CORE_RELATED" if importance >= 0.75 else "CONTEXTUAL"
        edge_id = _clean_text(edge.get("edge_id")) or f"recovered_edge_{index + 1}"
        relationships.append(
            {
                "relationship_id": f"graph_{edge_id}",
                "related_subject": subject,
                "related_subject_type": "OTHER",
                "relationship_class": relationship_class,
                "relationship_family": _clean_text(edge.get("relationship_family")),
                "factual_bridge": bridge,
                "evidence_summary": bridge,
                "factual_confidence": edge.get("factual_confidence", 0.75),
                "bridge_confidence": edge.get("editorial_confidence", 0.75),
                "editorial_confidence": edge.get("editorial_confidence", 0.75),
                "false_positive_risk": edge.get("false_positive_risk", 0.25),
                "acceptance_condition": (
                    f"The title must materially concern {subject} in the context of: {bridge}"
                ),
                "allowed_story_angles": [bridge],
                "excluded_story_angles": [
                    f"Incidental or ambiguous mentions of {subject} without the stated bridge"
                ],
                "rejection_rule": (
                    f"Reject titles about {subject} that do not express the supported relationship."
                ),
            }
        )
    return relationships


def _recover_named_array_objects(
    research_text: str,
    field_name: str,
) -> list[dict[str, Any]]:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[', research_text)
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
        elif character == "}" and depth:
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


def _build_editorial_manifestation_lookup(
    profile: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(profile, dict):
        return {}
    manifestations = profile.get("editorial_manifestations", [])
    if not isinstance(manifestations, list):
        return {}
    return {
        manifestation_id: item
        for item in manifestations
        if isinstance(item, dict)
        and (manifestation_id := _clean_text(item.get("manifestation_id")))
    }


def _normalize_editorial_v2_relationship(
    relationship: dict[str, Any],
    manifestation_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = dict(relationship)
    references: list[object] = []
    for field in (
        "editorial_manifestations",
        "editorial_manifestation_ids",
        "linked_editorial_manifestation_ids",
    ):
        value = relationship.get(field)
        if isinstance(value, list):
            references.extend(value)

    manifestation_texts: list[str] = []
    seen: set[str] = set()
    for reference in references:
        manifestation = reference if isinstance(reference, dict) else None
        if manifestation is None:
            manifestation = manifestation_lookup.get(_clean_text(reference))
        if manifestation is None:
            text_candidates: list[object] = [reference]
        else:
            text_candidates = [
                manifestation.get("manifestation_subject"),
                manifestation.get("headline_angle"),
            ]
            positive_cues = manifestation.get("positive_title_cues")
            if isinstance(positive_cues, list):
                text_candidates.extend(positive_cues)
        for candidate in text_candidates:
            cleaned = _clean_text(candidate)
            normalized_text = cleaned.casefold()
            if cleaned and normalized_text not in seen:
                seen.add(normalized_text)
                manifestation_texts.append(cleaned)

    normalized["editorial_manifestations"] = manifestation_texts
    return normalized


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

    confidence_values = [
        relationship.get(field)
        for field in (
            "factual_confidence",
            "bridge_confidence",
            "editorial_confidence",
        )
        if relationship.get(field) is not None
    ]
    confidence = (
        min(_confidence_score(value) for value in confidence_values)
        if confidence_values
        else _confidence_score(relationship.get("confidence"))
    )
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
    risk_value = relationship.get("false_positive_risk")
    if isinstance(risk_value, (int, float)):
        risk_score = 1.0 - max(0.0, min(1.0, float(risk_value)))
    else:
        risk = _clean_text(risk_value).casefold()
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
) -> list[str]:
    subject = _clean_text(relationship.get("related_subject"))
    bridge = _clean_text(relationship.get("factual_bridge"))
    family = _clean_text(relationship.get("relationship_family"))
    candidates = [subject]
    manifestations = relationship.get("editorial_manifestations")
    if isinstance(manifestations, list):
        candidates.extend(_clean_text(item) for item in manifestations)
    for field in ("allowed_story_angles", "positive_title_cues", "required_co_cues"):
        values = relationship.get(field)
        if isinstance(values, list):
            candidates.extend(_clean_text(item) for item in values)
    candidates.append(". ".join(part for part in (subject, family, bridge) if part))

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
