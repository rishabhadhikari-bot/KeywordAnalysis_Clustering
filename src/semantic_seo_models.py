"""Lazy local-model adapters for the isolated Semantic SEO pipeline."""

from __future__ import annotations

import hashlib
import importlib.machinery
import os
import re
import sqlite3
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.search_aliases import SEARCH_ALIAS_ENTITIES


DEFAULT_SEMANTIC_SEO_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_SEMANTIC_SEO_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_SEMANTIC_SEO_GLINER_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_SEMANTIC_SEO_GLINER_ENCODER_MODEL = "microsoft/deberta-v3-base"
DEFAULT_SEMANTIC_SEO_NLI_MODEL = (
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)
DEFAULT_SEMANTIC_SEO_RELATION_MODEL = "Babelscape/mrebel-base"

SEMANTIC_SEO_ENTITY_LABELS = (
    "person",
    "organization",
    "country",
    "place",
    "geopolitical conflict",
    "event",
    "policy",
    "law",
    "commodity",
    "financial instrument",
    "shipping route",
    "industry",
    "product",
    "technology",
    "sports team",
    "creative work",
)
TITLE_ENTITY_CACHE_VERSION = "2026-08-25-gliner-canonical-v1"
DEFAULT_TITLE_ENTITY_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "semantic_seo"
    / "title_entities.db"
)


@dataclass(frozen=True)
class SemanticSEOModelSettings:
    embedding_model: str
    reranker_model: str
    gliner_model: str
    nli_model: str
    relation_model: str
    enable_mrebel: bool


def load_semantic_seo_model_settings() -> SemanticSEOModelSettings:
    return SemanticSEOModelSettings(
        embedding_model=_setting(
            "SEMANTIC_SEO_EMBEDDING_MODEL",
            DEFAULT_SEMANTIC_SEO_EMBEDDING_MODEL,
        ),
        reranker_model=_setting(
            "SEMANTIC_SEO_RERANKER_MODEL",
            DEFAULT_SEMANTIC_SEO_RERANKER_MODEL,
        ),
        gliner_model=_setting(
            "SEMANTIC_SEO_GLINER_MODEL",
            DEFAULT_SEMANTIC_SEO_GLINER_MODEL,
        ),
        nli_model=_setting(
            "SEMANTIC_SEO_NLI_MODEL",
            DEFAULT_SEMANTIC_SEO_NLI_MODEL,
        ),
        relation_model=_setting(
            "SEMANTIC_SEO_RELATION_MODEL",
            DEFAULT_SEMANTIC_SEO_RELATION_MODEL,
        ),
        # mREBEL's published checkpoint is non-commercial. It must be an
        # explicit deployment decision rather than an automatic dependency.
        enable_mrebel=_environment_flag("SEMANTIC_SEO_ENABLE_MREBEL", False),
    )


def local_model_status(settings: SemanticSEOModelSettings) -> dict[str, dict[str, object]]:
    models = {
        "BGE-M3 embeddings": settings.embedding_model,
        "BGE reranker": settings.reranker_model,
        "GLiNER": settings.gliner_model,
        "DeBERTa title relevance": settings.nli_model,
        "mREBEL": settings.relation_model,
    }
    status = {}
    for label, model_name in models.items():
        available, detail = _local_snapshot_status(model_name)
        if (
            label == "GLiNER"
            and available
            and model_name == DEFAULT_SEMANTIC_SEO_GLINER_MODEL
        ):
            encoder_available, encoder_detail = _local_snapshot_status(
                DEFAULT_SEMANTIC_SEO_GLINER_ENCODER_MODEL
            )
            if not encoder_available:
                available = False
                detail = (
                    "GLiNER's encoder dependency is unavailable: "
                    f"{encoder_detail}"
                )
        enabled = label != "mREBEL" or settings.enable_mrebel
        status[label] = {
            "model": model_name,
            "available": available,
            "enabled": enabled,
            "detail": detail if enabled else "Disabled pending license approval",
        }
    return status


def rerank_relationship_candidates(
    keyword_query: str,
    candidates: list[dict[str, Any]],
    *,
    model_name: str,
    max_candidates: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    selected = [dict(item) for item in candidates[: max(1, int(max_candidates))]]
    if not selected:
        return [], {"stage": "bge_reranker", "processed": 0, "applied": False}
    tokenizer, model, torch = _load_sequence_classifier(model_name)
    pairs = []
    for candidate in selected:
        primary = _primary_evidence(candidate)
        route_text = ". ".join(
            part
            for part in (
                str(keyword_query).strip(),
                str(primary.get("related_subject", "")).strip(),
                str(primary.get("factual_bridge", "")).strip(),
                str(primary.get("acceptance_condition", "")).strip(),
            )
            if part
        )
        pairs.append((route_text, str(candidate.get("page_title", ""))))

    scores: list[float] = []
    batch_size = 16
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        inputs = tokenizer(
            [pair[0] for pair in batch],
            [pair[1] for pair in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().float().numpy()
        if logits.ndim == 2 and logits.shape[1] == 1:
            batch_scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        elif logits.ndim == 1:
            batch_scores = 1.0 / (1.0 + np.exp(-logits))
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            batch_scores = probabilities[:, -1]
        scores.extend(float(value) for value in batch_scores)

    for candidate, score in zip(selected, scores):
        candidate["bge_reranker_score"] = round(score, 6)
        primary = _primary_evidence(candidate)
        semantic_score = float(primary.get("semantic_score", 0.0))
        gliner_score = candidate.get("gliner_entity_score")
        if gliner_score is None:
            local_score = 0.70 * score + 0.30 * semantic_score
        else:
            local_score = (
                0.65 * score
                + 0.25 * semantic_score
                + 0.10 * float(gliner_score)
            )
        candidate["semantic_seo_local_score"] = round(local_score, 6)
    selected.sort(
        key=lambda row: (
            float(row.get("semantic_seo_local_score", 0.0)),
            int(row.get("total_views", 0) or 0),
        ),
        reverse=True,
    )
    return selected, {
        "stage": "bge_reranker",
        "processed": len(selected),
        "applied": True,
        "model": model_name,
    }


def add_gliner_entity_compatibility(
    candidates: list[dict[str, Any]],
    *,
    model_name: str,
    max_candidates: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    selected = [dict(item) for item in candidates]
    if not selected:
        return selected, {"stage": "gliner", "processed": 0, "applied": False}
    model = _load_gliner(model_name)
    processed = min(len(selected), max(1, int(max_candidates)))
    for candidate in selected[:processed]:
        primary = _primary_evidence(candidate)
        route_text = ". ".join(
            part
            for part in (
                str(primary.get("related_subject", "")),
                str(primary.get("factual_bridge", "")),
                " ".join(str(item) for item in primary.get("required_title_cues", [])),
            )
            if part.strip()
        )
        title = str(candidate.get("page_title", ""))
        route_entities = _predict_gliner(model, route_text)
        title_entities = _predict_gliner(model, title)
        route_keys = {_entity_key(item) for item in route_entities}
        title_keys = {_entity_key(item) for item in title_entities}
        exact_overlap = route_keys & title_keys
        route_labels = {str(item.get("label", "")).casefold() for item in route_entities}
        title_labels = {str(item.get("label", "")).casefold() for item in title_entities}
        label_overlap = route_labels & title_labels
        compatibility = min(
            1.0,
            (0.65 if exact_overlap else 0.0)
            + min(0.35, 0.12 * len(label_overlap)),
        )
        candidate["gliner_entity_score"] = round(compatibility, 6)
        semantic_score = float(primary.get("semantic_score", 0.0))
        candidate["semantic_seo_local_score"] = round(
            0.80 * semantic_score + 0.20 * compatibility,
            6,
        )
        candidate["gliner_title_entities"] = title_entities
        candidate["gliner_route_entities"] = route_entities
    selected.sort(
        key=lambda row: (
            float(row.get("semantic_seo_local_score", 0.0)),
            int(row.get("total_views", 0) or 0),
        ),
        reverse=True,
    )
    return selected, {
        "stage": "gliner",
        "processed": processed,
        "applied": True,
        "model": model_name,
    }


def extract_and_link_title_entities(
    title_by_story_id: dict[str, str],
    *,
    model_name: str,
    batch_size: int = 32,
    cache_path: Path | None = DEFAULT_TITLE_ENTITY_CACHE_PATH,
) -> dict[str, list[dict[str, object]]]:
    """Extract and canonically link title entities during index ingestion.

    The returned data is deliberately query-independent so GLiNER never needs to
    run over the candidate set in the online search path.
    """
    ordered = [
        (str(story_id), str(title).strip())
        for story_id, title in title_by_story_id.items()
        if str(story_id).strip() and str(title).strip()
    ]
    if not ordered:
        return {}

    cache_keys = {
        story_id: _title_entity_cache_key(title, model_name)
        for story_id, title in ordered
    }
    cached = (
        _load_title_entity_cache(cache_path, list(cache_keys.values()))
        if cache_path is not None
        else {}
    )
    result = {
        story_id: cached[cache_key]
        for story_id, cache_key in cache_keys.items()
        if cache_key in cached
    }
    missing = [
        (story_id, title)
        for story_id, title in ordered
        if story_id not in result
    ]
    if not missing:
        return result

    model = _load_gliner(model_name)
    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(missing), safe_batch_size):
        batch = missing[start : start + safe_batch_size]
        texts = [title for _, title in batch]
        raw_entities = model.inference(
            texts,
            list(SEMANTIC_SEO_ENTITY_LABELS),
            threshold=0.45,
            batch_size=min(safe_batch_size, len(texts)),
        )
        if len(raw_entities) != len(batch):
            raise RuntimeError("GLiNER did not return one entity set per title.")
        cache_batch: dict[str, list[dict[str, object]]] = {}
        for (story_id, _), entities in zip(batch, raw_entities):
            normalized = [
                {
                    "text": str(item.get("text", "")).strip(),
                    "label": str(item.get("label", "")).strip(),
                    "score": round(float(item.get("score", 0.0)), 4),
                }
                for item in entities
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            linked = link_title_entities(normalized)
            result[story_id] = linked
            if cache_path is not None:
                cache_batch[cache_keys[story_id]] = linked
        if cache_path is not None and cache_batch:
            _save_title_entity_cache_batch(cache_path, model_name, cache_batch)
    return result


def link_title_entities(
    entities: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Assign stable canonical IDs to GLiNER entities without a remote service."""
    alias_targets = _canonical_alias_targets()
    linked_by_id: dict[str, dict[str, object]] = {}
    for entity in entities:
        surface = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "")).strip().casefold()
        normalized = _normalize_entity_text(surface)
        if not normalized:
            continue
        known = alias_targets.get(normalized, ())
        if len(known) == 1:
            canonical_id, canonical_name = known[0]
        else:
            digest = hashlib.sha256(
                f"{label}|{normalized}".encode("utf-8")
            ).hexdigest()[:20]
            canonical_id = f"title_entity_{digest}"
            canonical_name = surface
        linked = {
            "entity_id": canonical_id,
            "canonical_name": canonical_name,
            "surface": surface,
            "label": label,
            "score": round(float(entity.get("score", 0.0)), 4),
        }
        existing = linked_by_id.get(canonical_id)
        if existing is None or float(linked["score"]) > float(existing["score"]):
            linked_by_id[canonical_id] = linked
    return sorted(
        linked_by_id.values(),
        key=lambda item: (-float(item["score"]), str(item["entity_id"])),
    )


def add_indexed_entity_compatibility(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Use ingestion-time GLiNER entities without loading GLiNER during search."""
    selected = [dict(item) for item in candidates]
    processed = min(len(selected), max(1, int(max_candidates)))
    for candidate in selected[:processed]:
        primary = _primary_evidence(candidate)
        route_text = " ".join(
            str(part)
            for part in (
                primary.get("related_subject", ""),
                primary.get("factual_bridge", ""),
                " ".join(str(item) for item in primary.get("required_title_cues", [])),
            )
            if str(part).strip()
        ).casefold()
        names = [
            str(value).strip()
            for value in candidate.get("title_entity_names", [])
            if str(value).strip()
        ]
        candidate_ids = {
            str(value).strip()
            for value in candidate.get("title_entity_ids", [])
            if str(value).strip()
        }
        route_ids = canonical_entity_ids_in_text(route_text)
        matched_ids = sorted(candidate_ids & route_ids)
        matched = [name for name in names if _phrase_in_text(name, route_text)]
        compatibility = min(
            1.0,
            (0.65 if matched_ids else 0.0) + min(0.35, 0.35 * len(matched)),
        )
        candidate["indexed_entity_score"] = round(compatibility, 6)
        candidate["matched_indexed_entities"] = matched
        candidate["matched_canonical_entity_ids"] = matched_ids
        semantic_score = float(primary.get("semantic_score", 0.0))
        candidate["semantic_seo_local_score"] = round(
            0.85 * semantic_score + 0.15 * compatibility,
            6,
        )
    selected.sort(
        key=lambda row: (
            float(row.get("semantic_seo_local_score", 0.0)),
            int(row.get("total_views", 0) or 0),
        ),
        reverse=True,
    )
    return selected, {
        "stage": "indexed_gliner_entities",
        "processed": processed,
        "applied": bool(processed),
        "online_model_inference": False,
    }


def canonical_entity_ids_in_text(text: str) -> set[str]:
    """Resolve unambiguous curated entity names/aliases present in free text."""
    normalized_text = _normalize_entity_text(text)
    resolved: set[str] = set()
    for phrase, targets in _canonical_alias_targets().items():
        if len(targets) == 1 and _phrase_in_text(phrase, normalized_text):
            resolved.add(targets[0][0])
    return resolved


def score_deberta_title_relevance(
    keyword_query: str,
    candidates: list[dict[str, Any]],
    *,
    model_name: str,
    max_candidates: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Soft-score title relevance with DeBERTa; never require article bodies.

    NLI entailment is used as a bounded compatibility signal, not as proof of a
    factual relationship and not as a hard filter. This preserves indirect-title
    recall while allowing title-only deployments to use the model explicitly.
    """
    selected = [dict(item) for item in candidates]
    eligible = selected[: max(1, int(max_candidates))]
    if not eligible:
        return selected, {
            "stage": "deberta_title_relevance",
            "processed": 0,
            "applied": False,
        }
    tokenizer, model, torch = _load_sequence_classifier(model_name)
    entailment_index = _entailment_label_index(model.config)
    premises: list[str] = []
    hypotheses: list[str] = []
    for candidate in eligible:
        primary = _primary_evidence(candidate)
        premises.append(str(candidate.get("page_title", "")).strip())
        route = ". ".join(
            value
            for value in (
                str(keyword_query).strip(),
                str(primary.get("related_subject", "")).strip(),
                str(primary.get("factual_bridge", "")).strip(),
            )
            if value
        )
        hypotheses.append(f"This title is relevant to: {route}")

    scores: list[float] = []
    for start in range(0, len(eligible), 8):
        inputs = tokenizer(
            premises[start : start + 8],
            hypotheses[start : start + 8],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().float().numpy()
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        scores.extend(float(row[entailment_index]) for row in probabilities)

    for candidate, score in zip(eligible, scores):
        candidate["deberta_title_relevance_score"] = round(score, 6)
        prior = float(
            candidate.get(
                "semantic_seo_local_score",
                _primary_evidence(candidate).get("semantic_score", 0.0),
            )
        )
        candidate["semantic_seo_local_score"] = round(
            0.90 * prior + 0.10 * score,
            6,
        )
    selected.sort(
        key=lambda row: (
            float(row.get("semantic_seo_local_score", 0.0)),
            int(row.get("total_views", 0) or 0),
        ),
        reverse=True,
    )
    return selected, {
        "stage": "deberta_title_relevance",
        "processed": len(eligible),
        "applied": True,
        "model": model_name,
        "hard_filter": False,
        "evidence_mode": "title_only",
    }


def apply_deberta_evidence_gate(
    candidates: list[dict[str, Any]],
    *,
    model_name: str,
    max_candidates: int = 30,
    minimum_entailment: float = 0.55,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Apply NLI only when real article evidence text is present."""
    selected = [dict(item) for item in candidates]
    evidence_fields = ("article_text", "summary", "description")
    eligible = [
        item
        for item in selected[: max(1, int(max_candidates))]
        if any(str(item.get(field, "")).strip() for field in evidence_fields)
    ]
    if not eligible:
        return selected, {
            "stage": "deberta_nli",
            "processed": 0,
            "applied": False,
            "reason": "Article summaries or bodies are not available; title-only NLI was skipped.",
        }

    tokenizer, model, torch = _load_sequence_classifier(model_name)
    entailment_index = _entailment_label_index(model.config)
    accepted_ids = set()
    for candidate in eligible:
        premise = next(
            str(candidate.get(field, "")).strip()
            for field in evidence_fields
            if str(candidate.get(field, "")).strip()
        )
        primary = _primary_evidence(candidate)
        hypothesis = (
            "This article is related to the original query because "
            + str(primary.get("factual_bridge", "")).strip()
        )
        inputs = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0].detach().cpu().float().numpy()
        exp = np.exp(logits - logits.max())
        probabilities = exp / exp.sum()
        score = float(probabilities[entailment_index])
        candidate["deberta_entailment_score"] = round(score, 6)
        candidate["deberta_evidence_pass"] = score >= minimum_entailment
        if score >= minimum_entailment:
            accepted_ids.add(str(candidate.get("story_id", "")))

    result = [
        item
        for item in selected
        if item not in eligible or str(item.get("story_id", "")) in accepted_ids
    ]
    return result, {
        "stage": "deberta_nli",
        "processed": len(eligible),
        "accepted": len(accepted_ids),
        "applied": True,
        "model": model_name,
        "minimum_entailment": minimum_entailment,
    }


def extract_mrebel_relationships(
    texts: list[str],
    *,
    model_name: str,
    max_texts: int = 100,
) -> list[list[dict[str, str]]]:
    """Optional ingestion-time relationship extraction from article evidence."""
    tokenizer, model, torch = _load_seq2seq(model_name)
    selected = [str(text).strip() for text in texts if str(text).strip()][
        : max(1, int(max_texts))
    ]
    results: list[list[dict[str, str]]] = []
    for start in range(0, len(selected), 8):
        batch = selected[start : start + 8]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256, num_beams=3)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
        results.extend(_parse_rebel_triplets(text) for text in decoded)
    return results


def _parse_rebel_triplets(text: str) -> list[dict[str, str]]:
    cleaned = (
        str(text)
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .strip()
    )
    subject = ""
    object_value = ""
    relation = ""
    current = ""
    triplets = []

    def append_triplet() -> None:
        if subject.strip() and object_value.strip() and relation.strip():
            triplets.append(
                {
                    "subject": subject.strip(),
                    "object": object_value.strip(),
                    "relation": relation.strip(),
                }
            )

    for token in cleaned.split():
        if token == "<triplet>":
            append_triplet()
            subject, object_value, relation = "", "", ""
            current = "subject"
        elif token == "<subj>":
            current = "object"
        elif token == "<obj>":
            current = "relation"
        elif current == "subject":
            subject += " " + token
        elif current == "object":
            object_value += " " + token
        elif current == "relation":
            relation += " " + token
    append_triplet()
    return triplets


def _primary_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("retrieval_evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                return item
    return {}


def _predict_gliner(model: Any, text: str) -> list[dict[str, object]]:
    if not text.strip():
        return []
    entities = model.predict_entities(
        text,
        list(SEMANTIC_SEO_ENTITY_LABELS),
        threshold=0.45,
    )
    return [
        {
            "text": str(item.get("text", "")).strip(),
            "label": str(item.get("label", "")).strip(),
            "score": round(float(item.get("score", 0.0)), 4),
        }
        for item in entities
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]


def _entity_key(entity: dict[str, object]) -> tuple[str, str]:
    normalized_text = _normalize_entity_text(entity.get("text", ""))
    return str(entity.get("label", "")).casefold(), normalized_text


def _normalize_entity_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _canonical_alias_targets() -> dict[str, tuple[tuple[str, str], ...]]:
    targets: dict[str, set[tuple[str, str]]] = {}
    for entity in SEARCH_ALIAS_ENTITIES:
        for value in (entity.canonical, *entity.aliases):
            normalized = _normalize_entity_text(value)
            if normalized:
                targets.setdefault(normalized, set()).add(
                    (entity.entity_id, entity.canonical)
                )
    return {
        key: tuple(sorted(values))
        for key, values in targets.items()
    }


def _phrase_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = _normalize_entity_text(phrase)
    normalized_text = _normalize_entity_text(text)
    return bool(
        normalized_phrase
        and re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_text)
    )


def _title_entity_cache_key(title: str, model_name: str) -> str:
    value = (
        f"{TITLE_ENTITY_CACHE_VERSION}|{str(model_name).strip()}|"
        f"{' '.join(str(title).split()).casefold()}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_title_entity_cache(
    cache_path: Path,
    cache_keys: list[str],
) -> dict[str, list[dict[str, object]]]:
    if not cache_path.exists() or not cache_keys:
        return {}
    result: dict[str, list[dict[str, object]]] = {}
    try:
        connection = sqlite3.connect(cache_path, timeout=30)
        try:
            for start in range(0, len(cache_keys), 500):
                selected = cache_keys[start : start + 500]
                placeholders = ",".join("?" for _ in selected)
                document_rows = connection.execute(
                    f"SELECT cache_key FROM title_entity_documents "
                    f"WHERE cache_key IN ({placeholders})",
                    selected,
                ).fetchall()
                present = {str(row[0]) for row in document_rows}
                for cache_key in present:
                    result[cache_key] = []
                if not present:
                    continue
                mention_placeholders = ",".join("?" for _ in present)
                mention_rows = connection.execute(
                    f"SELECT cache_key, entity_id, canonical_name, surface, label, score "
                    f"FROM title_entity_mentions WHERE cache_key IN "
                    f"({mention_placeholders}) ORDER BY cache_key, ordinal",
                    sorted(present),
                ).fetchall()
                for row in mention_rows:
                    result[str(row[0])].append(
                        {
                            "entity_id": str(row[1]),
                            "canonical_name": str(row[2]),
                            "surface": str(row[3]),
                            "label": str(row[4]),
                            "score": float(row[5]),
                        }
                    )
        finally:
            connection.close()
    except sqlite3.Error:
        return {}
    return result


def _save_title_entity_cache_batch(
    cache_path: Path,
    model_name: str,
    entities_by_cache_key: dict[str, list[dict[str, object]]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path, timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS title_entity_documents (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                pipeline_version TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS title_entity_mentions (
                cache_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                surface TEXT NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (cache_key, ordinal)
            )
            """
        )
        connection.executemany(
            "INSERT OR REPLACE INTO title_entity_documents "
            "(cache_key, model_name, pipeline_version) VALUES (?, ?, ?)",
            [
                (cache_key, model_name, TITLE_ENTITY_CACHE_VERSION)
                for cache_key in entities_by_cache_key
            ],
        )
        connection.executemany(
            "DELETE FROM title_entity_mentions WHERE cache_key = ?",
            [(cache_key,) for cache_key in entities_by_cache_key],
        )
        connection.executemany(
            """
            INSERT INTO title_entity_mentions (
                cache_key, ordinal, entity_id, canonical_name, surface, label, score
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row
                for cache_key, entities in entities_by_cache_key.items()
                for ordinal, entity in enumerate(entities)
                for row in [
                    (
                        cache_key,
                        ordinal,
                        str(entity.get("entity_id", "")),
                        str(entity.get("canonical_name", "")),
                        str(entity.get("surface", "")),
                        str(entity.get("label", "")),
                        float(entity.get("score", 0.0)),
                    )
                ]
            ],
        )
        connection.commit()
    finally:
        connection.close()


@lru_cache(maxsize=4)
def _load_sequence_classifier(model_name: str):
    model_path = _require_local_snapshot(model_name)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers and torch are required for local scoring.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, torch


@lru_cache(maxsize=1)
def _load_gliner(model_name: str):
    model_path = _require_local_snapshot(model_name)
    if model_name == DEFAULT_SEMANTIC_SEO_GLINER_MODEL:
        _require_local_snapshot(DEFAULT_SEMANTIC_SEO_GLINER_ENCODER_MODEL)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in (sys.prefix, os.path.join(sys.prefix, "Scripts")):
            if os.path.isdir(directory):
                os.add_dll_directory(directory)
        _install_pytorch_only_onnxruntime_stub()
    try:
        from gliner import GLiNER
    except Exception as exc:
        raise RuntimeError(f"GLiNER could not be loaded: {exc}") from exc
    return GLiNER.from_pretrained(model_path, local_files_only=True)


def _install_pytorch_only_onnxruntime_stub() -> None:
    """Avoid a Windows ONNX Runtime/PyTorch DLL-order crash.

    The lab uses GLiNER's PyTorch checkpoint, but GLiNER imports ONNX Runtime
    eagerly. On this Windows runtime, importing ONNX Runtime after PyTorch causes
    a native access violation. A minimal stub keeps the unused ONNX path disabled
    while leaving GLiNER's PyTorch inference fully functional.
    """
    existing = sys.modules.get("onnxruntime")
    if existing is not None:
        return
    runtime_stub = types.ModuleType("onnxruntime")

    class UnavailableInferenceSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "ONNX inference is disabled in the Semantic SEO Windows process."
            )

    class UnavailableSessionOptions:
        pass

    runtime_stub.InferenceSession = UnavailableInferenceSession
    runtime_stub.SessionOptions = UnavailableSessionOptions
    runtime_stub.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=0)
    runtime_stub.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", loader=None)
    runtime_stub.__semantic_seo_stub__ = True
    sys.modules["onnxruntime"] = runtime_stub


@lru_cache(maxsize=1)
def _load_seq2seq(model_name: str):
    model_path = _require_local_snapshot(model_name)
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers and torch are required for mREBEL.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return tokenizer, model, torch


def _entailment_label_index(config: Any) -> int:
    id_to_label = getattr(config, "id2label", {}) or {}
    for raw_index, raw_label in id_to_label.items():
        if "entail" in str(raw_label).casefold():
            return int(raw_index)
    # The configured DeBERTa checkpoint documents index zero as entailment.
    return 0


def _local_snapshot_status(model_name: str) -> tuple[bool, str]:
    try:
        path = _require_local_snapshot(model_name)
    except RuntimeError as exc:
        return False, str(exc)
    return True, path


@lru_cache(maxsize=16)
def _require_local_snapshot(model_name: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is unavailable.") from exc
    try:
        return str(snapshot_download(repo_id=model_name, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(f"Model '{model_name}' is not in the local cache.") from exc


def _setting(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _environment_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}
