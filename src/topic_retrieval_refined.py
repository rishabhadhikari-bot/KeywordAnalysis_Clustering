"""Independent corpus topic discovery for the BERTopic refined tab."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from src.data_processing import simple_query_tokens, simple_title_tokens
from src.search_aliases import normalize_refined_alias_text, opensearch_synonym_rules

REFINED_MODEL_VERSION = "bertopic-refined-v1"
EMBEDDING_MODEL = "BAAI/bge-m3"
_FIT_LOCK = RLock()
_EMBED_LOCK = RLock()
HINDI_STOPWORDS = "hai ke ki ka mein se ko par aur hain ho tha thi the bhi ek ne toh to ye yeh vo woh nahi liye tak hi kar kya ab".split()


def _rows(title_summary):
    rows = title_summary.to_dict("records") if isinstance(title_summary, pd.DataFrame) else title_summary
    valid = []
    for row in rows:
        story_id, title = row.get("story_id"), row.get("page_title")
        if pd.isna(story_id) or pd.isna(title):
            continue
        story_id, title = str(story_id).strip(), str(title).strip()
        if story_id and title:
            valid.append({"story_id": story_id, "page_title": title})
    valid.sort(key=lambda row: (row["story_id"], row["page_title"]))
    if len({row["story_id"] for row in valid}) != len(valid):
        raise ValueError("BERTopic refined requires one title per story_id.")
    return valid


def refined_corpus_fingerprint(title_summary):
    payload = {"version": REFINED_MODEL_VERSION, "embedder": EMBEDDING_MODEL, "rows": _rows(title_summary)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _tokens(text):
    normalized = normalize_refined_alias_text(str(text))
    # Retain two-letter tokens identically in lowercase corpus titles and queries.
    query_tokens = simple_query_tokens(normalized)
    title_tokens = simple_title_tokens(normalized)
    return title_tokens if title_tokens else query_tokens


def keyword_variant_tokens(keyword_query):
    """Expand safe aliases, without recursively expanding ambiguous abbreviations."""
    query = _tokens(keyword_query)
    result = set(query)
    for rule in opensearch_synonym_rules():
        phrases = [_tokens(phrase) for phrase in rule.split(", ")]
        if any(phrase and any(query[i:i + len(phrase)] == phrase
                              for i in range(len(query) - len(phrase) + 1)) for phrase in phrases):
            result.update(token for phrase in phrases for token in phrase)
    return result


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _encode(texts):
    with _EMBED_LOCK:
        return np.asarray(_embedder().encode(texts, batch_size=16, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


def _train(rows):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP

    titles = [row["page_title"] for row in rows]
    embeddings = _encode(titles)
    model = BERTopic(
        embedding_model=_embedder(),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=15, min_samples=5,
                              cluster_selection_method="leaf", prediction_data=True),
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words=sorted(
            set(ENGLISH_STOP_WORDS) | set(HINDI_STOPWORDS))),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        calculate_probabilities=False,
    )
    assignments, _ = model.fit_transform(titles, embeddings)
    assignments = np.asarray(assignments, dtype=np.int64)
    topic_ids = np.asarray(sorted(set(assignments) - {-1}), dtype=np.int64)
    centroids = np.asarray([embeddings[assignments == topic].mean(axis=0) for topic in topic_ids],
                           dtype=np.float32).reshape(len(topic_ids), embeddings.shape[1])
    if len(centroids):
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    labels = {str(topic): ", ".join(term for term, _ in (model.get_topic(int(topic)) or [])[:8])
              for topic in topic_ids}
    return model, assignments, topic_ids, centroids, labels


def fit_bertopic_refined_model(title_summary, *, index_dir=None):
    """Fit once per corpus/version, publishing a complete immutable disk artifact."""
    rows = _rows(title_summary)
    if len(rows) < 16:
        raise ValueError("BERTopic refined needs at least 16 non-empty titles to fit its topic model.")
    fingerprint = refined_corpus_fingerprint(rows)
    root = Path(index_dir) if index_dir is not None else Path(__file__).resolve().parents[1] / "data" / "bertopic_refined"
    target = root / fingerprint
    with _FIT_LOCK:
        if not target.exists():
            root.mkdir(parents=True, exist_ok=True)
            model, assignments, topic_ids, centroids, labels = _train(rows)
            stage = root.resolve() / f"fit-{uuid.uuid4().hex}"
            stage.mkdir()
            try:
                model.save(str(stage / "model"), serialization="safetensors",
                           save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL)
                np.savez_compressed(stage / "assignments.npz",
                                    story_ids=np.asarray([row["story_id"] for row in rows]),
                                    assignments=assignments, topic_ids=topic_ids, centroids=centroids)
                metadata = {"fingerprint": fingerprint, "version": REFINED_MODEL_VERSION,
                            "embedding_model": EMBEDDING_MODEL, "labels": labels,
                            "title_count": len(rows), "outlier_count": int(np.sum(assignments == -1))}
                (stage / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
                # Another process may already have published the same complete fit.
                try:
                    os.rename(stage, target)
                except OSError:
                    if not target.exists():
                        raise
            finally:
                if stage.exists() and stage.resolve().parent == root.resolve():
                    shutil.rmtree(stage)
        return _load_artifact(str(target))


@lru_cache(maxsize=2)
def _load_artifact(path):
    root = Path(path)
    metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with np.load(root / "assignments.npz", allow_pickle=False) as data:
        artifact = {key: data[key] for key in data.files}
    artifact["manifest"] = metadata
    artifact["members"] = {int(topic): np.flatnonzero(artifact["assignments"] == topic)
                           for topic in artifact["topic_ids"]}
    return artifact


def get_bertopic_refined_topic_catalog(title_summary=None, *, artifact=None):
    if artifact is None:
        if title_summary is None:
            raise ValueError("Provide the current title corpus or a fitted artifact.")
        artifact = fit_bertopic_refined_model(title_summary)
    return [{"topic_id": int(topic), "label": artifact["manifest"]["labels"].get(str(topic), ""),
             "size": len(artifact["members"][int(topic)])} for topic in artifact["topic_ids"]]


def retrieve_bertopic_refined_candidates(keyword_query, title_summary, excluded_story_ids=(), *,
                                         top_k=5, min_similarity=0.45, artifact=None):
    """Select centroid topics, subtract IDs, filter any keyword token, then rank."""
    if top_k < 1 or not -1 <= min_similarity <= 1:
        raise ValueError("Choose a positive topic count and a cosine threshold between -1 and 1.")
    variants = keyword_variant_tokens(keyword_query)
    if not variants:
        raise ValueError("Enter a query with searchable keywords.")
    if artifact is None:
        artifact = fit_bertopic_refined_model(title_summary)
    query = _encode([keyword_query])[0]
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = artifact["centroids"] @ query
    selected = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] >= min_similarity][:top_k]
    excluded = {str(story_id).strip() for story_id in excluded_story_ids}
    rows = {row["story_id"]: row for row in _rows(title_summary)}
    candidates, selected_topics = [], []
    removed_ids = removed_keywords = raw_count = 0
    for index in selected:
        topic = int(artifact["topic_ids"][index])
        score = float(scores[index])
        label = artifact["manifest"]["labels"].get(str(topic), "")
        selected_topics.append({"topic_id": topic, "label": label, "similarity": score})
        for position in artifact["members"][topic]:
            raw_count += 1
            story_id = str(artifact["story_ids"][position])
            if story_id in excluded:
                removed_ids += 1
                continue
            row = rows[story_id]
            if variants.intersection(_tokens(row["page_title"])):
                removed_keywords += 1
                continue
            candidates.append({**row, "topic_id": topic, "topic_label": label,
                               "topic_relevance_score": score})
    candidates.sort(key=lambda row: (-row["topic_relevance_score"], row["story_id"]))
    metadata = artifact["manifest"]
    diagnostics = {"selected_topics": selected_topics, "topic_count": len(scores),
                   "raw_candidates": raw_count, "removed_excluded_ids": removed_ids,
                   "removed_keyword_titles": removed_keywords, "keyword_variant_tokens": sorted(variants),
                   "min_similarity": min_similarity, "top_k": top_k,
                   "outlier_count": metadata["outlier_count"],
                   "outlier_ratio": metadata["outlier_count"] / max(metadata["title_count"], 1),
                   "corpus_fingerprint": metadata["fingerprint"], "model_version": REFINED_MODEL_VERSION}
    return candidates, diagnostics
