"""Retrieve full-title vectors, then extract keywords from their source titles."""

from collections import Counter
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from src.daily_keywords import title_tokens
from src.word_embeddings import STOPWORDS, _normalize, load_or_build_index

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "title_keyword_embeddings"


def prepare_titles(titles: pd.DataFrame) -> pd.DataFrame:
    rows = titles[["story_id", "page_title"]].drop_duplicates("story_id", keep="last").dropna().copy()
    rows["story_id"] = rows["story_id"].astype(str).str.strip()
    rows["page_title"] = rows["page_title"].map(
        lambda text: " ".join(unicodedata.normalize("NFKC", str(text)).split()))
    return rows.loc[(rows.story_id != "") & (rows.page_title != "")].sort_values("story_id").reset_index(drop=True)


def load_title_vectors(texts: tuple[str, ...], model_name: str) -> np.ndarray:
    # Cache complete title strings in row order, separately from every other index.
    return load_or_build_index(texts, model_name, index_dir=INDEX_DIR)


def retrieve_keywords(titles: pd.DataFrame, vectors: np.ndarray, query: str,
                      query_vector: np.ndarray, title_limit: int = 50,
                      min_similarity: float = .4, keyword_limit: int = 20,
                      min_occurrences: int = 2, exclusions: str = "",
                      exclude_query: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    titles = titles.reset_index(drop=True)
    scores = _normalize(vectors, len(titles)) @ _normalize(np.asarray(query_vector).reshape(1, -1), 1)[0]
    ranked = titles.copy()
    ranked["title_similarity"] = np.clip(scores, -1, 1)
    matched = ranked.loc[ranked.title_similarity >= min_similarity].sort_values(
        ["title_similarity", "story_id"], ascending=[False, True]).head(title_limit)
    excluded = STOPWORDS | set(title_tokens(exclusions))
    if exclude_query:
        excluded |= set(title_tokens(query))
    token_sets = [{w for w in title_tokens(t) if w not in excluded and any(c.isalpha() for c in w)}
                  for t in titles.page_title]
    frequencies = Counter(w for words in token_sets for w in words)
    evidence = {}
    for index, row in matched.iterrows():
        for word in token_sets[index]:
            entry = evidence.setdefault(word, {"keyword": word, "matched_title_occurrences": 0,
                "relevance": 0., "best_title_similarity": float(row.title_similarity),
                "example_title": row.page_title})
            entry["matched_title_occurrences"] += 1
            entry["relevance"] += max(0., float(row.title_similarity))
    columns = ["query", "keyword", "relevance", "matched_title_occurrences", "title_occurrences",
               "best_title_similarity", "example_title"]
    records = []
    for word, entry in evidence.items():
        if entry["matched_title_occurrences"] < min_occurrences:
            continue
        entry["query"] = query
        entry["title_occurrences"] = frequencies[word]
        entry["relevance"] *= np.log1p(len(titles) / frequencies[word]) / max(1, len(matched))
        records.append(entry)
    results = pd.DataFrame(records, columns=columns).sort_values(
        ["relevance", "keyword"], ascending=[False, True]).head(keyword_limit).reset_index(drop=True)
    return results, matched.reset_index(drop=True)
