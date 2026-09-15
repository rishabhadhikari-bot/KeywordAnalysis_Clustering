"""Independent page-title vocabulary and cosine retrieval for the Embeddings tab."""

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.daily_keywords import title_tokens
from src.relationship_embeddings import get_relationship_embedding_model_name


STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | frozenset(
    "s t d ll m re ve don doesn didn isn aren wasn weren won wouldn couldn shouldn "
    "का की के को से में पर और या है हैं था थी थे हो भी एक यह वह ये वो ने लिए तक तो ही".split()
)
# Common grammatical Hindi words also appear transliterated in the title corpus.
STOPWORDS = STOPWORDS | frozenset("ka ki ke ko hai hain tha thi ho bhi aur mein".split())
INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "word_embeddings"


def build_vocabulary(titles: pd.DataFrame, extra_stopwords: str = "") -> pd.DataFrame:
    """Count each word once per story; repeated monthly rows cannot inflate counts."""
    rows = titles[["story_id", "page_title"]].drop_duplicates("story_id", keep="last")
    excluded = STOPWORDS | set(title_tokens(extra_stopwords))
    counts = Counter()
    for title in rows["page_title"].dropna():
        counts.update({word for word in title_tokens(str(title))
                       if word not in excluded and any(char.isalpha() for char in word)})
    return pd.DataFrame(sorted(counts.items()), columns=["keyword", "title_occurrences"])


def index_key(words: tuple[str, ...], model_name: str) -> str:
    payload = json.dumps(["word-index-v1", model_name, words], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=2)
def _model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"Cannot load the local embedding model '{model_name}'. "
                           "Make sure its model files and sentence-transformers are installed.") from exc


def _normalize(vectors: np.ndarray, expected_rows: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != expected_rows or not np.isfinite(vectors).all():
        raise ValueError("The embedding model returned invalid vectors.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The embedding model returned an empty vector.")
    return vectors / norms


def encode_words(texts: list[str], model_name: str) -> np.ndarray:
    return _normalize(_model(model_name).encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ), len(texts))


def load_or_build_index(words: tuple[str, ...], model_name: str,
                        index_dir: Path = INDEX_DIR) -> np.ndarray:
    if not words:
        raise ValueError("No keywords remain after exclusions.")
    path = index_dir / f"{index_key(words, model_name)}.npy"
    if path.exists():
        try:
            return _normalize(np.load(path, allow_pickle=False), len(words))
        except (OSError, ValueError):
            pass  # Rebuild an incomplete or invalid cache.
    vectors = encode_words(list(words), model_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=index_dir, suffix=".npy", delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, vectors, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return vectors


def query_texts(query: str, separate: bool) -> list[str]:
    if separate:
        return list(dict.fromkeys(word for word in title_tokens(query) if word not in STOPWORDS))
    return [" ".join(title_tokens(query))] if title_tokens(query) else []


def title_evidence(titles: pd.DataFrame, query: str) -> tuple[pd.DataFrame, int]:
    """Evidence uses whole tokens, all non-stop query words, and distinct stories.

    Co-occurrence establishes corpus support, not synonymy. An unseen query
    deliberately has no evidence rather than silently falling back to raw vectors.
    """
    query_words = set(title_tokens(query)) - STOPWORDS
    counts = Counter()
    examples = {}
    matched = 0
    rows = titles[["story_id", "page_title"]].drop_duplicates("story_id", keep="last")
    for title in rows["page_title"].dropna() if query_words else []:
        tokens = set(title_tokens(str(title)))
        if query_words.issubset(tokens):
            matched += 1
            counts.update(tokens)
            for token in tokens:
                examples.setdefault(token, str(title))
    return pd.DataFrame(
        [(word, count, examples[word]) for word, count in sorted(counts.items())],
        columns=["keyword", "shared_titles", "example_title"],
    ), matched


def rank_keywords(vocabulary: pd.DataFrame, vectors: np.ndarray, query: str,
                  query_vector: np.ndarray, top_k: int = 20, min_similarity: float = 0,
                  min_occurrences: int = 1, exclude_query: bool = True,
                  evidence: pd.DataFrame | None = None, min_shared_titles: int = 2) -> pd.DataFrame:
    vectors = _normalize(vectors, len(vocabulary))
    query_vector = _normalize(np.asarray(query_vector).reshape(1, -1), 1)[0]
    result = vocabulary.copy()
    result["similarity"] = np.clip(vectors @ query_vector, -1, 1)
    if evidence is not None:
        result = result.merge(evidence, on="keyword", how="left", validate="one_to_one")
        result["shared_titles"] = result["shared_titles"].fillna(0).astype(int)
    keep = (result["similarity"] >= min_similarity) & (result["title_occurrences"] >= min_occurrences)
    if exclude_query:
        keep &= ~result["keyword"].isin(title_tokens(query))
    if evidence is not None:
        keep &= result["shared_titles"] >= min_shared_titles
    result = result.loc[keep].sort_values(["similarity", "keyword"], ascending=[False, True]).head(top_k)
    result.insert(0, "query", query)
    return result.reset_index(drop=True)
