import os
from functools import lru_cache

import numpy as np

from src.nlp.keybert_scoring import KEYBERT_MODEL_NAME


def cosine_similarity_scores(query: str, candidate_texts: list[str]) -> list[float]:
    if not query or not candidate_texts:
        return []

    model = _load_sentence_transformer()
    if model is None:
        return [0.0 for _ in candidate_texts]

    try:
        embeddings = model.encode(
            [query] + candidate_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        return [0.0 for _ in candidate_texts]

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]
    scores = np.dot(candidate_embeddings, query_embedding)
    return [float(score) for score in scores]


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return SentenceTransformer(KEYBERT_MODEL_NAME, local_files_only=True)
    except TypeError:
        try:
            return SentenceTransformer(KEYBERT_MODEL_NAME)
        except Exception:
            return None
    except Exception:
        return None
