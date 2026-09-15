from functools import lru_cache

from src.nlp.normalization import normalize_text


KEYBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def score_keyword_records_with_keybert(
    titles: list[str],
    records_by_title: list[list[dict[str, object]]],
) -> list[list[dict[str, object]]]:
    if not titles or not records_by_title:
        return records_by_title

    keybert_model = _load_keybert_model()
    if keybert_model is None:
        return _with_default_keybert_scores(records_by_title)

    scored_records_by_title = []
    for title, records in zip(titles, records_by_title):
        scored_records_by_title.append(_score_title_records(title, records, keybert_model))

    return scored_records_by_title


def _score_title_records(title: str, records: list[dict[str, object]], keybert_model) -> list[dict[str, object]]:
    if not title or not records:
        return _with_default_keybert_scores([records])[0]

    candidates = _candidate_terms(records)
    if not candidates:
        return _with_default_keybert_scores([records])[0]

    try:
        extracted_keywords = keybert_model.extract_keywords(
            docs=title,
            candidates=candidates,
            stop_words=None,
            top_n=len(candidates),
        )
    except Exception:
        return _with_default_keybert_scores([records])[0]

    keybert_scores = {
        normalize_text(keyword): float(score)
        for keyword, score in extracted_keywords
        if keyword
    }

    scored_records = []
    for record in records:
        scored_record = dict(record)
        base_score = float(scored_record.get("relevance_score", 0))
        normalized_keyword = normalize_text(str(scored_record.get("normalized_keyword", "")))
        keybert_score = keybert_scores.get(normalized_keyword, 0.0)

        scored_record["base_relevance_score"] = round(base_score, 3)
        scored_record["keybert_score"] = round(keybert_score, 3)
        scored_record["relevance_score"] = round(min(0.99, (base_score * 0.4) + (keybert_score * 0.6)), 3)
        scored_records.append(scored_record)

    return scored_records


def _candidate_terms(records: list[dict[str, object]]) -> list[str]:
    terms = []
    seen_terms = set()
    for record in records:
        keyword = normalize_text(str(record.get("normalized_keyword", "")))
        if not keyword or keyword in seen_terms:
            continue
        seen_terms.add(keyword)
        terms.append(keyword)
    return terms


def _with_default_keybert_scores(records_by_title: list[list[dict[str, object]]]) -> list[list[dict[str, object]]]:
    defaulted = []
    for records in records_by_title:
        defaulted_records = []
        for record in records:
            defaulted_record = dict(record)
            base_score = float(defaulted_record.get("relevance_score", 0))
            defaulted_record["base_relevance_score"] = round(base_score, 3)
            defaulted_record["keybert_score"] = 0.0
            defaulted_records.append(defaulted_record)
        defaulted.append(defaulted_records)
    return defaulted


@lru_cache(maxsize=1)
def _load_keybert_model():
    try:
        from keybert import KeyBERT
    except Exception:
        return None

    try:
        return KeyBERT(model=KEYBERT_MODEL_NAME)
    except Exception:
        return None
