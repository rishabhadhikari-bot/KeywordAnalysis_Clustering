from dataclasses import asdict, dataclass

from src.nlp.lemmatization import _load_spacy_model
from src.nlp.normalization import tokenize


@dataclass(frozen=True)
class KeywordCandidate:
    keyword: str
    normalized_keyword: str
    keyword_type: str
    keyword_source: str
    relevance_score: float
    semantic_group: str
    extraction_method: str
    pos_pattern: str
    entity_label: str


def extract_keyword_candidates(title: str) -> list[dict[str, object]]:
    tokens = tokenize(title)
    if not tokens:
        return []

    candidates = {}
    _add_spacy_candidates(title, candidates)

    if not candidates:
        _add_fallback_single_tokens(tokens, candidates)

    ordered = sorted(candidates.values(), key=lambda item: item.relevance_score, reverse=True)
    return [asdict(candidate) for candidate in ordered]


def classify_keyword(keyword: str) -> str:
    tokens = keyword.split()

    if keyword.replace(" ", "").isdigit():
        return "metric"
    if len(tokens) >= 2:
        return "topic_phrase"
    return "keyword"


def get_semantic_group(keyword: str, keyword_type: str) -> str:
    return keyword_type.replace("_", " ")


def score_keyword(
    phrase: str,
    start_index: int,
    phrase_size: int,
    keyword_type: str,
    extraction_method: str,
) -> float:
    score = 0.35
    score += min(phrase_size, 3) * 0.12
    score += max(0, 4 - start_index) * 0.04

    if keyword_type in {"person_or_entity", "organization", "named_entity"}:
        score += 0.18
    elif keyword_type in {"place", "topic_phrase"}:
        score += 0.11

    if any(char.isdigit() for char in phrase):
        score += 0.04

    if extraction_method == "named_entity":
        score += 0.2
    elif extraction_method == "noun_chunk":
        score += 0.12
    elif extraction_method == "proper_noun_phrase":
        score += 0.1

    return round(min(score, 0.99), 3)


def _is_low_value_phrase(phrase: str, phrase_size: int) -> bool:
    tokens = phrase.split()
    if not tokens:
        return True
    if phrase_size == 1 and tokens[0].isdigit() and len(tokens[0]) != 4:
        return True
    return False


def _add_spacy_candidates(title: str, candidates: dict[str, KeywordCandidate]) -> None:
    try:
        nlp = _load_spacy_model()
    except OSError:
        nlp = None
    if nlp is None:
        return

    doc = nlp(title)

    for entity in doc.ents:
        phrase = " ".join(token.lemma_.lower() for token in entity if not token.is_stop and not token.is_punct).strip()
        if _is_low_value_phrase(phrase, len(phrase.split())):
            continue
        keyword_type = _map_entity_label(entity.label_)
        _upsert_candidate(
            candidates=candidates,
            phrase=phrase,
            keyword_type=keyword_type,
            score=score_keyword(
                phrase=phrase,
                start_index=entity.start,
                phrase_size=len(phrase.split()),
                keyword_type=keyword_type,
                extraction_method="named_entity",
            ),
            extraction_method="named_entity",
            pos_pattern=" ".join(token.pos_ for token in entity),
            entity_label=entity.label_,
        )

    for chunk in doc.noun_chunks:
        phrase_tokens = [
            token.lemma_.lower()
            for token in chunk
            if not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "PROPN", "NUM"}
        ]
        phrase = " ".join(phrase_tokens).strip()
        if _is_low_value_phrase(phrase, len(phrase_tokens)):
            continue
        keyword_type = classify_keyword(phrase)
        _upsert_candidate(
            candidates=candidates,
            phrase=phrase,
            keyword_type=keyword_type,
            score=score_keyword(
                phrase=phrase,
                start_index=chunk.start,
                phrase_size=len(phrase_tokens),
                keyword_type=keyword_type,
                extraction_method="noun_chunk",
            ),
            extraction_method="noun_chunk",
            pos_pattern=" ".join(token.pos_ for token in chunk),
            entity_label="",
        )

    for token in doc:
        if token.is_stop or token.is_punct or token.pos_ not in {"NOUN", "PROPN"}:
            continue

        phrase = token.lemma_.lower().strip()
        if _is_low_value_phrase(phrase, 1):
            continue

        keyword_type = classify_keyword(phrase)
        _upsert_candidate(
            candidates=candidates,
            phrase=phrase,
            keyword_type=keyword_type,
            score=score_keyword(
                phrase=phrase,
                start_index=token.i,
                phrase_size=1,
                keyword_type=keyword_type,
                extraction_method="content_token",
            ),
            extraction_method="content_token",
            pos_pattern=token.pos_,
            entity_label="",
        )

    proper_noun_tokens = []
    proper_start = 0
    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop:
            if not proper_noun_tokens:
                proper_start = token.i
            proper_noun_tokens.append(token.lemma_.lower())
            continue
        _flush_proper_nouns(candidates, proper_noun_tokens, proper_start)
        proper_noun_tokens = []
    _flush_proper_nouns(candidates, proper_noun_tokens, proper_start)


def _add_fallback_single_tokens(tokens: list[str], candidates: dict[str, KeywordCandidate]) -> None:
    for start_index, token in enumerate(tokens):
        if _is_low_value_phrase(token, 1):
            continue

        keyword_type = classify_keyword(token)
        _upsert_candidate(
            candidates=candidates,
            phrase=token,
            keyword_type=keyword_type,
            score=score_keyword(
                phrase=token,
                start_index=start_index,
                phrase_size=1,
                keyword_type=keyword_type,
                extraction_method="fallback_token",
            ),
            extraction_method="fallback_token",
            pos_pattern="",
            entity_label="",
        )


def _flush_proper_nouns(candidates: dict[str, KeywordCandidate], tokens: list[str], start_index: int) -> None:
    if len(tokens) < 2:
        return
    phrase = " ".join(tokens)
    keyword_type = classify_keyword(phrase)
    _upsert_candidate(
        candidates=candidates,
        phrase=phrase,
        keyword_type=keyword_type,
        score=score_keyword(
            phrase=phrase,
            start_index=start_index,
            phrase_size=len(tokens),
            keyword_type=keyword_type,
            extraction_method="proper_noun_phrase",
        ),
        extraction_method="proper_noun_phrase",
        pos_pattern="PROPN",
        entity_label="",
    )


def _upsert_candidate(
    candidates: dict[str, KeywordCandidate],
    phrase: str,
    keyword_type: str,
    score: float,
    extraction_method: str,
    pos_pattern: str,
    entity_label: str,
) -> None:
    phrase = phrase.strip()
    if not phrase:
        return

    current = candidates.get(phrase)
    if current is not None and score <= current.relevance_score:
        return

    candidates[phrase] = KeywordCandidate(
        keyword=phrase,
        normalized_keyword=phrase,
        keyword_type=keyword_type,
        keyword_source="extracted",
        relevance_score=score,
        semantic_group=get_semantic_group(phrase, keyword_type),
        extraction_method=extraction_method,
        pos_pattern=pos_pattern,
        entity_label=entity_label,
    )


def _map_entity_label(label: str) -> str:
    if label == "PERSON":
        return "person_or_entity"
    if label in {"GPE", "LOC"}:
        return "place"
    if label == "ORG":
        return "organization"
    if label in {"DATE", "TIME"}:
        return "date_or_time"
    if label in {"MONEY", "PERCENT", "CARDINAL", "QUANTITY"}:
        return "metric"
    if label == "EVENT":
        return "geopolitical_event"
    return "named_entity"
