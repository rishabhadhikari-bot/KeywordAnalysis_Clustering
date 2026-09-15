import re

from src.config import MIN_KEYWORD_LENGTH, STOP_WORDS
from src.nlp.lemmatization import lemmatize_tokens


TERM_NORMALIZATION = {
    "ayodhyas": "ayodhya",
    "crises": "crisis",
    "indias": "india",
    "isreali": "israeli",
    "isreal": "israel",
    "usa": "us",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    raw_tokens = []

    for token in normalized.split():
        normalized_token = normalize_token(token)
        if not normalized_token or normalized_token in STOP_WORDS:
            continue
        raw_tokens.append(normalized_token)

    tokens = []
    for lemma in lemmatize_tokens(raw_tokens):
        if not is_keyword_token(lemma):
            continue
        tokens.append(lemma)

    return tokens


def normalize_token(token: str) -> str:
    token = TERM_NORMALIZATION.get(token, token)

    return token


def is_keyword_token(token: str) -> bool:
    if not token:
        return False
    if token in STOP_WORDS:
        return False
    if token.isdigit():
        return True
    return len(token) >= MIN_KEYWORD_LENGTH


def make_ngrams(tokens: list[str], max_size: int = 3) -> list[tuple[str, int, int]]:
    ngrams = []
    for size in range(1, max_size + 1):
        for start in range(0, len(tokens) - size + 1):
            phrase_tokens = tokens[start : start + size]
            ngrams.append((" ".join(phrase_tokens), start, size))
    return ngrams
