from functools import lru_cache


IRREGULAR_LEMMAS = {
    "children": "child",
    "crises": "crisis",
    "men": "man",
    "people": "person",
    "teeth": "tooth",
    "women": "woman",
}


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy
    except (ImportError, OSError):
        return None

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue

    return None


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    nlp = _load_spacy_model()
    if nlp is None:
        return [_fallback_lemma(token) for token in tokens]

    doc = nlp(" ".join(tokens))
    lemmas = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        if not lemma or lemma == "-pron-":
            lemma = token.text.lower()
        lemmas.append(lemma)
    return lemmas


def _fallback_lemma(token: str) -> str:
    if token in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[token]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "sis")):
        return token[:-1]
    return token
