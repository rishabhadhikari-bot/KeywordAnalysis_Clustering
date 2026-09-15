import pandas as pd
from rapidfuzz import fuzz

from src.nlp.embedding_similarity import cosine_similarity_scores
from src.nlp.normalization import normalize_text


def filter_keyword_results(
    keyword_query: str,
    keyword_monthly: pd.DataFrame,
    related_stories: pd.DataFrame,
    match_type: str = "Semantic search",
    fuzzy_threshold: int = 75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_query = normalize_text(keyword_query)
    if not normalized_query:
        return keyword_monthly.head(0), related_stories.head(0), _empty_matches()

    match_candidates = _build_match_candidates(keyword_monthly)
    if match_candidates.empty:
        return keyword_monthly.head(0), related_stories.head(0), _empty_matches()

    matched_keywords = _match_keywords(
        normalized_query=normalized_query,
        match_candidates=match_candidates,
        match_type=match_type,
        fuzzy_threshold=fuzzy_threshold,
    )

    if matched_keywords.empty:
        return keyword_monthly.head(0), related_stories.head(0), matched_keywords

    keyword_values = matched_keywords["keyword"].unique()
    match_labels = matched_keywords[["keyword", "match_reason"]].drop_duplicates()
    match_labels["match_category"] = match_labels["match_reason"].map(_match_category)

    monthly = keyword_monthly.loc[keyword_monthly["keyword"].isin(keyword_values)].copy()
    stories = related_stories.loc[related_stories["keyword"].isin(keyword_values)].copy()

    monthly = monthly.merge(match_labels, on="keyword", how="left")
    stories = stories.merge(match_labels, on="keyword", how="left")
    return monthly, stories, matched_keywords


def _build_match_candidates(keyword_monthly: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "keyword",
        "normalized_keyword",
        "keyword_type",
        "keyword_source",
        "semantic_group",
        "parent_keyword",
        "related_keywords",
    ]
    available_columns = [column for column in columns if column in keyword_monthly.columns]
    candidates = keyword_monthly[available_columns].drop_duplicates().copy()

    for column in columns:
        if column not in candidates.columns:
            candidates[column] = ""

    return candidates


def _match_keywords(normalized_query: str, match_candidates: pd.DataFrame, match_type: str, fuzzy_threshold: int) -> pd.DataFrame:
    matched_rows = []
    semantic_scores = _semantic_scores(normalized_query, match_candidates)

    for row_index, row in enumerate(match_candidates.itertuples(index=False)):
        keyword = str(row.keyword)
        normalized_keyword = str(row.normalized_keyword)
        semantic_group = str(row.semantic_group)
        parent_keyword = str(row.parent_keyword)
        related_keywords = str(row.related_keywords)

        fuzzy_score = calculate_fuzzy_score(normalized_query, normalized_keyword)
        semantic_score = semantic_scores[row_index]

        is_match = False
        match_reason = ""

        if match_type == "Exact match":
            is_match = normalized_query == normalized_keyword
            match_reason = "exact"
        elif match_type == "Contains match":
            is_match = _contains_phrase(normalized_query, normalized_keyword) or _contains_phrase(normalized_keyword, normalized_query)
            match_reason = "contains"
        elif match_type == "Fuzzy match":
            is_match = passes_fuzzy_threshold(normalized_query, normalized_keyword, fuzzy_score, fuzzy_threshold)
            match_reason = "fuzzy"
        else:
            is_match = (
                normalized_query == normalized_keyword
                or _contains_phrase(normalized_query, normalized_keyword)
                or passes_fuzzy_threshold(normalized_query, normalized_keyword, fuzzy_score, fuzzy_threshold)
                or semantic_score >= 0.35
            )
            match_reason = _semantic_reason(normalized_query, normalized_keyword, fuzzy_score, semantic_score, fuzzy_threshold)

        if not is_match:
            continue

        matched_rows.append(
            {
                "keyword": keyword,
                "keyword_type": row.keyword_type,
                "keyword_source": row.keyword_source,
                "semantic_group": semantic_group,
                "parent_keyword": parent_keyword,
                "related_keywords": related_keywords,
                "fuzzy_score": fuzzy_score,
                "semantic_score": round(semantic_score, 3),
                "match_category": _match_category(match_reason),
                "match_reason": match_reason,
            }
        )

    if not matched_rows:
        return _empty_matches()

    return (
        pd.DataFrame(matched_rows)
        .sort_values(["semantic_score", "fuzzy_score", "keyword"], ascending=[False, False, True])
        .drop_duplicates(subset=["keyword"], keep="first")
        .reset_index(drop=True)
    )


def calculate_fuzzy_score(left: str, right: str) -> int:
    if not left or not right:
        return 0
    return round(fuzz.token_sort_ratio(left, right))


def _semantic_scores(normalized_query: str, match_candidates: pd.DataFrame) -> list[float]:
    query_token_count = len(normalized_query.split())
    candidate_texts = []
    for row in match_candidates.itertuples(index=False):
        text_parts = [
            str(row.normalized_keyword),
            str(row.semantic_group),
            str(row.parent_keyword),
        ]
        if query_token_count >= 2:
            text_parts.append(str(row.related_keywords))
        candidate_texts.append(" ".join(text_parts))

    if not candidate_texts:
        return []

    cosine_scores = cosine_similarity_scores(normalized_query, candidate_texts)

    scores = []
    for index, row in enumerate(match_candidates.itertuples(index=False)):
        normalized_keyword = str(row.normalized_keyword)
        semantic_group = str(row.semantic_group)
        parent_keyword = str(row.parent_keyword)
        related_keywords = str(row.related_keywords)
        score = float(cosine_scores[index]) if index < len(cosine_scores) else 0.0

        if (
            _contains_phrase(normalized_query, normalized_keyword)
            or _contains_phrase(normalized_query, semantic_group)
            or _contains_phrase(normalized_query, parent_keyword)
            or (query_token_count >= 2 and _contains_phrase(normalized_query, related_keywords))
        ):
            score = max(score, 0.8)

        scores.append(score)

    return scores


def _semantic_reason(
    normalized_query: str,
    normalized_keyword: str,
    fuzzy_score: int,
    semantic_score: float,
    fuzzy_threshold: int,
) -> str:
    if normalized_query == normalized_keyword:
        return "exact"
    if _contains_phrase(normalized_query, normalized_keyword):
        return "contains"
    if passes_fuzzy_threshold(normalized_query, normalized_keyword, fuzzy_score, fuzzy_threshold):
        return "fuzzy"
    if semantic_score >= 0.35:
        return "semantic"
    return "matched"


def _match_category(match_reason: str) -> str:
    if match_reason == "exact":
        return "Exact keyword match"
    if match_reason == "semantic":
        return "Semantic match"
    if match_reason == "fuzzy":
        return "Fuzzy keyword match"
    if match_reason == "contains":
        return "Contains keyword match"
    return "Keyword match"


def _contains_phrase(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    if needle == haystack:
        return True

    needle_tokens = needle.split()
    haystack_tokens = haystack.split()
    if len(needle_tokens) > len(haystack_tokens):
        return False

    for start_index in range(0, len(haystack_tokens) - len(needle_tokens) + 1):
        if haystack_tokens[start_index : start_index + len(needle_tokens)] == needle_tokens:
            return True

    return False


def passes_fuzzy_threshold(
    normalized_query: str,
    normalized_keyword: str,
    fuzzy_score: int,
    fuzzy_threshold: int,
) -> bool:
    if len(normalized_query) <= 3 or len(normalized_keyword) <= 3:
        return normalized_query == normalized_keyword
    return fuzzy_score >= fuzzy_threshold


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "keyword",
            "keyword_type",
            "keyword_source",
            "semantic_group",
            "parent_keyword",
            "related_keywords",
            "fuzzy_score",
            "semantic_score",
            "match_category",
            "match_reason",
        ]
    )
