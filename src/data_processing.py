from pathlib import Path
import re

import pandas as pd

from src.config import MIN_KEYWORD_LENGTH, PV_THRESHOLD_PERCENT, TWO_LETTER_SEARCH_ABBREVIATIONS
from src.nlp.keybert_scoring import score_keyword_records_with_keybert
from src.nlp.keyword_extraction import extract_keyword_candidates
from src.nlp.normalization import is_keyword_token, normalize_text, normalize_token
from src.nlp.search import (
    calculate_fuzzy_score,
    filter_keyword_results,
    passes_fuzzy_threshold,
)
from src.nlp.semantic_expansion import add_corpus_semantic_context


REQUIRED_STORY_COLUMNS = {"story_id", "page_title"}
REQUIRED_TRAFFIC_COLUMNS = {"story_id", "Month", "Event count"}

STORY_ID_COLUMN_ALIASES = {"storyid", "story_id"}
PAGE_TITLE_COLUMN_ALIASES = {"pagetitle", "page_title", "pagetitel", "pagettile"}
WORD_OVERVIEW_EXCLUDED_TOKENS = {
    "april",
    "august",
    "check",
    "december",
    "february",
    "january",
    "july",
    "june",
    "latest",
    "march",
    "may",
    "new",
    "november",
    "october",
    "said",
    "says",
    "september",
    "today",
    "tomorrow",
    "watch",
    "year",
}


def load_story_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype="string")
    metadata = _normalize_story_metadata_columns(metadata, path)
    _validate_columns(metadata, REQUIRED_STORY_COLUMNS, path)

    metadata = metadata.copy()
    metadata["story_id"] = metadata["story_id"].str.strip()
    metadata["page_title"] = metadata["page_title"].fillna("").str.strip()
    metadata = metadata.drop_duplicates(subset=["story_id"], keep="last")
    return metadata


def load_story_metrics(path: Path) -> pd.DataFrame:
    traffic = pd.read_csv(path, dtype={"story_id": "string"})
    _validate_columns(traffic, REQUIRED_TRAFFIC_COLUMNS, path)

    traffic = traffic.copy()
    traffic["story_id"] = traffic["story_id"].str.strip()
    traffic["month"] = pd.to_datetime(traffic["Month"], format="%b-%Y", errors="coerce").dt.to_period("M").dt.to_timestamp()
    traffic["views"] = pd.to_numeric(traffic["Event count"], errors="coerce").fillna(0)

    invalid_months = traffic["month"].isna().sum()
    if invalid_months:
        raise ValueError(f"{path} has {invalid_months} rows with invalid Month values.")

    return traffic[["story_id", "month", "views"]]


def build_keyword_traffic_model(
    story_metadata: pd.DataFrame,
    monthly_views: pd.DataFrame,
    threshold_percent: float = PV_THRESHOLD_PERCENT,
) -> dict[str, pd.DataFrame]:
    monthly_story_views = (
        monthly_views.groupby(["month", "story_id"], as_index=False)
        .agg(views=("views", "sum"))
        .sort_values(["month", "views"], ascending=[True, False])
    )

    monthly_totals = monthly_story_views.groupby("month", as_index=False).agg(monthly_total_views=("views", "sum"))
    monthly_totals["monthly_threshold"] = monthly_totals["monthly_total_views"] * threshold_percent

    eligible_stories = monthly_story_views.merge(monthly_totals, on="month", how="left")
    eligible_stories["is_eligible"] = eligible_stories["views"] >= eligible_stories["monthly_threshold"]
    eligible_stories = eligible_stories.loc[eligible_stories["is_eligible"]].copy()

    enriched_stories = eligible_stories.merge(story_metadata, on="story_id", how="left")
    enriched_stories["page_title"] = enriched_stories["page_title"].fillna("")
    extracted_records = enriched_stories["page_title"].apply(extract_keyword_candidates).tolist()
    scored_records = score_keyword_records_with_keybert(
        titles=enriched_stories["page_title"].tolist(),
        records_by_title=extracted_records,
    )
    enriched_stories["keyword_records"] = scored_records
    enriched_stories["keywords"] = enriched_stories["keyword_records"].apply(
        lambda records: [record["keyword"] for record in records if record["keyword_source"] == "extracted"]
    )

    exploded_keywords = (
        enriched_stories.explode("keyword_records")
        .dropna(subset=["keyword_records"])
        .reset_index(drop=True)
    )
    keyword_details = pd.json_normalize(exploded_keywords["keyword_records"])
    exploded_keywords = pd.concat(
        [
            exploded_keywords.drop(columns=["keyword_records"]).reset_index(drop=True),
            keyword_details.reset_index(drop=True),
        ],
        axis=1,
    )
    if "parent_keyword" not in exploded_keywords.columns:
        exploded_keywords["parent_keyword"] = ""
    exploded_keywords["parent_keyword"] = exploded_keywords["parent_keyword"].fillna("")
    exploded_keywords = exploded_keywords.loc[exploded_keywords["keyword"].str.len() > 0].copy()

    keyword_monthly = (
        exploded_keywords.groupby(
            [
                "keyword",
                "normalized_keyword",
                "keyword_type",
                "keyword_source",
                "semantic_group",
                "parent_keyword",
                "month",
            ],
            as_index=False,
        )
        .agg(
            associated_views=("views", "sum"),
            related_story_count=("story_id", "nunique"),
            avg_relevance_score=("relevance_score", "mean"),
            avg_keybert_score=("keybert_score", "mean"),
        )
        .sort_values(["month", "associated_views"], ascending=[True, False])
    )
    keyword_monthly["avg_relevance_score"] = keyword_monthly["avg_relevance_score"].round(3)
    keyword_monthly["avg_keybert_score"] = keyword_monthly["avg_keybert_score"].round(3)
    keyword_monthly = add_corpus_semantic_context(keyword_monthly, exploded_keywords)

    keyword_summary = (
        keyword_monthly.groupby(
            [
                "keyword",
                "normalized_keyword",
                "keyword_type",
                "keyword_source",
                "semantic_group",
                "parent_keyword",
                "related_keywords",
            ],
            as_index=False,
        )
        .agg(
            total_associated_views=("associated_views", "sum"),
            avg_monthly_associated_views=("associated_views", "mean"),
            active_months=("month", "nunique"),
            avg_relevance_score=("avg_relevance_score", "mean"),
            avg_keybert_score=("avg_keybert_score", "mean"),
        )
        .sort_values("total_associated_views", ascending=False)
    )
    keyword_summary["avg_relevance_score"] = keyword_summary["avg_relevance_score"].round(3)
    keyword_summary["avg_keybert_score"] = keyword_summary["avg_keybert_score"].round(3)

    related_stories = exploded_keywords[
        [
            "keyword",
            "normalized_keyword",
            "keyword_type",
            "keyword_source",
            "relevance_score",
            "base_relevance_score",
            "keybert_score",
            "semantic_group",
            "parent_keyword",
            "extraction_method",
            "pos_pattern",
            "entity_label",
            "month",
            "story_id",
            "page_title",
            "views",
            "monthly_total_views",
            "monthly_threshold",
        ]
    ].sort_values(["keyword", "month", "views"], ascending=[True, True, False])

    return {
        "monthly_totals": monthly_totals,
        "eligible_stories": enriched_stories,
        "exploded_keywords": exploded_keywords,
        "keyword_monthly": keyword_monthly,
        "keyword_summary": keyword_summary,
        "related_stories": related_stories,
    }


def build_simple_keyword_lookup_model(
    story_metadata: pd.DataFrame,
    monthly_views: pd.DataFrame,
    threshold_percent: float = PV_THRESHOLD_PERCENT,
) -> dict[str, pd.DataFrame]:
    monthly_story_views = (
        monthly_views.groupby(["month", "story_id"], as_index=False)
        .agg(views=("views", "sum"))
        .sort_values(["month", "views"], ascending=[True, False])
    )

    monthly_totals = monthly_story_views.groupby("month", as_index=False).agg(monthly_total_views=("views", "sum"))
    monthly_totals["threshold_value"] = monthly_totals["monthly_total_views"] * threshold_percent
    enriched_stories = monthly_story_views.merge(monthly_totals, on="month", how="left")
    enriched_stories["above_monthly_threshold"] = enriched_stories["views"] > enriched_stories["threshold_value"]
    enriched_stories = enriched_stories.merge(story_metadata, on="story_id", how="left")
    enriched_stories["page_title"] = enriched_stories["page_title"].fillna("")
    enriched_stories["title_tokens"] = enriched_stories["page_title"].apply(simple_title_tokens)
    enriched_stories["clean_title"] = enriched_stories["title_tokens"].apply(lambda tokens: " ".join(tokens))

    title_summary = (
        enriched_stories.groupby(["story_id", "page_title", "clean_title"], as_index=False)
        .agg(
            total_views=("views", "sum"),
            active_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
        .sort_values("total_views", ascending=False)
    )
    word_overview = build_title_word_overview(title_summary)

    return {
        "monthly_totals": monthly_totals,
        "story_months": enriched_stories,
        "title_summary": title_summary,
        "word_overview": word_overview,
    }


def build_title_word_overview(title_summary: pd.DataFrame) -> pd.DataFrame:
    if title_summary.empty:
        return pd.DataFrame(columns=["word", "total_views", "title_count", "avg_views_per_title"])

    word_records = title_summary[["story_id", "clean_title", "total_views"]].copy()
    word_records["word"] = word_records["clean_title"].apply(lambda text: sorted(set(str(text).split())))
    word_records = word_records.explode("word").dropna(subset=["word"])
    word_records = word_records[
        word_records["word"].str.contains(r"[a-z]", regex=True)
        & (word_records["word"].str.len() >= MIN_KEYWORD_LENGTH)
        & ~word_records["word"].isin(WORD_OVERVIEW_EXCLUDED_TOKENS)
    ]

    if word_records.empty:
        return pd.DataFrame(columns=["word", "total_views", "title_count", "avg_views_per_title"])

    word_overview = (
        word_records.groupby("word", as_index=False)
        .agg(
            total_views=("total_views", "sum"),
            title_count=("story_id", "nunique"),
        )
        .sort_values(["total_views", "title_count", "word"], ascending=[False, False, True])
    )
    word_overview["avg_views_per_title"] = word_overview["total_views"] / word_overview["title_count"]
    return word_overview


def search_titles_by_keywords(
    keyword_query: str,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
    match_mode: str = "All keywords",
    include_similar_spellings: bool = False,
    fuzzy_threshold: int = 75,
) -> dict[str, pd.DataFrame | list[str]]:
    query_tokens = simple_query_tokens(keyword_query)
    query_tokens = list(dict.fromkeys(query_tokens))
    if not query_tokens:
        return {
            "query_tokens": [],
            "matched_title_summary": title_summary.head(0),
            "exact_matched_title_summary": title_summary.head(0),
            "fuzzy_matched_title_summary": title_summary.head(0),
            "matched_story_months": story_months.head(0),
            "monthly_summary": story_months.head(0),
        }

    exact_story_ids = _match_story_ids_by_tokens(
        title_summary=title_summary,
        query_tokens=query_tokens,
        match_mode=match_mode,
    )

    match_details = pd.DataFrame(
        {
            "story_id": exact_story_ids,
            "direct_match_tier": "Exact lexical match",
            "direct_match_score": 100,
            "direct_match_detail": "Exact normalized token match",
        }
    )
    if include_similar_spellings:
        fuzzy_details = _match_story_ids_by_fuzzy_tokens(
            title_summary=title_summary.loc[~title_summary["story_id"].isin(exact_story_ids)],
            query_tokens=query_tokens,
            match_mode=match_mode,
            fuzzy_threshold=fuzzy_threshold,
        )
        match_details = pd.concat([match_details, fuzzy_details], ignore_index=True)

    matched_story_ids = match_details["story_id"]
    matched_title_summary = title_summary.loc[title_summary["story_id"].isin(matched_story_ids)].copy()
    matched_title_summary = matched_title_summary.merge(match_details, on="story_id", how="left")
    matched_story_months = story_months.loc[story_months["story_id"].isin(matched_story_ids)].copy()
    matched_story_months = matched_story_months.merge(
        match_details[["story_id", "direct_match_tier"]],
        on="story_id",
        how="left",
    )
    matched_title_summary = add_threshold_split_columns(matched_title_summary, matched_story_months)
    exact_matched_titles = matched_title_summary.loc[
        matched_title_summary["direct_match_tier"] == "Exact lexical match"
    ].copy()
    fuzzy_matched_titles = matched_title_summary.loc[
        matched_title_summary["direct_match_tier"] == "Similar spelling/root match"
    ].copy()
    above_threshold_titles = matched_title_summary.loc[
        matched_title_summary["threshold_matched_months"] > 0
    ].copy()
    below_threshold_titles = matched_title_summary.loc[
        matched_title_summary["threshold_matched_months"] == 0
    ].copy()

    if not matched_story_months.empty:
        matched_story_months = matched_story_months.sort_values(["views", "month"], ascending=[False, True])
        monthly_summary = (
            matched_story_months.groupby("month", as_index=False)
            .agg(
                titles_containing_keywords=("story_id", "nunique"),
                views=("views", "sum"),
            )
            .sort_values("month")
        )
    else:
        monthly_summary = pd.DataFrame(columns=["month", "titles_containing_keywords", "views"])

    return {
        "query_tokens": query_tokens,
        "matched_title_summary": matched_title_summary,
        "exact_matched_title_summary": exact_matched_titles,
        "fuzzy_matched_title_summary": fuzzy_matched_titles,
        "above_threshold_titles": above_threshold_titles,
        "below_threshold_titles": below_threshold_titles,
        "matched_story_months": matched_story_months,
        "monthly_summary": monthly_summary,
    }


def add_threshold_split_columns(title_summary: pd.DataFrame, story_months: pd.DataFrame) -> pd.DataFrame:
    if title_summary.empty or story_months.empty or "above_monthly_threshold" not in story_months.columns:
        enriched_titles = title_summary.copy()
        enriched_titles["threshold_matched_months"] = 0
        enriched_titles["views_above_threshold"] = 0
        return enriched_titles

    threshold_rows = story_months.copy()
    threshold_rows["threshold_matched_months"] = threshold_rows["above_monthly_threshold"].astype(int)
    threshold_rows["views_above_threshold"] = threshold_rows["views"].where(
        threshold_rows["above_monthly_threshold"],
        0,
    )
    threshold_stats = (
        threshold_rows.groupby("story_id", as_index=False)
        .agg(
            threshold_matched_months=("threshold_matched_months", "sum"),
            views_above_threshold=("views_above_threshold", "sum"),
        )
    )

    enriched_titles = title_summary.merge(threshold_stats, on="story_id", how="left")
    enriched_titles["threshold_matched_months"] = enriched_titles["threshold_matched_months"].fillna(0).astype(int)
    enriched_titles["views_above_threshold"] = enriched_titles["views_above_threshold"].fillna(0).astype(int)
    return enriched_titles


def _match_story_ids_by_tokens(
    title_summary: pd.DataFrame,
    query_tokens: list[str],
    match_mode: str,
) -> pd.Series:
    query_phrase = " ".join(query_tokens)

    if match_mode == "Exact cleaned phrase":
        mask = title_summary["clean_title"].apply(lambda text: _contains_token_phrase(str(text), query_phrase))
    elif match_mode == "Any keyword":
        mask = title_summary["clean_title"].apply(
            lambda text: any(_contains_token_phrase(str(text), token) for token in query_tokens)
        )
    else:
        mask = title_summary["clean_title"].apply(
            lambda text: all(_contains_token_phrase(str(text), token) for token in query_tokens)
        )

    return title_summary.loc[mask, "story_id"]


def _contains_token_phrase(clean_title: str, phrase: str) -> bool:
    phrase_tokens = phrase.split()
    title_tokens = _title_tokens_for_query(clean_title, phrase_tokens)
    if not title_tokens or not phrase_tokens or len(phrase_tokens) > len(title_tokens):
        return False

    for start_index in range(0, len(title_tokens) - len(phrase_tokens) + 1):
        if title_tokens[start_index : start_index + len(phrase_tokens)] == phrase_tokens:
            return True
    return False


def _match_story_ids_by_fuzzy_tokens(
    title_summary: pd.DataFrame,
    query_tokens: list[str],
    match_mode: str,
    fuzzy_threshold: int,
) -> pd.DataFrame:
    matched_rows = []
    for row in title_summary.itertuples(index=False):
        title_tokens = _title_tokens_for_query(str(row.clean_title), query_tokens)
        matched, score, detail = _fuzzy_token_match(
            title_tokens=title_tokens,
            query_tokens=query_tokens,
            match_mode=match_mode,
            fuzzy_threshold=fuzzy_threshold,
        )
        if matched:
            matched_rows.append(
                {
                    "story_id": row.story_id,
                    "direct_match_tier": "Similar spelling/root match",
                    "direct_match_score": score,
                    "direct_match_detail": detail,
                }
            )

    return pd.DataFrame(
        matched_rows,
        columns=[
            "story_id",
            "direct_match_tier",
            "direct_match_score",
            "direct_match_detail",
        ],
    )


def _title_tokens_for_query(clean_title: str, query_tokens: list[str]) -> list[str]:
    """Remove non-requested short title words while retaining queried abbreviations.

    The stored title stream keeps every two-letter token so an uppercase unknown
    abbreviation can be found. Ignoring unrelated short words at match time
    preserves the previous cleaned-phrase behavior for ordinary searches.
    """
    requested_tokens = set(query_tokens)
    return [
        token
        for token in clean_title.split()
        if len(token) != 2 or token in requested_tokens
    ]


def _fuzzy_token_match(
    title_tokens: list[str],
    query_tokens: list[str],
    match_mode: str,
    fuzzy_threshold: int,
) -> tuple[bool, int, str]:
    if not title_tokens or not query_tokens:
        return False, 0, ""

    def token_score(query_token: str, title_token: str) -> int:
        # Acronyms and other very short terms must match exactly. Fuzzy matching
        # them creates noisy pairs such as AI -> rain or PM -> PMO.
        score = calculate_fuzzy_score(query_token, title_token)
        return (
            score
            if passes_fuzzy_threshold(
                query_token,
                title_token,
                score,
                fuzzy_threshold,
            )
            else 0
        )

    def summarize(pairs: list[tuple[str, str, int]]) -> tuple[bool, int, str]:
        if not pairs or any(score < fuzzy_threshold for _, _, score in pairs):
            return False, 0, ""
        # At least one token must be approximate; exact-only titles belong in
        # tier one and have already been removed from this pass.
        approximate_pairs = [pair for pair in pairs if pair[0] != pair[1]]
        if not approximate_pairs:
            return False, 0, ""
        score = min(pair[2] for pair in pairs)
        detail = ", ".join(
            f"{query_token} -> {title_token} ({pair_score})"
            for query_token, title_token, pair_score in approximate_pairs
        )
        return True, score, detail

    if match_mode == "Exact cleaned phrase":
        if len(query_tokens) > len(title_tokens):
            return False, 0, ""
        candidates = []
        for start_index in range(len(title_tokens) - len(query_tokens) + 1):
            window = title_tokens[start_index : start_index + len(query_tokens)]
            pairs = [
                (query_token, title_token, token_score(query_token, title_token))
                for query_token, title_token in zip(query_tokens, window)
            ]
            candidate = summarize(pairs)
            if candidate[0]:
                candidates.append(candidate)
        return max(candidates, key=lambda item: item[1]) if candidates else (False, 0, "")

    best_pairs = []
    for query_token in query_tokens:
        title_token, score = max(
            ((title_token, token_score(query_token, title_token)) for title_token in title_tokens),
            key=lambda item: item[1],
        )
        best_pairs.append((query_token, title_token, score))

    if match_mode == "Any keyword":
        best_pair = max(best_pairs, key=lambda item: item[2])
        return summarize([best_pair])
    return summarize(best_pairs)


def simple_title_tokens(text: str) -> list[str]:
    """Build the normalized token stream stored for direct title search.

    Two-letter alphabetic tokens are retained regardless of source casing so
    lowercase source titles can match an editor's uppercase abbreviation query.
    Longer tokens continue to use the existing keyword and stop-word rules.
    """
    tokens = []
    if not isinstance(text, str):
        return tokens

    raw_tokens = _simple_search_raw_tokens(text)
    for raw_token in raw_tokens:
        normalized_token = normalize_token(raw_token.lower())
        normalized_token = _simple_lookup_lemma(normalized_token)
        is_two_letter_token = normalized_token.isalpha() and len(normalized_token) == 2
        if is_keyword_token(normalized_token) or is_two_letter_token:
            tokens.append(normalized_token)
    return tokens


def simple_query_tokens(text: str) -> list[str]:
    """Build direct-search query tokens with explicit abbreviation intent.

    Unknown uppercase abbreviations work without registration. Registered
    abbreviations also work in lowercase or title case. Ordinary lowercase
    two-letter words remain subject to the existing stop-word/length rules.
    """
    tokens = []
    if not isinstance(text, str):
        return tokens

    raw_tokens = _simple_search_raw_tokens(text)
    for raw_token in raw_tokens:
        normalized_token = normalize_token(raw_token.lower())
        normalized_token = _simple_lookup_lemma(normalized_token)
        is_uppercase_abbreviation = (
            raw_token.isalpha()
            and raw_token.isupper()
            and len(normalized_token) == 2
        )
        is_registered_abbreviation = normalized_token in TWO_LETTER_SEARCH_ABBREVIATIONS
        if (
            is_keyword_token(normalized_token)
            or is_uppercase_abbreviation
            or is_registered_abbreviation
        ):
            tokens.append(normalized_token)
    return tokens


def _simple_search_raw_tokens(text: str) -> list[str]:
    # Collapse dotted alphabetic abbreviations before punctuation tokenization:
    # U.S. -> US, M.S. -> MS. The normalized matcher remains case-insensitive.
    collapsed_text = re.sub(
        r"\b(?:[A-Za-z]\.)+[A-Za-z]\b\.?",
        lambda match: match.group(0).replace(".", ""),
        text.replace("&", " and "),
    )
    return re.findall(r"[A-Za-z0-9]+", collapsed_text)


def _simple_lookup_lemma(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4 and not token.endswith(("ss", "sis")):
        return token[:-1]
    return token


def build_title_keyword_records(title: str) -> list[dict[str, object]]:
    extracted_keywords = extract_keyword_candidates(title)
    return score_keyword_records_with_keybert([title], [extracted_keywords])[0]


def _validate_columns(df: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required column(s): {missing}")


def _normalize_story_metadata_columns(metadata: pd.DataFrame, path: Path) -> pd.DataFrame:
    rename_map = {}
    normalized_columns = {_normalize_column_name(column): column for column in metadata.columns}

    story_id_column = _find_column(normalized_columns, STORY_ID_COLUMN_ALIASES)
    page_title_column = _find_column(normalized_columns, PAGE_TITLE_COLUMN_ALIASES)

    if story_id_column:
        rename_map[story_id_column] = "story_id"
    if page_title_column:
        rename_map[page_title_column] = "page_title"

    normalized_metadata = metadata.rename(columns=rename_map)
    if "story_id" in normalized_metadata.columns and "page_title" in normalized_metadata.columns:
        return normalized_metadata[["story_id", "page_title"]]

    return normalized_metadata


def _find_column(normalized_columns: dict[str, str], aliases: set[str]) -> str | None:
    for alias in aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]
    return None


def _normalize_column_name(column: str) -> str:
    return "".join(character for character in column.lower() if character.isalnum())
