"""Independent daily title-token aggregation; never modifies the lookup model."""

from collections import Counter
import unicodedata

import numpy as np
import pandas as pd


DEFAULT_STOPWORDS = frozenset(
    "a an and are as at be been being but by for from had has have he her his "
    "i in is it its of on or our she that the their them they this to was we "
    "were will with you your".split()
)
OUTPUT_COLUMNS = [
    "date", "token", "occurrences", "total_token_occurrences",
    "total_views", "views_per_occurrence",
]


def title_tokens(title: str) -> list[str]:
    """Unicode words/numbers; punctuation and hyphens separate tokens."""
    text = unicodedata.normalize("NFKC", title).casefold()
    text = "".join(
        char if unicodedata.category(char)[0] in {"L", "M", "N"} else " "
        for char in text
    )
    return text.split()


def build_daily_keyword_dataset(
    traffic: pd.DataFrame,
    metadata: pd.DataFrame,
    excluded_tokens: frozenset[str] = DEFAULT_STOPWORDS,
) -> tuple[pd.DataFrame, dict]:
    """Sum additive traffic rows per page/day before assigning views to tokens.

    Accept ISO YYYY-MM-DD or analytics YYYYMMDD dates only. Uploaded titles
    take precedence; conflicting titles per page/day must be resolved upstream.
    Pages absent on a day are not assumed to have zero views.
    """
    aliases = {"Date": "date", "Date (YYYYMMDD)": "date", "Event count": "views",
               "Views": "views", "PageTtile": "page_title", "Page title": "page_title"}
    rows = traffic.rename(columns=aliases).copy()
    if rows.columns.duplicated().any():
        raise ValueError("Duplicate input columns after normalizing column names.")
    missing = {"date", "story_id", "views"} - set(rows.columns)
    if missing:
        raise ValueError("Daily CSV is missing columns: " + ", ".join(sorted(missing)))
    if rows.empty:
        raise ValueError("The daily CSV contains no records.")
    dates = rows["date"].astype("string").str.strip()
    compact = dates.str.fullmatch(r"\d{8}", na=False)
    iso = dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    parsed = pd.to_datetime(dates.where(iso), format="%Y-%m-%d", errors="coerce")
    parsed.loc[compact] = pd.to_datetime(dates.loc[compact], format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("Dates must be valid daily dates in YYYY-MM-DD or YYYYMMDD format; monthly totals cannot be used.")
    rows["date"] = parsed
    rows["story_id"] = rows["story_id"].astype("string").str.strip()
    if (rows["story_id"].isna() | rows["story_id"].eq("")).any():
        raise ValueError("Every daily row must have a story_id.")
    rows["views"] = pd.to_numeric(rows["views"], errors="coerce")
    if (rows["views"].isna() | ~np.isfinite(rows["views"]) | rows["views"].lt(0)).any():
        raise ValueError("Views must be finite, non-negative numbers.")

    titles = metadata[["story_id", "page_title"]].copy()
    titles["story_id"] = titles["story_id"].astype("string").str.strip()
    titles = titles.drop_duplicates("story_id", keep="last").set_index("story_id")["page_title"]
    if "page_title" not in rows:
        rows["page_title"] = ""
    rows["page_title"] = rows["page_title"].fillna("").astype(str).str.strip()
    supplied = rows["page_title"].ne("")
    title_counts = rows.loc[supplied].groupby(["date", "story_id"])["page_title"].nunique()
    if title_counts.gt(1).any():
        raise ValueError("Conflicting titles for the same story_id and day. Supply one title per page/day.")
    rows["page_title"] = rows["page_title"].replace("", pd.NA)
    daily = rows.groupby(["date", "story_id"], as_index=False).agg(
        views=("views", "sum"), page_title=("page_title", "first")
    )
    fallback = daily["page_title"].isna()
    daily["page_title"] = daily["page_title"].fillna(daily["story_id"].map(titles)).fillna("")
    missing_titles = daily["page_title"].str.strip().eq("")
    audit = {
        "input_rows": len(rows), "page_days": len(daily),
        "missing_title_page_days": int(missing_titles.sum()),
        "missing_title_views": float(daily.loc[missing_titles, "views"].sum()),
        "metadata_title_page_days": int((fallback & ~missing_titles).sum()),
    }
    counts = {
        title: {token: count for token, count in Counter(title_tokens(title)).items()
                if token not in excluded_tokens}
        for title in daily["page_title"].unique()
    }
    daily["token"] = daily["page_title"].map(lambda title: list(counts[title]))
    expanded = daily.explode("token").dropna(subset=["token"])
    if expanded.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), audit
    expanded["token_count"] = [counts[title][token] for title, token in
                               zip(expanded["page_title"], expanded["token"])]
    result = expanded.groupby(["date", "token"], as_index=False).agg(
        occurrences=("story_id", "size"), total_token_occurrences=("token_count", "sum"),
        total_views=("views", "sum"),
    )
    result["views_per_occurrence"] = result["total_views"] / result["occurrences"]
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result[OUTPUT_COLUMNS].sort_values(
        ["date", "total_views", "token"], ascending=[True, False, True], ignore_index=True
    ), audit
