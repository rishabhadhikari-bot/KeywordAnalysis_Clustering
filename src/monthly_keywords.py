"""Monthly title-token aggregation using the existing story-month model."""

import pandas as pd

from src.daily_keywords import DEFAULT_STOPWORDS, build_daily_keyword_dataset


def build_monthly_keyword_dataset(
    story_months: pd.DataFrame,
    excluded_tokens: frozenset[str] = DEFAULT_STOPWORDS,
) -> tuple[pd.DataFrame, dict]:
    """Normalize each month to one grouping key, then reuse token accounting.

    The date key is internal only; source views remain monthly totals throughout.
    No input rows or shared model columns are modified.
    """
    required = {"month", "story_id", "page_title", "views"}
    if missing := required - set(story_months.columns):
        raise ValueError("Missing monthly data columns: " + ", ".join(sorted(missing)))
    rows = story_months[["month", "story_id", "page_title", "views"]].copy()
    columns = ["month", "token", "occurrences", "total_token_occurrences",
               "total_views", "views_per_occurrence"]
    if rows.empty:
        return pd.DataFrame(columns=columns), {"page_months": 0, "missing_title_page_months": 0,
                                              "missing_title_views": 0}
    months = pd.to_datetime(rows.pop("month"), errors="coerce")
    if months.isna().any():
        raise ValueError("Monthly data contains invalid months.")
    rows["date"] = months.dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    try:
        result, audit = build_daily_keyword_dataset(
            rows, pd.DataFrame(columns=["story_id", "page_title"]), excluded_tokens
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("daily", "monthly").replace("page/day", "page/month")
                         .replace("and day", "and month")) from exc
    result = result.rename(columns={"date": "month"})
    result["month"] = result["month"].str.slice(0, 7)
    return result, {"page_months": audit["page_days"],
                    "missing_title_page_months": audit["missing_title_page_days"],
                    "missing_title_views": audit["missing_title_views"]}
