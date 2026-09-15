from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd


def add_corpus_semantic_context(
    keyword_df: pd.DataFrame,
    exploded_keywords: pd.DataFrame,
    max_related_terms: int = 8,
) -> pd.DataFrame:
    if keyword_df.empty or exploded_keywords.empty:
        return keyword_df

    related_terms = build_keyword_cooccurrence_map(
        exploded_keywords=exploded_keywords,
        max_related_terms=max_related_terms,
    )

    enriched = keyword_df.copy()
    enriched["related_keywords"] = enriched["normalized_keyword"].map(
        lambda keyword: ", ".join(related_terms.get(str(keyword), []))
    )
    return enriched


def build_keyword_cooccurrence_map(
    exploded_keywords: pd.DataFrame,
    max_related_terms: int = 8,
) -> dict[str, list[str]]:
    required_columns = {"month", "story_id", "normalized_keyword"}
    if not required_columns.issubset(exploded_keywords.columns):
        return {}

    cooccurrence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    story_groups = exploded_keywords.groupby(["month", "story_id"], dropna=False)

    for _, group in story_groups:
        keywords = sorted(
            {
                str(keyword).strip()
                for keyword in group["normalized_keyword"].dropna()
                if str(keyword).strip()
            }
        )
        if len(keywords) < 2:
            continue

        for left_keyword, right_keyword in combinations(keywords, 2):
            cooccurrence_counts[left_keyword][right_keyword] += 1
            cooccurrence_counts[right_keyword][left_keyword] += 1

    return {
        keyword: [
            related_keyword
            for related_keyword, _ in counts.most_common(max_related_terms)
        ]
        for keyword, counts in cooccurrence_counts.items()
    }
