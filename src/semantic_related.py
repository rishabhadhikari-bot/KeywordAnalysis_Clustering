"""Pure result assembly for relationship-aware related-title performance."""

from __future__ import annotations

import re
from typing import Any, Collection

import pandas as pd

from src.data_processing import simple_title_tokens


DEFAULT_HIGH_CONFIDENCE_ROUTE_SCORE = 0.86
DEFAULT_BOUNDED_TITLE_EVIDENCE_MIN_SCORE = 0.78


def build_fast_relationship_titles(
    relationship_candidates: list[dict[str, Any]],
    *,
    high_confidence_score: float = DEFAULT_HIGH_CONFIDENCE_ROUTE_SCORE,
    bounded_title_evidence_min_score: float = (
        DEFAULT_BOUNDED_TITLE_EVIDENCE_MIN_SCORE
    ),
) -> pd.DataFrame:
    """Apply conservative title-level checks to relationship-route candidates."""
    rows: list[dict[str, Any]] = []
    for candidate in relationship_candidates:
        story_id = str(candidate.get("story_id", "")).strip()
        page_title = str(candidate.get("page_title", "")).strip()
        evidence = candidate.get("retrieval_evidence", [])
        if not story_id or not page_title or not isinstance(evidence, list):
            continue

        accepted_evidence = []
        for route in evidence:
            if not isinstance(route, dict):
                continue
            title_match, matched_phrase = _route_has_title_evidence(page_title, route)
            semantic_score = float(route.get("semantic_score", 0.0))
            standalone = route.get("can_retrieve_standalone") is not False
            lexical_route = (
                route.get("match_method") == "opensearch_relationship_phrase"
            )
            semantic_route = not lexical_route
            high_confidence_match = (
                semantic_route
                and standalone
                and semantic_score >= float(high_confidence_score)
            )
            bounded_title_match = (
                semantic_route
                and not standalone
                and title_match
                and semantic_score >= float(bounded_title_evidence_min_score)
            )
            standalone_title_match = standalone and title_match
            if not (
                standalone_title_match
                or bounded_title_match
                or high_confidence_match
            ):
                continue
            accepted = dict(route)
            accepted["title_evidence"] = matched_phrase
            if bounded_title_match:
                accepted["validation_status"] = (
                    "Query-anchored route and relationship cue agree"
                )
            elif standalone_title_match:
                accepted["validation_status"] = (
                    "Standalone related subject or required cue appears in title"
                )
            else:
                accepted["validation_status"] = (
                    "High-confidence standalone relationship route"
                )
            accepted_evidence.append(accepted)

        if not accepted_evidence:
            continue
        accepted_evidence.sort(
            key=lambda item: (
                bool(item.get("title_evidence")),
                float(item.get("semantic_score", 0.0)),
                float(item.get("relationship_quality", 0.0)),
            ),
            reverse=True,
        )
        primary = accepted_evidence[0]
        lexical_evidence = bool(primary.get("title_evidence"))
        semantic_score = float(primary.get("semantic_score", 0.0))
        relationship_quality = float(primary.get("relationship_quality", 0.0))
        relationship_score = min(
            1.0,
            semantic_score * 0.70
            + relationship_quality * 0.20
            + (0.08 if lexical_evidence else 0.0)
            + min(0.02, 0.01 * (len(accepted_evidence) - 1)),
        )
        rows.append(
            {
                "story_id": story_id,
                "page_title": page_title,
                "ai_term": str(primary.get("related_subject", "")),
                "ai_category": str(primary.get("relationship_class", "")),
                "ai_relationship": str(primary.get("factual_bridge", "")),
                "ai_audit_reason": str(primary.get("acceptance_condition", "")),
                "ai_confidence": round(relationship_score, 4),
                "ai_relevance_level": 2 if lexical_evidence else 1,
                "ai_relevance_type": (
                    "core_related" if lexical_evidence else "contextual"
                ),
                "ai_supporting_routes": len(accepted_evidence),
                "relationship_semantic_score": round(semantic_score, 6),
                "title_relationship_evidence": str(
                    primary.get("title_evidence", "")
                ),
                "validation_status": str(primary.get("validation_status", "")),
                "total_views": candidate.get("total_views", 0),
                "active_months": candidate.get("active_months", 0),
                "first_month": candidate.get("first_month"),
                "last_month": candidate.get("last_month"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "story_id",
                "page_title",
                "ai_term",
                "ai_category",
                "ai_relationship",
                "ai_confidence",
                "ai_relevance_level",
                "total_views",
            ]
        )
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["story_id"], keep="first")
        .sort_values(
            ["ai_relevance_level", "ai_confidence", "total_views"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def _route_has_title_evidence(
    page_title: str,
    route: dict[str, Any],
) -> tuple[bool, str]:
    normalized_title = " ".join(simple_title_tokens(page_title))
    if not normalized_title:
        return False, ""
    phrases = _relationship_title_phrases(route)
    for phrase in phrases:
        normalized_phrase = " ".join(simple_title_tokens(phrase))
        if not normalized_phrase:
            continue
        if f" {normalized_phrase} " in f" {normalized_title} ":
            return True, phrase
    return False, ""


def _relationship_title_phrases(route: dict[str, Any]) -> list[str]:
    subject = str(route.get("related_subject", "")).strip()
    relationship_role = str(route.get("relationship_role", "")).upper()
    compact_subject = re.sub(r"[^A-Za-z0-9]", "", subject)
    ambiguous_short_intelligence_name = (
        len(simple_title_tokens(subject)) == 1
        and len(compact_subject) <= 4
        and "INTELLIGENCE" in relationship_role
    )
    # Names such as Israel's intelligence directorate "Aman" collide with
    # ordinary personal names in headlines.  Its longer manifestations or
    # query-anchored semantic route may still qualify; the bare token may not.
    phrases = [] if ambiguous_short_intelligence_name else [subject]
    if subject and not ambiguous_short_intelligence_name:
        phrases.extend(re.findall(r"\(([^)]+)\)", subject))
        phrases.append(re.sub(r"\s*\([^)]*\)\s*", " ", subject).strip())
    required_cues = route.get("required_title_cues", [])
    if isinstance(required_cues, list):
        phrases.extend(str(cue).strip() for cue in required_cues)
    safe_phrases = []
    for phrase in phrases:
        if not phrase:
            continue
        compact_original = re.sub(r"[^A-Za-z0-9]", "", phrase)
        if (
            len(simple_title_tokens(phrase)) == 1
            and len(compact_original) <= 3
            and compact_original.isupper()
        ):
            continue
        safe_phrases.append(phrase)
    return list(dict.fromkeys(safe_phrases))


def build_relationship_performance_result(
    related_titles: pd.DataFrame,
    excluded_story_ids: Collection[str],
    story_months: pd.DataFrame,
) -> dict[str, Any]:
    """Enforce refined-result exclusion and aggregate accepted-title traffic."""
    excluded = {
        str(story_id).strip()
        for story_id in excluded_story_ids
        if str(story_id).strip()
    }
    candidates = related_titles.copy()
    if "story_id" not in candidates.columns:
        candidates["story_id"] = pd.Series(dtype="string")
    candidates["story_id"] = candidates["story_id"].astype(str).str.strip()
    candidates = candidates.loc[candidates["story_id"].ne("")].copy()

    candidate_ids = set(candidates["story_id"])
    excluded_overlap = candidate_ids & excluded
    candidates = candidates.loc[~candidates["story_id"].isin(excluded)].copy()
    candidates = candidates.drop_duplicates(subset=["story_id"], keep="first")

    sort_columns = [
        column
        for column in ("ai_relevance_level", "ai_confidence", "total_views")
        if column in candidates.columns
    ]
    if sort_columns:
        candidates = candidates.sort_values(
            sort_columns,
            ascending=[False] * len(sort_columns),
        )
    candidates = candidates.reset_index(drop=True)
    candidates.insert(0, "relationship_rank", range(1, len(candidates) + 1))

    months = story_months.copy()
    if "story_id" not in months.columns:
        months["story_id"] = pd.Series(dtype="string")
    months["story_id"] = months["story_id"].astype(str).str.strip()
    accepted_ids = set(candidates["story_id"])
    matched_story_months = months.loc[months["story_id"].isin(accepted_ids)].copy()

    if matched_story_months.empty:
        monthly_summary = pd.DataFrame(
            columns=["month", "related_titles_without_query", "views"]
        )
    else:
        monthly_summary = (
            matched_story_months.groupby("month", as_index=False)
            .agg(
                related_titles_without_query=("story_id", "nunique"),
                views=("views", "sum"),
            )
            .sort_values("month")
        )

    return {
        "matched_titles": candidates,
        "matched_story_months": matched_story_months,
        "monthly_summary": monthly_summary,
        "retrieved_relationship_titles": len(candidate_ids),
        "excluded_refined_titles": len(excluded_overlap),
        "refined_story_ids": excluded,
    }
