import html
import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.miscellaneous_tab import show_miscellaneous_tab
from src.embeddings_tab import show_embeddings_tab
from pathlib import Path

from src.config import STORY_METADATA_PATH, STORY_METRICS_PATH
from src.data_processing import (
    add_threshold_split_columns,
    build_simple_keyword_lookup_model,
    load_story_metrics,
    load_story_metadata,
    search_titles_by_keywords,
    simple_query_tokens,
)
from src.vertex_related import (
    DEFAULT_GENERATIVE_RETRIEVAL_BATCH_SIZE,
    get_related_ai_mode,
    get_prompt_contract_fingerprint,
    get_vertex_location,
    get_vertex_model_name,
    related_ai_uses_legacy_behavior,
    research_query_relationships_with_vertex,
    select_titles_generatively_with_vertex,
)
from src.relationship_embeddings import (
    RELATIONSHIP_RETRIEVAL_VERSION,
    extract_relationship_routes,
    get_relationship_embedding_model_name,
    passes_relationship_evidence_gate,
    retrieve_relationship_candidates,
)
from src.topic_retrieval import (
    BERTOPIC_INDEX_VERSION,
    BERTOPIC_RETRIEVAL_VERSION,
    get_bertopic_training_config,
    get_bertopic_topic_catalog,
    retrieve_bertopic_candidates,
)
from src.opensearch_refined import (
    MATCH_TIER_ORDER,
    OpenSearchRefinedClient,
    OpenSearchRefinedError,
    build_refined_primary_analysis_result,
    collect_all_refined_primary_story_ids,
    collect_all_refined_story_id_sets,
    load_opensearch_settings,
    start_local_opensearch,
    title_corpus_fingerprint,
)
from src.wikidata_source_related import (
    SOURCE_RELATED_PIPELINE_VERSION,
    SourceProfile,
    SourceRelatedError,
    SourceSettings,
    WikidataLocalClient,
    find_wikidata_related_stories,
    load_source_settings,
)
from src.wikidata_relationship_policy import POLICY_VERSION, RetrievalPolicy
from src.search_aliases import find_ambiguous_aliases, normalize_refined_alias_text
from src.topic_retrieval_refined import (
    REFINED_MODEL_VERSION,
    fit_bertopic_refined_model,
    get_bertopic_refined_topic_catalog,
    refined_corpus_fingerprint,
    retrieve_bertopic_refined_candidates,
)
from src.semantic_search_tab import show_semantic_search_tab
from src.recent_stories_tab import show_recent_stories_tab
from src.semantic_seo_tab import show_semantic_seo_tab
from src.knowledge_source_related import (
    SOURCE_PROFILE_SCHEMA_VERSION,
    SOURCE_RETRIEVAL_VERSION,
    build_source_related_result,
    get_knowledge_source_user_agent,
)


LOOKUP_PIPELINE_VERSION = "2026-08-17-two-letter-abbreviation-search-v4"
RELATED_STORY_INITIAL_LIMIT = 100
RELATED_STORY_LOAD_INCREMENT = 100
REFINED_SEARCH_INITIAL_LIMIT = 1000
REFINED_SEARCH_LOAD_INCREMENT = 1000
ALL_RELATED_RRF_K = 60
ALL_RELATED_AI_WEIGHT = 1.0
ALL_RELATED_BERTOPIC_WEIGHT = 0.75
TITLE_TABLE_SCROLL_AFTER_ROWS = 100
TITLE_TABLE_100_ROW_HEIGHT = 4440
GENERATIVE_RETRIEVAL_VERSION = "2026-08-25-title-normalization-v18"
GROUNDED_PROFILE_MAX_AGE_HOURS = 24
BERTOPIC_RETRIEVAL_CACHE_VERSION = BERTOPIC_RETRIEVAL_VERSION
FLOW_STATUS_POLL_INTERVAL = "2s"
RELATED_LIVE_RESULT_LIMIT = 100
RELATED_FLOW_PROGRESS: dict[str, dict[str, object]] = {}
RELATED_FLOW_PROGRESS_LOCK = threading.Lock()
# GROUNDED_PROFILE_VERSION = "2026-07-30-refined-prompts-v5"
# CANDIDATE_JUDGE_VERSION = "2026-07-30-refined-prompts-v4"
LEGACY_GROUNDED_PROFILE_VERSION = "2026-07-30-refined-prompts-v5"
LEGACY_CANDIDATE_JUDGE_VERSION = "2026-08-18-plain-editorial-summary-v5"
# Previous cache versions, retained so the prompt rollout can be rolled back:
# ENTITY_SAFE_GROUNDED_PROFILE_VERSION = "2026-07-31-query-type-context-v9"
# ENTITY_SAFE_CANDIDATE_JUDGE_VERSION = "2026-07-31-query-type-context-v8"
# Previous implemented research prompt cache version, retained for rollback:
# ENTITY_SAFE_GROUNDED_PROFILE_VERSION = "2026-07-31-global-editorial-prompt-v10"
# Previous manifestation-unaware prompt versions, retained for rollback:
# ENTITY_SAFE_GROUNDED_PROFILE_VERSION = "2026-07-31-search-audit-prompt-v11"
# ENTITY_SAFE_CANDIDATE_JUDGE_VERSION = "2026-07-31-global-editorial-prompt-v9"
ENTITY_SAFE_GROUNDED_PROFILE_VERSION = "2026-08-26-utf8-grounding-consensus-v23"
ENTITY_SAFE_CANDIDATE_JUDGE_VERSION = "2026-08-25-title-normalization-v19"
GROUNDED_PROFILE_VERSION = (
    LEGACY_GROUNDED_PROFILE_VERSION
    if related_ai_uses_legacy_behavior()
    else ENTITY_SAFE_GROUNDED_PROFILE_VERSION
)
CANDIDATE_JUDGE_VERSION = (
    LEGACY_CANDIDATE_JUDGE_VERSION
    if related_ai_uses_legacy_behavior()
    else ENTITY_SAFE_CANDIDATE_JUDGE_VERSION
)


@st.cache_resource(show_spinner=False)
def get_tab_flow_executor() -> ThreadPoolExecutor:
    """Keep slow, explicitly requested tab flows off the Streamlit script thread."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="tab-flow")


def get_tab_flow_state(state_key: str) -> dict[str, object]:
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "future": None,
            "result": None,
            "error": "",
        }
    return st.session_state[state_key]


def build_tab_flow_state_key(prefix: str, *parts: object) -> str:
    serialized_parts = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(serialized_parts.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def build_live_related_results_html(rows: list[dict[str, object]]) -> str:
    """Render progressive results without invoking Streamlit's PyArrow grid."""
    if not rows:
        return ""
    body_rows = []
    for row in rows[:RELATED_LIVE_RESULT_LIMIT]:
        try:
            confidence = float(row.get("ai_confidence", 0.0)) * 100
        except (TypeError, ValueError):
            confidence = 0.0
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('story_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('page_title', '')))}</td>"
            f"<td>{html.escape(str(row.get('ai_relationship', '')))}</td>"
            f"<td>{confidence:.1f}%</td>"
            "</tr>"
        )
    return (
        "<style>"
        ".live-related-scroll{max-height:680px;overflow:auto;border:1px solid #e5e7eb;border-radius:8px;}"
        ".live-related-table{width:100%;border-collapse:separate;border-spacing:0;font-size:14px;line-height:1.35;}"
        ".live-related-table th,.live-related-table td{padding:9px 10px;border-right:1px solid #e5e7eb;"
        "border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top;}"
        ".live-related-table th{position:sticky;top:0;z-index:1;background:#f8fafc;color:#64748b;}"
        ".live-related-table th:nth-child(2),.live-related-table td:nth-child(2){min-width:360px;white-space:normal;}"
        ".live-related-table th:nth-child(3),.live-related-table td:nth-child(3){min-width:520px;white-space:normal;}"
        ".live-related-table th:last-child,.live-related-table td:last-child{border-right:0;}"
        "</style>"
        "<div class='live-related-scroll'><table class='live-related-table'>"
        "<thead><tr><th>Story ID</th><th>Page title</th><th>AI relationship</th>"
        "<th>Confidence</th></tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def build_running_loader_html(message: str) -> str:
    """Render a persistent animated loader for background tab work."""
    return (
        "<style>"
        "@keyframes related-loader-spin{to{transform:rotate(360deg);}}"
        ".related-loader{display:flex;align-items:center;gap:10px;padding:10px 12px;"
        "margin:4px 0 10px;border:1px solid #fecaca;border-radius:8px;background:#fff7f7;"
        "color:#7f1d1d;font-size:14px;font-weight:600;}"
        ".related-loader-icon{width:18px;height:18px;border:3px solid #fecaca;"
        "border-top-color:#dc2626;border-radius:50%;animation:related-loader-spin .8s linear infinite;"
        "flex:0 0 auto;}"
        "</style>"
        "<div class='related-loader' role='status' aria-live='polite'>"
        "<span class='related-loader-icon' aria-hidden='true'></span>"
        f"<span>{html.escape(message)}</span></div>"
    )


def format_process_duration(seconds: object) -> str:
    """Format a process duration compactly without hiding short operations."""
    try:
        duration = max(0.0, float(seconds))
    except (TypeError, ValueError):
        duration = 0.0
    if duration < 1:
        return f"{duration * 1000:.0f} ms"
    if duration < 60:
        return f"{duration:.1f} s"
    minutes, remaining_seconds = divmod(duration, 60)
    return f"{int(minutes)}m {remaining_seconds:.1f}s"


def format_relationship_path(value: object) -> str:
    """Render auditable structured graph paths compactly in editorial tables."""
    if not isinstance(value, list):
        return ""
    labels: list[str] = []
    for step in value:
        if not isinstance(step, dict):
            continue
        source = str(step.get("source_label", "")).strip()
        predicate = str(step.get("predicate_label", "")).strip()
        target = str(step.get("target_label", "")).strip()
        if not labels and source:
            labels.append(source)
        if predicate and target:
            labels.append(f"--{predicate}--> {target}")
        elif target:
            labels.append(f"--> {target}")
    return " ".join(labels)


def get_process_timing_snapshot(
    progress: dict[str, object],
) -> list[dict[str, object]]:
    """Return completed timings plus the elapsed portion of the active phase."""
    raw_timings = progress.get("process_timings", [])
    timings = [
        dict(item)
        for item in raw_timings
        if isinstance(item, dict) and str(item.get("step", "")).strip()
    ]
    active_phase = str(progress.get("phase", "")).strip()
    phase_started_at = progress.get("phase_started_at")
    if active_phase and isinstance(phase_started_at, (int, float)):
        timings.append(
            {
                "step": active_phase,
                "duration_seconds": max(0.0, time.perf_counter() - phase_started_at),
                "is_active": True,
            }
        )
    return timings


def build_process_timings_html(
    timings: list[dict[str, object]],
    total_elapsed_seconds: float | None = None,
) -> str:
    """Build the Related Stories process-timing component."""
    normalized = []
    for item in timings:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "")).strip()
        if not step:
            continue
        try:
            duration = max(0.0, float(item.get("duration_seconds", 0.0)))
        except (TypeError, ValueError):
            duration = 0.0
        normalized.append(
            {
                "step": step,
                "duration_seconds": duration,
                "is_active": bool(item.get("is_active", False)),
            }
        )
    if not normalized:
        return ""

    slowest = max(normalized, key=lambda item: item["duration_seconds"])
    longest_duration = max(float(slowest["duration_seconds"]), 0.001)
    total = (
        max(0.0, float(total_elapsed_seconds))
        if total_elapsed_seconds is not None
        else sum(float(item["duration_seconds"]) for item in normalized)
    )
    rows = []
    for item in normalized:
        duration = float(item["duration_seconds"])
        width = max(2.0, (duration / longest_duration) * 100)
        is_slowest = item is slowest
        row_class = " timing-row-slowest" if is_slowest else ""
        active_label = " <span class='timing-active'>RUNNING</span>" if item["is_active"] else ""
        slowest_label = " <span class='timing-slowest'>SLOWEST</span>" if is_slowest else ""
        rows.append(
            f"<div class='timing-row{row_class}'>"
            "<div class='timing-label'>"
            f"<span>{html.escape(str(item['step']))}{active_label}{slowest_label}</span>"
            f"<strong>{html.escape(format_process_duration(duration))}</strong>"
            "</div>"
            "<div class='timing-track'>"
            f"<div class='timing-bar' style='width:{width:.1f}%'></div>"
            "</div></div>"
        )

    return (
        "<style>"
        ".process-timing{margin:10px 0 14px;padding:14px 16px;border:1px solid #e2e8f0;"
        "border-radius:10px;background:#fff;}"
        ".timing-summary{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;"
        "margin-bottom:12px;}"
        ".timing-title{font-weight:700;color:#0f172a;}"
        ".timing-subtitle{font-size:12px;color:#64748b;margin-top:2px;}"
        ".timing-total{text-align:right;white-space:nowrap;color:#475569;font-size:12px;}"
        ".timing-total strong{display:block;color:#0f172a;font-size:17px;}"
        ".timing-row{padding:7px 8px;margin:3px -8px;border-radius:7px;}"
        ".timing-row-slowest{background:#fff7f7;}"
        ".timing-label{display:flex;justify-content:space-between;gap:12px;font-size:13px;"
        "color:#334155;margin-bottom:5px;}"
        ".timing-label strong{white-space:nowrap;color:#0f172a;}"
        ".timing-track{height:7px;border-radius:999px;background:#e2e8f0;overflow:hidden;}"
        ".timing-bar{height:100%;border-radius:999px;background:#64748b;}"
        ".timing-row-slowest .timing-bar{background:#c62828;}"
        ".timing-active,.timing-slowest{display:inline-block;margin-left:6px;padding:1px 5px;"
        "border-radius:999px;font-size:9px;font-weight:700;vertical-align:1px;}"
        ".timing-active{background:#dbeafe;color:#1d4ed8;}"
        ".timing-slowest{background:#fee2e2;color:#b91c1c;}"
        "</style>"
        "<div class='process-timing'>"
        "<div class='timing-summary'><div>"
        "<div class='timing-title'>Process timing</div>"
        f"<div class='timing-subtitle'>Most time: {html.escape(str(slowest['step']))}</div>"
        "</div>"
        f"<div class='timing-total'>Total elapsed<strong>{html.escape(format_process_duration(total))}</strong></div>"
        "</div>"
        + "".join(rows)
        + "</div>"
    )


def show_related_process_timings(
    timings: list[dict[str, object]],
    total_elapsed_seconds: float | None = None,
) -> None:
    timing_html = build_process_timings_html(timings, total_elapsed_seconds)
    if timing_html:
        st.markdown(timing_html, unsafe_allow_html=True)


@st.fragment(run_every=FLOW_STATUS_POLL_INTERVAL)
def show_tab_flow_progress(state_key: str, message: str) -> None:
    """Poll only a lightweight status fragment while a background flow is active."""
    flow_state = get_tab_flow_state(state_key)
    future = flow_state.get("future")
    if isinstance(future, Future) and not future.done():
        st.markdown(build_running_loader_html(message), unsafe_allow_html=True)
        st.info(message)
        with RELATED_FLOW_PROGRESS_LOCK:
            progress = dict(RELATED_FLOW_PROGRESS.get(state_key, {}))
        if progress:
            phase = str(progress.get("phase", "Starting Related Stories")).strip()
            st.caption(f"Current stage: {phase}")
            completed = int(progress.get("completed_batches", 0))
            total = int(progress.get("total_batches", 0))
            evaluated = int(progress.get("evaluated_count", 0))
            accepted_count = int(progress.get("accepted_count", 0))
            st.caption(
                f"Gemini batches completed: {completed:,} of {total:,}. "
                f"Evaluated {evaluated:,} candidates and found {accepted_count:,} related stories so far."
            )
            show_related_process_timings(get_process_timing_snapshot(progress))
            live_rows = progress.get("live_rows", [])
            if isinstance(live_rows, list) and live_rows:
                st.markdown("#### Verified stories available so far")
                st.markdown(
                    build_live_related_results_html(live_rows),
                    unsafe_allow_html=True,
                )
                if accepted_count > len(live_rows):
                    st.caption(
                        f"Showing the top {len(live_rows):,} verified stories while evaluation continues. "
                        "The complete result will appear when all batches finish."
                    )
        return
    if isinstance(future, Future):
        st.rerun()


@st.fragment(run_every=FLOW_STATUS_POLL_INTERVAL)
def show_source_flow_progress(state_key: str) -> None:
    """Poll the source-only background flow without AI-specific status text."""
    flow_state = get_tab_flow_state(state_key)
    future = flow_state.get("future")
    if isinstance(future, Future) and not future.done():
        message = "Building source-based related-story results"
        st.markdown(build_running_loader_html(message), unsafe_allow_html=True)
        with RELATED_FLOW_PROGRESS_LOCK:
            progress = dict(RELATED_FLOW_PROGRESS.get(state_key, {}))
        phase = str(progress.get("phase", "Contacting knowledge sources")).strip()
        st.caption(f"Current stage: {phase}")
        return
    if isinstance(future, Future):
        st.rerun()


@st.fragment(run_every=FLOW_STATUS_POLL_INTERVAL)
def show_all_related_progress(state_key: str) -> None:
    """Poll the two independent flows without streaming an unstable merged table."""
    flow_state = st.session_state.get(state_key, {})
    ai_future = flow_state.get("ai_future")
    bertopic_future = flow_state.get("bertopic_future")
    ai_running = isinstance(ai_future, Future) and not ai_future.done()
    bertopic_running = isinstance(bertopic_future, Future) and not bertopic_future.done()
    if not ai_running and not bertopic_running:
        st.rerun()
        return

    st.markdown(
        build_running_loader_html("Building and validating related-story results"),
        unsafe_allow_html=True,
    )
    st.info("Building the final combined result. The table will appear after both flows finish.")
    ai_status = "Running" if ai_running else "Complete"
    if flow_state.get("ai_skipped"):
        ai_status = "Skipped (no direct matches)"
    bertopic_status = "Running" if bertopic_running else "Complete"
    st.caption(f"AI-related: {ai_status} · BERTopic: {bertopic_status}")

    with RELATED_FLOW_PROGRESS_LOCK:
        progress = dict(RELATED_FLOW_PROGRESS.get(state_key, {}))
    if ai_running and progress:
        phase = str(progress.get("phase", "Starting Related Stories")).strip()
        completed = int(progress.get("completed_batches", 0))
        total = int(progress.get("total_batches", 0))
        evaluated = int(progress.get("evaluated_count", 0))
        accepted = int(progress.get("accepted_count", 0))
        st.caption(f"AI stage: {phase}")
        if total:
            st.caption(
                f"Gemini batches: {completed:,} of {total:,}; "
                f"evaluated {evaluated:,}, accepted {accepted:,}."
            )
        show_related_process_timings(get_process_timing_snapshot(progress))
QUERY_PROFILE_DB_PATH = Path(__file__).resolve().parent / "query_profiles.db"
SERVICE_ACCOUNT_PATH = Path(__file__).resolve().parent / "service_account.json"
CATEGORY_PATH = Path(__file__).resolve().parent / "StoryID-Category.csv"
WORD_COUNT_PATH = Path(__file__).resolve().parent / "StoryID-WordCount.csv"
REGION_PATH = Path(__file__).resolve().parent / "StoryID-Region_withViews.csv"
PLANNED_TRENDING_PATH = Path(__file__).resolve().parent / "StoryID-planned_trending.csv"
AUDIENCE_TYPE_PATH = Path(__file__).resolve().parent / "StoryID- AudienceType.csv"
WORD_COUNT_BUCKETS = [
    (0, 99, "Under 100 words"),
    (100, 199, "100-199 words"),
    (200, 299, "200-299 words"),
    (300, 399, "300-399 words"),
    (400, 499, "400-499 words"),
    (500, 699, "500-699 words"),
    (700, 999, "700-999 words"),
    (1000, None, "1000+ words"),
]
AUDIENCE_SEGMENT_LABELS = {
    "brand lover": "Brand Lovers",
    "casual readers": "Casual Readers",
    "loyal users": "Loyal Users",
}
EDITORIAL_COLUMN_LABELS = {
    "story_id": "Story ID",
    "page_title": "Page title",
    "month": "Month",
    "word": "Title word",
    "title_count": "Titles",
    "titles_containing_keywords": "Matching titles",
    "views": "Views",
    "total_views": "Total views",
    "views_above_threshold": "Views from high-traffic months",
    "threshold_matched_months": "High-traffic months",
    "active_months": "Months with views",
    "first_month": "First month",
    "last_month": "Last month",
    "monthly_total_views": "Total monthly views",
    "threshold_value": "High-traffic cutoff",
    "monthly_threshold": "High-traffic cutoff",
    "above_monthly_threshold": "Is high-traffic month",
    "avg_views_per_title": "Avg views per title",
    "article_count": "Article count",
    "avg_page_views": "Avg page views",
    "region": "Region",
    "total_articles": "Total articles",
    "page_views": "Page views",
    "views_per_article": "Views per Article",
    "category": "Category",
    "word_count_bucket": "Word count bucket",
    "planned_trending": "Planned/Trending",
    "audience_name": "Audience segment",
    "audience_views": "Audience views",
    "total_users": "Total users",
    "users_per_article": "Users per article",
    "views_per_article": "Views per article",
    "views_per_user": "Views per user",
    "brand_lovers_views": "Brand Lovers views",
    "casual_readers_views": "Casual Readers views",
    "loyal_users_views": "Loyal Users views",
    "ai_relationship": "AI relationship",
    "ai_audit_reason": "AI audit reason",
    "ai_confidence": "AI confidence",
    "ai_term": "AI search term",
    "ai_category": "AI term category",
    "ai_relevance_level": "AI relevance level",
    "ai_relevance_type": "AI relevance type",
    "ai_supporting_routes": "Supporting AI routes",
    "ai_research_source_count": "Web research sources",
    "would_pass_evidence_gate": "Passes strict evidence gate",
    "relationship_or_cluster": "Relationship / cluster",
    "unified_rank_score": "Unified rank score",
    "direct_match_tier": "Match tier",
    "direct_match_score": "Similarity score",
    "direct_match_detail": "Approximate token matches",
    "decision": "Evidence decision",
    "supporting_relationship_id": "Evidence relationship ID",
    "related_subject": "Evidence subject",
    "factual_bridge": "Factual bridge",
    "evidence_kind": "Title evidence type",
    "title_evidence": "Title evidence",
    "reason": "Evidence decision reason",
    "confidence": "Evidence confidence",
    "relevance_level": "Evidence relevance level",
    "relevance_type": "Evidence relevance type",
    "retrieval_channels_display": "Retrieval channels",
    "hybrid_score": "Hybrid RRF score",
    "validation_status": "Validation status",
    "relationship": "Source relationship",
    "matched_title_evidence": "Matched title evidence",
    "source_names": "Knowledge sources",
    "result_tier": "Result tier",
    "predicate_label": "Predicate",
    "relationship_class": "Relationship class",
    "relationship_family": "Relationship family",
    "relationship_role": "Relationship role",
    "relationship_hops": "Graph hops",
    "relationship_path": "Relationship path",
    "supporting_related_subjects": "All supporting subjects",
    "supporting_relationship_count": "Supporting relationships",
    "can_retrieve_standalone": "Standalone route",
    "temporal_scope": "Time scope",
    "source_confidence": "Source confidence",
    "source_related_subject": "Related subject",
    "source_evidence_urls": "Source evidence URLs",
}


st.set_page_config(
    page_title="Keyword Traffic",
    page_icon="chart",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_model(
    story_metadata_path: str,
    story_metrics_path: str,
    story_metadata_mtime: float,
    story_metrics_mtime: float,
    lookup_pipeline_version: str,
) -> dict[str, pd.DataFrame]:
    story_metadata = load_story_metadata(Path(story_metadata_path))
    monthly_views = load_story_metrics(Path(story_metrics_path))
    return build_simple_keyword_lookup_model(
        story_metadata=story_metadata,
        monthly_views=monthly_views,
    )


@st.cache_data(show_spinner=False)
def load_story_categories(category_path: str, category_mtime: float) -> pd.DataFrame:
    categories = pd.read_csv(category_path, dtype={"story_id": "string", "category": "string"})
    required_columns = {"Month", "story_id", "category"}
    missing_columns = required_columns.difference(categories.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{category_path} is missing required column(s): {missing}")

    categories = categories.copy()
    categories["story_id"] = categories["story_id"].str.strip()
    categories["month"] = parse_month_labels(categories["Month"])
    categories["category"] = categories["category"].fillna("Uncategorized").str.strip()
    categories["category"] = categories["category"].replace("", "Uncategorized")

    invalid_months = categories["month"].isna().sum()
    if invalid_months:
        raise ValueError(f"{category_path} has {invalid_months} rows with invalid Month values.")

    return categories[["story_id", "month", "category"]].drop_duplicates()


def parse_month_labels(month_values: pd.Series) -> pd.Series:
    parsed_months = pd.to_datetime(month_values, format="%b-%Y", errors="coerce")
    missing_months = parsed_months.isna()
    if missing_months.any():
        parsed_months.loc[missing_months] = pd.to_datetime(
            month_values.loc[missing_months],
            format="%B-%Y",
            errors="coerce",
        )
    return parsed_months.dt.to_period("M").dt.to_timestamp()


@st.cache_data(show_spinner=False)
def load_story_word_counts(word_count_path: str, word_count_mtime: float) -> pd.DataFrame:
    word_counts = pd.read_csv(word_count_path, dtype={"story_id": "string"})
    normalized_columns = {
        column: column.strip().lower().replace(" ", "_") for column in word_counts.columns
    }
    word_counts = word_counts.rename(columns=normalized_columns)
    required_columns = {"month", "story_id", "word_count"}
    missing_columns = required_columns.difference(word_counts.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{word_count_path} is missing required column(s): {missing}")

    word_counts = word_counts.copy()
    word_counts["story_id"] = word_counts["story_id"].str.strip()
    word_counts["month"] = parse_month_labels(word_counts["month"])
    word_counts["word_count"] = pd.to_numeric(word_counts["word_count"], errors="coerce")

    invalid_months = word_counts["month"].isna().sum()
    if invalid_months:
        raise ValueError(f"{word_count_path} has {invalid_months} rows with invalid Month values.")

    return word_counts[["story_id", "month", "word_count"]].drop_duplicates(
        subset=["story_id", "month"],
        keep="last",
    )


@st.cache_data(show_spinner=False)
def load_story_planned_trending(
    planned_trending_path: str,
    planned_trending_mtime: float,
) -> pd.DataFrame:
    planned_trending = pd.read_csv(planned_trending_path, dtype={"story_id": "string"})
    normalized_columns = {
        column: column.strip().lower().replace(" ", "_") for column in planned_trending.columns
    }
    planned_trending = planned_trending.rename(columns=normalized_columns)
    required_columns = {"month", "story_id", "planned_trending"}
    missing_columns = required_columns.difference(planned_trending.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{planned_trending_path} is missing required column(s): {missing}")

    planned_trending = planned_trending.copy()
    planned_trending["story_id"] = planned_trending["story_id"].str.strip()
    planned_trending["month"] = parse_month_labels(planned_trending["month"])
    planned_trending["planned_trending"] = (
        planned_trending["planned_trending"].fillna("Unknown").str.strip().str.lower()
    )
    planned_trending["planned_trending"] = planned_trending["planned_trending"].replace(
        {"": "Unknown", "na": "Unknown", "(not set)": "Unknown"}
    )
    planned_trending["planned_trending"] = planned_trending["planned_trending"].str.title()

    invalid_months = planned_trending["month"].isna().sum()
    if invalid_months:
        raise ValueError(
            f"{planned_trending_path} has {invalid_months} rows with invalid Month values."
        )

    return planned_trending[["story_id", "month", "planned_trending"]].drop_duplicates(
        subset=["story_id", "month"],
        keep="last",
    )


@st.cache_data(show_spinner=False)
def load_story_regions(region_path: str, region_mtime: float) -> pd.DataFrame:
    regions = pd.read_csv(region_path, dtype={"story_id": "string", "Region": "string"})
    normalized_columns = {
        column: column.strip().lower().replace(" ", "_") for column in regions.columns
    }
    regions = regions.rename(columns=normalized_columns)
    required_columns = {"month", "story_id", "region", "event_count"}
    missing_columns = required_columns.difference(regions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{region_path} is missing required column(s): {missing}")

    regions = regions.copy()
    regions["story_id"] = regions["story_id"].str.strip()
    regions["month"] = parse_month_labels(regions["month"])
    regions["region"] = regions["region"].fillna("Unknown").str.strip()
    regions["region"] = regions["region"].replace("", "Unknown")
    regions["event_count"] = pd.to_numeric(regions["event_count"], errors="coerce").fillna(0)

    invalid_months = regions["month"].isna().sum()
    if invalid_months:
        raise ValueError(f"{region_path} has {invalid_months} rows with invalid Month values.")

    return regions[["story_id", "month", "region", "event_count"]].drop_duplicates()


@st.cache_data(show_spinner=False)
def load_story_audience_types(audience_type_path: str, audience_type_mtime: float) -> pd.DataFrame:
    audience_types = pd.read_csv(audience_type_path, dtype={"story_id": "string"})
    normalized_columns = {
        column: column.strip().lower().replace(" ", "_") for column in audience_types.columns
    }
    audience_types = audience_types.rename(columns=normalized_columns)
    required_columns = {"month", "story_id", "audience_name", "event_count", "total_users"}
    missing_columns = required_columns.difference(audience_types.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{audience_type_path} is missing required column(s): {missing}")

    audience_types = audience_types.copy()
    audience_types["story_id"] = audience_types["story_id"].str.strip()
    audience_types["month"] = parse_month_labels(audience_types["month"])
    audience_types["audience_name"] = audience_types["audience_name"].fillna("Unknown").str.strip()
    audience_types["audience_name"] = audience_types["audience_name"].replace("", "Unknown")
    normalized_audience_names = audience_types["audience_name"].str.lower()
    for raw_label, display_label in AUDIENCE_SEGMENT_LABELS.items():
        audience_types.loc[
            normalized_audience_names.str.contains(raw_label, na=False),
            "audience_name",
        ] = display_label
    audience_types["event_count"] = pd.to_numeric(
        audience_types["event_count"],
        errors="coerce",
    ).fillna(0)
    audience_types["total_users"] = pd.to_numeric(
        audience_types["total_users"],
        errors="coerce",
    ).fillna(0)

    invalid_months = audience_types["month"].isna().sum()
    if invalid_months:
        raise ValueError(
            f"{audience_type_path} has {invalid_months} rows with invalid Month values."
        )

    return audience_types[
        ["story_id", "month", "audience_name", "event_count", "total_users"]
    ].drop_duplicates()


def format_month_column(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    if "month" in formatted.columns:
        formatted["month"] = pd.to_datetime(formatted["month"]).dt.strftime("%b %Y")
    if "first_month" in formatted.columns:
        formatted["first_month"] = pd.to_datetime(formatted["first_month"]).dt.strftime("%b %Y")
    if "last_month" in formatted.columns:
        formatted["last_month"] = pd.to_datetime(formatted["last_month"]).dt.strftime("%b %Y")
    return formatted


def format_indian_number(value: object, decimals: int | None = None) -> str:
    if pd.isna(value):
        return ""

    numeric_value = float(value)
    sign = "-" if numeric_value < 0 else ""
    absolute_value = abs(numeric_value)
    decimal_places = decimals
    if decimal_places is None:
        decimal_places = 0 if absolute_value.is_integer() else 2

    rounded_value = round(absolute_value, decimal_places)
    integer_part = str(int(rounded_value))
    decimal_part = ""
    if decimal_places > 0:
        decimal_text = f"{rounded_value:.{decimal_places}f}".split(".")[1].rstrip("0")
        decimal_part = f".{decimal_text}" if decimal_text else ""

    if len(integer_part) <= 3:
        grouped_integer = integer_part
    else:
        last_three_digits = integer_part[-3:]
        leading_digits = integer_part[:-3]
        grouped_leading_digits = []
        while len(leading_digits) > 2:
            grouped_leading_digits.insert(0, leading_digits[-2:])
            leading_digits = leading_digits[:-2]
        if leading_digits:
            grouped_leading_digits.insert(0, leading_digits)
        grouped_integer = f"{','.join(grouped_leading_digits)},{last_three_digits}"

    return f"{sign}{grouped_integer}{decimal_part}"


def format_numeric_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    excluded_columns = {
        "story_id",
        "story id",
        "page title",
        "month",
        "first month",
        "last month",
        "title word",
        "word count bucket",
        "region",
        "category",
        "planned/trending",
        "audience segment",
        "ai search term",
        "ai term category",
        "ai relationship",
        "ai confidence",
        "is high-traffic month",
    }
    for column in formatted.columns:
        if str(column).strip().lower() in excluded_columns:
            continue
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].apply(format_indian_number)
    return formatted


def format_editorial_table(df: pd.DataFrame) -> pd.DataFrame:
    return format_numeric_display_columns(
        format_month_column(df).rename(columns=EDITORIAL_COLUMN_LABELS)
    )


def format_indian_count(value: object) -> str:
    return format_indian_number(value, decimals=0) or "0"


def format_indian_plot_text(values: pd.Series) -> list[str]:
    return [format_indian_count(value) for value in values]


@st.cache_resource(show_spinner=False)
def streamlit_dataframe_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def show_editorial_dataframe(df: pd.DataFrame) -> None:
    if streamlit_dataframe_available():
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
        return

    st.markdown(
        """
        <style>
            .editorial-table-scroll {
                width: 100%;
                overflow-x: auto;
            }
            .editorial-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                overflow: hidden;
                font-size: 14px;
                line-height: 1.35;
            }
            .editorial-table th,
            .editorial-table td {
                border-right: 1px solid #e5e7eb;
                border-bottom: 1px solid #e5e7eb;
                padding: 9px 10px;
                vertical-align: top;
                white-space: nowrap;
            }
            .editorial-table th {
                background: #f8fafc;
                color: #64748b;
                font-weight: 500;
                text-align: left;
            }
            .editorial-table th:last-child,
            .editorial-table td:last-child {
                border-right: 0;
            }
            .editorial-table tr:last-child td {
                border-bottom: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="editorial-table-scroll">{df.to_html(index=False, escape=True, classes="editorial-table")}</div>',
        unsafe_allow_html=True,
    )


def add_high_traffic_month_display(
    title_rows: pd.DataFrame,
    monthly_rows: pd.DataFrame,
) -> pd.DataFrame:
    displayed = title_rows.copy()
    required_title_columns = {"story_id", "threshold_matched_months"}
    required_monthly_columns = {"story_id", "month", "above_monthly_threshold"}
    if (
        displayed.empty
        or not required_title_columns.issubset(displayed.columns)
        or not required_monthly_columns.issubset(monthly_rows.columns)
    ):
        return displayed

    high_traffic_months = monthly_rows.loc[monthly_rows["above_monthly_threshold"]].copy()
    if high_traffic_months.empty:
        return displayed

    high_traffic_months["month_label"] = pd.to_datetime(high_traffic_months["month"]).dt.strftime("%b %Y")
    month_labels = (
        high_traffic_months.sort_values(["story_id", "month"])
        .groupby("story_id")["month_label"]
        .apply(lambda labels: ", ".join(labels))
    )

    displayed["threshold_matched_months"] = displayed.apply(
        lambda row: (
            f"{int(row['threshold_matched_months'])} ({month_labels.get(row['story_id'])})"
            if month_labels.get(row["story_id"])
            else row["threshold_matched_months"]
        ),
        axis=1,
    )
    return displayed


def show_wrapped_title_table(
    df: pd.DataFrame,
    extra_wrapped_columns: dict[str, int] | None = None,
    max_rows: int | None = None,
    scroll_height: int = 420,
    scroll_after_rows: int | None = None,
) -> None:
    displayed_df = df.copy()
    row_count = len(displayed_df)
    if max_rows is not None and row_count > max_rows:
        sort_column = next(
            (column for column in ("Total views", "Views") if column in displayed_df.columns),
            None,
        )
        if sort_column:
            displayed_df = displayed_df.sort_values(sort_column, ascending=False)
        else:
            st.caption(f"Showing all {format_indian_count(row_count)} rows in a scrollable table.")

    title_column_number = list(df.columns).index("Page title") + 1 if "Page title" in df.columns else 2
    title_column_class = f"wrapped-title-table-title-{title_column_number}"
    table_instance_token = uuid.uuid4().hex
    table_instance_class = f"wrapped-title-table-{table_instance_token}"
    scroll_container_class = f"wrapped-title-table-scroll-{table_instance_token}"
    wrapped_columns = {title_column_number: 420}
    if extra_wrapped_columns:
        for column_name, min_width in extra_wrapped_columns.items():
            if column_name in df.columns:
                wrapped_columns[list(df.columns).index(column_name) + 1] = min_width

    wrapped_column_css = "\n".join(
        f"""
            .{table_instance_class}.{title_column_class} th:nth-child({column_number}),
            .{table_instance_class}.{title_column_class} td:nth-child({column_number}) {{
                min-width: {min_width}px;
                max-width: {min_width + 120}px;
                white-space: normal;
                overflow-wrap: anywhere;
                word-break: normal;
            }}
        """
        for column_number, min_width in wrapped_columns.items()
    )
    enable_vertical_scroll = (
        scroll_after_rows is None or row_count > scroll_after_rows
    )
    max_height_css = f"{scroll_height}px" if enable_vertical_scroll else "none"
    overflow_y_css = "auto" if enable_vertical_scroll else "visible"

    table_style = """
        <style>
            .__TABLE_INSTANCE_CLASS__ {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                overflow: hidden;
                font-size: 14px;
                line-height: 1.35;
            }
            .__SCROLL_CONTAINER_CLASS__ {
                max-height: __MAX_HEIGHT__;
                overflow-x: auto;
                overflow-y: __OVERFLOW_Y__;
                width: 100%;
            }
            .__TABLE_INSTANCE_CLASS__ th,
            .__TABLE_INSTANCE_CLASS__ td {
                border-right: 1px solid #e5e7eb;
                border-bottom: 1px solid #e5e7eb;
                padding: 9px 10px;
                vertical-align: top;
            }
            .__TABLE_INSTANCE_CLASS__ th {
                background: #f8fafc;
                color: #64748b;
                font-weight: 500;
                text-align: left;
                white-space: nowrap;
                position: sticky;
                top: 0;
                z-index: 1;
            }
            .__TABLE_INSTANCE_CLASS__.__TITLE_COLUMN_CLASS__ td {
                white-space: nowrap;
            }
            __WRAPPED_COLUMN_CSS__
            .__TABLE_INSTANCE_CLASS__ th:last-child,
            .__TABLE_INSTANCE_CLASS__ td:last-child {
                border-right: 0;
            }
            .__TABLE_INSTANCE_CLASS__ tr:last-child td {
                border-bottom: 0;
            }
        </style>
        """
    table_style = table_style.replace(
        "__TABLE_INSTANCE_CLASS__",
        table_instance_class,
    ).replace(
        "__SCROLL_CONTAINER_CLASS__",
        scroll_container_class,
    ).replace(
        "__TITLE_COLUMN_CLASS__",
        title_column_class,
    ).replace(
        "__WRAPPED_COLUMN_CSS__",
        wrapped_column_css,
    ).replace(
        "__MAX_HEIGHT__",
        max_height_css,
    ).replace(
        "__OVERFLOW_Y__",
        overflow_y_css,
    )
    st.markdown(
        table_style,
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            f'<div class="{scroll_container_class}">'
            f'{displayed_df.to_html(index=False, escape=True, classes=f"{table_instance_class} {title_column_class}")}'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def show_monthly_views_heatmap(monthly_rows: pd.DataFrame) -> None:
    required_columns = {"story_id", "page_title", "month", "views"}
    if monthly_rows.empty or not required_columns.issubset(monthly_rows.columns):
        return

    heatmap_rows = monthly_rows[list(required_columns)].copy()
    heatmap_rows["month"] = pd.to_datetime(heatmap_rows["month"], errors="coerce")
    heatmap_rows["views"] = pd.to_numeric(heatmap_rows["views"], errors="coerce").fillna(0)
    heatmap_rows = heatmap_rows.dropna(subset=["month"])
    if heatmap_rows.empty:
        return

    heatmap_rows["month_label"] = heatmap_rows["month"].dt.strftime("%b %Y")
    month_order = (
        heatmap_rows[["month", "month_label"]]
        .drop_duplicates()
        .sort_values("month")["month_label"]
        .tolist()
    )
    pivot = heatmap_rows.pivot_table(
        index=["story_id", "page_title"],
        columns="month_label",
        values="views",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=month_order, fill_value=0)

    row_summary = pivot.assign(
        active_months=(pivot > 0).sum(axis=1),
        total_views=pivot.sum(axis=1),
    ).sort_values(["active_months", "total_views"], ascending=[False, False])
    view_matrix = row_summary.drop(columns=["active_months", "total_views"])

    max_views = float(view_matrix.to_numpy().max()) if not view_matrix.empty else 0

    st.markdown("#### Monthly views heatmap")

    def cell_background(value: float) -> str:
        if value <= 0 or max_views <= 0:
            return "#ffffff"
        intensity = min(value / max_views, 1)
        if intensity >= 0.75:
            return "#8fd19e"
        if intensity >= 0.45:
            return "#b7dfbf"
        if intensity >= 0.2:
            return "#d9f0dc"
        return "#edf7ee"

    header_cells = "".join(
        f"<th class='monthly-views-heatmap-month'>{html.escape(month)}</th>"
        for month in view_matrix.columns
    )
    body_rows = []
    for story_id, page_title in view_matrix.index:
        month_cells = []
        for month in view_matrix.columns:
            value = float(view_matrix.loc[(story_id, page_title), month])
            display_value = format_indian_count(value) if value > 0 else ""
            month_cells.append(
                (
                    "<td class='monthly-views-heatmap-value' "
                    f"style='background-color: {cell_background(value)};'>"
                    f"{display_value}</td>"
                )
            )
        body_rows.append(
            (
                "<tr>"
                f"<td class='monthly-views-heatmap-id'>{html.escape(str(story_id))}</td>"
                f"<td class='monthly-views-heatmap-title'>{html.escape(str(page_title))}</td>"
                f"{''.join(month_cells)}"
                "</tr>"
            )
        )

    st.markdown(
        f"""
        <style>
            .monthly-views-heatmap-scroll {{
                width: 100%;
                max-height: 460px;
                overflow: auto;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }}
            .monthly-views-heatmap {{
                width: 100%;
                min-width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                color: #374151;
                font-size: 14px;
                line-height: 1.35;
            }}
            .monthly-views-heatmap th,
            .monthly-views-heatmap td {{
                border-right: 1px solid #e5e7eb;
                border-bottom: 1px solid #e5e7eb;
                padding: 9px 10px;
                text-align: left;
                white-space: nowrap;
                vertical-align: top;
            }}
            .monthly-views-heatmap th {{
                position: sticky;
                top: 0;
                z-index: 2;
                background: #f8fafc;
                color: #64748b;
                font-weight: 500;
            }}
            .monthly-views-heatmap th:last-child,
            .monthly-views-heatmap td:last-child {{
                border-right: 0;
            }}
            .monthly-views-heatmap tr:last-child td {{
                border-bottom: 0;
            }}
            .monthly-views-heatmap-id {{
                min-width: 105px;
            }}
            .monthly-views-heatmap-title {{
                min-width: 420px;
                max-width: 540px;
                white-space: normal;
                overflow-wrap: anywhere;
                word-break: normal;
            }}
            .monthly-views-heatmap-month,
            .monthly-views-heatmap-value {{
                min-width: 105px;
            }}
            .monthly-views-heatmap-value {{
                text-align: right;
            }}
        </style>
        <div class="monthly-views-heatmap-scroll">
            <table class="monthly-views-heatmap">
                <thead>
                    <tr>
                        <th class="monthly-views-heatmap-id">Story ID</th>
                        <th class="monthly-views-heatmap-title">Page title</th>
                        {header_cells}
                    </tr>
                </thead>
                <tbody>
                    {''.join(body_rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_file_mtime(path: Path) -> float:
    return path.stat().st_mtime


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]]


def build_category_analytics(
    matched_story_months: pd.DataFrame,
    story_categories: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["category", "article_count", "total_views", "avg_page_views"]
    if matched_story_months.empty:
        return pd.DataFrame(columns=columns)

    category_rows = matched_story_months[["story_id", "month", "views"]].merge(
        story_categories,
        on=["story_id", "month"],
        how="left",
    )
    category_rows["category"] = category_rows["category"].fillna("Uncategorized")

    category_summary = (
        category_rows.groupby("category", as_index=False)
        .agg(
            article_count=("story_id", "nunique"),
            total_views=("views", "sum"),
        )
        .sort_values(["total_views", "article_count", "category"], ascending=[False, False, True])
    )
    category_summary["total_views"] = category_summary["total_views"].round(0).astype(int)
    category_summary["avg_page_views"] = (
        category_summary["total_views"] / category_summary["article_count"]
    ).round(0).astype(int)
    return category_summary[columns]


def assign_word_count_bucket(word_count: float) -> str:
    if pd.isna(word_count):
        return "Unknown"
    for lower_bound, upper_bound, label in WORD_COUNT_BUCKETS:
        if word_count >= lower_bound and (upper_bound is None or word_count <= upper_bound):
            return label
    return "Unknown"


def get_word_count_bucket_order(bucket: str) -> int:
    bucket_order = {label: index for index, (_, _, label) in enumerate(WORD_COUNT_BUCKETS)}
    return bucket_order.get(bucket, len(bucket_order))


def build_word_count_analytics(
    matched_story_months: pd.DataFrame,
    story_word_counts: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["word_count_bucket", "article_count", "total_views", "avg_page_views"]
    if matched_story_months.empty:
        return pd.DataFrame(columns=columns)

    word_count_rows = matched_story_months[["story_id", "month", "views"]].merge(
        story_word_counts,
        on=["story_id", "month"],
        how="left",
    )
    word_count_rows["word_count_bucket"] = word_count_rows["word_count"].apply(assign_word_count_bucket)

    word_count_summary = (
        word_count_rows.groupby("word_count_bucket", as_index=False)
        .agg(
            article_count=("story_id", "nunique"),
            total_views=("views", "sum"),
        )
    )
    all_buckets = pd.DataFrame(
        {"word_count_bucket": [label for _, _, label in WORD_COUNT_BUCKETS] + ["Unknown"]}
    )
    word_count_summary = all_buckets.merge(
        word_count_summary,
        on="word_count_bucket",
        how="left",
    )
    word_count_summary[["article_count", "total_views"]] = word_count_summary[
        ["article_count", "total_views"]
    ].fillna(0)
    word_count_summary["bucket_order"] = word_count_summary["word_count_bucket"].apply(
        get_word_count_bucket_order
    )
    word_count_summary = word_count_summary.sort_values(
        ["bucket_order", "total_views", "article_count"],
        ascending=[True, False, False],
    )
    word_count_summary["total_views"] = word_count_summary["total_views"].round(0).astype(int)
    word_count_summary["avg_page_views"] = 0
    has_articles = word_count_summary["article_count"] > 0
    word_count_summary.loc[has_articles, "avg_page_views"] = (
        word_count_summary.loc[has_articles, "total_views"]
        / word_count_summary.loc[has_articles, "article_count"]
    ).round(0).astype(int)
    word_count_summary["article_count"] = word_count_summary["article_count"].astype(int)
    return word_count_summary[columns]


def build_planned_trending_analytics(
    matched_story_months: pd.DataFrame,
    story_planned_trending: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["planned_trending", "article_count", "total_views", "avg_page_views"]
    if matched_story_months.empty:
        return pd.DataFrame(columns=columns)

    planned_trending_rows = matched_story_months[["story_id", "month", "views"]].merge(
        story_planned_trending,
        on=["story_id", "month"],
        how="left",
    )
    planned_trending_rows["planned_trending"] = planned_trending_rows[
        "planned_trending"
    ].fillna("Unknown")

    planned_trending_summary = (
        planned_trending_rows.groupby("planned_trending", as_index=False)
        .agg(
            article_count=("story_id", "nunique"),
            total_views=("views", "sum"),
        )
        .sort_values(
            ["total_views", "article_count", "planned_trending"],
            ascending=[False, False, True],
        )
    )
    planned_trending_summary["total_views"] = (
        planned_trending_summary["total_views"].round(0).astype(int)
    )
    planned_trending_summary["article_count"] = (
        planned_trending_summary["article_count"].fillna(0).astype(int)
    )
    planned_trending_summary["avg_page_views"] = 0
    has_articles = planned_trending_summary["article_count"] > 0
    planned_trending_summary.loc[has_articles, "avg_page_views"] = (
        planned_trending_summary.loc[has_articles, "total_views"]
        / planned_trending_summary.loc[has_articles, "article_count"]
    ).round(0).astype(int)
    return planned_trending_summary[columns]


def build_region_analytics(
    matched_story_months: pd.DataFrame,
    story_regions: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["region", "total_articles", "page_views", "views_per_article"]
    if matched_story_months.empty:
        return pd.DataFrame(columns=columns)

    region_rows = matched_story_months[["story_id", "month"]].merge(
        story_regions,
        on=["story_id", "month"],
        how="left",
    )
    region_rows["region"] = region_rows["region"].fillna("Unknown")
    region_rows["event_count"] = region_rows["event_count"].fillna(0)

    region_summary = (
        region_rows.groupby("region", as_index=False)
        .agg(
            total_articles=("story_id", "nunique"),
            page_views=("event_count", "sum"),
        )
        .sort_values(["page_views", "total_articles", "region"], ascending=[False, False, True])
        .head(15)
    )
    region_summary["page_views"] = region_summary["page_views"].round(0).astype(int)
    region_summary["views_per_article"] = (
        region_summary["page_views"] / region_summary["total_articles"]
    ).round(0).astype(int)
    return region_summary[columns]


def build_audience_segment_analytics(
    matched_story_months: pd.DataFrame,
    story_audience_types: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "audience_name",
        "article_count",
        "audience_views",
        "total_users",
        "users_per_article",
        "views_per_article",
        "views_per_user",
    ]
    if matched_story_months.empty:
        return pd.DataFrame(columns=columns)

    audience_rows = matched_story_months[["story_id", "month"]].merge(
        story_audience_types,
        on=["story_id", "month"],
        how="left",
    )
    audience_rows["audience_name"] = audience_rows["audience_name"].fillna("Unknown")
    audience_rows["event_count"] = audience_rows["event_count"].fillna(0)
    audience_rows["total_users"] = audience_rows["total_users"].fillna(0)

    audience_summary = (
        audience_rows.groupby("audience_name", as_index=False)
        .agg(
            article_count=("story_id", "nunique"),
            audience_views=("event_count", "sum"),
            total_users=("total_users", "sum"),
        )
        .sort_values(
            ["audience_views", "total_users", "article_count", "audience_name"],
            ascending=[False, False, False, True],
        )
    )
    audience_summary["audience_views"] = audience_summary["audience_views"].round(0).astype(int)
    audience_summary["total_users"] = audience_summary["total_users"].round(0).astype(int)
    audience_summary["users_per_article"] = (
        audience_summary["total_users"] / audience_summary["article_count"]
    ).round(0).astype(int)
    audience_summary["views_per_article"] = (
        audience_summary["audience_views"] / audience_summary["article_count"]
    ).round(0).astype(int)
    audience_summary["views_per_user"] = 0.0
    has_users = audience_summary["total_users"] > 0
    audience_summary.loc[has_users, "views_per_user"] = (
        audience_summary.loc[has_users, "audience_views"]
        / audience_summary.loc[has_users, "total_users"]
    ).round(2)
    return audience_summary[columns]


def build_audience_story_report(
    matched_story_months: pd.DataFrame,
    story_audience_types: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "month",
        "story_id",
        "page_title",
        "audience_name",
        "audience_views",
        "total_users",
        "views_per_user",
    ]
    if matched_story_months.empty:
        return pd.DataFrame(columns=columns)

    story_columns = ["story_id", "month", "page_title"]
    story_segment_rows = matched_story_months[story_columns].merge(
        story_audience_types,
        on=["story_id", "month"],
        how="left",
    )
    story_segment_rows["audience_name"] = story_segment_rows["audience_name"].fillna("Unknown")
    story_segment_rows["event_count"] = story_segment_rows["event_count"].fillna(0)
    story_segment_rows["total_users"] = story_segment_rows["total_users"].fillna(0)

    story_segment_summary = (
        story_segment_rows.groupby(["month", "story_id", "page_title", "audience_name"], as_index=False)
        .agg(
            audience_views=("event_count", "sum"),
            total_users=("total_users", "sum"),
        )
        .sort_values(["audience_views", "total_users"], ascending=[False, False])
    )
    story_segment_summary["audience_views"] = (
        story_segment_summary["audience_views"].round(0).astype(int)
    )
    story_segment_summary["total_users"] = story_segment_summary["total_users"].round(0).astype(int)
    story_segment_summary["views_per_user"] = 0.0
    has_users = story_segment_summary["total_users"] > 0
    story_segment_summary.loc[has_users, "views_per_user"] = (
        story_segment_summary.loc[has_users, "audience_views"]
        / story_segment_summary.loc[has_users, "total_users"]
    ).round(2)
    return story_segment_summary[columns]


def add_audience_segment_view_columns(
    title_rows: pd.DataFrame,
    matched_story_months: pd.DataFrame,
    story_audience_types: pd.DataFrame,
) -> pd.DataFrame:
    displayed = title_rows.copy()
    segment_columns = {
        "Brand Lovers": "brand_lovers_views",
        "Casual Readers": "casual_readers_views",
        "Loyal Users": "loyal_users_views",
    }
    for output_column in segment_columns.values():
        displayed[output_column] = 0

    if displayed.empty or matched_story_months.empty or story_audience_types.empty:
        return displayed

    audience_rows = matched_story_months[["story_id", "month"]].merge(
        story_audience_types,
        on=["story_id", "month"],
        how="left",
    )
    if audience_rows.empty:
        return displayed

    audience_rows["event_count"] = audience_rows["event_count"].fillna(0)
    audience_pivot = (
        audience_rows.pivot_table(
            index="story_id",
            columns="audience_name",
            values="event_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns=segment_columns)
        .reset_index()
    )
    existing_segment_columns = [
        column for column in segment_columns.values() if column in audience_pivot.columns
    ]
    if not existing_segment_columns:
        return displayed

    displayed = displayed.drop(columns=list(segment_columns.values()), errors="ignore").merge(
        audience_pivot[["story_id", *existing_segment_columns]],
        on="story_id",
        how="left",
    )
    for output_column in segment_columns.values():
        if output_column not in displayed.columns:
            displayed[output_column] = 0
        displayed[output_column] = displayed[output_column].fillna(0).round(0).astype(int)
    return displayed


def format_compact_number(value: float) -> str:
    if pd.isna(value):
        return "0"
    return format_indian_count(value)


def apply_executive_chart_style(fig, height: int = 380):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=24),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", font_size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)", zeroline=False)
    return fig


def show_dashboard_empty_state() -> None:
    st.info("Enter a keyword or group of keywords to unlock the executive dashboard.")


def render_kpi_card_group(cards: list[dict[str, str]]) -> None:
    card_markup = "\n".join(
        (
            '<div class="executive-kpi-card">'
            f'<div class="executive-kpi-label">{html.escape(card["label"])}</div>'
            f'<div class="executive-kpi-value">{html.escape(card["value"])}</div>'
            "</div>"
        )
        for card in cards
    )
    st.markdown(
        (
            "<style>"
            ".executive-kpi-group{margin:0.7rem 0 1rem;}"
            ".executive-kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.9rem;}"
            ".executive-kpi-card{min-height:112px;padding:1rem 0.85rem;border:1px solid #e5e7eb;"
            "border-left:5px solid #14b8a6;border-radius:8px;background:linear-gradient(135deg,#ffffff 0%,#f8fafc 72%,#ecfeff 100%);"
            "box-shadow:0 10px 24px rgba(15,23,42,0.07);display:flex;flex-direction:column;"
            "align-items:center;justify-content:center;text-align:center;}"
            ".executive-kpi-label{color:#64748b;font-size:0.82rem;font-weight:600;line-height:1.25;}"
            ".executive-kpi-value{margin-top:0.5rem;color:#111827;font-size:clamp(1.45rem,2.4vw,2.05rem);"
            "font-weight:600;line-height:1.08;overflow-wrap:anywhere;}"
            "@media (max-width:980px){.executive-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}"
            "@media (max-width:560px){.executive-kpi-grid{grid-template-columns:1fr;}}"
            "</style>"
            '<div class="executive-kpi-group">'
            f'<div class="executive-kpi-grid">{card_markup}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def build_grounded_profile_key(
    keyword_query: str,
    match_type: str,
    matched_titles: list[str],
    query_central_policy: bool = False,
) -> tuple[str, str]:
    normalized_query = " ".join(keyword_query.casefold().split())
    normalized_titles = sorted(
        " ".join(str(title).casefold().split())
        for title in matched_titles
        if str(title).strip()
    )
    title_fingerprint = hashlib.sha256(
        "\n".join(normalized_titles).encode("utf-8")
    ).hexdigest()
    model_profile_suffix = ""
    if not related_ai_uses_legacy_behavior():
        model_profile_suffix = (
            f"|mode={get_related_ai_mode()}"
            f"|research_model={get_vertex_model_name('research')}"
            f"|route_model={get_vertex_model_name('route')}"
            f"|prompt_contract={get_prompt_contract_fingerprint()}"
        )
    comparison_policy_suffix = "|query_central_policy=1" if query_central_policy else ""
    profile_key = hashlib.sha256(
        (
            f"{GROUNDED_PROFILE_VERSION}|{normalized_query}|"
            f"{match_type}|{title_fingerprint}{model_profile_suffix}"
            f"{comparison_policy_suffix}"
        ).encode("utf-8")
    ).hexdigest()
    return profile_key, title_fingerprint


def saved_grounded_profile_needs_recovery(
    keyword_query: str,
    grounded_research: dict[str, object],
) -> bool:
    """Refresh malformed profiles created before relationship recovery existed."""
    warnings = " ".join(
        str(item) for item in grounded_research.get("quality_warnings", [])
    ).casefold()
    if any(
        marker in warnings
        for marker in (
            "no executed google search",
            "no grounding source",
            "not valid structured json",
            "quality gate did not pass",
        )
    ):
        return True

    recovered_routes = extract_relationship_routes(
        keyword_query=keyword_query,
        grounded_research=grounded_research,
    )
    if recovered_routes:
        return False

    return True


def init_grounded_profile_db() -> None:
    with sqlite3.connect(QUERY_PROFILE_DB_PATH, timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS web_grounded_query_profiles (
                profile_key TEXT PRIMARY KEY,
                normalized_query TEXT NOT NULL,
                match_type TEXT NOT NULL,
                title_fingerprint TEXT NOT NULL,
                profile_version TEXT NOT NULL,
                research_json TEXT NOT NULL,
                topics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS web_grounded_candidate_evaluations (
                profile_key TEXT NOT NULL,
                story_id TEXT NOT NULL,
                judge_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                decision TEXT NOT NULL,
                ai_relationship TEXT NOT NULL,
                ai_audit_reason TEXT NOT NULL DEFAULT '',
                ai_confidence REAL NOT NULL,
                ai_relevance_level INTEGER NOT NULL,
                ai_relevance_type TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                PRIMARY KEY(profile_key, story_id, judge_version, model_name)
            )
            """
        )
        candidate_evaluation_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(web_grounded_candidate_evaluations)"
            ).fetchall()
        }
        if "title_fingerprint" not in candidate_evaluation_columns:
            connection.execute(
                "ALTER TABLE web_grounded_candidate_evaluations "
                "ADD COLUMN title_fingerprint TEXT NOT NULL DEFAULT ''"
            )
        if "ai_audit_reason" not in candidate_evaluation_columns:
            connection.execute(
                "ALTER TABLE web_grounded_candidate_evaluations "
                "ADD COLUMN ai_audit_reason TEXT NOT NULL DEFAULT ''"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS web_grounded_generative_retrievals (
                profile_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                retrieval_version TEXT NOT NULL,
                corpus_fingerprint TEXT NOT NULL,
                selected_json TEXT NOT NULL,
                screened_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    profile_key,
                    model_name,
                    retrieval_version,
                    corpus_fingerprint
                )
            )
            """
        )
        connection.commit()


def load_saved_grounded_profile(
    profile_key: str,
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    init_grounded_profile_db()
    with sqlite3.connect(QUERY_PROFILE_DB_PATH, timeout=30) as connection:
        row = connection.execute(
            """
            SELECT research_json, topics_json, updated_at
            FROM web_grounded_query_profiles
            WHERE profile_key = ? AND profile_version = ?
            """,
            (profile_key, GROUNDED_PROFILE_VERSION),
        ).fetchone()
    if row is None:
        return None
    if not grounded_profile_timestamp_is_fresh(row[2]):
        return None
    try:
        research = json.loads(row[0])
        topics = json.loads(row[1])
    except json.JSONDecodeError:
        return None
    if not isinstance(research, dict) or not isinstance(topics, list):
        return None
    return research, topics


def grounded_profile_timestamp_is_fresh(
    value: object,
    *,
    now: datetime | None = None,
) -> bool:
    try:
        updated_at = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    reference_time = now or datetime.now(timezone.utc)
    age_hours = (reference_time - updated_at).total_seconds() / 3600
    return 0 <= age_hours <= GROUNDED_PROFILE_MAX_AGE_HOURS


def save_grounded_profile(
    profile_key: str,
    keyword_query: str,
    match_type: str,
    title_fingerprint: str,
    grounded_research: dict[str, object],
    related_topics: list[dict[str, object]],
) -> None:
    init_grounded_profile_db()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(QUERY_PROFILE_DB_PATH, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO web_grounded_query_profiles (
                profile_key,
                normalized_query,
                match_type,
                title_fingerprint,
                profile_version,
                research_json,
                topics_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_key) DO UPDATE SET
                research_json = excluded.research_json,
                topics_json = excluded.topics_json,
                updated_at = excluded.updated_at
            """,
            (
                profile_key,
                " ".join(keyword_query.casefold().split()),
                match_type,
                title_fingerprint,
                GROUNDED_PROFILE_VERSION,
                json.dumps(grounded_research, ensure_ascii=False),
                json.dumps(related_topics, ensure_ascii=False),
                now,
                now,
            ),
        )
        connection.commit()


@st.cache_data(show_spinner=False)
def refresh_fast_relationship_profile(
    keyword_query: str,
    match_type: str,
    matched_title_context: tuple[str, ...],
    refresh_request_id: str,
) -> dict[str, object]:
    """Refresh only the reusable relationship graph, without judging titles."""
    del refresh_request_id
    profile_key, title_fingerprint = build_grounded_profile_key(
        keyword_query=keyword_query,
        match_type=match_type,
        matched_titles=list(matched_title_context),
        query_central_policy=False,
    )
    grounded_research = research_query_relationships_with_vertex(
        keyword_query=keyword_query,
        matched_titles=list(matched_title_context),
        service_account_path=SERVICE_ACCOUNT_PATH,
        match_type=match_type,
        location=get_vertex_location(),
        model_name=get_vertex_model_name("research"),
        query_central_policy=False,
    )
    save_grounded_profile(
        profile_key=profile_key,
        keyword_query=keyword_query,
        match_type=match_type,
        title_fingerprint=title_fingerprint,
        grounded_research=grounded_research,
        related_topics=[],
    )
    return grounded_research


def load_cached_generative_retrieval(
    profile_key: str,
    model_name: str,
    corpus_fingerprint: str,
) -> tuple[list[dict[str, object]], int] | None:
    init_grounded_profile_db()
    with sqlite3.connect(QUERY_PROFILE_DB_PATH, timeout=30) as connection:
        row = connection.execute(
            """
            SELECT selected_json, screened_count
            FROM web_grounded_generative_retrievals
            WHERE profile_key = ?
              AND model_name = ?
              AND retrieval_version = ?
              AND corpus_fingerprint = ?
            """,
            (
                profile_key,
                model_name,
                GENERATIVE_RETRIEVAL_VERSION,
                corpus_fingerprint,
            ),
        ).fetchone()
    if row is None:
        return None
    try:
        selected = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(selected, list):
        return None
    return selected, int(row[1])


def save_generative_retrieval(
    profile_key: str,
    model_name: str,
    corpus_fingerprint: str,
    selected_titles: list[dict[str, object]],
    screened_count: int,
) -> None:
    init_grounded_profile_db()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(QUERY_PROFILE_DB_PATH, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO web_grounded_generative_retrievals (
                profile_key,
                model_name,
                retrieval_version,
                corpus_fingerprint,
                selected_json,
                screened_count,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                profile_key,
                model_name,
                retrieval_version,
                corpus_fingerprint
            ) DO UPDATE SET
                selected_json = excluded.selected_json,
                screened_count = excluded.screened_count,
                updated_at = excluded.updated_at
            """,
            (
                profile_key,
                model_name,
                GENERATIVE_RETRIEVAL_VERSION,
                corpus_fingerprint,
                json.dumps(selected_titles, ensure_ascii=False),
                int(screened_count),
                now,
                now,
            ),
        )
        connection.commit()


def load_cached_candidate_evaluations(
    profile_key: str,
    story_ids: list[str],
    model_name: str,
    title_fingerprints: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    if not story_ids:
        return {}
    init_grounded_profile_db()
    placeholders = ",".join("?" for _ in story_ids)
    parameters = [
        profile_key,
        CANDIDATE_JUDGE_VERSION,
        model_name,
        *story_ids,
    ]
    with sqlite3.connect(QUERY_PROFILE_DB_PATH, timeout=30) as connection:
        rows = connection.execute(
            f"""
            SELECT
                story_id,
                decision,
                ai_relationship,
                ai_audit_reason,
                ai_confidence,
                ai_relevance_level,
                ai_relevance_type,
                title_fingerprint
            FROM web_grounded_candidate_evaluations
            WHERE profile_key = ?
              AND judge_version = ?
              AND model_name = ?
              AND story_id IN ({placeholders})
            """,
            parameters,
        ).fetchall()
    evaluations = {
        str(row[0]): {
            "story_id": str(row[0]),
            "decision": str(row[1]),
            "ai_relationship": str(row[2]),
            "ai_audit_reason": str(row[3]),
            "ai_confidence": float(row[4]),
            "ai_relevance_level": int(row[5]),
            "ai_relevance_type": str(row[6]),
        }
        for row in rows
        if title_fingerprints is None
        or str(row[7]) == title_fingerprints.get(str(row[0]), "")
    }
    return evaluations


def save_candidate_evaluations(
    profile_key: str,
    evaluations: list[dict[str, object]],
    model_name: str,
) -> None:
    if not evaluations:
        return
    init_grounded_profile_db()
    evaluated_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(QUERY_PROFILE_DB_PATH, timeout=30) as connection:
        connection.executemany(
            """
            INSERT INTO web_grounded_candidate_evaluations (
                profile_key,
                story_id,
                judge_version,
                model_name,
                decision,
                ai_relationship,
                ai_audit_reason,
                ai_confidence,
                ai_relevance_level,
                ai_relevance_type,
                title_fingerprint,
                evaluated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_key, story_id, judge_version, model_name)
            DO UPDATE SET
                decision = excluded.decision,
                ai_relationship = excluded.ai_relationship,
                ai_audit_reason = excluded.ai_audit_reason,
                ai_confidence = excluded.ai_confidence,
                ai_relevance_level = excluded.ai_relevance_level,
                ai_relevance_type = excluded.ai_relevance_type,
                title_fingerprint = excluded.title_fingerprint,
                evaluated_at = excluded.evaluated_at
            """,
            [
                (
                    profile_key,
                    str(evaluation["story_id"]),
                    CANDIDATE_JUDGE_VERSION,
                    model_name,
                    str(evaluation["decision"]),
                    str(evaluation.get("ai_relationship", "")),
                    str(evaluation.get("ai_audit_reason", "")),
                    float(evaluation.get("ai_confidence", 0.0)),
                    int(evaluation.get("ai_relevance_level", 0)),
                    str(evaluation.get("ai_relevance_type", "")),
                    str(evaluation.get("title_fingerprint", "")),
                    evaluated_at,
                )
                for evaluation in evaluations
            ],
        )
        connection.commit()


def build_related_titles_from_evaluations(
    candidate_pool: pd.DataFrame,
    evaluations_by_story_id: dict[str, dict[str, object]],
    research_source_count: int,
) -> pd.DataFrame:
    columns = [
        "story_id",
        "page_title",
        "ai_term",
        "ai_category",
        "ai_relationship",
        "ai_audit_reason",
        "ai_confidence",
        "ai_relevance_level",
        "ai_relevance_type",
        "ai_supporting_routes",
        "ai_research_source_count",
        "would_pass_evidence_gate",
        "total_views",
        "active_months",
        "first_month",
        "last_month",
    ]
    rows = []
    for candidate in candidate_pool.to_dict("records"):
        story_id = str(candidate["story_id"])
        evaluation = evaluations_by_story_id.get(story_id)
        if not evaluation or evaluation.get("decision") != "include":
            continue
        evidence = sorted(
            candidate.get("retrieval_evidence", []),
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        primary_evidence = evidence[0] if evidence else {}
        row = candidate.copy()
        row.pop("retrieval_evidence", None)
        row["ai_term"] = str(primary_evidence.get("term", ""))
        row["ai_category"] = str(primary_evidence.get("category", ""))
        row["ai_relationship"] = str(evaluation.get("ai_relationship", ""))
        row["ai_audit_reason"] = str(evaluation.get("ai_audit_reason", ""))
        row["ai_confidence"] = float(evaluation.get("ai_confidence", 0.0))
        row["ai_relevance_level"] = int(evaluation.get("ai_relevance_level", 0))
        row["ai_relevance_type"] = str(evaluation.get("ai_relevance_type", ""))
        row["ai_supporting_routes"] = len(evidence)
        row["ai_research_source_count"] = research_source_count
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["ai_relevance_level", "ai_confidence", "total_views"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def combine_all_related_results(
    ai_related_titles: pd.DataFrame,
    bertopic_candidates: list[dict[str, object]],
    *,
    ai_weight: float = ALL_RELATED_AI_WEIGHT,
    bertopic_weight: float = ALL_RELATED_BERTOPIC_WEIGHT,
    rrf_k: int = ALL_RELATED_RRF_K,
) -> pd.DataFrame:
    """Union final AI and BERTopic outputs using weighted reciprocal-rank fusion."""
    output_columns = [
        "story_id",
        "page_title",
        "relationship_or_cluster",
        "total_views",
        "unified_rank_score",
        "ai_rank",
        "bertopic_rank",
        "retrieval_sources",
        "rrf_score",
    ]
    rows_by_story_id: dict[str, dict[str, object]] = {}

    if not ai_related_titles.empty:
        ranked_ai = ai_related_titles.sort_values(
            ["ai_relevance_level", "ai_confidence", "total_views"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        for rank, row in enumerate(ranked_ai.to_dict("records"), start=1):
            story_id = str(row.get("story_id", "")).strip()
            if not story_id:
                continue
            rows_by_story_id[story_id] = {
                "story_id": story_id,
                "page_title": str(row.get("page_title", "")),
                "relationship_or_cluster": str(row.get("ai_relationship", "")),
                "total_views": float(row.get("total_views", 0) or 0),
                "ai_rank": rank,
                "bertopic_rank": None,
                "retrieval_sources": ["AI-related"],
            }

    for rank, candidate in enumerate(bertopic_candidates, start=1):
        story_id = str(candidate.get("story_id", "")).strip()
        if not story_id:
            continue
        evidence = candidate.get("retrieval_evidence", [])
        primary_evidence = evidence[0] if isinstance(evidence, list) and evidence else {}
        topic_label = (
            str(primary_evidence.get("related_subject", ""))
            if isinstance(primary_evidence, dict)
            else ""
        )
        existing = rows_by_story_id.get(story_id)
        if existing is None:
            rows_by_story_id[story_id] = {
                "story_id": story_id,
                "page_title": str(candidate.get("page_title", "")),
                "relationship_or_cluster": topic_label,
                "total_views": float(candidate.get("total_views", 0) or 0),
                "ai_rank": None,
                "bertopic_rank": rank,
                "retrieval_sources": ["BERTopic"],
            }
        else:
            existing["bertopic_rank"] = rank
            existing["retrieval_sources"] = ["AI-related", "BERTopic"]
            if not str(existing.get("relationship_or_cluster", "")).strip():
                existing["relationship_or_cluster"] = topic_label
            existing["total_views"] = max(
                float(existing.get("total_views", 0) or 0),
                float(candidate.get("total_views", 0) or 0),
            )

    if not rows_by_story_id:
        return pd.DataFrame(columns=output_columns)

    rows = []
    safe_k = max(1, int(rrf_k))
    for row in rows_by_story_id.values():
        ai_rank = row.get("ai_rank")
        bertopic_rank = row.get("bertopic_rank")
        rrf_score = 0.0
        if isinstance(ai_rank, int):
            rrf_score += float(ai_weight) / (safe_k + ai_rank)
        if isinstance(bertopic_rank, int):
            rrf_score += float(bertopic_weight) / (safe_k + bertopic_rank)
        row["rrf_score"] = rrf_score
        row["retrieval_sources"] = ", ".join(row["retrieval_sources"])
        rows.append(row)

    maximum_rrf_score = max(float(row["rrf_score"]) for row in rows)
    for row in rows:
        row["unified_rank_score"] = round(
            100.0 * float(row["rrf_score"]) / maximum_rrf_score,
            1,
        ) if maximum_rrf_score else 0.0

    return (
        pd.DataFrame(rows, columns=output_columns)
        .sort_values(
            ["rrf_score", "total_views"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def get_bertopic_view_data(
    keyword_query: str,
    title_summary: pd.DataFrame,
    direct_story_ids: tuple[str, ...],
    index_version: str,
    retrieval_version: str,
    excluded_story_ids: tuple[str, ...] = (),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    del index_version, retrieval_version
    return retrieve_bertopic_candidates(
        keyword_query=keyword_query,
        candidate_titles=title_summary.to_dict("records"),
        direct_story_ids=direct_story_ids,
        excluded_story_ids=excluded_story_ids,
    )


def get_bertopic_catalog_data(
    title_summary: pd.DataFrame,
    index_version: str,
    excluded_story_ids: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    del index_version
    return get_bertopic_topic_catalog(title_summary.to_dict("records"), excluded_story_ids=excluded_story_ids)


def run_bertopic_tab_flow(
    keyword_query: str,
    title_summary: pd.DataFrame,
    direct_story_ids: tuple[str, ...],
    excluded_story_ids: tuple[str, ...] = (),
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Compute all BERTopic-tab data in one background job."""
    # Also suppress identical displayed titles stored under different story IDs.
    excluded = set(excluded_story_ids) | set(direct_story_ids)
    title_keys = title_summary["page_title"].astype(str).str.casefold().str.split().str.join(" ")
    excluded_titles = set(title_keys[title_summary["story_id"].astype(str).isin(excluded)])
    excluded.update(title_summary.loc[title_keys.isin(excluded_titles), "story_id"].astype(str))
    excluded_story_ids = tuple(sorted(excluded))
    candidates, diagnostics = get_bertopic_view_data(
        keyword_query=keyword_query,
        title_summary=title_summary,
        direct_story_ids=direct_story_ids,
        index_version=BERTOPIC_INDEX_VERSION,
        retrieval_version=BERTOPIC_RETRIEVAL_CACHE_VERSION,
        excluded_story_ids=excluded_story_ids,
    )
    catalog = get_bertopic_catalog_data(
        title_summary=title_summary,
        index_version=BERTOPIC_INDEX_VERSION,
        excluded_story_ids=excluded_story_ids,
    )
    candidates = [row for row in candidates if str(row["story_id"]) not in excluded]
    # Direct titles can inform discovery internally without reappearing in previews.
    for row in candidates:
        for evidence in row.get("retrieval_evidence", []):
            evidence["prototype_anchor_titles"] = []
    for prototype in diagnostics.get("query_prototypes", []):
        prototype.pop("representative_titles", None)
    diagnostics["excluded_story_count"] = len(excluded)
    return candidates, diagnostics, catalog


@st.cache_data(show_spinner=False)
def get_ai_related_candidate_pool(
    keyword_query: str,
    match_type: str,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
    matched_titles: pd.DataFrame,
    direct_story_ids: tuple[str, ...],
    force_profile_refresh: bool = False,
    apply_strict_evidence_gate: bool = False,
    query_central_policy: bool = False,
    refresh_request_id: str = "",
    _progress_key: str = "",
) -> pd.DataFrame:
    # The value is intentionally used only by Streamlit's cache key. A unique
    # ID makes an explicit refresh a one-time cache miss without clearing
    # cached results for every other query and session.
    del refresh_request_id

    columns = [
        "story_id",
        "page_title",
        "retrieval_evidence",
        "total_views",
        "active_months",
        "first_month",
        "last_month",
        "would_pass_evidence_gate",
    ]
    if title_summary.empty or not keyword_query.strip():
        return pd.DataFrame(columns=columns)

    process_started_at = time.perf_counter()
    process_timings: list[dict[str, object]] = []
    active_phase = ""
    active_phase_started_at = process_started_at

    if _progress_key:
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS[_progress_key] = {
                "phase": "",
                "phase_started_at": process_started_at,
                "process_timings": [],
                "completed_batches": 0,
                "total_batches": 0,
                "evaluated_count": 0,
                "accepted_count": 0,
                "live_rows": [],
            }

    def start_process_phase(phase: str) -> None:
        nonlocal active_phase, active_phase_started_at
        now = time.perf_counter()
        if active_phase:
            process_timings.append(
                {
                    "step": active_phase,
                    "duration_seconds": max(0.0, now - active_phase_started_at),
                }
            )
        active_phase = phase
        active_phase_started_at = now
        if _progress_key:
            with RELATED_FLOW_PROGRESS_LOCK:
                progress = RELATED_FLOW_PROGRESS.setdefault(_progress_key, {})
                progress["phase"] = phase
                progress["phase_started_at"] = now
                progress["process_timings"] = [dict(item) for item in process_timings]

    def finish_process_timing(result: pd.DataFrame) -> pd.DataFrame:
        nonlocal active_phase
        now = time.perf_counter()
        if active_phase:
            process_timings.append(
                {
                    "step": active_phase,
                    "duration_seconds": max(0.0, now - active_phase_started_at),
                }
            )
            active_phase = ""
        result.attrs["process_timings"] = [dict(item) for item in process_timings]
        result.attrs["total_elapsed_seconds"] = max(0.0, now - process_started_at)
        return result

    # Remove every primary-traffic story before relationship research or candidate
    # retrieval begins. This also prevents an unnecessary Vertex research call
    # when the exclusion set leaves no corpus to evaluate.
    start_process_phase("Prepare title corpus")
    direct_story_id_set = set(direct_story_ids)
    sweep_corpus = title_summary.loc[
        ~title_summary["story_id"].astype(str).isin(direct_story_id_set)
    ].copy()
    sweep_corpus = sweep_corpus.loc[
        sweep_corpus["page_title"].fillna("").astype(str).str.strip().ne("")
    ].reset_index(drop=True)
    if sweep_corpus.empty:
        return finish_process_timing(pd.DataFrame(columns=columns))

    start_process_phase("Prepare query context")
    research_context_titles = matched_titles
    if "story_id" in research_context_titles.columns:
        research_context_titles = research_context_titles.loc[
            ~research_context_titles["story_id"].astype(str).isin(direct_story_id_set)
        ]
    matched_title_context = (
        research_context_titles["page_title"].dropna().astype(str).map(str.strip)
    )
    matched_title_context = matched_title_context.loc[
        matched_title_context.ne("")
    ].tolist()
    profile_key, title_fingerprint = build_grounded_profile_key(
        keyword_query=keyword_query,
        match_type=match_type,
        matched_titles=matched_title_context,
        query_central_policy=query_central_policy,
    )
    saved_profile = None
    if not force_profile_refresh:
        start_process_phase("Load saved query profile")
        try:
            saved_profile = load_saved_grounded_profile(profile_key)
        except sqlite3.Error:
            saved_profile = None
        if saved_profile is not None and saved_grounded_profile_needs_recovery(
            keyword_query=keyword_query,
            grounded_research=saved_profile[0],
        ):
            saved_profile = None

    if saved_profile is not None:
        grounded_research, _ = saved_profile
    else:
        start_process_phase("Research query with Gemini")
        try:
            grounded_research = research_query_relationships_with_vertex(
                keyword_query=keyword_query,
                matched_titles=matched_title_context,
                service_account_path=SERVICE_ACCOUNT_PATH,
                match_type=match_type,
                location=get_vertex_location(),
                model_name=get_vertex_model_name("research"),
                query_central_policy=query_central_policy,
            )
        except Exception as exc:
            # Do not convert transient external failures into an empty successful
            # result: st.cache_data would retain that failure for later runs.
            raise RuntimeError(
                f"Live grounded relationship research failed: {exc}"
            ) from exc
        recovery_error = str(
            grounded_research.get("relationship_recovery_error", "")
        ).strip()
        if recovery_error:
            raise RuntimeError(
                f"Live grounded relationship research failed: {recovery_error}"
            )
        try:
            save_grounded_profile(
                profile_key=profile_key,
                keyword_query=keyword_query,
                match_type=match_type,
                title_fingerprint=title_fingerprint,
                grounded_research=grounded_research,
                related_topics=[],
            )
        except sqlite3.Error:
            pass
    corpus_fingerprint = hashlib.sha256(
        "\n".join(
            f"{row.story_id}\t{row.page_title}"
            for row in sweep_corpus[["story_id", "page_title"]].itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()
    research_fingerprint = hashlib.sha256(
        json.dumps(grounded_research, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    embedding_model = get_relationship_embedding_model_name()
    retrieval_profile_key = hashlib.sha256(
        (
            f"{profile_key}|{research_fingerprint}|{RELATIONSHIP_RETRIEVAL_VERSION}|"
            f"{embedding_model}|{GENERATIVE_RETRIEVAL_VERSION}"
        ).encode("utf-8")
    ).hexdigest()
    selector_model = get_vertex_model_name("judge")
    if force_profile_refresh:
        cached_sweep = None
    else:
        start_process_phase("Check saved candidate results")
        cached_sweep = load_cached_generative_retrieval(
            profile_key=retrieval_profile_key,
            model_name=selector_model,
            corpus_fingerprint=corpus_fingerprint,
        )
    cached_relationship_routes = extract_relationship_routes(
        keyword_query=keyword_query,
        grounded_research=grounded_research,
        query_central_policy=query_central_policy,
    )
    retrieval_diagnostics: dict[str, object] = {
        "mode": "relationship_embeddings",
        "relationship_count": len(
            {route.relationship_id for route in cached_relationship_routes}
        ),
        "route_count": len(cached_relationship_routes),
        "candidate_count": 0,
        "fallback_reason": "",
        "embedding_model": embedding_model,
        "excluded_story_count": len(direct_story_id_set),
        "eligible_corpus_count": len(sweep_corpus),
    }
    exclusion_reason_by_story_id: dict[str, str] = {}
    if cached_sweep is None:
        start_process_phase("Retrieve embedding candidates")
        retrieval_candidates: list[dict[str, object]] = []
        try:
            retrieval_candidates, retrieval_diagnostics = retrieve_relationship_candidates(
                keyword_query=keyword_query,
                candidate_titles=title_summary.to_dict("records"),
                grounded_research=grounded_research,
                model_name=embedding_model,
                query_central_policy=query_central_policy,
                excluded_story_ids=direct_story_id_set,
            )
        except Exception as exc:
            # Embedding/model/index errors are operational failures, not a valid
            # zero-result retrieval, and must remain retryable on the next click.
            raise RuntimeError(
                f"Relationship candidate retrieval failed: {exc}"
            ) from exc

        retrieval_candidates = [
            row
            for row in retrieval_candidates
            if str(row.get("story_id", "")) not in direct_story_id_set
        ]
        judge_candidates = retrieval_candidates
        if judge_candidates:
            retrieval_diagnostics.setdefault("mode", "relationship_embeddings")
            retrieval_diagnostics["candidate_count"] = len(judge_candidates)
        else:
            judge_candidates = []
            retrieval_diagnostics["mode"] = "bounded_retrieval_no_candidates"
            retrieval_diagnostics["candidate_count"] = 0

        candidate_story_ids = [
            str(row.get("story_id", "")).strip() for row in judge_candidates
        ]
        title_fingerprints = {
            str(row.get("story_id", "")).strip(): hashlib.sha256(
                str(row.get("page_title", "")).strip().encode("utf-8")
            ).hexdigest()
            for row in judge_candidates
        }
        if force_profile_refresh:
            cached_evaluations = {}
        else:
            start_process_phase("Check saved candidate evaluations")
            cached_evaluations = load_cached_candidate_evaluations(
                profile_key=retrieval_profile_key,
                story_ids=candidate_story_ids,
                model_name=selector_model,
                title_fingerprints=title_fingerprints,
            )
        pending_candidates = [
            row
            for row in judge_candidates
            if str(row.get("story_id", "")).strip() not in cached_evaluations
        ]
        selected_titles = [
            evaluation
            for evaluation in cached_evaluations.values()
            if evaluation.get("decision") == "include"
        ]
        exclusion_reason_by_story_id = {
            story_id: str(evaluation.get("ai_audit_reason", "")).strip()
            or "No exclusion reason was recorded."
            for story_id, evaluation in cached_evaluations.items()
            if evaluation.get("decision") == "exclude"
        }
        accepted_by_story_id = {
            str(item.get("story_id", "")): dict(item) for item in selected_titles
        }
        candidate_by_story_id = {
            str(item.get("story_id", "")).strip(): item
            for item in judge_candidates
        }
        evaluated_count = len(cached_evaluations)

        def current_live_rows() -> list[dict[str, object]]:
            rows = []
            for story_id, evaluation in accepted_by_story_id.items():
                source = candidate_by_story_id.get(story_id, {})
                rows.append(
                    {
                        "story_id": story_id,
                        "page_title": str(source.get("page_title", "")),
                        "ai_relationship": str(
                            evaluation.get("ai_relationship", "")
                        ),
                        "ai_confidence": float(
                            evaluation.get("ai_confidence", 0.0)
                        ),
                        "ai_relevance_level": int(
                            evaluation.get("ai_relevance_level", 0)
                        ),
                        "total_views": float(source.get("total_views", 0) or 0),
                    }
                )
            rows.sort(
                key=lambda item: (
                    int(item["ai_relevance_level"]),
                    float(item["ai_confidence"]),
                    float(item["total_views"]),
                ),
                reverse=True,
            )
            return rows[:RELATED_LIVE_RESULT_LIMIT]

        if _progress_key:
            pending_batch_count = (
                len(pending_candidates) + DEFAULT_GENERATIVE_RETRIEVAL_BATCH_SIZE - 1
            ) // DEFAULT_GENERATIVE_RETRIEVAL_BATCH_SIZE
            with RELATED_FLOW_PROGRESS_LOCK:
                progress = RELATED_FLOW_PROGRESS.setdefault(_progress_key, {})
                progress.update(
                    {
                        "completed_batches": 0,
                        "total_batches": pending_batch_count,
                        "evaluated_count": evaluated_count,
                        "accepted_count": len(accepted_by_story_id),
                        "live_rows": current_live_rows(),
                    }
                )

        def publish_batch(
            batch: list[dict[str, object]],
            batch_selected: list[dict[str, object]],
            completed_batches: int,
            total_batches: int,
        ) -> None:
            nonlocal evaluated_count
            selected_in_batch = {
                str(item.get("story_id", "")).strip(): item
                for item in batch_selected
            }
            evaluations = []
            for candidate in batch:
                story_id = str(candidate.get("story_id", "")).strip()
                selected = selected_in_batch.get(story_id)
                if selected is None:
                    evaluations.append(
                        {
                            "story_id": story_id,
                            "decision": "exclude",
                            "ai_relationship": "",
                            "ai_audit_reason": str(
                                candidate.get(
                                    "_selection_audit_reason",
                                    "Gemini did not return this title as a match.",
                                )
                            ),
                            "ai_confidence": 0.0,
                            "ai_relevance_level": 0,
                            "ai_relevance_type": "",
                        }
                    )
                else:
                    evaluation = dict(selected)
                    evaluation["decision"] = "include"
                    evaluations.append(evaluation)
                    accepted_by_story_id[story_id] = evaluation
                if evaluations[-1]["decision"] == "exclude":
                    exclusion_reason_by_story_id[story_id] = str(
                        evaluations[-1].get("ai_audit_reason", "")
                    ).strip() or "No exclusion reason was recorded."
                else:
                    exclusion_reason_by_story_id.pop(story_id, None)
                evaluations[-1]["title_fingerprint"] = title_fingerprints.get(
                    story_id, ""
                )
            save_candidate_evaluations(
                profile_key=retrieval_profile_key,
                evaluations=evaluations,
                model_name=selector_model,
            )
            evaluated_count += len(batch)
            if _progress_key:
                with RELATED_FLOW_PROGRESS_LOCK:
                    progress = RELATED_FLOW_PROGRESS.setdefault(_progress_key, {})
                    progress.update(
                        {
                            "completed_batches": completed_batches,
                            "total_batches": total_batches,
                            "evaluated_count": evaluated_count,
                            "accepted_count": len(accepted_by_story_id),
                            "live_rows": current_live_rows(),
                        }
                    )

        if pending_candidates:
            start_process_phase("Validate candidates with Gemini")
        newly_selected = (
            select_titles_generatively_with_vertex(
                keyword_query=keyword_query,
                candidate_titles=pending_candidates,
                grounded_research=grounded_research,
                service_account_path=SERVICE_ACCOUNT_PATH,
                location=get_vertex_location(),
                model_name=selector_model,
                batch_size=DEFAULT_GENERATIVE_RETRIEVAL_BATCH_SIZE,
                batch_callback=publish_batch,
                query_central_policy=query_central_policy,
            )
            if pending_candidates
            else []
        )
        for selected in newly_selected:
            accepted_by_story_id[str(selected.get("story_id", ""))] = selected
        start_process_phase("Save and assemble candidate results")
        selected_titles = list(accepted_by_story_id.values())
        retrieval_evidence_by_story_id = {
            str(row.get("story_id", "")): list(row.get("retrieval_evidence", []))
            for row in judge_candidates
            if row.get("retrieval_evidence")
        }
        for selected in selected_titles:
            story_id = str(selected.get("story_id", ""))
            if story_id in retrieval_evidence_by_story_id:
                selected["retrieval_evidence"] = retrieval_evidence_by_story_id[story_id]
        screened_count = len(judge_candidates)
        if judge_candidates and selected_titles:
            save_generative_retrieval(
                profile_key=retrieval_profile_key,
                model_name=selector_model,
                corpus_fingerprint=corpus_fingerprint,
                selected_titles=selected_titles,
                screened_count=screened_count,
            )
    else:
        selected_titles, screened_count = cached_sweep
        retrieval_diagnostics["candidate_count"] = screened_count

    start_process_phase("Filter and assemble final results")
    evidence_gate_pass_by_story_id = {
        str(selected.get("story_id", "")).strip(): passes_relationship_evidence_gate(
            selected
        )
        for selected in selected_titles
        if str(selected.get("story_id", "")).strip()
    }
    selected_by_story_id = {
        str(selected.get("story_id", "")).strip(): selected
        for selected in selected_titles
        if str(selected.get("story_id", "")).strip()
        and (
            not apply_strict_evidence_gate
            or evidence_gate_pass_by_story_id.get(
                str(selected.get("story_id", "")).strip(), False
            )
        )
    }
    raw_selected_count = len(selected_titles)
    would_fail_evidence_gate_count = sum(
        not passed for passed in evidence_gate_pass_by_story_id.values()
    )
    evidence_rejected_count = (
        would_fail_evidence_gate_count if apply_strict_evidence_gate else 0
    )
    selected_ids = set(selected_by_story_id)
    candidate_pool = sweep_corpus.loc[
        sweep_corpus["story_id"].astype(str).isin(selected_ids)
    ].copy()
    if candidate_pool.empty:
        candidate_pool = pd.DataFrame(columns=columns)
    else:
        def display_retrieval_evidence(story_id: object) -> list[dict[str, object]]:
            selected = selected_by_story_id[str(story_id)]
            embedding_evidence = selected.get("retrieval_evidence", [])
            if isinstance(embedding_evidence, list) and embedding_evidence:
                return [
                    {
                        "term": str(item.get("related_subject", "")),
                        "category": str(item.get("relationship_class", "")),
                        "relationship": str(item.get("factual_bridge", "")),
                        "confidence": float(item.get("similarity", 0.0)),
                        **item,
                    }
                    for item in embedding_evidence
                    if isinstance(item, dict)
                ]
            return [
                {
                    "term": "Generative corpus sweep",
                    "category": "generative_selection",
                    "relationship": str(selected.get("ai_relationship", "")),
                    "confidence": float(selected.get("ai_confidence", 0.0)),
                }
            ]

        candidate_pool["retrieval_evidence"] = candidate_pool["story_id"].map(
            display_retrieval_evidence
        )
        candidate_pool["would_pass_evidence_gate"] = candidate_pool["story_id"].map(
            lambda story_id: evidence_gate_pass_by_story_id.get(str(story_id), False)
        )
        candidate_pool = candidate_pool[
            [column for column in columns if column in candidate_pool.columns]
        ].reset_index(drop=True)

    evaluations = {
        story_id: {
            "story_id": story_id,
            "decision": "include",
            "ai_relationship": str(selected.get("ai_relationship", "")),
            "ai_audit_reason": str(selected.get("ai_audit_reason", "")),
            "ai_confidence": float(selected.get("ai_confidence", 0.0)),
            "ai_relevance_level": int(selected.get("ai_relevance_level", 0)),
            "ai_relevance_type": str(selected.get("ai_relevance_type", "")),
        }
        for story_id, selected in selected_by_story_id.items()
    }

    result_key = hashlib.sha256(
        (
            f"{retrieval_profile_key}|{GENERATIVE_RETRIEVAL_VERSION}|{selector_model}|"
            f"{corpus_fingerprint}|strict_gate={int(apply_strict_evidence_gate)}"
            f"{'|query_central_policy=1' if query_central_policy else ''}"
        ).encode("utf-8")
    ).hexdigest()
    candidate_pool.attrs["profile_key"] = result_key
    candidate_pool.attrs["query_profile"] = []
    candidate_pool.attrs["preverified_evaluations"] = evaluations
    candidate_pool.attrs["screened_count"] = screened_count
    candidate_pool.attrs["raw_selected_count"] = raw_selected_count
    candidate_pool.attrs["evidence_rejected_count"] = evidence_rejected_count
    candidate_pool.attrs["would_fail_evidence_gate_count"] = (
        would_fail_evidence_gate_count
    )
    candidate_pool.attrs["gemini_excluded_count"] = sum(
        reason.startswith("Gemini did not return")
        for reason in exclusion_reason_by_story_id.values()
    )
    candidate_pool.attrs["local_validation_rejected_count"] = sum(
        reason.startswith("Local validation rejected")
        for reason in exclusion_reason_by_story_id.values()
    )
    candidate_pool.attrs["unclassified_excluded_count"] = sum(
        not reason.startswith(("Gemini did not return", "Local validation rejected"))
        for reason in exclusion_reason_by_story_id.values()
    )
    candidate_pool.attrs["strict_evidence_gate_enabled"] = apply_strict_evidence_gate
    candidate_pool.attrs["query_central_policy_enabled"] = query_central_policy
    candidate_pool.attrs["corpus_count"] = len(sweep_corpus)
    candidate_pool.attrs["retrieval_mode"] = retrieval_diagnostics.get("mode", "")
    candidate_pool.attrs["relationship_count"] = int(
        retrieval_diagnostics.get("relationship_count", 0)
    )
    candidate_pool.attrs["route_count"] = int(retrieval_diagnostics.get("route_count", 0))
    candidate_pool.attrs["embedding_model"] = str(
        retrieval_diagnostics.get("embedding_model", embedding_model)
    )
    candidate_pool.attrs["retrieval_excluded_primary_count"] = int(
        retrieval_diagnostics.get("excluded_story_count", len(direct_story_id_set))
    )
    candidate_pool.attrs["retrieval_eligible_corpus_count"] = int(
        retrieval_diagnostics.get("eligible_corpus_count", len(sweep_corpus))
    )
    candidate_pool.attrs["retrieval_embedding_search_depth"] = int(
        retrieval_diagnostics.get("embedding_search_depth", 0)
    )
    candidate_pool.attrs["retrieval_fallback_reason"] = str(
        retrieval_diagnostics.get("fallback_reason", "")
    )
    candidate_pool.attrs["sweep_batch_count"] = (
        screened_count + DEFAULT_GENERATIVE_RETRIEVAL_BATCH_SIZE - 1
    ) // DEFAULT_GENERATIVE_RETRIEVAL_BATCH_SIZE
    candidate_pool.attrs["grounding_sources"] = grounded_research.get("sources", [])
    candidate_pool.attrs["web_search_queries"] = grounded_research.get("web_search_queries", [])
    candidate_pool.attrs["research_error"] = ""
    return finish_process_timing(candidate_pool)


def show_dashboard_tab(
    keyword_query: str,
    search_result: dict[str, object] | None,
    story_categories: pd.DataFrame,
    story_word_counts: pd.DataFrame,
    story_planned_trending: pd.DataFrame,
    story_regions: pd.DataFrame,
    story_audience_types: pd.DataFrame,
) -> None:
    st.subheader("Executive dashboard")
    if not keyword_query.strip():
        show_dashboard_empty_state()
        return
    if search_result is None:
        st.warning("No page titles match this keyword search.")
        return

    matched_story_months = search_result["matched_story_months"]
    matched_titles = search_result["matched_title_summary"]
    monthly_summary = search_result["monthly_summary"]
    if (
        not isinstance(matched_story_months, pd.DataFrame)
        or matched_story_months.empty
        or not isinstance(matched_titles, pd.DataFrame)
        or matched_titles.empty
    ):
        st.warning("No page titles match this keyword search.")
        return

    available_months = (
        matched_story_months[["month"]]
        .dropna()
        .drop_duplicates()
        .sort_values("month")["month"]
        .tolist()
    )
    month_options = ["All months"] + [
        pd.to_datetime(month).strftime("%b %Y") for month in available_months
    ]
    selected_month_label = st.selectbox(
        "Dashboard month",
        options=month_options,
        help="Use All months for the full executive view, or isolate one month for a focused readout.",
    )

    filtered_story_months = matched_story_months
    if selected_month_label != "All months":
        selected_month = available_months[month_options.index(selected_month_label) - 1]
        filtered_story_months = matched_story_months.loc[
            matched_story_months["month"] == selected_month
        ].copy()

    category_summary = build_category_analytics(filtered_story_months, story_categories)
    word_count_summary = build_word_count_analytics(filtered_story_months, story_word_counts)
    planned_trending_summary = build_planned_trending_analytics(
        filtered_story_months,
        story_planned_trending,
    )
    region_summary = build_region_analytics(filtered_story_months, story_regions)
    audience_segment_summary = build_audience_segment_analytics(
        filtered_story_months,
        story_audience_types,
    )

    total_views = int(filtered_story_months["views"].sum())
    article_count = filtered_story_months["story_id"].nunique()
    avg_views_per_article = int(round(total_views / article_count)) if article_count else 0
    high_traffic_articles = (
        filtered_story_months.loc[filtered_story_months["above_monthly_threshold"], "story_id"]
        .nunique()
        if "above_monthly_threshold" in filtered_story_months.columns
        else 0
    )
    high_traffic_views = (
        int(filtered_story_months.loc[filtered_story_months["above_monthly_threshold"], "views"].sum())
        if "above_monthly_threshold" in filtered_story_months.columns
        else 0
    )
    top_category = category_summary.iloc[0]["category"] if not category_summary.empty else "N/A"
    top_region = region_summary.iloc[0]["region"] if not region_summary.empty else "N/A"
    top_audience_segment = (
        audience_segment_summary.iloc[0]["audience_name"]
        if not audience_segment_summary.empty
        else "N/A"
    )

    analysis_source = str(search_result.get("analysis_source", "")).strip()
    source_note = (
        " Based on Search Results refined primary tiers; Similar spelling is excluded."
        if analysis_source == "Search Results refined"
        else ""
    )
    st.caption(
        f"Executive view for `{keyword_query.strip()}` across "
        f"{selected_month_label.lower()}.{source_note}"
    )
    render_kpi_card_group(
        [
            {"label": "Matched views", "value": format_compact_number(total_views)},
            {"label": "Unique titles", "value": format_indian_count(article_count)},
            {"label": "Avg views/title", "value": format_compact_number(avg_views_per_article)},
            {"label": "High-traffic views", "value": format_compact_number(high_traffic_views)},
        ],
    )
    render_kpi_card_group(
        [
            {"label": "Leading category", "value": str(top_category)},
            {"label": "Leading region", "value": str(top_region)},
            {"label": "Leading segment", "value": str(top_audience_segment)},
            {"label": "High-traffic titles", "value": format_indian_count(high_traffic_articles)},
        ],
    )

    st.divider()

    if (
        selected_month_label == "All months"
        and isinstance(monthly_summary, pd.DataFrame)
        and not monthly_summary.empty
    ):
        timeline_df = monthly_summary.copy()
        timeline_df["month_label"] = pd.to_datetime(timeline_df["month"]).dt.strftime("%b %Y")
        timeline_df["views_label"] = format_indian_plot_text(timeline_df["views"])
        timeline_df["titles_label"] = format_indian_plot_text(
            timeline_df["titles_containing_keywords"]
        )
        fig = px.area(
            timeline_df,
            x="month_label",
            y="views",
            markers=True,
            title="Matched traffic trend",
            labels={
                "month_label": "Month",
                "views": "Views",
                "titles_containing_keywords": "Titles",
            },
            custom_data=["views_label", "titles_label"],
            hover_data={"views_label": False, "titles_label": False},
        )
        fig.update_traces(
            line=dict(width=3),
            fillcolor="rgba(37,99,235,0.16)",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Views: %{customdata[0]}<br>"
                "Titles: %{customdata[1]}<extra></extra>"
            ),
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(apply_executive_chart_style(fig, height=340), use_container_width=True)

    chart_col_left, chart_col_right = st.columns(2)
    with chart_col_left:
        if not category_summary.empty:
            category_chart = category_summary.head(12).sort_values("total_views")
            category_chart["total_views_label"] = format_indian_plot_text(
                category_chart["total_views"]
            )
            category_chart["article_count_label"] = format_indian_plot_text(
                category_chart["article_count"]
            )
            category_chart["avg_page_views_label"] = format_indian_plot_text(
                category_chart["avg_page_views"]
            )
            fig = px.bar(
                category_chart,
                x="total_views",
                y="category",
                orientation="h",
                title="Category contribution",
                text="total_views_label",
                labels={
                    "category": "Category",
                    "total_views": "Views",
                    "article_count": "Titles",
                    "avg_page_views": "Avg views/title",
                },
                custom_data=[
                    "total_views_label",
                    "article_count_label",
                    "avg_page_views_label",
                ],
                hover_data={
                    "total_views_label": False,
                    "article_count_label": False,
                    "avg_page_views_label": False,
                },
            )
            fig.update_traces(
                marker_color="#2563eb",
                texttemplate="%{text}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Views: %{customdata[0]}<br>"
                    "Titles: %{customdata[1]}<br>"
                    "Avg views/title: %{customdata[2]}<extra></extra>"
                ),
            )
            st.plotly_chart(apply_executive_chart_style(fig, height=430), use_container_width=True)
        else:
            st.info("No category data is available for this selection.")

    with chart_col_right:
        if not planned_trending_summary.empty:
            chart_summary = planned_trending_summary.copy()
            chart_summary["article_count"] = pd.to_numeric(
                chart_summary["article_count"],
                errors="coerce",
            ).fillna(0).astype(int)
            chart_summary["avg_page_views"] = pd.to_numeric(
                chart_summary["avg_page_views"],
                errors="coerce",
            ).fillna(0).astype(int)
            chart_summary["total_views"] = pd.to_numeric(
                chart_summary["total_views"],
                errors="coerce",
            ).fillna(0).astype(int)
            chart_summary["hover_text"] = chart_summary.apply(
                lambda row: (
                    f"planned_trending={row['planned_trending']}<br>"
                    f"total_views={format_indian_count(row['total_views'])}<br>"
                    f"title_count={format_indian_count(row['article_count'])}<br>"
                    f"views_per_article={format_indian_count(row['avg_page_views'])}"
                ),
                axis=1,
            )

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=chart_summary["planned_trending"].astype(str).tolist(),
                        values=chart_summary["total_views"].tolist(),
                        hole=0.58,
                        hovertext=chart_summary["hover_text"].tolist(),
                        hoverinfo="text",
                        textinfo="percent+label",
                        sort=False,
                        marker=dict(line=dict(color="white", width=2)),
                    )
                ]
            )
            fig.update_layout(title="Planned vs trending mix")
            st.plotly_chart(apply_executive_chart_style(fig, height=430), use_container_width=True)
        else:
            st.info("No planned/trending data is available for this selection.")

    chart_col_left, chart_col_right = st.columns(2)
    with chart_col_left:
        if not word_count_summary.empty:
            word_chart = word_count_summary.loc[word_count_summary["total_views"] > 0].copy()
            if word_chart.empty:
                word_chart = word_count_summary.copy()
            word_chart["article_count_label"] = format_indian_plot_text(
                word_chart["article_count"]
            )
            word_chart["total_views_label"] = format_indian_plot_text(word_chart["total_views"])
            word_chart["avg_page_views_label"] = format_indian_plot_text(
                word_chart["avg_page_views"]
            )
            fig = px.bar(
                word_chart,
                x="word_count_bucket",
                y="total_views",
                title="Performance by story length",
                text="article_count_label",
                labels={
                    "word_count_bucket": "Word count bucket",
                    "total_views": "Views",
                    "article_count": "Titles",
                    "avg_page_views": "Avg views/title",
                },
                custom_data=[
                    "total_views_label",
                    "article_count_label",
                    "avg_page_views_label",
                ],
                hover_data={
                    "total_views_label": False,
                    "article_count_label": False,
                    "avg_page_views_label": False,
                },
            )
            fig.update_traces(
                marker_color="#16a34a",
                texttemplate="%{text} titles",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Views: %{customdata[0]}<br>"
                    "Titles: %{customdata[1]}<br>"
                    "Avg views/title: %{customdata[2]}<extra></extra>"
                ),
            )
            fig.update_xaxes(tickangle=-25)
            st.plotly_chart(apply_executive_chart_style(fig, height=430), use_container_width=True)
        else:
            st.info("No word count data is available for this selection.")

    with chart_col_right:
        if not region_summary.empty:
            region_chart = region_summary.head(12).sort_values("page_views")
            region_chart["page_views_label"] = format_indian_plot_text(region_chart["page_views"])
            region_chart["total_articles_label"] = format_indian_plot_text(
                region_chart["total_articles"]
            )
            region_chart["views_per_article_label"] = format_indian_plot_text(
                region_chart["views_per_article"]
            )
            fig = px.bar(
                region_chart,
                x="page_views",
                y="region",
                orientation="h",
                title="Regional demand",
                text="page_views_label",
                labels={
                    "region": "Region",
                    "page_views": "Views",
                    "total_articles": "Titles",
                    "views_per_article": "Views/title",
                },
                custom_data=[
                    "page_views_label",
                    "total_articles_label",
                    "views_per_article_label",
                ],
                hover_data={
                    "page_views_label": False,
                    "total_articles_label": False,
                    "views_per_article_label": False,
                },
            )
            fig.update_traces(
                marker_color="#db2777",
                texttemplate="%{text}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Views: %{customdata[0]}<br>"
                    "Titles: %{customdata[1]}<br>"
                    "Views/title: %{customdata[2]}<extra></extra>"
                ),
            )
            st.plotly_chart(apply_executive_chart_style(fig, height=430), use_container_width=True)
        else:
            st.info("No region data is available for this selection.")

    if not audience_segment_summary.empty:
        st.subheader("Audience segment lens")
        audience_chart = audience_segment_summary.sort_values("audience_views", ascending=False)
        audience_chart["audience_views_label"] = format_indian_plot_text(
            audience_chart["audience_views"]
        )
        audience_chart["views_per_article_label"] = format_indian_plot_text(
            audience_chart["views_per_article"]
        )
        audience_chart["total_users_label"] = format_indian_plot_text(
            audience_chart["total_users"]
        )

        audience_col_left, audience_col_right = st.columns([1.35, 1])
        with audience_col_left:
            fig = px.bar(
                audience_chart,
                x="audience_name",
                y="audience_views",
                title="Views by segment",
                text="audience_views_label",
                labels={
                    "audience_name": "Audience segment",
                    "audience_views": "Views",
                    "views_per_article": "Views per article",
                },
                custom_data=["audience_views_label", "views_per_article_label"],
            )
            fig.update_traces(
                marker_color="#2563eb",
                texttemplate="%{text}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Views: %{customdata[0]}<br>"
                    "Views per article: %{customdata[1]}<extra></extra>"
                ),
            )
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(apply_executive_chart_style(fig, height=420), use_container_width=True)

        with audience_col_right:
            fig = px.pie(
                audience_chart,
                names="audience_name",
                values="total_users",
                hole=0.62,
                title="Total users by segment",
                labels={
                    "audience_name": "Audience segment",
                    "total_users": "Users",
                },
                color="audience_name",
                color_discrete_map={
                    "Brand Lovers": "#2563eb",
                    "Casual Readers": "#f59e0b",
                    "Loyal Users": "#16a34a",
                },
            )
            fig.update_traces(
                textinfo="percent+label",
                sort=False,
                hovertemplate="<b>%{label}</b><br>Users: %{hovertext}<extra></extra>",
                hovertext=audience_chart["total_users_label"].tolist(),
                marker=dict(line=dict(color="white", width=2)),
            )
            st.plotly_chart(apply_executive_chart_style(fig, height=420), use_container_width=True)
    else:
        st.info("No audience segment data is available for this selection.")

    st.subheader("Top executive opportunities")
    opportunity_rows = matched_titles.loc[
        matched_titles["story_id"].isin(filtered_story_months["story_id"].unique())
    ].copy()
    opportunity_rows = opportunity_rows.sort_values("total_views", ascending=False).head(10)
    opportunity_rows = add_audience_segment_view_columns(
        opportunity_rows,
        filtered_story_months,
        story_audience_types,
    )
    show_wrapped_title_table(
        format_editorial_table(
            select_existing_columns(
                add_high_traffic_month_display(opportunity_rows, filtered_story_months),
                [
                    "story_id",
                    "page_title",
                    "total_views",
                    "brand_lovers_views",
                    "casual_readers_views",
                    "loyal_users_views",
                    "views_above_threshold",
                    "threshold_matched_months",
                    "active_months",
                    "first_month",
                    "last_month",
                ],
            )
        ),
        extra_wrapped_columns={
            "Brand Lovers views": 120,
            "Casual Readers views": 130,
            "Loyal Users views": 120,
        },
    )


def show_analytics_tab(
    keyword_query: str,
    search_result: dict[str, object] | None,
    story_categories: pd.DataFrame,
    story_word_counts: pd.DataFrame,
    story_planned_trending: pd.DataFrame,
    story_regions: pd.DataFrame,
    story_audience_types: pd.DataFrame,
) -> None:
    st.subheader("Analytics")
    if not keyword_query.strip():
        st.info("Enter a keyword or group of keywords to see dimension analytics.")
        return
    if search_result is None:
        st.warning("No page titles match this keyword search.")
        return

    matched_story_months = search_result["matched_story_months"]
    if not isinstance(matched_story_months, pd.DataFrame) or matched_story_months.empty:
        st.warning("No page titles match this keyword search.")
        return

    if search_result.get("analysis_source") == "Search Results refined":
        st.caption(
            "Based on Search Results refined primary tiers; Similar spelling is excluded."
        )

    available_months = (
        matched_story_months[["month"]]
        .dropna()
        .drop_duplicates()
        .sort_values("month")["month"]
        .tolist()
    )
    month_options = ["All months"] + [
        pd.to_datetime(month).strftime("%b %Y") for month in available_months
    ]
    selected_month_label = st.radio(
        "Month",
        options=month_options,
        horizontal=True,
    )
    filtered_story_months = matched_story_months
    if selected_month_label != "All months":
        selected_month = available_months[month_options.index(selected_month_label) - 1]
        filtered_story_months = matched_story_months.loc[
            matched_story_months["month"] == selected_month
        ].copy()

    category_summary = build_category_analytics(
        matched_story_months=filtered_story_months,
        story_categories=story_categories,
    )
    if category_summary.empty:
        st.info("No category analytics are available for this keyword search.")
        return

    st.caption("Category matrix for titles containing the searched keyword.")
    show_editorial_dataframe(format_editorial_table(category_summary))

    word_count_summary = build_word_count_analytics(
        matched_story_months=filtered_story_months,
        story_word_counts=story_word_counts,
    )
    if word_count_summary.empty:
        st.info("No word count analytics are available for this keyword search.")
        return

    st.caption("Word count bucket matrix for titles containing the searched keyword.")
    show_editorial_dataframe(format_editorial_table(word_count_summary))

    planned_trending_summary = build_planned_trending_analytics(
        matched_story_months=filtered_story_months,
        story_planned_trending=story_planned_trending,
    )
    if planned_trending_summary.empty:
        st.info("No planned/trending analytics are available for this keyword search.")
        return

    st.caption("Planned/trending matrix for titles containing the searched keyword.")
    show_editorial_dataframe(format_editorial_table(planned_trending_summary))

    region_summary = build_region_analytics(
        matched_story_months=filtered_story_months,
        story_regions=story_regions,
    )
    if region_summary.empty:
        st.info("No region analytics are available for this keyword search.")
        return

    st.caption("Region matrix for titles containing the searched keyword.")
    show_editorial_dataframe(format_editorial_table(region_summary))

    audience_segment_summary = build_audience_segment_analytics(
        matched_story_months=filtered_story_months,
        story_audience_types=story_audience_types,
    )
    if audience_segment_summary.empty:
        st.info("No audience segment analytics are available for this keyword search.")
        return

    st.caption("Audience segment matrix for titles containing the searched keyword.")
    show_editorial_dataframe(format_editorial_table(audience_segment_summary))

    audience_story_report = build_audience_story_report(
        matched_story_months=filtered_story_months,
        story_audience_types=story_audience_types,
    )
    if audience_story_report.empty:
        return

    st.caption("Top story-by-audience segment report, ranked by audience views.")
    show_wrapped_title_table(
        format_editorial_table(audience_story_report.head(100)),
        extra_wrapped_columns={"Audience segment": 280},
    )


def show_search_tab(
    keyword_query: str,
    match_mode: str,
    search_result: dict[str, pd.DataFrame | list[str]] | None,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> None:
    if not keyword_query.strip():
        st.info("Enter a keyword or group of keywords to see matching page titles and views.")
        st.subheader("Highest-view titles")
        show_wrapped_title_table(format_editorial_table(title_summary.head(25)))
        return

    if search_result is None:
        st.warning("No page titles match this keyword search.")
        return

    matched_titles = search_result["matched_title_summary"]
    exact_matched_titles = search_result["exact_matched_title_summary"]
    fuzzy_matched_titles = search_result["fuzzy_matched_title_summary"]
    above_threshold_titles = search_result["above_threshold_titles"]
    below_threshold_titles = search_result["below_threshold_titles"]
    matched_story_months = search_result["matched_story_months"]
    monthly_summary = search_result["monthly_summary"]

    if matched_titles.empty:
        st.warning("No page titles match this keyword search.")
        return

    matched_views = int(matched_story_months["views"].sum())
    matched_story_count = matched_titles["story_id"].nunique()

    result_cols = st.columns(3)
    result_cols[0].metric("Titles containing this search", format_indian_count(matched_story_count))
    result_cols[1].metric("Views from matching titles", format_indian_count(matched_views))
    result_cols[2].metric("Monthly title matches", format_indian_count(len(matched_story_months)))

    tier_cols = st.columns(2)
    tier_cols[0].metric("Exact lexical matches", format_indian_count(len(exact_matched_titles)))
    tier_cols[1].metric(
        "Similar spelling/root matches",
        format_indian_count(len(fuzzy_matched_titles)),
    )
    if not fuzzy_matched_titles.empty:
        st.caption(
            "Approximate matches are a separate second tier based only on token spelling "
            "similarity. They do not include synonyms or meaning-based matches."
        )
        with st.expander("Review similar spelling/root matches", expanded=True):
            fuzzy_review_columns = [
                "story_id",
                "page_title",
                "direct_match_score",
                "direct_match_detail",
                "total_views",
            ]
            show_wrapped_title_table(
                format_editorial_table(
                    select_existing_columns(fuzzy_matched_titles, fuzzy_review_columns)
                )
            )

    st.subheader("Monthly views from matching titles")
    if not monthly_summary.empty:
        chart_df = monthly_summary.copy()
        chart_df["month_label"] = pd.to_datetime(chart_df["month"]).dt.strftime("%b %Y")
        fig = px.bar(
            chart_df,
            x="month_label",
            y="views",
            text="titles_containing_keywords",
            labels={
                "month_label": "Month",
                "views": "Views",
                "titles_containing_keywords": "Titles",
            },
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        show_editorial_dataframe(format_editorial_table(monthly_summary))

    title_result_columns = [
        "story_id",
        "page_title",
        "direct_match_tier",
        "total_views",
        "views_above_threshold",
        "threshold_matched_months",
        "active_months",
        "first_month",
        "last_month",
    ]
    other_matching_title_columns = [
        "story_id",
        "page_title",
        "direct_match_tier",
        "total_views",
        "active_months",
        "first_month",
        "last_month",
    ]

    st.subheader("Titles with at least one high-traffic month")
    st.caption(
        f"{len(above_threshold_titles):,} matching titles had enough views to count "
        "as high-traffic in one or more months."
    )
    if above_threshold_titles.empty:
        st.info("No matching titles reached the high-traffic cutoff.")
    else:
        show_wrapped_title_table(
            format_editorial_table(
                select_existing_columns(
                    add_high_traffic_month_display(above_threshold_titles, matched_story_months),
                    title_result_columns,
                )
            ),
            scroll_after_rows=30,
            scroll_height=3640,
        )

    st.subheader("Other matching titles")
    st.caption(
        f"{len(below_threshold_titles):,} titles matched the search but never reached "
        "the high-traffic cutoff in any month."
    )
    if below_threshold_titles.empty:
        st.info("All matching titles reached the high-traffic cutoff in at least one month.")
    else:
        show_wrapped_title_table(
            format_editorial_table(
                select_existing_columns(
                    below_threshold_titles,
                    other_matching_title_columns,
                )
            ),
            scroll_after_rows=10,
            scroll_height=420,
        )

    st.subheader("Monthly breakdown for matching titles")
    show_wrapped_title_table(
        format_editorial_table(
            select_existing_columns(
                matched_story_months,
                [
                    "month",
                    "story_id",
                    "page_title",
                    "views",
                    "threshold_value",
                    "above_monthly_threshold",
                ],
            )
        ),
        max_rows=10,
    )
    show_monthly_views_heatmap(matched_story_months)


def build_refined_search_result(
    hits: list[dict[str, object]],
    total_hits: int,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> dict[str, object]:
    """Join ranked OpenSearch hits to the existing local traffic model."""
    hit_columns = [
        "story_id",
        "refined_match_tier",
        "opensearch_score",
        "matched_queries",
        "refined_rank",
    ]
    ranked_hits = []
    for rank, hit in enumerate(hits, start=1):
        ranked_hit = dict(hit)
        ranked_hit["story_id"] = str(ranked_hit.get("story_id", ""))
        ranked_hit["refined_rank"] = rank
        ranked_hits.append(ranked_hit)
    hit_df = pd.DataFrame(ranked_hits)
    if hit_df.empty:
        hit_df = pd.DataFrame(columns=hit_columns)
    else:
        hit_df = hit_df[[column for column in hit_columns if column in hit_df.columns]]

    local_titles = title_summary.copy()
    local_titles["story_id"] = local_titles["story_id"].astype(str)
    matched_titles = hit_df.merge(local_titles, on="story_id", how="inner")
    if not matched_titles.empty:
        matched_titles = matched_titles.sort_values("refined_rank")

    local_months = story_months.copy()
    local_months["story_id"] = local_months["story_id"].astype(str)
    match_tiers = hit_df[["story_id", "refined_match_tier"]]
    matched_story_months = local_months.merge(match_tiers, on="story_id", how="inner")
    matched_titles = add_threshold_split_columns(matched_titles, matched_story_months)
    primary_story_ids = set(
        hit_df.loc[
            hit_df["refined_match_tier"] != "Similar spelling",
            "story_id",
        ].astype(str)
    )
    primary_story_months = matched_story_months.loc[
        matched_story_months["story_id"].isin(primary_story_ids)
    ].copy()
    if primary_story_months.empty:
        monthly_summary = pd.DataFrame(
            columns=["month", "titles_containing_keywords", "views"]
        )
    else:
        monthly_summary = (
            primary_story_months.groupby("month", as_index=False)
            .agg(
                titles_containing_keywords=("story_id", "nunique"),
                views=("views", "sum"),
            )
            .sort_values("month")
        )

    return {
        "matched_titles": matched_titles,
        "matched_story_months": matched_story_months,
        "primary_story_months": primary_story_months,
        "monthly_summary": monthly_summary,
        "total_hits": int(total_hits),
        "returned_hits": len(hit_df),
    }


@st.cache_data(show_spinner=False)
def get_complete_refined_primary_story_ids(
    normalized_query: str,
    match_mode: str,
    index_fingerprint: str,
) -> tuple[str, ...]:
    """Load the complete primary-tier set used to protect Related Stories."""
    settings = load_opensearch_settings()
    if not settings.configured:
        raise OpenSearchRefinedError("OPENSEARCH_URL is not configured.")
    client = OpenSearchRefinedClient(settings)
    index_state = client.index_state()
    if (
        not index_state.get("ready")
        or str(index_state.get("fingerprint", "")) != str(index_fingerprint)
    ):
        raise OpenSearchRefinedError(
            "The refined index changed before its primary exclusion set was collected."
        )
    story_ids = collect_all_refined_primary_story_ids(
        client=client,
        normalized_query=normalized_query,
        match_mode=match_mode,
    )
    return tuple(sorted(story_ids))


@st.cache_data(show_spinner=False)
def get_complete_refined_story_id_sets(
    normalized_query: str,
    match_mode: str,
    index_fingerprint: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load all displayed refined IDs and the primary traffic subset."""
    settings = load_opensearch_settings()
    if not settings.configured:
        raise OpenSearchRefinedError("OPENSEARCH_URL is not configured.")
    client = OpenSearchRefinedClient(settings)
    index_state = client.index_state()
    if (
        not index_state.get("ready")
        or str(index_state.get("fingerprint", "")) != str(index_fingerprint)
    ):
        raise OpenSearchRefinedError(
            "The refined index changed before its exclusion sets were collected."
        )
    all_story_ids, primary_story_ids = collect_all_refined_story_id_sets(
        client=client,
        normalized_query=normalized_query,
        match_mode=match_mode,
    )
    return tuple(sorted(all_story_ids)), tuple(sorted(primary_story_ids))


def show_refined_search_tab(
    keyword_query: str,
    match_mode: str,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> dict[str, object] | None:
    st.subheader("OpenSearch refined results")
    st.caption(
        "Ranked title search using exact phrase, exact keyword, known-alias, "
        "root-variant, and similar-spelling tiers."
    )

    settings = load_opensearch_settings()
    if not settings.configured:
        st.info(
            "OpenSearch is not configured. Start the local service and set "
            "OPENSEARCH_URL to enable this tab."
        )
        st.code(
            "$env:OPENSEARCH_URL='http://localhost:9200'\n"
            "$env:OPENSEARCH_VERIFY_CERTS='false'\n"
            "streamlit run app.py",
            language="powershell",
        )
        return

    client = OpenSearchRefinedClient(settings)
    try:
        connection = client.connection_info()
        index_state = client.index_state()
    except OpenSearchRefinedError as initial_error:
        if not settings.auto_start:
            st.error("The refined search tab could not connect to OpenSearch.")
            st.caption(str(initial_error))
            return
        try:
            with st.spinner("Starting local OpenSearch..."):
                start_local_opensearch(settings)
            st.rerun()
            connection = client.connection_info()
            index_state = client.index_state()
        except OpenSearchRefinedError as exc:
            st.error("The refined search tab could not start or connect to OpenSearch.")
            st.caption(str(exc))
            return

    status_cols = st.columns(3)
    status_cols[0].metric("Cluster", connection["cluster_name"])
    status_cols[1].metric("OpenSearch version", connection["version"])
    status_cols[2].metric(
        "Indexed titles",
        format_indian_count(index_state.get("document_count", 0)),
    )

    expected_fingerprint = title_corpus_fingerprint(title_summary)
    needs_sync = (
        not index_state.get("ready")
        or index_state.get("fingerprint") != expected_fingerprint
    )
    if needs_sync:
        st.warning(
            "The refined index is missing or does not match the current CSV data. "
            "Syncing creates a new versioned index and switches only the refined-search alias."
        )
    if st.button(
        "Sync OpenSearch refined index",
        key="sync_opensearch_refined_index",
        type="primary" if needs_sync else "secondary",
    ):
        try:
            with st.spinner("Indexing titles in OpenSearch..."):
                index_state = client.sync_title_index(title_summary)
            st.success(
                f"Indexed {int(index_state['document_count']):,} titles for refined search."
            )
            needs_sync = False
        except (OpenSearchRefinedError, ValueError) as exc:
            st.error("OpenSearch indexing could not be completed.")
            st.caption(str(exc))
            return

    if needs_sync:
        return
    if not keyword_query.strip():
        st.info("Enter a keyword or keyword group above to run refined search.")
        return

    normalized_query = " ".join(
        dict.fromkeys(simple_query_tokens(normalize_refined_alias_text(keyword_query)))
    )
    if not normalized_query:
        st.warning("The query contains no searchable title keywords.")
        return

    for ambiguity in find_ambiguous_aliases(keyword_query):
        st.warning(
            f"{ambiguity.alias} is ambiguous ({', '.join(ambiguity.candidates)}). "
            "Automatic alias expansion was skipped; enter the intended full name."
        )

    paging_state_key = build_tab_flow_state_key(
        "opensearch_refined_paging",
        normalized_query,
        match_mode,
        index_state.get("fingerprint", ""),
        index_state.get("physical_index", ""),
    )
    for existing_key in list(st.session_state):
        if (
            str(existing_key).startswith("opensearch_refined_paging_")
            and existing_key != paging_state_key
        ):
            del st.session_state[existing_key]
    if paging_state_key not in st.session_state:
        try:
            hits, total_hits, next_search_after = client.search_page(
                normalized_query=normalized_query,
                match_mode=match_mode,
                size=REFINED_SEARCH_INITIAL_LIMIT,
            )
        except OpenSearchRefinedError as exc:
            st.error("OpenSearch could not complete the refined query.")
            st.caption(str(exc))
            return
        st.session_state[paging_state_key] = {
            "hits": hits,
            "total_hits": total_hits,
            "next_search_after": next_search_after,
        }

    paging_state = st.session_state[paging_state_key]
    hits = list(paging_state.get("hits", []))
    total_hits = int(paging_state.get("total_hits", len(hits)))
    remaining_hits = max(0, total_hits - len(hits))

    if remaining_hits:
        st.warning(
            f"OpenSearch found {total_hits:,} results; this view shows "
            f"the top {len(hits):,}."
        )

    result = build_refined_search_result(
        hits=hits,
        total_hits=total_hits,
        story_months=story_months,
        title_summary=title_summary,
    )
    primary_analysis_result = build_refined_primary_analysis_result(result)
    matched_titles = result["matched_titles"]
    if not isinstance(matched_titles, pd.DataFrame) or matched_titles.empty:
        st.warning("No refined title matches were found.")
        return

    if remaining_hits:
        try:
            (
                complete_refined_story_ids,
                complete_primary_story_ids,
            ) = get_complete_refined_story_id_sets(
                normalized_query=normalized_query,
                match_mode=match_mode,
                index_fingerprint=str(index_state.get("fingerprint", "")),
            )
        except OpenSearchRefinedError as exc:
            st.error(
                "The complete primary-traffic exclusion set could not be built, so "
                "Related Stories is disabled for this query."
            )
            st.caption(str(exc))
            return
    else:
        primary_titles = primary_analysis_result["matched_title_summary"]
        complete_refined_story_ids = tuple(
            matched_titles["story_id"].astype(str).drop_duplicates().tolist()
        )
        complete_primary_story_ids = tuple(
            primary_titles["story_id"].astype(str).drop_duplicates().tolist()
        )
    primary_analysis_result["complete_refined_story_ids"] = complete_refined_story_ids
    primary_analysis_result["complete_primary_story_ids"] = complete_primary_story_ids

    tier_counts = matched_titles["refined_match_tier"].value_counts()
    tier_columns = st.columns(len(MATCH_TIER_ORDER))
    for column, tier in zip(tier_columns, MATCH_TIER_ORDER):
        column.metric(tier, format_indian_count(tier_counts.get(tier, 0)))

    primary_story_months = primary_analysis_result["matched_story_months"]
    primary_matched_titles = primary_analysis_result["matched_title_summary"]
    primary_title_count = primary_matched_titles["story_id"].nunique()
    summary_cols = st.columns(3)
    summary_cols[0].metric(
        "Primary titles (exact + alias + root)",
        format_indian_count(primary_title_count),
    )
    summary_cols[1].metric(
        "Primary traffic views",
        format_indian_count(primary_story_months["views"].sum()),
    )
    summary_cols[2].metric(
        "All retrieved titles",
        format_indian_count(matched_titles["story_id"].nunique()),
    )
    st.caption(
        "Primary traffic metrics include exact, known-alias, and root-variant matches. "
        "Similar-spelling matches are visible for review but do not change the primary totals."
    )

    monthly_summary = primary_analysis_result["monthly_summary"]
    if isinstance(monthly_summary, pd.DataFrame) and not monthly_summary.empty:
        st.subheader("Monthly primary-match traffic")
        chart_df = monthly_summary.copy()
        chart_df["month_label"] = pd.to_datetime(chart_df["month"]).dt.strftime("%b %Y")
        figure = px.bar(
            chart_df,
            x="month_label",
            y="views",
            text="titles_containing_keywords",
            labels={
                "month_label": "Month",
                "views": "Views",
                "titles_containing_keywords": "Titles",
            },
        )
        figure.update_xaxes(type="category")
        st.plotly_chart(
            figure,
            use_container_width=True,
            key="refined_search_monthly_chart",
        )

    st.subheader("Ranked refined matches")
    result_columns = [
        "refined_rank",
        "story_id",
        "page_title",
        "refined_match_tier",
        "opensearch_score",
        "total_views",
        "active_months",
        "first_month",
        "last_month",
    ]
    show_wrapped_title_table(
        format_editorial_table(select_existing_columns(matched_titles, result_columns)),
        scroll_after_rows=30,
        scroll_height=1600,
    )

    if remaining_hits:
        next_batch_size = min(REFINED_SEARCH_LOAD_INCREMENT, remaining_hits)
        if st.button(
            f"Load {next_batch_size:,} more results",
            key=f"load_more_{paging_state_key}",
        ):
            try:
                with st.spinner("Loading more refined results..."):
                    page_hits, refreshed_total, next_search_after = client.search_page(
                        normalized_query=normalized_query,
                        match_mode=match_mode,
                        size=next_batch_size,
                        search_after=paging_state.get("next_search_after"),
                    )
                paging_state["hits"] = hits + page_hits
                paging_state["total_hits"] = refreshed_total
                paging_state["next_search_after"] = next_search_after
                st.rerun()
            except OpenSearchRefinedError as exc:
                st.error("OpenSearch could not load the next result batch.")
                st.caption(str(exc))

    similar_titles = matched_titles.loc[
        matched_titles["refined_match_tier"] == "Similar spelling"
    ]
    if not similar_titles.empty:
        with st.expander("Review similar-spelling matches", expanded=False):
            show_wrapped_title_table(
                format_editorial_table(
                    select_existing_columns(similar_titles, result_columns)
                ),
                scroll_after_rows=20,
                scroll_height=900,
            )

    return primary_analysis_result


def get_bertopic_refined_exclusions(
    keyword_query: str, match_mode: str, title_summary: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    settings = load_opensearch_settings()
    if not settings.configured:
        raise OpenSearchRefinedError("Configure and sync OpenSearch refined search first.")
    normalized_query = " ".join(dict.fromkeys(
        simple_query_tokens(normalize_refined_alias_text(keyword_query))
    ))
    if not normalized_query:
        raise ValueError("Enter a query with searchable keywords.")
    client = OpenSearchRefinedClient(settings)
    state = client.index_state()
    fingerprint = title_corpus_fingerprint(title_summary)
    if not state.get("ready") or state.get("fingerprint") != fingerprint:
        raise OpenSearchRefinedError("Sync the refined index to the current title corpus first.")
    # Uncached collection protects against index switches and partial result pages.
    all_ids, primary_ids = collect_all_refined_story_id_sets(
        client, normalized_query=normalized_query, match_mode=match_mode,
    )
    final_state = client.index_state()
    if final_state.get("physical_index") != state.get("physical_index") or final_state.get("fingerprint") != fingerprint:
        raise OpenSearchRefinedError("The refined index changed during exclusion collection; retry the search.")
    return tuple(sorted(all_ids)), tuple(sorted(primary_ids))


def run_bertopic_refined_tab_flow(keyword_query, title_summary, excluded_story_ids,
                                  top_k, min_similarity):
    artifact = fit_bertopic_refined_model(title_summary)
    candidates, diagnostics = retrieve_bertopic_refined_candidates(
        keyword_query, title_summary, excluded_story_ids,
        top_k=top_k, min_similarity=min_similarity, artifact=artifact,
    )
    catalog = get_bertopic_refined_topic_catalog(artifact=artifact)
    return candidates, diagnostics, catalog


def show_bertopic_refined_tab(keyword_query, match_mode, title_summary):
    st.subheader("BERTopic refined")
    st.caption(
        "Discover articles in topics related to your keyword. Refined-search story IDs "
        "and titles containing any searched keyword or known alias token are excluded."
    )
    if not keyword_query.strip():
        st.info("Enter a keyword or keyword group to discover related topics.")
        return
    with st.expander("Topic selection settings"):
        top_k = st.slider("Maximum topics", 1, 30, 5, key="bertopic_refined_top_k")
        threshold = st.slider("Minimum topic similarity", 0.0, 1.0, 0.45, 0.01,
                              key="bertopic_refined_threshold")
        st.caption("The default threshold is a starting point and needs validation on editorial queries.")
    try:
        excluded_story_ids, _ = get_bertopic_refined_exclusions(keyword_query, match_mode, title_summary)
        fingerprint = refined_corpus_fingerprint(title_summary)
    except (OpenSearchRefinedError, ValueError) as exc:
        st.warning("BERTopic refined requires a complete, current refined-search exclusion set.")
        st.caption(str(exc))
        return
    state_key = build_tab_flow_state_key(
        "bertopic_refined_tab_flow", keyword_query.strip(), match_mode,
        excluded_story_ids, fingerprint, REFINED_MODEL_VERSION, top_k, threshold,
    )
    flow_state = get_tab_flow_state(state_key)
    future = flow_state.get("future")
    running = isinstance(future, Future) and not future.done()
    if st.button("Run BERTopic refined", key=f"run_{state_key}", type="primary", disabled=running):
        flow_state.update(result=None, error="")
        future = get_tab_flow_executor().submit(
            run_bertopic_refined_tab_flow, keyword_query, title_summary.copy(deep=True),
            excluded_story_ids, top_k, threshold,
        )
        flow_state["future"] = future
        running = True
    if isinstance(future, Future) and future.done():
        try:
            flow_state["result"] = future.result()
        except Exception as exc:
            flow_state["error"] = str(exc)
        flow_state["future"] = None
        running = False
    if running:
        show_tab_flow_progress(state_key,
            "BERTopic refined is running in the background. The first run fits and saves "
            "the corpus model; later queries reuse it. Other tabs remain available.")
    if flow_state.get("error"):
        st.warning("Unable to complete BERTopic refined analysis.")
        st.caption(flow_state["error"])
        return
    if flow_state.get("result") is None:
        if not running:
            st.info("Click Run BERTopic refined to discover topics for this query.")
        return
    candidates, diagnostics, catalog = flow_state["result"]
    columns = st.columns(3)
    columns[0].metric("Corpus topics", diagnostics["topic_count"])
    columns[1].metric("Selected topics", len(diagnostics["selected_topics"]))
    columns[2].metric("Related articles", len(candidates))
    st.caption(
        f"{diagnostics['removed_excluded_ids']:,} candidates removed because they appear in refined search. "
        f"{diagnostics['removed_keyword_titles']:,} candidates removed for containing the keyword or an alias token. "
        f"Corpus outlier ratio: {diagnostics['outlier_ratio']:.1%}."
    )
    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
    else:
        st.info("No articles survived topic selection and keyword filtering. Try another query or adjust topic selection settings.")
    with st.expander("Matched topics and filtering diagnostics"):
        st.json(diagnostics)
    with st.expander("Browse all topics"):
        st.dataframe(pd.DataFrame(catalog, columns=["topic_id", "label", "size"]),
                     use_container_width=True, hide_index=True)


def show_bertopic_tab(
    keyword_query: str,
    search_result: dict[str, pd.DataFrame | list[str]] | None,
    title_summary: pd.DataFrame,
    match_mode: str = "All keywords",
) -> None:
    st.subheader("BERTopic corpus topics")
    st.caption(
        "This tab shows BERTopic output before Gemini or relationship validation. "
        "Topic membership is thematic discovery evidence, not final story relevance."
    )
    if not keyword_query.strip():
        st.info("Enter a keyword or keyword group to inspect matching BERTopic clusters.")
        return

    # Never infer exclusions from a paginated/primary-only display result.
    try:
        excluded_story_ids, direct_story_ids = get_bertopic_refined_exclusions(
            keyword_query, match_mode, title_summary
        )
        training_config = get_bertopic_training_config()
    except (OpenSearchRefinedError, ValueError) as exc:
        st.warning("BERTopic requires a complete, current refined-search exclusion set.")
        st.caption(str(exc))
        return
    st.caption("All refined-search matches, including similar spelling, are excluded from results and title previews.")

    state_key = build_tab_flow_state_key(
        "bertopic_tab_flow",
        keyword_query.casefold().strip(),
        direct_story_ids,
        excluded_story_ids,
        match_mode,
        hashlib.sha256(pd.util.hash_pandas_object(title_summary.astype(str), index=True).values.tobytes()).hexdigest(),
        json.dumps(training_config, sort_keys=True),
        BERTOPIC_INDEX_VERSION,
        BERTOPIC_RETRIEVAL_CACHE_VERSION,
    )
    flow_state = get_tab_flow_state(state_key)
    future = flow_state.get("future")
    is_running = isinstance(future, Future) and not future.done()
    run_clicked = st.button(
        "Run BERTopic Analysis",
        key=f"run_{state_key}",
        type="primary",
        disabled=is_running,
        help="Run only the BERTopic discovery flow for the current search.",
    )
    if run_clicked:
        flow_state["result"] = None
        flow_state["error"] = ""
        flow_state["future"] = get_tab_flow_executor().submit(
            run_bertopic_tab_flow,
            keyword_query,
            title_summary.copy(deep=False),
            direct_story_ids,
            excluded_story_ids,
        )
        future = flow_state["future"]
        is_running = True

    if isinstance(future, Future) and future.done():
        try:
            flow_state["result"] = future.result()
            flow_state["error"] = ""
        except Exception as exc:
            flow_state["result"] = None
            flow_state["error"] = str(exc)
        flow_state["future"] = None
        future = None
        is_running = False

    if is_running:
        show_tab_flow_progress(
            state_key,
            "BERTopic analysis is running in the background. Other tabs remain available.",
        )

    error = str(flow_state.get("error", "")).strip()
    if error:
        st.warning("Unable to load the BERTopic index.")
        st.caption(error)
        return

    result = flow_state.get("result")
    if result is None:
        if not is_running:
            st.info("Click Run BERTopic Analysis to load topics and candidates for this query.")
        return
    candidates, diagnostics, catalog = result

    metric_columns = st.columns(5)
    metric_columns[0].metric("Corpus topics", int(diagnostics.get("topic_count", 0)))
    metric_columns[1].metric(
        "Selected topics", int(diagnostics.get("selected_topic_count", 0))
    )
    metric_columns[2].metric("Discovery candidates", len(candidates))
    metric_columns[3].metric("Outlier titles", int(diagnostics.get("outlier_count", 0)))
    metric_columns[4].metric(
        "Outliers rescued", int(diagnostics.get("outlier_rescue_count", 0))
    )
    if diagnostics.get("used_direct_title_context"):
        st.caption(
            "Topic selection contextualized the query with "
            f"{int(diagnostics.get('direct_context_title_count', 0)):,} direct-match "
            "title embeddings. Semantic-only topics must agree at both centroid and "
            "topic-label level."
        )
    query_prototypes = diagnostics.get("query_prototypes", [])
    if query_prototypes:
        st.caption(
            f"Built {len(query_prototypes):,} query-specific title groups from "
            f"{int(diagnostics.get('direct_context_title_count', 0)):,} direct matches. "
            f"They contributed {int(diagnostics.get('prototype_candidate_count', 0)):,} "
            "nearest-title candidates, including BERTopic outliers."
        )
        with st.expander("Direct-title groups used for discovery", expanded=False):
            st.dataframe(
                pd.DataFrame(query_prototypes),
                use_container_width=True,
                hide_index=True,
            )
    contextual_fallback_reason = str(
        diagnostics.get("contextual_query_fallback_reason", "")
    ).strip()
    if contextual_fallback_reason:
        st.caption(
            "Direct-title query context was unavailable; raw query embedding was used. "
            f"{contextual_fallback_reason}"
        )

    selected_topics = diagnostics.get("selected_topics", [])
    direct_topic_matches = diagnostics.get("direct_topic_matches", {})
    candidate_count_by_topic: dict[int, int] = {}
    candidate_rows = []
    for candidate in candidates:
        evidence = candidate.get("retrieval_evidence", [])
        primary = evidence[0] if isinstance(evidence, list) and evidence else {}
        relationship_id = str(primary.get("relationship_id", ""))
        try:
            topic_id = int(relationship_id.split(":", 1)[1])
        except (IndexError, ValueError):
            topic_id = -1
        candidate_count_by_topic[topic_id] = candidate_count_by_topic.get(topic_id, 0) + 1
        candidate_rows.append(
            {
                "story_id": str(candidate.get("story_id", "")),
                "page_title": str(candidate.get("page_title", "")),
                "topic_id": topic_id,
                "topic_label": str(primary.get("related_subject", "")),
                "query_topic_similarity": round(
                    float(primary.get("similarity", 0.0)) * 100, 2
                ),
                "topic_label_similarity": round(
                    float(primary.get("topic_label_similarity", 0.0)) * 100, 2
                ),
                "title_query_similarity": round(
                    float(primary.get("title_query_similarity", 0.0)) * 100, 2
                ),
                "prototype_similarity": round(
                    float(primary.get("prototype_similarity", 0.0)) * 100, 2
                ),
                "prototype_id": primary.get("prototype_id"),
                "anchor_titles": " | ".join(
                    str(item)
                    for item in primary.get("prototype_anchor_titles", [])
                ),
                "membership_probability": round(
                    float(primary.get("topic_probability", 0.0)) * 100, 2
                ),
                "candidate_relevance_score": round(
                    float(primary.get("candidate_relevance_score", 0.0)) * 100, 2
                ),
                "selection_route": str(primary.get("selection_route", "")),
                "total_views": int(float(candidate.get("total_views", 0) or 0)),
            }
        )

    st.markdown("#### Global topics selected as supporting routes")
    if not selected_topics:
        st.info(
            str(diagnostics.get("fallback_reason", "")).strip()
            or "No topic passed the configured similarity and direct-support thresholds."
        )
    else:
        selected_topic_rows = [
            {
                "topic_id": int(item.get("topic_id", -1)),
                "topic_label": str(item.get("label", "")),
                "query_topic_similarity": round(
                    float(item.get("similarity", 0.0)) * 100, 2
                ),
                "raw_query_similarity": round(
                    float(item.get("raw_query_similarity", 0.0)) * 100, 2
                ),
                "topic_label_similarity": round(
                    float(item.get("label_similarity", 0.0)) * 100, 2
                ),
                "selection_route": str(item.get("selection_route", "")),
                "supporting_direct_titles": int(
                    direct_topic_matches.get(str(item.get("topic_id", "")), 0)
                ),
                "candidates_contributed": candidate_count_by_topic.get(
                    int(item.get("topic_id", -1)), 0
                ),
            }
            for item in selected_topics
        ]
        st.dataframe(
            pd.DataFrame(selected_topic_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### BERTopic discovery candidates")
    if not candidate_rows:
        st.info("No non-direct titles were contributed by the selected topics.")
    else:
        show_wrapped_title_table(
            pd.DataFrame(candidate_rows).rename(
                columns={"story_id": "Story ID", "page_title": "Page title"}
            ),
            extra_wrapped_columns={
                "topic_label": 240,
                "anchor_titles": 420,
            },
            scroll_after_rows=TITLE_TABLE_SCROLL_AFTER_ROWS,
            scroll_height=TITLE_TABLE_100_ROW_HEIGHT,
        )

    with st.expander(f"Browse all {len(catalog):,} corpus topics", expanded=False):
        st.caption("Counts and representative titles below exclude refined matches. Unassigned titles remain visible for review.")
        st.dataframe(
            pd.DataFrame(catalog),
            use_container_width=True,
            hide_index=True,
            height=620,
        )


def show_all_related_stories_tab(
    keyword_query: str,
    match_mode: str,
    search_result: dict[str, pd.DataFrame | list[str]] | None,
    primary_traffic_story_ids: tuple[str, ...] | None,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> None:
    st.subheader("All Related Stories")
    st.caption(
        "Combines the completed AI-related and BERTopic result sets, removes duplicate "
        "Story IDs, and ranks the union with weighted reciprocal-rank fusion."
    )
    if not keyword_query.strip():
        st.info("Enter a keyword or keyword group to find all related stories.")
        return
    if primary_traffic_story_ids is None:
        st.warning(
            "The complete refined primary-traffic set is unavailable, so related-story "
            "processing is disabled to prevent overlap."
        )
        return

    matched_titles = pd.DataFrame(columns=["story_id", "page_title"])
    if search_result is not None:
        candidate_matches = search_result.get("matched_title_summary")
        if isinstance(candidate_matches, pd.DataFrame):
            matched_titles = candidate_matches
    legacy_direct_story_ids = tuple(
        matched_titles.get("story_id", pd.Series(dtype=str)).astype(str)
    )
    direct_story_ids = tuple(
        dict.fromkeys((*primary_traffic_story_ids, *legacy_direct_story_ids))
    )
    normalized_query = " ".join(keyword_query.casefold().split())
    state_key = build_tab_flow_state_key(
        "all_related_stories_tab_flow",
        normalized_query,
        match_mode,
        direct_story_ids,
        len(title_summary),
        GENERATIVE_RETRIEVAL_VERSION,
        BERTOPIC_INDEX_VERSION,
        BERTOPIC_RETRIEVAL_CACHE_VERSION,
        ALL_RELATED_RRF_K,
        ALL_RELATED_AI_WEIGHT,
        ALL_RELATED_BERTOPIC_WEIGHT,
    )
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "ai_future": None,
            "bertopic_future": None,
            "ai_candidate_pool": None,
            "bertopic_result": None,
            "ai_error": "",
            "bertopic_error": "",
            "ai_skipped": False,
            "attempted": False,
            "final_result": None,
            "summary": {},
            "show_all": False,
        }
    flow_state = st.session_state[state_key]
    ai_future = flow_state.get("ai_future")
    bertopic_future = flow_state.get("bertopic_future")
    is_running = (
        isinstance(ai_future, Future) and not ai_future.done()
    ) or (
        isinstance(bertopic_future, Future) and not bertopic_future.done()
    )

    run_clicked = st.button(
        "Run All Related Stories",
        key=f"run_{state_key}",
        type="primary",
        disabled=is_running,
        help="Run the existing AI-related and BERTopic flows and combine their final outputs.",
    )
    if run_clicked:
        flow_state.update(
            {
                "ai_future": None,
                "bertopic_future": None,
                "ai_candidate_pool": None,
                "bertopic_result": None,
                "ai_error": "",
                "bertopic_error": "",
                "ai_skipped": matched_titles.empty,
                "attempted": True,
                "final_result": None,
                "summary": {},
                "show_all": False,
            }
        )
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS.pop(state_key, None)
        executor = get_tab_flow_executor()
        if not matched_titles.empty:
            flow_state["ai_future"] = executor.submit(
                get_ai_related_candidate_pool,
                keyword_query=keyword_query,
                match_type=match_mode,
                story_months=story_months.copy(deep=False),
                title_summary=title_summary.copy(deep=False),
                matched_titles=matched_titles.copy(deep=False),
                direct_story_ids=direct_story_ids,
                force_profile_refresh=False,
                apply_strict_evidence_gate=False,
                _progress_key=state_key,
            )
        flow_state["bertopic_future"] = executor.submit(
            get_bertopic_view_data,
            keyword_query=keyword_query,
            title_summary=title_summary.copy(deep=False),
            direct_story_ids=direct_story_ids,
            index_version=BERTOPIC_INDEX_VERSION,
            retrieval_version=BERTOPIC_RETRIEVAL_CACHE_VERSION,
        )
        ai_future = flow_state.get("ai_future")
        bertopic_future = flow_state.get("bertopic_future")
        is_running = True

    if isinstance(ai_future, Future) and ai_future.done():
        try:
            flow_state["ai_candidate_pool"] = ai_future.result()
        except Exception as exc:
            flow_state["ai_error"] = str(exc)
        flow_state["ai_future"] = None
        ai_future = None
    if isinstance(bertopic_future, Future) and bertopic_future.done():
        try:
            flow_state["bertopic_result"] = bertopic_future.result()
        except Exception as exc:
            flow_state["bertopic_error"] = str(exc)
        flow_state["bertopic_future"] = None
        bertopic_future = None

    is_running = (
        isinstance(ai_future, Future) and not ai_future.done()
    ) or (
        isinstance(bertopic_future, Future) and not bertopic_future.done()
    )
    if is_running:
        show_all_related_progress(state_key)
        return

    if flow_state.get("attempted") and flow_state.get("final_result") is None:
        ai_candidate_pool = flow_state.get("ai_candidate_pool")
        ai_related_titles = pd.DataFrame()
        if isinstance(ai_candidate_pool, pd.DataFrame):
            ai_related_titles = build_related_titles_from_evaluations(
                candidate_pool=ai_candidate_pool,
                evaluations_by_story_id=ai_candidate_pool.attrs.get(
                    "preverified_evaluations", {}
                ),
                research_source_count=len(
                    ai_candidate_pool.attrs.get("grounding_sources", [])
                ),
            )
        bertopic_result = flow_state.get("bertopic_result")
        bertopic_candidates: list[dict[str, object]] = []
        bertopic_diagnostics: dict[str, object] = {}
        if isinstance(bertopic_result, tuple) and len(bertopic_result) == 2:
            bertopic_candidates, bertopic_diagnostics = bertopic_result

        combined = combine_all_related_results(
            ai_related_titles=ai_related_titles,
            bertopic_candidates=bertopic_candidates,
        )
        ai_count = int(ai_related_titles["story_id"].nunique()) if not ai_related_titles.empty else 0
        bertopic_count = len(
            {
                str(candidate.get("story_id", "")).strip()
                for candidate in bertopic_candidates
                if str(candidate.get("story_id", "")).strip()
            }
        )
        flow_state["final_result"] = combined
        flow_state["summary"] = {
            "ai_count": ai_count,
            "bertopic_count": bertopic_count,
            "combined_count": len(combined),
            "duplicate_count": max(0, ai_count + bertopic_count - len(combined)),
            "bertopic_fallback_reason": str(
                bertopic_diagnostics.get("fallback_reason", "")
            ).strip(),
        }
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS.pop(state_key, None)

    ai_error = str(flow_state.get("ai_error", "")).strip()
    bertopic_error = str(flow_state.get("bertopic_error", "")).strip()
    if flow_state.get("ai_skipped"):
        st.warning(
            "No direct-title matches were available, so the AI-related flow was skipped. "
            "The combined table contains BERTopic results only."
        )
    elif ai_error:
        st.warning("AI-related search was unavailable; showing available BERTopic results.")
        st.caption(ai_error)
    if bertopic_error:
        st.warning("BERTopic was unavailable; showing available AI-related results.")
        st.caption(bertopic_error)

    combined = flow_state.get("final_result")
    if not isinstance(combined, pd.DataFrame):
        st.info("Click Run All Related Stories to build the combined result.")
        return
    if combined.empty:
        fallback_reason = str(
            flow_state.get("summary", {}).get("bertopic_fallback_reason", "")
        ).strip()
        st.info("Neither flow returned related stories for this query.")
        if fallback_reason:
            st.caption(fallback_reason)
        return

    summary = flow_state.get("summary", {})
    st.caption(
        f"Combined {int(summary.get('ai_count', 0)):,} AI-related and "
        f"{int(summary.get('bertopic_count', 0)):,} BERTopic results into "
        f"{int(summary.get('combined_count', len(combined))):,} unique stories; "
        f"removed {int(summary.get('duplicate_count', 0)):,} duplicates. "
        f"Ranking uses weighted RRF (AI {ALL_RELATED_AI_WEIGHT:g}, "
        f"BERTopic {ALL_RELATED_BERTOPIC_WEIGHT:g}, k={ALL_RELATED_RRF_K}) "
        "with views as the final tie-breaker."
    )
    display_limit = len(combined) if flow_state.get("show_all") else min(
        RELATED_STORY_INITIAL_LIMIT, len(combined)
    )
    display_rows = combined.head(display_limit)[
        [
            "story_id",
            "page_title",
            "relationship_or_cluster",
            "total_views",
            "unified_rank_score",
        ]
    ]
    show_wrapped_title_table(
        format_editorial_table(display_rows),
        extra_wrapped_columns={"Relationship / cluster": 520},
        scroll_after_rows=TITLE_TABLE_SCROLL_AFTER_ROWS,
        scroll_height=TITLE_TABLE_100_ROW_HEIGHT,
    )
    if display_limit < len(combined):
        if st.button(
            f"Show {len(combined) - display_limit:,} remaining stories",
            key=f"show_all_{state_key}",
        ):
            flow_state["show_all"] = True
            st.rerun()


def run_source_based_tab_flow(
    *,
    keyword_query: str,
    title_summary: pd.DataFrame,
    direct_story_ids: tuple[str, ...],
    selected_wikidata_qid: str,
    include_wikipedia: bool,
    include_wordnet: bool | None,
    include_conceptnet: bool | None,
    include_gdelt: bool | None,
    force_refresh: bool,
    progress_key: str,
) -> dict[str, object]:
    """Run source extraction and deterministic title matching off the UI thread."""

    def update_progress(phase: str) -> None:
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS[progress_key] = {
                "phase": phase,
                "phase_started_at": time.perf_counter(),
            }

    return build_source_related_result(
        keyword_query=keyword_query,
        title_summary=title_summary,
        excluded_story_ids=direct_story_ids,
        selected_wikidata_qid=selected_wikidata_qid,
        include_wikipedia=include_wikipedia,
        include_wordnet=include_wordnet,
        include_conceptnet=include_conceptnet,
        include_gdelt=include_gdelt,
        force_refresh=force_refresh,
        progress_callback=update_progress,
    )


def show_source_based_related_stories_tab(
    *,
    keyword_query: str,
    search_result: dict[str, pd.DataFrame | list[str]] | None,
    primary_story_ids: tuple[str, ...] | None,
    title_summary: pd.DataFrame,
) -> None:
    st.subheader("Source-Based Related Stories")
    st.caption(
        "Experimental deterministic results for free-form Latin-script topic "
        "queries using multi-mention Wikidata resolution, automatically routed "
        "open sources, exact title evidence, and a coverage-aware corpus fallback. "
        "Confirmed primary matches from Search Results refined are excluded; "
        "similar-spelling review matches remain eligible. This flow "
        "does not call Gemini and does not require Neo4j."
    )
    if not keyword_query.strip():
        st.info("Enter a keyword or group of keywords to search the knowledge sources.")
        return

    normalized_query = " ".join(keyword_query.casefold().split())
    query_token = hashlib.sha1(normalized_query.encode("utf-8")).hexdigest()[:12]
    with st.expander("Source configuration", expanded=False):
        selected_wikidata_qid = st.text_input(
            "Wikidata QID override",
            value="",
            key=f"source_wikidata_qid_{query_token}",
            placeholder="Optional, for example Q312",
            help=(
                "Leave blank to score candidates from Wikidata and local "
                "corpus evidence. Enter a QID only when editorial review identifies "
                "the intended interpretation."
            ),
        ).strip()
        include_wikipedia = st.checkbox(
            "Use English Wikipedia as an optional candidate fallback",
            value=False,
            key=f"source_wikipedia_{query_token}",
            help=(
                "Disabled by default. Latin-script input does not imply English "
                "language or select English Wikipedia as the knowledge base."
            ),
        )
        source_mode_options = ("Automatic", "Enabled", "Disabled")
        source_mode_values = {"Automatic": None, "Enabled": True, "Disabled": False}
        wordnet_mode = st.selectbox(
            "English WordNet routing",
            source_mode_options,
            index=0,
            key=f"source_wordnet_{query_token}",
            help=(
                "Automatic uses WordNet only when the query is not resolved as a "
                "named entity and WordNet supplies lexical evidence."
            ),
        )
        conceptnet_mode = st.selectbox(
            "ConceptNet routing",
            source_mode_options,
            index=0,
            key=f"source_conceptnet_{query_token}",
            help=(
                "Automatic uses the current /c/en adapter only after an English "
                "lexical route is supported; named-entity queries are not expanded."
            ),
        )
        gdelt_mode = st.selectbox(
            "GDELT recent-news routing",
            source_mode_options,
            index=0,
            key=f"source_gdelt_{query_token}",
            help=(
                "Automatic attempts GDELT for this news-retrieval flow. Failures "
                "fall back to other sources and the local corpus."
            ),
        )
        include_wordnet = source_mode_values[wordnet_mode]
        include_conceptnet = source_mode_values[conceptnet_mode]
        include_gdelt = source_mode_values[gdelt_mode]
        st.code(
            "\n".join(
                [
                    f"Profile version: {SOURCE_PROFILE_SCHEMA_VERSION}",
                    f"Retrieval version: {SOURCE_RETRIEVAL_VERSION}",
                    f"HTTP User-Agent: {get_knowledge_source_user_agent()}",
                ]
            ),
            language=None,
        )

    legacy_direct_ids: tuple[str, ...] = ()
    if search_result is not None:
        matched = search_result.get("matched_title_summary")
        if isinstance(matched, pd.DataFrame) and "story_id" in matched.columns:
            legacy_direct_ids = tuple(matched["story_id"].astype(str).tolist())
    direct_story_ids = tuple(
        dict.fromkeys((*(primary_story_ids or ()), *legacy_direct_ids))
    )
    direct_fingerprint = hashlib.sha1(
        "|".join(direct_story_ids).encode("utf-8")
    ).hexdigest()[:12]
    state_key = build_tab_flow_state_key(
        "source_related_tab_flow",
        normalized_query,
        selected_wikidata_qid.upper(),
        include_wikipedia,
        include_wordnet,
        include_conceptnet,
        include_gdelt,
        direct_fingerprint,
        len(title_summary),
        SOURCE_PROFILE_SCHEMA_VERSION,
        SOURCE_RETRIEVAL_VERSION,
    )
    flow_state = get_tab_flow_state(state_key)
    future = flow_state.get("future")
    is_running = isinstance(future, Future) and not future.done()
    run_col, refresh_col = st.columns(2)
    with run_col:
        run_clicked = st.button(
            "Find Source-Based Stories",
            key=f"run_{state_key}",
            type="primary",
            disabled=is_running,
            width="stretch",
            help="Reuse a compatible source profile from the local SQLite cache.",
        )
    with refresh_col:
        refresh_clicked = st.button(
            "Refresh Sources & Run",
            key=f"refresh_{state_key}",
            disabled=is_running,
            width="stretch",
            help="Fetch a fresh profile from the enabled public sources.",
        )
    if run_clicked or refresh_clicked:
        flow_state["result"] = None
        flow_state["error"] = ""
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS.pop(state_key, None)
        flow_state["future"] = get_tab_flow_executor().submit(
            run_source_based_tab_flow,
            keyword_query=keyword_query,
            title_summary=title_summary.copy(deep=False),
            direct_story_ids=direct_story_ids,
            selected_wikidata_qid=selected_wikidata_qid,
            include_wikipedia=include_wikipedia,
            include_wordnet=include_wordnet,
            include_conceptnet=include_conceptnet,
            include_gdelt=include_gdelt,
            force_refresh=refresh_clicked,
            progress_key=state_key,
        )
        future = flow_state["future"]
        is_running = True

    if isinstance(future, Future) and future.done():
        try:
            flow_state["result"] = future.result()
            flow_state["error"] = ""
        except Exception as exc:
            flow_state["result"] = None
            flow_state["error"] = str(exc)
        flow_state["future"] = None
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS.pop(state_key, None)
        is_running = False

    if is_running:
        show_source_flow_progress(state_key)

    error = str(flow_state.get("error", "")).strip()
    if error:
        st.warning("Unable to build source-based related-story results.")
        st.caption(error)
        return

    result = flow_state.get("result")
    if not isinstance(result, dict):
        if not is_running:
            st.info("Click Find Source-Based Stories to start the source-only flow.")
        return

    process_timings = result.get("process_timings", [])
    if isinstance(process_timings, list):
        show_related_process_timings(
            process_timings,
            result.get("total_elapsed_seconds"),
        )
    profile = result.get("profile", {})
    diagnostics = result.get("diagnostics", {})
    stories = result.get("stories")
    if not isinstance(profile, dict) or not isinstance(diagnostics, dict):
        st.warning("The source flow returned an invalid result.")
        return
    if not isinstance(stories, pd.DataFrame):
        stories = pd.DataFrame()

    resolved_entities = profile.get("resolved_entities", [])
    if not isinstance(resolved_entities, list):
        resolved_entities = []
    resolved = profile.get("resolved_entity", {})
    if not isinstance(resolved, dict):
        resolved = {}
    resolved_label = str(resolved.get("label", "")).strip()
    resolved_qid = str(resolved.get("qid", "")).strip()
    resolved_description = str(resolved.get("description", "")).strip()
    entity_resolution = profile.get("entity_resolution", {})
    if not isinstance(entity_resolution, dict):
        entity_resolution = {}
    selected_score = float(entity_resolution.get("selected_score", 0.0) or 0.0)
    resolved_labels = [
        str(item.get("label", "")).strip()
        for item in resolved_entities
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    ]
    if len(resolved_labels) > 1:
        st.success(
            "Resolved query mentions: "
            + ", ".join(resolved_labels)
            + (f" · combined confidence {selected_score * 100:.1f}%" if selected_score else "")
        )
    elif resolved_label:
        st.success(
            f"Resolved as {resolved_label}"
            + (f" ({resolved_qid})" if resolved_qid else "")
            + (f" — {resolved_description}" if resolved_description else "")
            + (f" · confidence {selected_score * 100:.1f}%" if selected_score else "")
        )
    elif str(profile.get("resolution_status", "")).strip() in {
        "corpus_or_lexical_only",
        "primary_corpus_only",
    }:
        st.info(
            "No external entity passed the automatic resolution gate. "
            "The workflow continued with normalized local-corpus and lexical evidence."
        )
    if bool(profile.get("cache_hit")):
        st.caption("Used a compatible source profile from the local SQLite cache.")

    interpretations = profile.get("interpretations", [])
    if isinstance(interpretations, list) and interpretations:
        if not resolved_qid:
            st.warning(
                "External sources returned possible interpretations, but none was "
                "automatically selected. Review the scored alternatives and use a QID "
                "override only when the intended entity is clear."
            )
        elif len(interpretations) > 1:
            st.caption(
                "The selected interpretation passed both the confidence threshold and "
                "the lead-over-alternatives threshold."
            )
        with st.expander("View alternative query interpretations", expanded=False):
            st.dataframe(
                pd.DataFrame(interpretations),
                use_container_width=True,
                hide_index=True,
            )

    status_rows = []
    source_status = profile.get("source_status", {})
    if isinstance(source_status, dict):
        for source_name, status in source_status.items():
            if not isinstance(status, dict):
                continue
            status_rows.append(
                {
                    "Source": source_name,
                    "Status": status.get("status", ""),
                    "Details": status.get("detail", ""),
                }
            )
    if status_rows:
        st.caption(
            " · ".join(
                f"{row['Source']}: {str(row['Status']).replace('_', ' ').title()}"
                for row in status_rows
            )
        )
    warnings = profile.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            if str(warning).strip():
                st.warning(str(warning))

    relationship_count = int(diagnostics.get("relationship_count", 0))
    primary_overlap_count = int(diagnostics.get("primary_overlap_count", 0))
    metric_columns = st.columns(4)
    metric_columns[0].metric("Source relationships", relationship_count)
    metric_columns[1].metric("Related titles", len(stories))
    metric_columns[2].metric("Excluded primary/direct", primary_overlap_count)
    metric_columns[3].metric(
        "Elapsed",
        format_process_duration(result.get("total_elapsed_seconds", 0)),
    )
    density_fallback = profile.get("density_fallback", {})
    if isinstance(density_fallback, dict) and density_fallback.get("activated"):
        source_coverage = density_fallback.get("source_coverage", {})
        if not isinstance(source_coverage, dict):
            source_coverage = {}
        st.caption(
            "Coverage enrichment used: "
            f"{str(density_fallback.get('activation_reason', 'profile sparsity'))}; "
            f"{int(density_fallback.get('source_relationship_count', 0)):,} source "
            "relationships produced "
            f"{int(source_coverage.get('matched_non_primary_title_count', 0)):,} "
            "usable non-primary title matches before enrichment; "
            f"{int(density_fallback.get('corpus_relationship_count', 0)):,} local "
            "corpus associations were retained for corroboration only."
        )

    centrality_rejections = int(
        diagnostics.get("centrality_rejection_count", 0)
    )
    if centrality_rejections:
        st.caption(
            f"Precision gate rejected {centrality_rejections:,} title(s) where a "
            "related subject lacked sufficient independent evidence or adaptive "
            "title centrality."
        )
    semantic_centrality = diagnostics.get("semantic_centrality", {})
    if isinstance(semantic_centrality, dict):
        if semantic_centrality.get("applied"):
            st.caption(
                "Local semantic subject-centrality checked "
                f"{int(semantic_centrality.get('processed', 0)):,} evidence-qualified "
                f"candidate(s) and rejected "
                f"{int(semantic_centrality.get('rejected', 0)):,} incidental or "
                "type-incompatible title(s). "
                f"Recovered {int(semantic_centrality.get('recovered', 0)):,} "
                "evidence-backed candidate(s) after an all-rejected outcome."
            )
        elif str(semantic_centrality.get("reason", "")).strip():
            st.caption(str(semantic_centrality.get("reason", "")).strip())

    coverage_audit = profile.get("relationship_coverage_audit", {})
    if isinstance(coverage_audit, dict):
        covered = int(coverage_audit.get("covered_branch_count", 0))
        applicable = int(coverage_audit.get("applicable_branch_count", 0))
        if applicable:
            st.caption(
                "Research-taxonomy coverage: "
                f"{covered:,} of {applicable:,} applicable "
                f"{coverage_audit.get('primary_type', 'topic')} relationship families "
                "have structured or current-event evidence."
            )

    if stories.empty:
        st.info(
            str(diagnostics.get("reason", "")).strip()
            or "No deterministic source-based matches were found."
        )
    else:
        display_stories = stories.copy()
        display_stories["source_confidence"] = display_stories["confidence"].map(
            lambda value: f"{float(value) * 100:.1f}%"
        )
        display_stories["relationship_path"] = display_stories[
            "relationship_path"
        ].map(format_relationship_path)
        display_stories = display_stories.rename(
            columns={"related_subject": "source_related_subject"}
        )
        display_stories = select_existing_columns(
            display_stories,
            [
                "story_id",
                "page_title",
                "source_related_subject",
                "relationship",
                "relationship_family",
                "relationship_role",
                "relationship_hops",
                "relationship_path",
                "supporting_related_subjects",
                "supporting_relationship_count",
                "matched_title_evidence",
                "retrieval_methods",
                "source_names",
                "source_confidence",
                "result_tier",
                "result_scope",
                "matched_query_components",
                "query_component_coverage",
                "title_subject_centrality",
                "semantic_subject_score",
                "evidence_score",
                "semantic_recovery",
                "rrf_score",
                "total_views",
                "active_months",
                "first_month",
                "last_month",
            ],
        )
        show_wrapped_title_table(
            format_editorial_table(display_stories),
            extra_wrapped_columns={
                "Source relationship": 520,
                "Relationship path": 520,
                "All supporting subjects": 360,
                "Matched title evidence": 260,
            },
            scroll_after_rows=TITLE_TABLE_SCROLL_AFTER_ROWS,
            scroll_height=TITLE_TABLE_100_ROW_HEIGHT,
        )

    relationships = profile.get("relationships", [])
    if isinstance(relationships, list) and relationships:
        with st.expander("View extracted source relationships", expanded=False):
            relationship_rows = []
            for relationship in relationships:
                if not isinstance(relationship, dict):
                    continue
                relationship_rows.append(
                    {
                        "related_subject": relationship.get("related_subject", ""),
                        "predicate_label": relationship.get("predicate_label", ""),
                        "relationship_family": relationship.get(
                            "relationship_family", ""
                        ),
                        "relationship_class": relationship.get(
                            "relationship_class", ""
                        ),
                        "relationship_role": relationship.get("relationship_role", ""),
                        "property_specificity": relationship.get(
                            "property_specificity", ""
                        ),
                        "can_retrieve_standalone": relationship.get(
                            "can_retrieve_standalone", False
                        ),
                        "retrieval_policy": relationship.get(
                            "retrieval_policy", ""
                        ),
                        "relationship_hops": relationship.get("hop_count", 1),
                        "relationship_path": format_relationship_path(
                            relationship.get("relationship_path", [])
                        ),
                        "temporal_scope": relationship.get("temporal_scope", ""),
                        "source_names": ", ".join(
                            str(item)
                            for item in relationship.get("source_names", [])
                        ),
                        "source_evidence_urls": "\n".join(
                            str(item)
                            for item in relationship.get("evidence_urls", [])
                        ),
                        "corpus_evidence_story_ids": ", ".join(
                            str(item.get("story_id", ""))
                            for item in relationship.get(
                                "evidence_story_rows", []
                            )
                            if isinstance(item, dict)
                            and str(item.get("story_id", "")).strip()
                        ),
                        "factual_bridge": relationship.get("factual_bridge", ""),
                        "acceptance_condition": relationship.get(
                            "acceptance_condition", ""
                        ),
                        "rejection_rule": relationship.get("rejection_rule", ""),
                        "factual_status": relationship.get("factual_status", ""),
                        "editorial_persistence": relationship.get(
                            "editorial_persistence", ""
                        ),
                        "corpus_support": relationship.get("corpus_support", ""),
                        "corpus_pmi": relationship.get("corpus_pmi", ""),
                        "recency_weighted_support": relationship.get(
                            "recency_weighted_support", ""
                        ),
                    }
                )
            show_wrapped_title_table(
                format_editorial_table(pd.DataFrame(relationship_rows)),
                extra_wrapped_columns={
                    "Factual bridge": 520,
                    "Relationship path": 520,
                    "Source evidence URLs": 420,
                },
                scroll_after_rows=50,
                scroll_height=720,
            )
    if status_rows:
        with st.expander("View source diagnostics", expanded=False):
            st.dataframe(
                pd.DataFrame(status_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.json(diagnostics)
            if isinstance(coverage_audit, dict) and coverage_audit:
                st.markdown("#### Research-taxonomy coverage audit")
                branches = coverage_audit.get("branches", [])
                if isinstance(branches, list) and branches:
                    st.dataframe(
                        pd.DataFrame(branches),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.json(coverage_audit.get("wikidata_graph_expansion", {}))


def show_related_stories_tab(
    keyword_query: str,
    match_mode: str,
    search_result: dict[str, pd.DataFrame | list[str]] | None,
    primary_traffic_story_ids: tuple[str, ...] | None,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> None:
    if not keyword_query.strip():
        st.info("Enter a keyword or group of keywords to discover AI-related stories.")
        return

    if search_result is None:
        st.warning("No page titles match this keyword search, so related stories cannot be evaluated.")
        return
    if primary_traffic_story_ids is None:
        st.warning(
            "The complete refined primary-traffic set is unavailable, so Related Stories "
            "is disabled to prevent primary titles from reaching AI processing."
        )
        return

    matched_titles = search_result["matched_title_summary"]
    if matched_titles.empty:
        st.warning("No page titles match this keyword search, so related stories cannot be evaluated.")
        return

    heading_col, configuration_col = st.columns(
        [30, 1],
        vertical_alignment="center",
    )
    with heading_col:
        st.subheader("Related Stories")
    with configuration_col.popover(
        "​",
        type="tertiary",
        help="Show AI configuration",
        icon=":material/settings:",
    ):
        st.caption("AI configuration")
        st.code(
            "\n".join(
                [
                    f"Mode: {get_related_ai_mode()}",
                    f"Research model: {get_vertex_model_name('research')}",
                    f"Generative selector: {get_vertex_model_name('judge')}",
                    f"Embedding model: {get_relationship_embedding_model_name()}",
                    "BERTopic: isolated to the BERTopic tab",
                    f"Sweep batch size: {DEFAULT_GENERATIVE_RETRIEVAL_BATCH_SIZE}",
                ]
            ),
            language=None,
        )
    normalized_ai_query = " ".join(keyword_query.casefold().split())
    query_central_policy = st.checkbox(
        "Apply query-central relevance policy (comparison mode)",
        value=False,
        key=(
            "query_central_relevance_policy_v2_"
            f"{hashlib.sha1(normalized_ai_query.encode('utf-8')).hexdigest()[:12]}"
        ),
        help=(
            "Run an isolated stricter variant where the title must centrally concern "
            "the query or a specific, non-replaceable manifestation. Context-only "
            "geography and generic topics cannot retrieve or qualify on their own."
        ),
    )
    apply_strict_evidence_gate = st.checkbox(
        "Apply strict post-Gemini evidence filter",
        value=False,
        key=(
            "strict_post_gemini_evidence_gate_v2_"
            f"{hashlib.sha1(normalized_ai_query.encode('utf-8')).hexdigest()[:12]}"
        ),
        help=(
            "When disabled, every title accepted by Gemini is displayed. The old "
            "embedding thresholds are still calculated and shown for comparison."
        ),
    )
    legacy_direct_story_ids = tuple(matched_titles["story_id"].astype(str).tolist())
    direct_story_ids = tuple(
        dict.fromkeys((*primary_traffic_story_ids, *legacy_direct_story_ids))
    )
    state_key = build_tab_flow_state_key(
        "related_stories_tab_flow",
        normalized_ai_query,
        match_mode,
        direct_story_ids,
        len(title_summary),
        GENERATIVE_RETRIEVAL_VERSION,
        apply_strict_evidence_gate,
        query_central_policy,
    )
    flow_state = get_tab_flow_state(state_key)
    future = flow_state.get("future")
    is_running = isinstance(future, Future) and not future.done()
    run_col, refresh_col = st.columns(2)
    with run_col:
        run_clicked = st.button(
            "Run Related Stories",
            key=f"run_{state_key}",
            type="primary",
            disabled=is_running,
            help="Reuse the saved AI profile and verified related-story results.",
            width="stretch",
        )
    with refresh_col:
        refresh_clicked = st.button(
            "Refresh AI Research & Run",
            key=f"refresh_{state_key}",
            disabled=is_running,
            help=(
                "Perform fresh web-grounded relationship research once, then save and "
                "reuse the new result on normal runs."
            ),
            width="stretch",
        )
    if run_clicked or refresh_clicked:
        refresh_request_id = uuid.uuid4().hex if refresh_clicked else ""
        flow_state["result"] = None
        flow_state["error"] = ""
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS.pop(state_key, None)
        flow_state["future"] = get_tab_flow_executor().submit(
            get_ai_related_candidate_pool,
            keyword_query=keyword_query,
            match_type=match_mode,
            story_months=story_months.copy(deep=False),
            title_summary=title_summary.copy(deep=False),
            matched_titles=matched_titles.copy(deep=False),
            direct_story_ids=direct_story_ids,
            force_profile_refresh=refresh_clicked,
            apply_strict_evidence_gate=apply_strict_evidence_gate,
            query_central_policy=query_central_policy,
            refresh_request_id=refresh_request_id,
            _progress_key=state_key,
        )
        future = flow_state["future"]
        is_running = True

    if isinstance(future, Future) and future.done():
        try:
            flow_state["result"] = future.result()
            flow_state["error"] = ""
        except Exception as exc:
            flow_state["result"] = None
            flow_state["error"] = str(exc)
        flow_state["future"] = None
        with RELATED_FLOW_PROGRESS_LOCK:
            RELATED_FLOW_PROGRESS.pop(state_key, None)
        future = None
        is_running = False

    if is_running:
        show_tab_flow_progress(
            state_key,
            "Related Stories is running in the background. Other tabs remain available.",
        )

    error = str(flow_state.get("error", "")).strip()
    if error:
        st.warning("Unable to build the AI-related story candidate pool.")
        st.caption(error)
        return

    candidate_pool = flow_state.get("result")
    if candidate_pool is None:
        if not is_running:
            st.info("Click Run Related Stories to start the AI relationship flow for this query.")
        return

    process_timings = candidate_pool.attrs.get("process_timings", [])
    if isinstance(process_timings, list):
        show_related_process_timings(
            process_timings,
            candidate_pool.attrs.get("total_elapsed_seconds"),
        )

    if candidate_pool.empty:
        if candidate_pool.attrs.get("retrieval_fallback_reason"):
            st.warning("AI relationship retrieval could not be completed safely.")
            st.caption(str(candidate_pool.attrs.get("retrieval_fallback_reason")))
        else:
            screened_count = int(candidate_pool.attrs.get("screened_count", 0))
            raw_selected_count = int(candidate_pool.attrs.get("raw_selected_count", 0))
            evidence_rejected_count = int(
                candidate_pool.attrs.get("evidence_rejected_count", 0)
            )
            strict_evidence_gate_enabled = bool(
                candidate_pool.attrs.get("strict_evidence_gate_enabled", False)
            )
            gemini_excluded_count = int(
                candidate_pool.attrs.get("gemini_excluded_count", 0)
            )
            local_validation_rejected_count = int(
                candidate_pool.attrs.get("local_validation_rejected_count", 0)
            )
            unclassified_excluded_count = int(
                candidate_pool.attrs.get("unclassified_excluded_count", 0)
            )
            if screened_count:
                if (
                    strict_evidence_gate_enabled
                    and raw_selected_count
                    and evidence_rejected_count == raw_selected_count
                ):
                    st.warning(
                        f"Gemini selected {raw_selected_count:,} of {screened_count:,} candidates, "
                        "but all failed final relationship-evidence validation. Refreshing the "
                        "relationship research can rebuild an incomplete profile."
                    )
                else:
                    st.info(
                        f"Gemini and the local evidence validator screened "
                        f"{screened_count:,} candidates and selected no additional "
                        "related stories."
                    )
                    if (
                        gemini_excluded_count
                        or local_validation_rejected_count
                        or unclassified_excluded_count
                    ):
                        st.caption(
                            "Exclusion audit: "
                            f"Gemini omitted {gemini_excluded_count:,}; "
                            f"local validation rejected {local_validation_rejected_count:,}; "
                            f"other/legacy exclusions {unclassified_excluded_count:,}."
                        )
            else:
                st.info("No validated indirect related stories were available for this query.")
    else:
        profile_key = str(candidate_pool.attrs.get("profile_key", "")).strip()
        grounding_sources = candidate_pool.attrs.get("grounding_sources", [])
        web_search_queries = candidate_pool.attrs.get("web_search_queries", [])
        research_error = str(candidate_pool.attrs.get("research_error", "")).strip()
        preverified_evaluations = candidate_pool.attrs.get("preverified_evaluations", {})
        screened_count = int(candidate_pool.attrs.get("screened_count", len(candidate_pool)))
        corpus_count = int(candidate_pool.attrs.get("corpus_count", screened_count))
        sweep_batch_count = int(candidate_pool.attrs.get("sweep_batch_count", 0))
        retrieval_mode = str(candidate_pool.attrs.get("retrieval_mode", ""))
        relationship_count = int(candidate_pool.attrs.get("relationship_count", 0))
        route_count = int(candidate_pool.attrs.get("route_count", 0))
        strict_evidence_gate_enabled = bool(
            candidate_pool.attrs.get("strict_evidence_gate_enabled", False)
        )
        query_central_policy_enabled = bool(
            candidate_pool.attrs.get("query_central_policy_enabled", False)
        )
        would_fail_evidence_gate_count = int(
            candidate_pool.attrs.get("would_fail_evidence_gate_count", 0)
        )
        retrieval_fallback_reason = str(
            candidate_pool.attrs.get("retrieval_fallback_reason", "")
        ).strip()
        retrieval_excluded_primary_count = int(
            candidate_pool.attrs.get("retrieval_excluded_primary_count", 0)
        )
        retrieval_eligible_corpus_count = int(
            candidate_pool.attrs.get("retrieval_eligible_corpus_count", corpus_count)
        )
        state_key = (
            f"ai_related_incremental_{CANDIDATE_JUDGE_VERSION}_{profile_key}"
        )
        if state_key not in st.session_state:
            st.session_state[state_key] = {
                "cursor": len(candidate_pool),
                "evaluations": dict(preverified_evaluations),
                "target": RELATED_STORY_INITIAL_LIMIT,
                "show_all": False,
            }
        incremental_state = st.session_state[state_key]
        incremental_state["cursor"] = len(candidate_pool)
        incremental_state["evaluations"] = dict(preverified_evaluations)
        if not incremental_state["show_all"]:
            incremental_state["target"] = max(
                int(incremental_state["target"]),
                RELATED_STORY_INITIAL_LIMIT,
            )
        target_count = (
            len(candidate_pool)
            if incremental_state["show_all"]
            else int(incremental_state["target"])
        )

        evaluated_pool = candidate_pool.iloc[
            : int(incremental_state["cursor"])
        ].copy()
        related_titles = build_related_titles_from_evaluations(
            candidate_pool=evaluated_pool,
            evaluations_by_story_id=incremental_state["evaluations"],
            research_source_count=len(grounding_sources),
        )
        display_limit = (
            len(related_titles)
            if incremental_state["show_all"]
            else min(int(incremental_state["target"]), len(related_titles))
        )
        displayed_related_titles = (
            related_titles.sort_values(
                ["ai_relevance_level", "ai_confidence", "total_views"],
                ascending=[False, False, False],
            )
            .head(display_limit)
            .reset_index(drop=True)
        )
        candidate_pool_exhausted = int(incremental_state["cursor"]) >= len(candidate_pool)

        hidden_related_count = max(0, len(related_titles) - len(displayed_related_titles))
        st.caption(
            f"Showing {len(displayed_related_titles):,} of {len(related_titles):,} verified "
            f"AI-related stories ({hidden_related_count:,} currently hidden). "
            f"Relationship retrieval selected {screened_count:,} candidates from "
            f"{corpus_count:,} non-primary titles using {relationship_count:,} relationships "
            f"and {route_count:,} embedding routes. Gemini verified them in "
            f"{sweep_batch_count:,} batches and selected {len(related_titles):,}. "
            f"Combined corpus coverage: {len(set(direct_story_ids)) + len(displayed_related_titles):,} "
            "unique primary/direct or AI-related titles."
        )
        st.caption(
            f"All {retrieval_eligible_corpus_count:,} non-primary titles were eligible "
            f"for indexed retrieval. {retrieval_excluded_primary_count:,} primary/direct titles "
            "were excluded before per-route and global candidate limits were applied."
        )
        if strict_evidence_gate_enabled:
            st.caption(
                "Strict post-Gemini evidence filter is ON. "
                f"It removed {would_fail_evidence_gate_count:,} Gemini-accepted titles."
            )
        else:
            st.caption(
                "Strict post-Gemini evidence filter is OFF. "
                f"{would_fail_evidence_gate_count:,} displayed Gemini-accepted titles "
                "would have failed the old application gate."
            )
        if query_central_policy_enabled:
            st.caption(
                "Query-central comparison policy is ON. Context-only geography and "
                "generic-topic routes were query-anchored, and every accepted title "
                "had to provide non-replaceable title-level evidence."
            )
        if retrieval_mode == "bounded_retrieval_no_candidates" and retrieval_fallback_reason:
            st.warning(
                "No candidates passed the bounded relationship-embedding retrieval. "
                "The full corpus was not sent to Gemini."
            )
            st.caption(retrieval_fallback_reason)
        if research_error:
            st.warning(
                "Live web grounding was unavailable. Indirect AI relationships were "
                "disabled instead of falling back to ungrounded model knowledge."
            )
            st.caption(research_error)
        if grounding_sources or web_search_queries:
            with st.expander("View AI web-grounding provenance", expanded=False):
                if web_search_queries:
                    st.caption("Google Search queries used by Gemini")
                    st.write(web_search_queries)
                if grounding_sources:
                    st.caption("Web sources returned in Vertex grounding metadata")
                    st.dataframe(pd.DataFrame(grounding_sources), use_container_width=True)
        if displayed_related_titles.empty:
            st.info("No candidates evaluated so far were accepted as related stories.")
        else:
            show_wrapped_title_table(
                format_editorial_table(
                    select_existing_columns(
                        displayed_related_titles,
                        [
                            "story_id",
                            "page_title",
                            "ai_term",
                            "ai_category",
                            "ai_relationship",
                            "ai_confidence",
                            "ai_relevance_level",
                            "ai_relevance_type",
                            "ai_supporting_routes",
                            "ai_research_source_count",
                            "would_pass_evidence_gate",
                            "total_views",
                            "active_months",
                            "first_month",
                            "last_month",
                        ],
                    )
                ),
                extra_wrapped_columns={"AI relationship": 520},
                scroll_after_rows=TITLE_TABLE_SCROLL_AFTER_ROWS,
                scroll_height=TITLE_TABLE_100_ROW_HEIGHT,
            )
            audit_details = displayed_related_titles.loc[
                displayed_related_titles.get(
                    "ai_audit_reason", pd.Series(index=displayed_related_titles.index, dtype=str)
                )
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            ]
            if not audit_details.empty:
                with st.expander("View detailed AI relationship audit", expanded=False):
                    st.caption(
                        "The main AI relationship is optimized for quick scanning. "
                        "This table preserves Gemini's fuller evidence explanation."
                    )
                    show_wrapped_title_table(
                        format_editorial_table(
                            select_existing_columns(
                                audit_details,
                                [
                                    "story_id",
                                    "page_title",
                                    "ai_relationship",
                                    "ai_audit_reason",
                                ],
                            )
                        ),
                        extra_wrapped_columns={
                            "AI relationship": 360,
                            "AI audit reason": 620,
                        },
                        scroll_after_rows=TITLE_TABLE_SCROLL_AFTER_ROWS,
                        scroll_height=TITLE_TABLE_100_ROW_HEIGHT,
                    )

        if len(related_titles) > display_limit or not candidate_pool_exhausted:
            current_target = int(incremental_state["target"])
            next_target = current_target + RELATED_STORY_LOAD_INCREMENT
            if next_target < len(candidate_pool):
                if st.button(
                    f"Load up to {RELATED_STORY_LOAD_INCREMENT} more related stories",
                    key=f"load_ai_related_{next_target}_{profile_key}",
                ):
                    incremental_state["target"] = next_target
                    st.rerun()
            elif not incremental_state["show_all"]:
                if st.button(
                    f"Show {len(related_titles) - display_limit:,} remaining related stories",
                    key=f"load_ai_related_all_{profile_key}",
                ):
                    incremental_state["show_all"] = True
                    st.rerun()
        elif len(related_titles) < target_count:
            st.caption(
                f"All candidates have been evaluated; {len(related_titles):,} related stories were accepted."
            )


def load_wikidata_source_profile(query: str, settings: SourceSettings, *, refresh: bool) -> SourceProfile:
    """Resolve against Wikidata, downloading missing subjects and reusing saved facts."""
    return WikidataLocalClient(settings).build_profile(query, refresh=refresh, download_missing=True)


def wikidata_profile_has_policies(profile) -> bool:
    """Older Streamlit session objects can survive a dataclass schema change."""
    return isinstance(profile, SourceProfile) and all(
        isinstance(getattr(route, "retrieval_policy", None), RetrievalPolicy)
        for route in profile.routes
    )


def show_wikidata_source_related_tab(
    keyword_query: str,
    match_mode: str,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> None:
    st.subheader("Wikidata Related Stories")
    st.caption(
        "Find additional articles through up to two Wikidata relationship hops. Missing subjects are downloaded "
        "on search and saved for reuse. Results come from your title corpus and include historical views."
    )
    st.caption("Shared events and other bounded connections require supporting title context. "
               "A related name alone cannot qualify through those connections.")
    st.caption("When a name matches several entities, relationships from all matching entities are searched. "
               "Each article appears once, with its supporting subjects retained.")
    st.caption(
        "Confirmed Search Results refined matches and direct mentions of your subject are excluded. "
        "A shared word inside another related entity's full name remains eligible for identity checks. "
        "Similar-spelling review matches remain eligible when a Wikidata relationship supports them."
    )

    if not keyword_query.strip():
        st.info("Enter a subject name or keyword group above to find Wikidata-related stories.")
        return

    normalized_query = " ".join(
        dict.fromkeys(simple_query_tokens(normalize_refined_alias_text(keyword_query)))
    )
    if not normalized_query:
        st.warning("The query contains no searchable title keywords.")
        return

    opensearch_settings = load_opensearch_settings()
    if not opensearch_settings.configured:
        st.warning(
            "Configure OpenSearch to exclude all confirmed Search Results refined matches "
            "before retrieving Wikidata-related articles."
        )
        return

    try:
        refined_client = OpenSearchRefinedClient(opensearch_settings)
        index_state = refined_client.index_state()
    except OpenSearchRefinedError as exc:
        st.error("The complete Search Results refined exclusion set is unavailable.")
        st.caption(str(exc))
        return

    expected_fingerprint = title_corpus_fingerprint(title_summary)
    if (
        not index_state.get("ready")
        or str(index_state.get("fingerprint", "")) != expected_fingerprint
    ):
        st.warning(
            "Sync the OpenSearch refined index first. The source-based flow will not run "
            "against an incomplete or stale exclusion set."
        )
        return

    source_settings = load_source_settings()
    # Only these scalar columns feed the monthly traffic lookup. The enriched
    # frame also contains list-valued title_tokens, which pandas cannot hash.
    traffic_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(
            story_months.reindex(columns=["story_id", "month", "views"]), index=False,
        ).values.tobytes()
    ).hexdigest()
    state_key = build_tab_flow_state_key(
        "wikidata_source_related",
        SOURCE_RELATED_PIPELINE_VERSION,
        normalized_query,
        match_mode,
        expected_fingerprint,
        index_state.get("physical_index", ""),
        source_settings.cache_path,
        source_settings.api_url,
        traffic_fingerprint,
    )
    for existing_key in list(st.session_state):
        if (
            str(existing_key).startswith("wikidata_source_related_")
            and existing_key != state_key
        ):
            del st.session_state[existing_key]
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "profile": None,
            "results": None,
            "excluded_count": 0,
            "error": "",
        }
    state = st.session_state[state_key]
    saved_profile, saved_results = state.get("profile"), state.get("results")
    rebuild_saved_results = (
        saved_profile is not None and not wikidata_profile_has_policies(saved_profile)
    ) or (
        isinstance(saved_results, pd.DataFrame)
        and (saved_results.attrs.get("relationship_policy_version") != POLICY_VERSION
             or saved_results.attrs.get("retrieval_pipeline_version") != SOURCE_RELATED_PIPELINE_VERSION)
    )
    if rebuild_saved_results:
        # Never display/export results computed before the article-context gate.
        # Rebuild from source facts, not from the incomplete route objects.
        state.update({"profile": None, "results": None, "excluded_count": 0, "error": ""})
        st.info("Updating saved results to apply the current relationship checks.")

    with st.expander("Source service configuration", expanded=False):
        st.code(f"Local database: {source_settings.cache_path}\nWikidata update API: {source_settings.api_url}", language="text")
        st.caption("Optional overrides: WIKIDATA_LOCAL_DB, WIKIDATA_API_URL, SOURCE_RELATED_TIMEOUT_SECONDS.")
        st.caption(WikidataLocalClient(source_settings).cache_summary())
    refresh_wikidata = st.checkbox(
        "Refresh saved Wikidata facts (requires internet)",
        value=False,
        key=f"refresh_{state_key}",
        help="New subjects are downloaded automatically when you search. Select this to update an existing subject.",
    )

    if st.button(
        "Find Wikidata Related Stories",
        key=f"run_{state_key}",
        type="primary",
    ) or rebuild_saved_results:
        state.update(
            {"profile": None, "results": None, "excluded_count": 0, "error": ""}
        )
        try:
            with st.spinner(
                "Building sourced relationships and checking the remaining title corpus..."
            ):
                profile = load_wikidata_source_profile(
                    query=keyword_query, settings=source_settings, refresh=refresh_wikidata,
                )
                if not wikidata_profile_has_policies(profile):
                    raise SourceRelatedError(
                        "The running app has outdated relationship definitions. Restart the app and search again. "
                        "Saved Wikidata facts can be reused."
                    )
                results = find_wikidata_related_stories(
                    query=keyword_query, profile=profile,
                    title_summary=title_summary,
                    story_months=story_months,
                    match_mode=match_mode, index_fingerprint=expected_fingerprint,
                    primary_id_lookup=get_complete_refined_primary_story_ids,
                    identity_client=WikidataLocalClient(source_settings),
                    refresh_identity=refresh_wikidata,
                )
            state["profile"] = profile
            state["results"] = results
            state["excluded_count"] = results.attrs["excluded_count"]
        except (OpenSearchRefinedError, SourceRelatedError, ValueError, sqlite3.Error, OSError) as exc:
            state["error"] = str(exc)

    if state.get("error"):
        st.error("Wikidata related stories could not be built.")
        st.caption(str(state["error"]))
        return

    profile = state.get("profile")
    results = state.get("results")
    if not isinstance(profile, SourceProfile) or not isinstance(results, pd.DataFrame):
        st.info(
            "Click Find Wikidata Related Stories to search. Multiword names and concepts are resolved "
            "as complete names; all matching entities are searched. If none match, try a fuller name."
        )
        return

    summary_columns = st.columns(3)
    summary_columns[0].metric(
        "Direct stories excluded",
        format_indian_count(state.get("excluded_count", 0)),
    )
    summary_columns[1].metric(
        "Wikidata retrieval routes",
        format_indian_count(results.attrs.get("retrieval_route_count", 0)),
    )
    summary_columns[2].metric(
        "Related corpus stories",
        format_indian_count(len(results)),
    )

    source_status = "; ".join(
        f"{source}: {status}" for source, status in profile.source_status.items()
    )
    if source_status:
        st.caption(source_status)
    for warning in profile.warnings:
        st.warning(warning)
    if profile.diagnostics:
        with st.expander("Source connection details", expanded=False):
            for source, detail in profile.diagnostics.items():
                st.text(f"{source}: {detail}")

    st.caption(
        f"Relationship checks withheld {results.attrs.get('relationship_withheld_story_count', 0):,} stories; "
        f"identity checks withheld another {results.attrs.get('identity_withheld_story_count', 0):,}. "
        "Missing relationship context and ambiguous names are withheld."
    )
    if results.attrs.get("relationship_audit"):
        with st.expander("Inspect withheld relationship matches (up to 500)", expanded=False):
            st.caption("A rejected connection does not block a story supported by another accepted connection.")
            st.dataframe(pd.DataFrame(results.attrs["relationship_audit"]), hide_index=True, width="stretch")
    if results.attrs.get("identity_audit"):
        with st.expander("Inspect withheld identity matches (up to 500)", expanded=False):
            st.caption("These are rejected relationship matches. A story can still qualify through another independently accepted match.")
            st.dataframe(pd.DataFrame(results.attrs["identity_audit"]), hide_index=True, width="stretch")

    with st.expander("Inspect entity resolution and source routes", expanded=False):
        if profile.linked_entities:
            query_details = {item["qid"]: item for item in getattr(profile, "query_entities", [])}
            entity_rows = [
                {
                    "Mention": entity.anchor,
                    "Description": query_details.get(entity.uri.rsplit("/", 1)[-1], {}).get("description", ""),
                    "Wikidata entity": entity.uri,
                    "Resolved by": entity.resolution_source,

                }
                for entity in profile.linked_entities
            ]
            st.markdown("**Query entities**")
            st.dataframe(
                pd.DataFrame(entity_rows),
                use_container_width=True,
                hide_index=True,
            )

        route_rows = [
            {
                "Related subject": route.related_subject,
                "Query subject": getattr(route, "query_entity_label", ""),
                "Query subject description": getattr(route, "query_entity_description", ""),
                "Relationship type": route.relationship_type,
                "Why related": route.why_related,
                "Evidence source": route.source_name,
                "Evidence URL": route.evidence_url,
                "Source heuristic score": route.confidence,
                "Supporting mentions": route.supporting_mentions,
                "Used for retrieval": route.source_kind == "wikidata_direct" and policy.enabled,
                "Relationship family": policy.family,
                "Standalone subject allowed": policy.can_retrieve_standalone,
                "Required title evidence": policy.acceptance_condition,
                "Rejection rule": policy.rejection_rule,
                "Related entity Q-ID": getattr(route, "target_qid", ""),
                "Hops": getattr(route, "hop_count", 1),
                "Relationship path": getattr(route, "relationship_path", ""),
                "Path source URLs": " ; ".join(getattr(route, "path_evidence_urls", ())),
            }
            for route in profile.routes
            for policy in (getattr(route, "retrieval_policy", None) or RetrievalPolicy(),)
        ]
        if route_rows:
            st.markdown("**Permitted relationship routes**")
            st.dataframe(
                pd.DataFrame(route_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Evidence URL": st.column_config.LinkColumn("Evidence URL"),
                },
            )

    if results.empty:
        if not results.attrs.get("retrieval_route_count", 0):
            st.info("No eligible Wikidata relationships are available for this subject. "
                    "Classification links alone are insufficient for discovery; this is a knowledge-coverage gap.")
        elif not results.attrs.get("candidate_count", 0):
            st.info("Wikidata relationships were found, but their names did not match any remaining corpus titles "
                    "after direct-story exclusions. Inspect the resolved names and routes above.")
        else:
            st.info("Related names matched corpus titles, but all candidates lacked sufficient relationship context "
                    "or identity evidence. Inspect the withheld matches above.")
        st.caption(f"Confirmed refined stories excluded: {results.attrs.get('primary_excluded_count', 0):,}; "
                   f"additional direct-title exclusions: {results.attrs.get('lexical_excluded_count', 0):,}; "
                   f"remaining name-match candidates: {results.attrs.get('candidate_count', 0):,}.")
        if match_mode == "Any keyword":
            st.caption("Any keyword can classify relatives sharing a surname as primary results. "
                       "Use All keywords to search the complete name and rerun both tabs.")
        return

    st.subheader("Wikidata-related corpus titles")
    st.caption(
        "Each result passes the relationship's title-context policy and an entity identity check. "
        "These checks do not establish the contents of the full article. Scores are identity heuristics, "
        "not measured probabilities of editorial relevance."
    )
    display = results.rename(
        columns={
            "story_id": "Story ID",
            "page_title": "Page title",
            "total_views": "Views (total)",
            "highest_view_month": "Month with highest views",
            "highest_views": "Highest views number",
            "relationship_type": "Relationship type",
            "why_related": "Why related",
            "evidence_source": "Evidence source",
            "evidence_url": "Evidence URL",
            "related_subject": "Matched related subject",
            "relevance_score": "Relevance score",
            "supporting_evidence_count": "Supporting evidence count",
            "matched_name": "Matched name or alias",
            "related_entity_qid": "Related entity Q-ID",
            "identity_reason": "Identity evidence",
            "identity_score": "Identity score",
            "hop_count": "Hops",
            "relationship_path": "Relationship path",
            "path_entity_ids": "Path entity IDs",
            "path_evidence_urls": "Path source URLs",
            "relationship_family": "Relationship family",
            "can_retrieve_standalone": "Standalone subject allowed",
            "relationship_title_evidence": "Relationship title evidence",
            "acceptance_condition": "Required title evidence",
            "rejection_rule": "Rejection rule",
            "matched_query_entities": "Matching query subjects",
            "matched_query_entity_ids": "Matching query entity IDs",
            "supporting_relationships": "All supporting relationships",
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=900,
        column_config={
            "Evidence URL": st.column_config.LinkColumn("Evidence URL"),
            "Relevance score": st.column_config.ProgressColumn(
                "Relevance score", min_value=0, max_value=100, format="%d"
            ),
        },
    )
    st.download_button(
        "Download Wikidata results CSV", display.to_csv(index=False).encode("utf-8-sig"),
        file_name="wikidata_related_stories.csv", mime="text/csv", key=f"export_{state_key}",
    )


def main() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #c62828;
            border-color: #c62828;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #a91f1f;
            border-color: #a91f1f;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button[kind="primary"]:disabled {
            background-color: #c62828;
            border-color: #c62828;
            color: #ffffff;
            opacity: 0.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Keyword Title Traffic Lookup")
    st.caption("Search title keywords and see views from matching page titles.")

    if st.button("Refresh CSV data"):
        st.cache_data.clear()
        st.rerun()

    try:
        model = load_model(
            story_metadata_path=str(STORY_METADATA_PATH),
            story_metrics_path=str(STORY_METRICS_PATH),
            story_metadata_mtime=get_file_mtime(STORY_METADATA_PATH),
            story_metrics_mtime=get_file_mtime(STORY_METRICS_PATH),
            lookup_pipeline_version=LOOKUP_PIPELINE_VERSION,
        )
    except Exception as exc:
        st.error("Unable to load the local CSV data.")
        st.exception(exc)
        return

    try:
        story_categories = load_story_categories(
            category_path=str(CATEGORY_PATH),
            category_mtime=get_file_mtime(CATEGORY_PATH),
        )
    except Exception as exc:
        st.error("Unable to load the story category CSV data.")
        st.exception(exc)
        return

    try:
        story_word_counts = load_story_word_counts(
            word_count_path=str(WORD_COUNT_PATH),
            word_count_mtime=get_file_mtime(WORD_COUNT_PATH),
        )
    except Exception as exc:
        st.error("Unable to load the story word count CSV data.")
        st.exception(exc)
        return

    try:
        story_regions = load_story_regions(
            region_path=str(REGION_PATH),
            region_mtime=get_file_mtime(REGION_PATH),
        )
    except Exception as exc:
        st.error("Unable to load the story region CSV data.")
        st.exception(exc)
        return

    try:
        story_planned_trending = load_story_planned_trending(
            planned_trending_path=str(PLANNED_TRENDING_PATH),
            planned_trending_mtime=get_file_mtime(PLANNED_TRENDING_PATH),
        )
    except Exception as exc:
        st.error("Unable to load the planned/trending CSV data.")
        st.exception(exc)
        return

    try:
        story_audience_types = load_story_audience_types(
            audience_type_path=str(AUDIENCE_TYPE_PATH),
            audience_type_mtime=get_file_mtime(AUDIENCE_TYPE_PATH),
        )
    except Exception as exc:
        st.error("Unable to load the audience segment CSV data.")
        st.exception(exc)
        return

    monthly_totals = model["monthly_totals"]
    story_months = model["story_months"]
    title_summary = model["title_summary"]

    total_views = int(monthly_totals["monthly_total_views"].sum()) if not monthly_totals.empty else 0
    total_story_months = len(story_months)
    unique_titles = title_summary["story_id"].nunique() if not title_summary.empty else 0

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total page views", format_indian_count(total_views))
    metric_cols[1].metric("Monthly title records", format_indian_count(total_story_months))
    metric_cols[2].metric("Unique titles", format_indian_count(unique_titles))

    with st.expander("Monthly high-traffic cutoff audit", expanded=False):
        audit_df = monthly_totals.copy()
        if "threshold_value" in audit_df.columns:
            audit_df["threshold_value"] = audit_df["threshold_value"].round(2)
        show_editorial_dataframe(format_editorial_table(audit_df))

    st.divider()

    st.subheader("Search title keywords")
    search_cols = st.columns([2, 1])
    with search_cols[0]:
        keyword_query = st.text_input(
            "Keyword or keyword group",
            placeholder="Try: lok sabha election, amavasya, custom duty",
        )
    with search_cols[1]:
        match_mode_options = {
            "All keywords - Match titles containing every searched word": "All keywords",
            "Exact phrase - Match titles with the words in the same order": "Exact phrase",
            "Any keyword - Match titles containing at least one searched word": "Any keyword",
        }
        selected_match_mode = st.selectbox(
            "Match mode",
            options=list(match_mode_options.keys()),
            help=(
                "Choose how strictly the searched keyword or keyword group should match page titles."
            ),
        )
        match_mode = match_mode_options[selected_match_mode]
    search_match_mode = "Exact cleaned phrase" if match_mode == "Exact phrase" else match_mode
    match_mode_notes = {
        "All keywords": "Best when you want titles that contain all searched words, even if the words appear in different parts of the title.",
        "Exact phrase": "Best when word order matters, such as a specific phrase, person, event, or topic name.",
        "Any keyword": "Best for a broader search where any one of the searched words can qualify a title.",
    }
    st.caption(match_mode_notes[match_mode])

    search_result = None
    if keyword_query.strip():
        search_result = search_titles_by_keywords(
            keyword_query=keyword_query,
            story_months=story_months,
            title_summary=title_summary,
            match_mode=search_match_mode,
        )
        if search_result["matched_title_summary"].empty:
            search_result = None

    refined_search_tab, source_related_tab, recent_stories_tab, dashboard_tab, bertopic_tab, bertopic_refined_tab, miscellaneous_tab, embeddings_tab = st.tabs(
        [
            "Search Results refined",
            "Wikidata",
            "Recent Stories",
            "Dashboard",
            "BERTopic",
            "BERTopic refined",
            "Miscellaneous",
            "Embeddings",
        ]
    )
    refined_primary_search_result = None
    with recent_stories_tab:
        show_recent_stories_tab(keyword_query)
    with refined_search_tab:
        refined_primary_search_result = show_refined_search_tab(
            keyword_query=keyword_query,
            match_mode=search_match_mode,
            story_months=story_months,
            title_summary=title_summary,
        )
    with dashboard_tab:
        show_dashboard_tab(
            keyword_query=keyword_query,
            search_result=refined_primary_search_result,
            story_categories=story_categories,
            story_word_counts=story_word_counts,
            story_planned_trending=story_planned_trending,
            story_regions=story_regions,
            story_audience_types=story_audience_types,
        )
    with bertopic_tab:
        show_bertopic_tab(
            keyword_query=keyword_query,
            search_result=search_result,
            title_summary=title_summary,
            match_mode=search_match_mode,
        )
    with bertopic_refined_tab:
        show_bertopic_refined_tab(keyword_query, search_match_mode, title_summary)
    with source_related_tab:
        show_wikidata_source_related_tab(
            keyword_query=keyword_query,
            match_mode=search_match_mode,
            story_months=story_months,
            title_summary=title_summary,
        )


    with miscellaneous_tab:
        show_miscellaneous_tab(story_months)

    with embeddings_tab:
        show_embeddings_tab(title_summary)


if __name__ == "__main__":
    main()
