"""Low-latency relationship-aware title search, isolated from existing tabs."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import simple_query_tokens
from src.opensearch_refined import (
    OpenSearchRefinedClient,
    OpenSearchRefinedError,
    load_opensearch_settings,
    start_local_opensearch,
    title_corpus_fingerprint,
)
from src.opensearch_semantic import (
    OpenSearchSemanticClient,
    OpenSearchSemanticError,
    collect_all_refined_story_ids,
    load_opensearch_semantic_settings,
    semantic_corpus_fingerprint,
)
from src.relationship_embeddings import extract_relationship_routes
from src.relationship_profile_store import load_latest_relationship_profile
from src.search_aliases import normalize_refined_alias_text
from src.semantic_related import (
    build_fast_relationship_titles,
    build_relationship_performance_result,
)


SEMANTIC_INITIAL_DISPLAY_LIMIT = 1000
SEMANTIC_DISPLAY_INCREMENT = 1000
RELATIONSHIP_SEMANTIC_FLOW_VERSION = "2026-08-21-evidence-bound-profile-v5"
# The cached relationship graph is already quality-gated.  Do not truncate it
# before retrieval: for broad entities, high-value conflict or institution
# relationships can rank below the first few generic profile relationships.
FAST_MAX_RELATIONSHIPS: int | None = None
FAST_MAX_TEXTS_PER_RELATIONSHIP = 3
FAST_MAX_ROUTES = 150
QUERY_PROFILE_DB_PATH = Path(__file__).resolve().parents[1] / "query_profiles.db"


def show_semantic_search_tab(
    keyword_query: str,
    match_mode: str,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
    *,
    relationship_profile_refresher: Callable[..., dict[str, object]],
    relationship_profile_version: str = "",
) -> None:
    st.subheader("Fast relationship-aware title performance")
    st.caption(
        "Normal searches use a cached factual relationship graph and OpenSearch "
        "vector retrieval. Gemini runs only when you explicitly build or refresh "
        "that graph."
    )

    if not keyword_query.strip():
        st.info("Enter a keyword or keyword group above to find related titles.")
        return

    normalized_query = " ".join(
        dict.fromkeys(simple_query_tokens(normalize_refined_alias_text(keyword_query)))
    )
    if not normalized_query:
        st.warning("The query contains no searchable title keywords.")
        return

    refined_settings = load_opensearch_settings()
    semantic_settings = load_opensearch_semantic_settings()
    if not refined_settings.configured or not semantic_settings.configured:
        st.info("Configure OPENSEARCH_URL to enable relationship-aware search.")
        return

    refined_client = OpenSearchRefinedClient(refined_settings)
    semantic_client = OpenSearchSemanticClient(semantic_settings)
    try:
        connection = refined_client.connection_info()
        refined_index_state = refined_client.index_state()
        semantic_index_state = semantic_client.index_state()
    except (OpenSearchRefinedError, OpenSearchSemanticError) as initial_error:
        if not refined_settings.auto_start:
            st.error("The relationship-aware tab could not connect to OpenSearch.")
            st.caption(str(initial_error))
            return
        try:
            with st.spinner("Starting local OpenSearch..."):
                start_local_opensearch(refined_settings)
            st.rerun()
            connection = refined_client.connection_info()
            refined_index_state = refined_client.index_state()
            semantic_index_state = semantic_client.index_state()
        except (OpenSearchRefinedError, OpenSearchSemanticError) as exc:
            st.error("The relationship-aware tab could not start OpenSearch.")
            st.caption(str(exc))
            return

    status_columns = st.columns(3)
    status_columns[0].metric("Cluster", connection["cluster_name"])
    status_columns[1].metric("OpenSearch version", connection["version"])
    status_columns[2].metric(
        "Related titles indexed",
        f"{int(semantic_index_state.get('document_count', 0)):,}",
    )

    expected_refined_fingerprint = title_corpus_fingerprint(title_summary)
    if (
        not refined_index_state.get("ready")
        or refined_index_state.get("fingerprint") != expected_refined_fingerprint
    ):
        st.warning(
            "Sync the Search Results refined index first. Its complete result set "
            "is the mandatory exclusion list."
        )
        return

    expected_semantic_fingerprint = semantic_corpus_fingerprint(
        title_summary,
        semantic_settings.embedding_model,
    )
    semantic_needs_sync = (
        not semantic_index_state.get("ready")
        or semantic_index_state.get("fingerprint") != expected_semantic_fingerprint
        or semantic_index_state.get("embedding_model")
        != semantic_settings.embedding_model
    )
    if semantic_needs_sync:
        st.warning(
            "The isolated OpenSearch related-title index is missing or stale. Sync it once; "
            "normal relationship searches reuse it."
        )
    if st.button(
        "Sync relationship title index",
        key="sync_fast_relationship_vector_index",
        type="primary" if semantic_needs_sync else "secondary",
    ):
        try:
            with st.spinner("Syncing reusable titles to OpenSearch..."):
                semantic_index_state = semantic_client.sync_title_index(title_summary)
            semantic_needs_sync = False
            st.success(
                f"Indexed {int(semantic_index_state['document_count']):,} titles."
            )
        except (OpenSearchSemanticError, RuntimeError, ValueError) as exc:
            st.error("The relationship vector index could not be synced.")
            st.caption(str(exc))
            return
    if semantic_needs_sync:
        return

    cached_profile = load_latest_relationship_profile(
        QUERY_PROFILE_DB_PATH,
        keyword_query,
        preferred_match_type=match_mode,
        required_profile_version=relationship_profile_version,
    )
    if cached_profile is None:
        st.info(
            "No cached relationship graph exists for this query. Build it once with "
            "Gemini; subsequent normal searches will use only OpenSearch."
        )
    else:
        _, profile_metadata = cached_profile
        updated_at = profile_metadata.get("updated_at", "")
        st.caption(
            "Cached relationship graph available"
            + (f" · updated {updated_at}" if updated_at else "")
            + "."
        )

    state_key = _semantic_state_key(
        RELATIONSHIP_SEMANTIC_FLOW_VERSION,
        normalized_query,
        match_mode,
        refined_index_state.get("fingerprint", ""),
        semantic_index_state.get("fingerprint", ""),
    )
    run_col, refresh_col = st.columns(2)
    with run_col:
        run_clicked = st.button(
            "Fast OpenSearch relationship search",
            key=f"run_{state_key}",
            type="primary",
            disabled=cached_profile is None,
            help="Uses no Gemini calls and reuses the saved relationship graph.",
            width="stretch",
        )
    with refresh_col:
        refresh_clicked = st.button(
            "Build/refresh relationship graph",
            key=f"refresh_{state_key}",
            help=(
                "Explicit slow path: Gemini and Google grounding rebuild the graph "
                "once, then OpenSearch runs the title search."
            ),
            width="stretch",
        )

    if run_clicked or refresh_clicked:
        started_at = time.perf_counter()
        try:
            with st.spinner(
                "Refreshing the relationship graph with Gemini..."
                if refresh_clicked
                else "Searching cached relationship routes in OpenSearch..."
            ):
                excluded_story_ids = collect_all_refined_story_ids(
                    refined_client,
                    normalized_query,
                    match_mode,
                    page_size=10_000,
                )
                profile_source = "cache"
                if refresh_clicked:
                    matched_title_context = tuple(
                        title_summary.loc[
                            title_summary["story_id"]
                            .astype(str)
                            .isin(excluded_story_ids),
                            "page_title",
                        ]
                        .dropna()
                        .astype(str)
                        .head(120)
                        .tolist()
                    )
                    grounded_research = relationship_profile_refresher(
                        keyword_query=keyword_query,
                        match_type=match_mode,
                        matched_title_context=matched_title_context,
                        refresh_request_id=uuid.uuid4().hex,
                    )
                    profile_source = "refreshed"
                else:
                    if cached_profile is None:
                        raise ValueError(
                            "No cached relationship graph is available for this query."
                        )
                    grounded_research = cached_profile[0]

                routes = extract_relationship_routes(
                    keyword_query=keyword_query,
                    grounded_research=grounded_research,
                    max_relationships=FAST_MAX_RELATIONSHIPS,
                    max_texts_per_relationship=FAST_MAX_TEXTS_PER_RELATIONSHIP,
                    # Anchor broad country, place, and generic-topic routes to
                    # the query so a bare endpoint does not retrieve arbitrary
                    # stories about that endpoint.
                    query_central_policy=True,
                )
                if not routes:
                    raise ValueError(
                        "The relationship graph contains no safe retrieval routes."
                    )
                candidates, retrieval_diagnostics = (
                    semantic_client.search_relationship_routes(
                        routes,
                        excluded_story_ids=excluded_story_ids,
                        max_routes=FAST_MAX_ROUTES,
                    )
                )
                related_titles = build_fast_relationship_titles(candidates)
                result = build_relationship_performance_result(
                    related_titles,
                    excluded_story_ids,
                    story_months,
                )
                elapsed_seconds = time.perf_counter() - started_at
                diagnostics = {
                    **retrieval_diagnostics,
                    "relationship_count": len(
                        {route.relationship_id for route in routes}
                    ),
                    "profile_source": profile_source,
                    "elapsed_seconds": elapsed_seconds,
                    "accepted_count": len(related_titles),
                }
            st.session_state[state_key] = {
                "result": result,
                "diagnostics": diagnostics,
                "display_limit": SEMANTIC_INITIAL_DISPLAY_LIMIT,
            }
        except (
            OpenSearchRefinedError,
            OpenSearchSemanticError,
            RuntimeError,
            ValueError,
        ) as exc:
            st.error("Relationship-aware title search could not be completed.")
            st.caption(str(exc))
            return
        except Exception as exc:
            st.error("The relationship graph could not be built or searched.")
            st.caption(str(exc))
            return

    semantic_state = st.session_state.get(state_key)
    if not isinstance(semantic_state, dict) or "result" not in semantic_state:
        st.info(
            "Use Fast OpenSearch relationship search for the low-latency path. "
            "Build/refresh is needed only when the cached graph is absent or outdated."
        )
        return

    result = semantic_state["result"]
    diagnostics = semantic_state.get("diagnostics", {})
    matched_titles = result.get("matched_titles")
    if not isinstance(matched_titles, pd.DataFrame) or matched_titles.empty:
        st.info(
            "No titles passed the relationship evidence checks after all Refined-tab "
            "matches were removed."
        )
        st.caption(
            f"OpenSearch candidates: {int(diagnostics.get('candidate_count', 0)):,}; "
            f"relationship routes: {int(diagnostics.get('route_count', 0)):,}; "
            f"elapsed: {float(diagnostics.get('elapsed_seconds', 0.0)):.2f}s."
        )
        return

    matched_story_months = result["matched_story_months"]
    related_title_count = int(matched_titles["story_id"].nunique())
    related_views = int(matched_story_months["views"].sum())
    average_views = float(matched_titles["total_views"].mean())
    median_views = float(matched_titles["total_views"].median())
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Evidence-checked related titles", f"{related_title_count:,}"
    )
    metric_columns[1].metric("Related traffic views", f"{related_views:,}")
    metric_columns[2].metric("Average views per title", f"{average_views:,.0f}")
    metric_columns[3].metric("Median views per title", f"{median_views:,.0f}")
    st.caption(
        f"Fast path completed in {float(diagnostics.get('elapsed_seconds', 0.0)):.2f}s "
        f"using {int(diagnostics.get('relationship_count', 0)):,} cached relationships "
        f"and {int(diagnostics.get('route_count', 0)):,} OpenSearch routes. "
        f"Accepted {int(diagnostics.get('accepted_count', 0)):,} of "
        f"{int(diagnostics.get('candidate_count', 0)):,} candidates. Displayed "
        "Refined-tab overlap is zero."
    )

    monthly_summary = result.get("monthly_summary")
    if isinstance(monthly_summary, pd.DataFrame) and not monthly_summary.empty:
        st.subheader("Monthly related-title traffic")
        chart_data = monthly_summary.copy()
        chart_data["month_label"] = pd.to_datetime(chart_data["month"]).dt.strftime(
            "%b %Y"
        )
        figure = px.bar(
            chart_data,
            x="month_label",
            y="views",
            text="related_titles_without_query",
            labels={
                "month_label": "Month",
                "views": "Views",
                "related_titles_without_query": "Related titles",
            },
        )
        figure.update_xaxes(type="category")
        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"semantic_monthly_chart_{state_key}",
        )

    st.subheader("Ranked relationship-aware titles")
    display_limit = min(
        int(semantic_state.get("display_limit", SEMANTIC_INITIAL_DISPLAY_LIMIT)),
        len(matched_titles),
    )
    display_columns = [
        "relationship_rank",
        "story_id",
        "page_title",
        "ai_term",
        "ai_category",
        "ai_relationship",
        "title_relationship_evidence",
        "validation_status",
        "relationship_semantic_score",
        "ai_confidence",
        "total_views",
        "active_months",
        "first_month",
        "last_month",
    ]
    display_table = matched_titles[
        [column for column in display_columns if column in matched_titles.columns]
    ].head(display_limit)
    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True,
        height=1600,
        column_config={
            "relationship_rank": "Relationship rank",
            "story_id": "Story ID",
            "page_title": st.column_config.TextColumn("Page title", width="large"),
            "ai_term": st.column_config.TextColumn("Related subject", width="medium"),
            "ai_category": "Relationship class",
            "ai_relationship": st.column_config.TextColumn(
                "Factual bridge", width="large"
            ),
            "title_relationship_evidence": "Title evidence",
            "validation_status": "Evidence decision",
            "relationship_semantic_score": st.column_config.NumberColumn(
                "OpenSearch route score", format="%.4f"
            ),
            "ai_confidence": st.column_config.NumberColumn(
                "Relationship score", format="%.2f"
            ),
            "total_views": st.column_config.NumberColumn("Total views", format="%d"),
            "active_months": "Active months",
            "first_month": "First month",
            "last_month": "Last month",
        },
    )

    remaining_titles = len(matched_titles) - display_limit
    if remaining_titles > 0:
        next_batch = min(SEMANTIC_DISPLAY_INCREMENT, remaining_titles)
        if st.button(
            f"Load {next_batch:,} more related results",
            key=f"load_more_{state_key}_{display_limit}",
        ):
            semantic_state["display_limit"] = display_limit + next_batch
            st.rerun()


def _semantic_state_key(*parts: object) -> str:
    serialized = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
    return f"fast_relationship_result_{digest}"
