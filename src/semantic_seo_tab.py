"""Streamlit UI for the isolated Semantic SEO experimental pipeline."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import SERVICE_ACCOUNT_PATH
from src.opensearch_refined import (
    OpenSearchRefinedClient,
    OpenSearchRefinedError,
    load_opensearch_settings,
    start_local_opensearch,
    title_corpus_fingerprint,
)
from src.opensearch_semantic import (
    OpenSearchSemanticError,
    collect_all_refined_story_ids,
    semantic_corpus_fingerprint,
)
from src.relationship_embeddings import extract_relationship_routes
from src.semantic_seo_models import (
    load_semantic_seo_model_settings,
    local_model_status,
)
from src.semantic_seo_pipeline import (
    DEFAULT_MAX_GEMINI_CANDIDATES,
    SEMANTIC_SEO_PIPELINE_VERSION,
    SemanticSEORunOptions,
    build_direct_query_route,
    build_semantic_seo_client,
    merge_gemini_validation,
    run_local_semantic_seo_stages,
    semantic_seo_result_frame,
)
from src.semantic_seo_routes import (
    SemanticSEORouteError,
    discover_verified_semantic_routes,
    load_semantic_route_profile,
    save_semantic_route_profile,
)
from src.vertex_related import (
    get_vertex_location,
    get_vertex_model_name,
    select_titles_generatively_with_vertex,
)


SEMANTIC_SEO_DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "semantic_seo"
    / "semantic_seo_profiles.db"
)
SEMANTIC_SEO_DISPLAY_LIMIT = 500


def show_semantic_seo_tab(
    keyword_query: str,
    match_mode: str,
    story_months: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> None:
    st.subheader("Semantic SEO Lab")
    st.caption(
        "Title-only, ingestion-first semantic retrieval. GLiNER entities and BGE-M3 "
        "vectors are stored in a separate OpenSearch alias; normal searches do not "
        "require Gemini and do not modify any existing tab."
    )

    lab_enabled = st.checkbox(
        "Enable Semantic SEO Lab for this session",
        value=False,
        key="semantic_seo_lab_enabled",
        help=(
            "The lab is opt-in and isolated. GLiNER runs during index ingestion; "
            "only bounded reranking runs during a normal search."
        ),
    )
    if not lab_enabled:
        st.info(
            "Enable the lab to check its isolated index and models. Existing tabs "
            "do not execute any Semantic SEO model or OpenSearch work while it is off."
        )
        return

    model_settings = load_semantic_seo_model_settings()
    model_status = local_model_status(model_settings)
    _show_model_status(model_status, title_summary)

    if not keyword_query.strip():
        st.info("Enter a query above to run the isolated Semantic SEO pipeline.")
        return

    refined_settings = load_opensearch_settings()
    if not refined_settings.configured:
        st.info("Configure OPENSEARCH_URL to enable the Semantic SEO Lab.")
        return

    refined_client = OpenSearchRefinedClient(refined_settings)
    semantic_client = build_semantic_seo_client(model_settings)
    try:
        connection = refined_client.connection_info()
        refined_state = refined_client.index_state()
        semantic_state = semantic_client.index_state()
    except (OpenSearchRefinedError, OpenSearchSemanticError) as initial_error:
        if not refined_settings.auto_start:
            st.error("The Semantic SEO Lab could not connect to OpenSearch.")
            st.caption(str(initial_error))
            return
        try:
            with st.spinner("Starting local OpenSearch..."):
                start_local_opensearch(refined_settings)
            st.rerun()
            connection = refined_client.connection_info()
            refined_state = refined_client.index_state()
            semantic_state = semantic_client.index_state()
        except (OpenSearchRefinedError, OpenSearchSemanticError) as exc:
            st.error("The Semantic SEO Lab could not start or connect to OpenSearch.")
            st.caption(str(exc))
            return

    status_columns = st.columns(4)
    status_columns[0].metric("Cluster", connection["cluster_name"])
    status_columns[1].metric("OpenSearch", connection["version"])
    status_columns[2].metric(
        "Lab titles indexed", f"{int(semantic_state.get('document_count', 0)):,}"
    )
    status_columns[3].metric("Embedding model", model_settings.embedding_model)

    expected_refined_fingerprint = title_corpus_fingerprint(title_summary)
    if (
        not refined_state.get("ready")
        or refined_state.get("fingerprint") != expected_refined_fingerprint
    ):
        st.warning(
            "Sync the Search Results refined index first. The lab uses its direct "
            "story IDs only as an exclusion boundary."
        )
        return

    expected_embedding_fingerprint = semantic_corpus_fingerprint(
        title_summary,
        model_settings.embedding_model,
    )
    expected_semantic_fingerprint = semantic_corpus_fingerprint(
        title_summary,
        model_settings.embedding_model,
        model_settings.gliner_model,
    )
    semantic_search_ready = (
        bool(semantic_state.get("ready"))
        and semantic_state.get("embedding_model") == model_settings.embedding_model
        and semantic_state.get("fingerprint")
        in {expected_embedding_fingerprint, expected_semantic_fingerprint}
    )
    ingestion_needs_sync = (
        not semantic_state.get("ready")
        or semantic_state.get("fingerprint") != expected_semantic_fingerprint
        or semantic_state.get("embedding_model") != model_settings.embedding_model
        or not semantic_state.get("title_entities_indexed")
        or semantic_state.get("entity_model") != model_settings.gliner_model
    )
    if ingestion_needs_sync:
        st.warning(
            "The isolated title-intelligence index is missing or stale. Its one-time "
            "CPU ingestion extracts and links GLiNER entities, generates BGE-M3 "
            "vectors, and stores both in OpenSearch."
        )
        if semantic_search_ready:
            st.info(
                "The existing BGE-M3 index remains searchable now. Indexed GLiNER "
                "entity scoring will activate after the offline enrichment completes."
            )
    embedding_available = bool(model_status["BGE-M3 embeddings"]["available"])
    gliner_available = bool(model_status["GLiNER"]["available"])
    if st.button(
        "Build/sync title intelligence index",
        key="semantic_seo_sync_bge_index",
        type="primary" if ingestion_needs_sync else "secondary",
        disabled=not (embedding_available and gliner_available),
        help=(
            "Uses only the traffic-semantic-seo-lab alias. Existing OpenSearch indexes "
            "are not changed."
        ),
    ):
        try:
            with st.spinner(
                "Extracting/linking title entities, encoding titles, and syncing "
                "the isolated index..."
            ):
                semantic_state = semantic_client.sync_title_index(title_summary)
            semantic_search_ready = True
            ingestion_needs_sync = False
            st.success(
                f"Indexed {int(semantic_state.get('document_count', 0)):,} titles "
                "with canonical entities and BGE-M3 vectors."
            )
        except (OpenSearchSemanticError, RuntimeError, ValueError) as exc:
            st.error("The isolated title-intelligence index could not be built.")
            st.caption(str(exc))
            return
    if not embedding_available:
        st.info(
            "BGE-M3 is not in the local model cache. The lab remains isolated and "
            "disabled until the approved model package is installed."
        )
        return
    if not gliner_available:
        st.info(
            "GLiNER is not locally available. The ingestion-first index remains "
            "disabled until the approved model package is installed."
        )
        return

    research_model = get_vertex_model_name("research")
    cached_profile = load_semantic_route_profile(
        SEMANTIC_SEO_DB_PATH,
        keyword_query,
        research_model,
    )
    if cached_profile is None:
        st.info(
            "Fast OpenSearch-first retrieval is ready. No reusable indirect route "
            "profile exists yet; Gemini enrichment below is optional and does not "
            "block the search button."
        )
    else:
        st.caption((
            f"Validated route cache: {cached_profile.relationship_count:,} relationships "
            f"· updated {cached_profile.created_at}."
        ).replace("\u00c2\u00b7", "\u00b7"))

    refresh_clicked = st.button(
        "Enrich/refresh reusable indirect routes with Gemini (optional)",
        key=f"semantic_seo_refresh_routes_{_query_key(keyword_query)}",
        width="stretch",
        help=(
            "Runs an explicit enrichment action only when clicked. "
            "Normal searches reuse saved routes and make no route-discovery request."
        ),
    )
    if refresh_clicked:
        try:
            with st.spinner("Discovering and verifying compact grounded routes..."):
                direct_ids = collect_all_refined_story_ids(
                    refined_client,
                    keyword_query,
                    match_mode,
                    page_size=10_000,
                )
                title_context = (
                    title_summary.loc[
                        title_summary["story_id"].astype(str).isin(direct_ids),
                        "page_title",
                    ]
                    .dropna()
                    .astype(str)
                    .head(30)
                    .tolist()
                )
                cached_profile = discover_verified_semantic_routes(
                    keyword_query=keyword_query,
                    matched_titles=title_context,
                    service_account_path=SERVICE_ACCOUNT_PATH,
                    location=get_vertex_location(),
                    model_name=research_model,
                )
                save_semantic_route_profile(SEMANTIC_SEO_DB_PATH, cached_profile)
            st.success(
                f"Saved {cached_profile.relationship_count:,} independently verified "
                "relationship routes."
            )
        except (SemanticSEORouteError, RuntimeError, ValueError) as exc:
            st.error("Validated Gemini route discovery could not be completed.")
            st.caption(str(exc))
            return
        except Exception as exc:
            st.error("Validated Gemini route discovery could not be completed.")
            st.caption(str(exc))
            return

    with st.expander("Experimental stage controls", expanded=False):
        use_indexed_entities = st.checkbox(
            "Use ingestion-time GLiNER entities",
            value=bool(semantic_state.get("title_entities_indexed")),
            disabled=not bool(semantic_state.get("title_entities_indexed")),
            key="semantic_seo_use_indexed_entities",
            help="Reads canonical entities from OpenSearch; GLiNER does not run here.",
        )
        use_reranker = st.checkbox(
            "BGE cross-encoder reranking",
            value=bool(model_status["BGE reranker"]["available"]),
            disabled=not bool(model_status["BGE reranker"]["available"]),
            key="semantic_seo_use_reranker",
        )
        use_deberta_title_relevance = st.checkbox(
            "DeBERTa title relevance score (optional, top 10)",
            value=False,
            disabled=not bool(model_status["DeBERTa title relevance"]["available"]),
            key="semantic_seo_use_deberta_title_relevance",
            help=(
                "Works without article bodies. It adds a soft title score and never "
                "rejects an indirect result as unsupported evidence."
            ),
        )
        use_gemini_validation = st.checkbox(
            "Gemini final validation (optional, adds network latency)",
            value=False,
            key="semantic_seo_use_gemini_validation",
        )
        maximum_gemini_candidates = st.slider(
            "Maximum candidates sent to Gemini",
            min_value=5,
            max_value=50,
            value=DEFAULT_MAX_GEMINI_CANDIDATES,
            step=5,
            disabled=not use_gemini_validation,
            key="semantic_seo_gemini_limit",
        )

    run_disabled = not semantic_search_ready or not embedding_available
    if not st.button(
        "Run isolated Semantic SEO search",
        key=f"semantic_seo_run_{_query_key(keyword_query)}",
        type="primary",
        disabled=run_disabled,
        width="stretch",
    ):
        _show_saved_result(keyword_query)
        return

    started_at = time.perf_counter()
    try:
        with st.spinner("Running the isolated candidate funnel..."):
            direct_started = time.perf_counter()
            direct_ids = collect_all_refined_story_ids(
                refined_client,
                keyword_query,
                match_mode,
                page_size=10_000,
            )
            direct_duration_ms = round(
                (time.perf_counter() - direct_started) * 1000,
                1,
            )
            routes = [build_direct_query_route(keyword_query)]
            if cached_profile is not None:
                routes.extend(
                    extract_relationship_routes(
                        keyword_query=keyword_query,
                        grounded_research=cached_profile.research,
                        max_relationships=4,
                        max_texts_per_relationship=3,
                        query_central_policy=True,
                    )
                )
            options = SemanticSEORunOptions(
                use_indexed_entities=use_indexed_entities,
                use_reranker=use_reranker,
                use_deberta_title_relevance=use_deberta_title_relevance,
                maximum_gemini_candidates=maximum_gemini_candidates,
            )
            candidates, diagnostics = run_local_semantic_seo_stages(
                keyword_query=keyword_query,
                routes=routes,
                client=semantic_client,
                excluded_story_ids=direct_ids,
                title_summary=title_summary,
                model_settings=model_settings,
                options=options,
            )
            diagnostics.insert(0, {
                "stage": "direct_result_exclusion",
                "applied": True,
                "processed": len(direct_ids),
                "duration_ms": direct_duration_ms,
            })
            if use_gemini_validation and candidates:
                gemini_started = time.perf_counter()
                validation_candidates = candidates[:maximum_gemini_candidates]
                selected = select_titles_generatively_with_vertex(
                    keyword_query=keyword_query,
                    candidate_titles=validation_candidates,
                    grounded_research=(
                        cached_profile.research
                        if cached_profile is not None
                        else {
                            "research_text": (
                                "Title-only semantic relevance to query: "
                                + keyword_query
                            )
                        }
                    ),
                    service_account_path=SERVICE_ACCOUNT_PATH,
                    location=get_vertex_location(),
                    model_name=get_vertex_model_name("judge"),
                    batch_size=20,
                    query_central_policy=True,
                )
                candidates = merge_gemini_validation(validation_candidates, selected)
                diagnostics.append(
                    {
                        "stage": "gemini_final_validation",
                        "applied": True,
                        "processed": len(validation_candidates),
                        "accepted": len(candidates),
                        "duration_ms": round(
                            (time.perf_counter() - gemini_started) * 1000,
                            1,
                        ),
                    }
                )
            else:
                diagnostics.append(
                    {
                        "stage": "gemini_final_validation",
                        "applied": False,
                        "processed": 0,
                        "reason": "Disabled or no local candidates were available.",
                    }
                )
            result = semantic_seo_result_frame(candidates)
            elapsed = time.perf_counter() - started_at
            state_key = _result_state_key(keyword_query)
            st.session_state[state_key] = {
                "result": result,
                "diagnostics": diagnostics,
                "sources": cached_profile.sources if cached_profile is not None else [],
                "web_search_queries": (
                    cached_profile.web_search_queries
                    if cached_profile is not None
                    else []
                ),
                "direct_excluded": len(direct_ids),
                "route_count": len(routes),
                "route_mode": (
                    "OpenSearch + reusable enriched routes"
                    if cached_profile is not None
                    else "OpenSearch title-only fast path"
                ),
                "elapsed": elapsed,
            }
    except Exception as exc:
        st.error("The isolated Semantic SEO search could not be completed.")
        st.caption(str(exc))
        return

    _show_saved_result(keyword_query)


def _show_model_status(
    model_status: dict[str, dict[str, object]],
    title_summary: pd.DataFrame,
) -> None:
    with st.expander("Pipeline readiness and isolation", expanded=False):
        rows = []
        for stage, status in model_status.items():
            rows.append(
                {
                    "Stage": stage,
                    "Model": status["model"],
                    "Enabled": "Yes" if status["enabled"] else "No",
                    "Locally available": "Yes" if status["available"] else "No",
                    "Status": status["detail"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info(
            "Title-only mode: GLiNER and entity linking run during ingestion. DeBERTa "
            "is available as an optional soft title-relevance scorer; it is not an "
            "article-evidence gate. mREBEL remains disabled because relationship "
            "extraction from headline fragments is not reliable."
        )


def _show_saved_result(keyword_query: str) -> None:
    state = st.session_state.get(_result_state_key(keyword_query))
    if not isinstance(state, dict):
        return
    result = state.get("result")
    if not isinstance(result, pd.DataFrame):
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Validated indirect titles", f"{len(result):,}")
    metric_columns[1].metric(
        "Direct titles excluded", f"{int(state.get('direct_excluded', 0)):,}"
    )
    metric_columns[2].metric("Verified routes", f"{int(state.get('route_count', 0)):,}")
    metric_columns[3].metric("Elapsed", f"{float(state.get('elapsed', 0.0)):.2f}s")
    st.caption(f"Result path: {state.get('route_mode', 'Semantic SEO Lab')}")

    diagnostics = state.get("diagnostics", [])
    if isinstance(diagnostics, list):
        with st.expander("Stage diagnostics", expanded=False):
            st.dataframe(pd.DataFrame(diagnostics), hide_index=True, width="stretch")

    if result.empty:
        st.info("No indirect title passed all enabled Semantic SEO stages.")
    else:
        st.subheader("Validated indirect Semantic SEO titles")
        st.dataframe(
            result.head(SEMANTIC_SEO_DISPLAY_LIMIT),
            hide_index=True,
            width="stretch",
            height=1200,
            column_config={
                "page_title": st.column_config.TextColumn("Page title", width="large"),
                "factual_bridge": st.column_config.TextColumn(
                    "Factual bridge", width="large"
                ),
                "gemini_audit_reason": st.column_config.TextColumn(
                    "Validation reason", width="large"
                ),
            },
        )

    sources = state.get("sources", [])
    queries = state.get("web_search_queries", [])
    if sources or queries:
        with st.expander("Grounding provenance", expanded=False):
            if queries:
                st.caption("Google Search queries")
                st.write(queries)
            if sources:
                st.caption("Grounding sources")
                st.dataframe(pd.DataFrame(sources), hide_index=True, width="stretch")


def _query_key(keyword_query: str) -> str:
    return hashlib.sha256(keyword_query.casefold().encode("utf-8")).hexdigest()[:16]


def _result_state_key(keyword_query: str) -> str:
    return f"semantic_seo_result_{SEMANTIC_SEO_PIPELINE_VERSION}_{_query_key(keyword_query)}"
