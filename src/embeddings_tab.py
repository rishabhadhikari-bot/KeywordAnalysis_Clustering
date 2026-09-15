"""On-demand full-title retrieval and keyword extraction, isolated from other tabs."""
import pandas as pd
import streamlit as st

from src.word_embeddings import encode_words as encode_texts, get_relationship_embedding_model_name, index_key, query_texts
from src.title_keyword_embeddings import prepare_titles, load_title_vectors, retrieve_keywords


@st.cache_data(show_spinner=False, max_entries=4)
def _titles(rows):
    return prepare_titles(rows)


@st.cache_resource(show_spinner=False, max_entries=2)
def _index(texts, model_name):
    return load_title_vectors(texts, model_name)


@st.fragment
def show_embeddings_tab(title_summary: pd.DataFrame) -> None:
    st.subheader("Embeddings: keywords from similar titles")
    st.caption("Full page titles → title vectors → query vector → nearest titles → related keywords.")
    titles = _titles(title_summary[["story_id", "page_title"]])
    texts = tuple(titles.page_title)
    model_name = get_relationship_embedding_model_name()
    signature = "full-titles-v1:" + index_key(texts, model_name)
    if st.session_state.get("emb_signature") != signature:
        st.session_state["emb_signature"] = signature
        st.session_state.pop("emb_ready", None)
    st.caption(f"{len(titles):,} story titles · Model: {model_name}")
    if st.button("Build / load title embeddings", key="emb_build", disabled=not texts):
        try:
            with st.spinner("Embedding full titles; the first build may take a few minutes..."):
                _index(texts, model_name)
            st.session_state["emb_ready"] = signature
        except Exception as exc:
            st.error(f"Could not prepare title embeddings: {exc}")
    if st.session_state.get("emb_ready") != signature:
        st.info("Build or load the title embeddings to start searching. Title vectors are cached separately.")
        return
    with st.form("emb_search_form"):
        query = st.text_input("Search keyword or phrase", key="emb_query", placeholder="ekadashi")
        mode = st.radio("Search mode", ["Combined meaning", "Separate keywords"], key="emb_mode", horizontal=True)
        cols = st.columns(3)
        title_limit = cols[0].number_input("Nearest titles per query", 1, 500, 50, key="emb_title_limit")
        min_similarity = cols[1].slider("Minimum title similarity", 0.0, 1.0, 0.4, .05, key="emb_title_similarity")
        top_k = cols[2].number_input("Keywords per query", 1, 200, 20, key="emb_keyword_limit")
        min_occurrences = st.number_input("Minimum occurrences in retrieved titles", 1, value=2, key="emb_matched_occurrences")
        exclusions = st.text_input("Additional keywords to exclude", key="emb_extra_stopwords")
        exclude_query = st.checkbox("Exclude searched words from keywords", value=True, key="emb_exclude_query")
        submitted = st.form_submit_button("Find similar keywords")
    st.caption("Full titles retain their wording and context during embedding. Stop words and standalone numbers "
               "are removed only when extracting result keywords. Retrieved titles do not need to contain the query.")
    st.caption("Keyword relevance sums supporting title similarities and downweights words common across the corpus. "
               "It is not a word-vector similarity or probability. Best title similarity is query-to-title cosine similarity.")
    if not submitted:
        return
    queries = query_texts(query, True) if mode == "Separate keywords" else ([query.strip()] if query.strip() else [])
    if not queries or not any(any(c.isalnum() for c in text) for text in queries):
        st.warning("Enter a keyword or phrase.")
        return
    try:
        with st.spinner("Finding nearest titles and extracting keywords..."):
            vectors = _index(texts, model_name)
            query_vectors = encode_texts(queries, model_name)
            batches, sources = [], []
            for text, vector in zip(queries, query_vectors):
                result, matched = retrieve_keywords(titles, vectors, text, vector, int(title_limit),
                    min_similarity, int(top_k), int(min_occurrences), exclusions, exclude_query)
                batches.append(result)
                matched.insert(0, "query", text)
                sources.append(matched)
                st.caption(f'"{text}": retrieved {len(matched):,} titles meeting the similarity threshold.')
            results = pd.concat(batches, ignore_index=True)
            matched_titles = pd.concat(sources, ignore_index=True)
        if results.empty:
            st.info("No keywords meet these filters. Try reducing the minimum occurrences or title similarity.")
        else:
            st.dataframe(results, hide_index=True, use_container_width=True)
            st.download_button("Download related keywords", results.to_csv(index=False).encode("utf-8-sig"),
                               "related_keywords.csv", "text/csv", key="emb_download", on_click="ignore")
        with st.expander("Inspect retrieved page titles"):
            st.dataframe(matched_titles, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f"Title keyword search could not complete: {exc}")
