"""On-demand monthly keyword reporting, isolated from shared search state."""

import hashlib

import pandas as pd
import streamlit as st

from src.daily_keywords import DEFAULT_STOPWORDS, title_tokens
from src.monthly_keywords import build_monthly_keyword_dataset


@st.fragment
def show_miscellaneous_tab(story_months: pd.DataFrame) -> None:
    st.subheader("Monthly title keywords")
    st.caption(
        "One row per month and token. Occurrences counts pages containing the token; "
        "total token occurrences also counts repeated words within a title. "
        "Views per occurrence = total views / occurrences."
    )
    st.caption("Uses the existing monthly traffic and page titles, across all stories.")
    exclude_common = st.checkbox("Exclude common English words", value=True, key="misc_stopwords")
    extra = st.text_input("Additional words to exclude (comma or space separated)", key="misc_exclusions")
    st.caption("Tokens are lowercased; punctuation and hyphens separate words. Numbers are retained.")
    exclusions = frozenset(title_tokens(extra)) | (DEFAULT_STOPWORDS if exclude_common else frozenset())
    source_hash = hashlib.sha256(pd.util.hash_pandas_object(
        story_months[["month", "story_id", "page_title", "views"]], index=False
    ).values.tobytes()).hexdigest()
    signature = (source_hash, tuple(sorted(exclusions)))
    if st.session_state.get("misc_monthly_signature") != signature:
        st.session_state.pop("misc_monthly_result", None)
    if st.button("Build monthly keyword dataset", disabled=story_months.empty, key="misc_build"):
        st.session_state.pop("misc_monthly_result", None)
        try:
            with st.spinner("Building monthly keyword dataset…"):
                result, audit = build_monthly_keyword_dataset(story_months, exclusions)
            st.session_state["misc_monthly_result"] = (result, audit)
            st.session_state["misc_monthly_signature"] = signature
        except ValueError as exc:
            st.error(str(exc))
    saved = st.session_state.get("misc_monthly_result")
    if saved is None:
        return
    result, audit = saved
    st.caption(f"{audit['page_months']:,} page/month records processed.")
    if audit["missing_title_page_months"]:
        st.warning(
            f"Excluded {audit['missing_title_page_months']:,} page/month records with missing titles "
            f"({audit['missing_title_views']:,.2f} views). Add titles to include them."
        )
    st.caption(
        "Each page contributes its views to every included token: token views overlap and "
        "must not be summed into site-wide views. Pages absent from a month are not counted. "
        "This dataset uses all monthly records, independent of the main keyword search and high-traffic cutoff. Existing titles are used for all months; explicit zero-view records are included."
    )
    if result.empty:
        st.info("No tokens remain after title matching and exclusions.")
        return
    months = sorted(result["month"].unique())
    selected = st.multiselect(
        "Months", options=months, default=months,
        key="misc_months_" + hashlib.sha256(repr(signature).encode()).hexdigest()[:16],
    )
    query = st.text_input("Filter tokens (contains)", key="misc_token_filter").strip().casefold()
    minimum = st.number_input("Minimum occurrences per month", min_value=1, value=1, step=1, key="misc_min_occurrences")
    visible = result.loc[
        result["month"].isin(selected)
        & result["token"].str.contains(query, regex=False)
        & result["occurrences"].ge(minimum)
    ]
    st.caption(f"{len(visible):,} matching rows. Preview shows up to 5,000; exports contain all matching rows.")
    st.dataframe(visible.head(5000), hide_index=True, use_container_width=True)
    st.download_button("Download filtered dataset", visible.to_csv(index=False).encode("utf-8-sig"),
                       "monthly_keywords_filtered.csv", "text/csv", key="misc_filtered_download")
    st.download_button("Download full dataset", result.to_csv(index=False).encode("utf-8-sig"),
                       "monthly_keywords.csv", "text/csv", key="misc_full_download")
