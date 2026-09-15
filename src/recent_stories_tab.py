"""Streamlit view for recent external news stories."""
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

from src.recent_stories import fetch_recent_stories
from src.wikidata_source_related import load_source_settings


@st.cache_data(ttl=600, show_spinner=False)
def load_recent_stories(query, days, rss_url, timeout, user_agent, languages=("en", "hi")):
    stories = fetch_recent_stories(query, days=days, rss_url=rss_url,
                                   timeout=timeout, user_agent=user_agent, languages=languages)
    return stories, datetime.now(timezone.utc)


def show_recent_stories_tab(keyword_query):
    st.subheader("Recent Stories")
    st.caption("Google News RSS headlines for your searched keyword, newest first. India editions. These are external news articles.")
    query = keyword_query.strip()
    if not query:
        st.info("Enter a keyword above to find recent news stories.")
        return
    date_column, language_column = st.columns(2)
    days = date_column.selectbox("News published within", [1, 7, 30], index=1,
                        format_func=lambda value: f"Past {value} day{'s' if value != 1 else ''}",
                        key="recent_stories_days")
    news_language = language_column.selectbox(
        "News language", ["English + Hindi", "English", "Hindi"],
        key="recent_stories_language",
        help="Search the English and/or Hindi Google News India editions. This selects feeds; it does not translate your keyword or enforce article language.",
    )
    languages = {"English + Hindi": ("en", "hi"), "English": ("en",), "Hindi": ("hi",)}[news_language]
    st.caption(f"Search: {query}. RSS results use Google News matching; the corpus match-mode selector does not apply.")
    settings = load_source_settings()
    key = (query, days, settings.google_news_rss_url, languages)
    stored = st.session_state.get("recent_stories_result")
    load = st.button("Find Recent Stories", key="find_recent_stories")
    refresh = st.button("Refresh news feed", key="refresh_recent_stories") if stored and stored[0] == key else False
    if load or refresh:
        args = (query, days, settings.google_news_rss_url, settings.timeout_seconds, settings.user_agent, languages)
        try:
            with st.spinner("Loading Google News RSS headlines…"):
                if refresh:
                    load_recent_stories.clear(*args)
                stories, fetched_at = load_recent_stories(*args)
            stored = (key, stories, fetched_at)
            st.session_state["recent_stories_result"] = stored
        except (requests.RequestException, ValueError, ET.ParseError) as exc:
            st.error("Google News RSS could not be loaded. Check your internet connection and try again.")
            with st.expander("Connection details"):
                st.code(str(exc))
            if stored and stored[0] == key:
                st.info("Showing the previously retrieved headlines below.")
    if not stored or stored[0] != key:
        st.info("Select Find Recent Stories to retrieve headlines for this keyword.")
        return
    _, stories, fetched_at = stored
    fetched_text = pd.Timestamp(fetched_at).tz_convert("Asia/Kolkata").strftime("%d %b %Y, %H:%M IST")
    st.caption(f"{len(stories)} articles · Retrieved {fetched_text} · Feed cached for 10 minutes. Links open via Google News.")
    if not stories:
        st.info("No matching headlines were returned. Try a wider date window, a different keyword, or another news language.")
        return
    frame = pd.DataFrame(stories)
    frame["Published (IST)"] = pd.to_datetime(frame.pop("Published"), utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M").fillna("Not provided")
    frame = frame[["Publisher", "Publisher domain", "News article title", "Published (IST)", "Article link"]]
    st.dataframe(frame, hide_index=True, use_container_width=True,
                 column_config={"News article title": st.column_config.TextColumn(width="large"),
                                "Article link": st.column_config.LinkColumn(display_text="Open article")})
