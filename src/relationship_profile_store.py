"""Read reusable grounded relationship profiles without coupling to Streamlit."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_latest_relationship_profile(
    db_path: Path,
    keyword_query: str,
    *,
    preferred_match_type: str = "",
    required_profile_version: str = "",
) -> tuple[dict[str, object], dict[str, str]] | None:
    """Load the newest cached profile for a canonical query.

    Profiles are deliberately reusable across match modes and direct-title
    fingerprints. Those inputs help research, but should not cause an
    interactive cache miss once a factual relationship graph already exists.
    """
    normalized_query = " ".join(str(keyword_query).casefold().split())
    if not normalized_query or not db_path.exists():
        return None

    try:
        with sqlite3.connect(db_path, timeout=5) as connection:
            rows = connection.execute(
                """
                SELECT research_json, match_type, profile_version, updated_at
                FROM web_grounded_query_profiles
                WHERE normalized_query = ?
                ORDER BY
                    CASE WHEN match_type = ? THEN 0 ELSE 1 END,
                    updated_at DESC
                """,
                (normalized_query, preferred_match_type),
            ).fetchall()
    except sqlite3.Error:
        return None

    for research_json, match_type, profile_version, updated_at in rows:
        if (
            required_profile_version
            and str(profile_version) != str(required_profile_version)
        ):
            continue
        try:
            research = json.loads(research_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(research, dict):
            continue
        return research, {
            "match_type": str(match_type),
            "profile_version": str(profile_version),
            "updated_at": str(updated_at),
        }
    return None
