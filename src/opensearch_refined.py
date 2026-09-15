"""Isolated OpenSearch backend for the refined title-search tab."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

from src.search_aliases import (
    opensearch_synonym_rules,
    query_uses_alias_expansion,
    refined_alias_title_text,
)


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


REFINED_SEARCH_PIPELINE_VERSION = "2026-08-20-opensearch-aliases-v2"
INDEX_META_DOCUMENT_ID = "__refined_search_index_meta__"
DEFAULT_INDEX_ALIAS = "traffic-title-refined"
MATCH_TIER_ORDER = (
    "Exact phrase",
    "Exact keywords",
    "Known alias",
    "Root variant",
    "Similar spelling",
)
PRIMARY_TRAFFIC_MATCH_TIERS = (
    "Exact phrase",
    "Exact keywords",
    "Known alias",
    "Root variant",
)
_LOCAL_OPENSEARCH_START_LOCK = threading.Lock()


class OpenSearchRefinedError(RuntimeError):
    """Raised when the refined-search OpenSearch service cannot complete an operation."""


@dataclass(frozen=True)
class OpenSearchSettings:
    url: str
    index_alias: str = DEFAULT_INDEX_ALIAS
    username: str = ""
    password: str = ""
    verify_certs: bool = True
    timeout_seconds: float = 10.0
    auto_start: bool = False
    home: str = ""
    startup_timeout_seconds: float = 120.0

    @property
    def configured(self) -> bool:
        return bool(self.url.strip())


def load_opensearch_settings() -> OpenSearchSettings:
    """Load optional connection settings without affecting the existing search path."""
    return OpenSearchSettings(
        url=os.getenv("OPENSEARCH_URL", "").strip().rstrip("/"),
        index_alias=_safe_index_name(
            os.getenv("OPENSEARCH_INDEX", DEFAULT_INDEX_ALIAS).strip()
            or DEFAULT_INDEX_ALIAS
        ),
        username=os.getenv("OPENSEARCH_USERNAME", "").strip(),
        password=os.getenv("OPENSEARCH_PASSWORD", ""),
        verify_certs=_environment_flag("OPENSEARCH_VERIFY_CERTS", default=True),
        timeout_seconds=max(
            1.0,
            float(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "10")),
        ),
        auto_start=_environment_flag("OPENSEARCH_AUTO_START", default=False),
        home=os.getenv("OPENSEARCH_HOME", "").strip(),
        startup_timeout_seconds=max(
            10.0,
            float(os.getenv("OPENSEARCH_STARTUP_TIMEOUT_SECONDS", "120")),
        ),
    )


def start_local_opensearch(settings: OpenSearchSettings) -> None:
    """Start the native Windows service through the project's local launcher."""
    if not settings.auto_start:
        raise OpenSearchRefinedError("Automatic OpenSearch startup is disabled.")

    hostname = (urlparse(settings.url).hostname or "").lower()
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise OpenSearchRefinedError(
            "Automatic OpenSearch startup is allowed only for a local loopback URL."
        )
    if os.name != "nt":
        raise OpenSearchRefinedError(
            "Automatic OpenSearch startup is currently supported only on Windows."
        )

    # Another Streamlit rerun or browser session may already have started the
    # shared local node. Avoid waiting on the process lock when it is usable.
    if _opensearch_is_reachable(settings):
        return

    if not settings.home:
        raise OpenSearchRefinedError(
            "OPENSEARCH_HOME is required when OPENSEARCH_AUTO_START is enabled."
        )

    project_root = Path(__file__).resolve().parents[1]
    launcher = project_root / "start-native-opensearch.ps1"
    opensearch_home = Path(os.path.expandvars(settings.home)).expanduser()
    if not launcher.is_file():
        raise OpenSearchRefinedError(f"OpenSearch launcher was not found at {launcher}.")
    if not (opensearch_home / "bin" / "opensearch.bat").is_file():
        raise OpenSearchRefinedError(
            f"OpenSearch was not found at {opensearch_home}. Check OPENSEARCH_HOME."
        )

    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(launcher),
        "-OpenSearchHome",
        str(opensearch_home),
        "-OpenSearchOnly",
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    deadline = time.monotonic() + settings.startup_timeout_seconds + 15

    while True:
        if _opensearch_is_reachable(settings):
            return
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise OpenSearchRefinedError(
                "Timed out waiting for another session to start local OpenSearch."
            )
        if _LOCAL_OPENSEARCH_START_LOCK.acquire(
            timeout=min(1.0, remaining_seconds)
        ):
            break

    try:
        if _opensearch_is_reachable(settings):
            return
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise OpenSearchRefinedError(
                "Timed out before the local OpenSearch launcher could run."
            )
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=remaining_seconds,
                creationflags=creation_flags,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OpenSearchRefinedError(
                f"OpenSearch could not be started automatically: {exc}"
            ) from exc
    finally:
        _LOCAL_OPENSEARCH_START_LOCK.release()

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise OpenSearchRefinedError(
            "OpenSearch could not be started automatically"
            + (f": {detail}" if detail else ".")
        )

    if not _opensearch_is_reachable(settings):
        raise OpenSearchRefinedError(
            f"OpenSearch launcher finished, but {settings.url} is not reachable."
        )


def _opensearch_is_reachable(settings: OpenSearchSettings) -> bool:
    """Return quickly when the configured local endpoint can accept requests."""
    auth = (
        (settings.username, settings.password)
        if settings.username or settings.password
        else None
    )
    try:
        response = requests.get(
            f"{settings.url}/",
            auth=auth,
            timeout=min(2.0, settings.timeout_seconds),
            verify=settings.verify_certs,
        )
    except requests.RequestException:
        return False
    return response.ok


def build_refined_query(
    normalized_query: str,
    match_mode: str,
    *,
    size: int = 1000,
    search_after: list[Any] | None = None,
) -> dict[str, Any]:
    """Build boosted exact, stemmed, and typo-tolerant retrieval tiers."""
    query = " ".join(str(normalized_query).split())
    if not query:
        raise ValueError("A normalized query is required.")

    operator = "or" if match_mode == "Any keyword" else "and"
    clauses: list[dict[str, Any]] = []

    if match_mode == "Exact cleaned phrase":
        clauses.extend(
            [
                {
                    "match_phrase": {
                        "clean_title": {
                            "query": query,
                            "boost": 12.0,
                            "_name": "exact_phrase",
                        }
                    }
                },
                {
                    "match_phrase": {
                        "clean_title.stemmed": {
                            "query": query,
                            "boost": 4.0,
                            "_name": "root_variant",
                        }
                    }
                },
            ]
        )
    else:
        clauses.extend(
            [
                {
                    "match_phrase": {
                        "clean_title": {
                            "query": query,
                            "boost": 12.0,
                            "_name": "exact_phrase",
                        }
                    }
                },
                {
                    "match": {
                        "clean_title": {
                            "query": query,
                            "operator": operator,
                            "boost": 7.0,
                            "_name": "exact_keywords",
                        }
                    }
                },
                {
                    "match": {
                        "clean_title.stemmed": {
                            "query": query,
                            "operator": operator,
                            "boost": 3.0,
                            "_name": "root_variant",
                        }
                    }
                },
            ]
        )

    if query_uses_alias_expansion(query):
        alias_query: dict[str, Any]
        if match_mode == "Exact cleaned phrase":
            alias_query = {
                "match_phrase": {
                    "alias_title": {
                        "query": query,
                        "analyzer": "refined_alias_search",
                        "boost": 5.0,
                        "_name": "known_alias",
                    }
                }
            }
        else:
            alias_query = {
                "match": {
                    "alias_title": {
                        "query": query,
                        "analyzer": "refined_alias_search",
                        "operator": operator,
                        "boost": 5.0,
                        "_name": "known_alias",
                    }
                }
            }
        clauses.append(alias_query)

    clauses.append(
        {
            "match": {
                "clean_title": {
                    "query": query,
                    "operator": operator,
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                    "max_expansions": 50,
                    "fuzzy_transpositions": True,
                    "boost": 1.0,
                    "_name": "similar_spelling",
                }
            }
        }
    )

    payload: dict[str, Any] = {
        "size": max(1, min(int(size), 10_000)),
        "track_total_hits": True,
        "query": {
            "bool": {
                "should": clauses,
                "minimum_should_match": 1,
                "filter": [{"term": {"document_type": "story"}}],
            }
        },
        "sort": [
            {"_score": {"order": "desc"}},
            {"total_views": {"order": "desc"}},
            {"story_id": {"order": "asc"}},
        ],
        "highlight": {
            "fields": {"page_title": {"number_of_fragments": 0}},
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
        },
    }
    if search_after is not None:
        payload["search_after"] = list(search_after)
    return payload


def classify_matched_queries(matched_queries: Iterable[str]) -> str:
    matched = set(matched_queries or [])
    for query_name, tier in (
        ("exact_phrase", "Exact phrase"),
        ("exact_keywords", "Exact keywords"),
        ("known_alias", "Known alias"),
        ("root_variant", "Root variant"),
        ("similar_spelling", "Similar spelling"),
    ):
        if query_name in matched:
            return tier
    return "Similar spelling"


def build_refined_primary_analysis_result(
    refined_result: dict[str, Any],
) -> dict[str, Any]:
    """Create the Dashboard/Analytics input from refined primary-tier matches."""
    matched_titles = refined_result.get("matched_titles")
    matched_story_months = refined_result.get("matched_story_months")
    if not isinstance(matched_titles, pd.DataFrame) or not isinstance(
        matched_story_months,
        pd.DataFrame,
    ):
        raise ValueError("Refined results must include title and monthly DataFrames.")

    primary_titles = matched_titles.loc[
        matched_titles["refined_match_tier"].isin(PRIMARY_TRAFFIC_MATCH_TIERS)
    ].copy()
    primary_story_months = matched_story_months.loc[
        matched_story_months["refined_match_tier"].isin(PRIMARY_TRAFFIC_MATCH_TIERS)
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
        "matched_title_summary": primary_titles,
        "matched_story_months": primary_story_months,
        "monthly_summary": monthly_summary,
        "analysis_source": "Search Results refined",
        "excluded_refined_match_tiers": ("Similar spelling",),
    }


def parse_search_hits(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw_hits = payload.get("hits", {}).get("hits", [])
    total_value = payload.get("hits", {}).get("total", 0)
    if isinstance(total_value, dict):
        total = int(total_value.get("value", len(raw_hits)))
    else:
        total = int(total_value or len(raw_hits))

    parsed = []
    for hit in raw_hits:
        source = hit.get("_source", {})
        matched_queries = [str(item) for item in hit.get("matched_queries", [])]
        parsed.append(
            {
                "story_id": str(source.get("story_id", hit.get("_id", ""))),
                "page_title": str(source.get("page_title", "")),
                "refined_match_tier": classify_matched_queries(matched_queries),
                "opensearch_score": round(float(hit.get("_score") or 0.0), 4),
                "matched_queries": matched_queries,
            }
        )
    return parsed, total


class OpenSearchRefinedClient:
    def __init__(self, settings: OpenSearchSettings):
        if not settings.configured:
            raise ValueError("OPENSEARCH_URL is not configured.")
        self.settings = settings
        self.session = requests.Session()
        if settings.username:
            self.session.auth = (settings.username, settings.password)

    def connection_info(self) -> dict[str, Any]:
        payload = self._request("GET", "/")
        version = payload.get("version", {}) if isinstance(payload, dict) else {}
        return {
            "cluster_name": str(payload.get("cluster_name", "OpenSearch")),
            "version": str(version.get("number", "unknown")),
        }

    def index_state(self) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"/{self.settings.index_alias}/_doc/{INDEX_META_DOCUMENT_ID}",
            allowed_statuses={404},
        )
        if response.get("status") == 404 or response.get("found") is False:
            return {"ready": False, "fingerprint": "", "document_count": 0}
        source = response.get("_source", {})
        return {
            "ready": True,
            "fingerprint": str(source.get("corpus_fingerprint", "")),
            "document_count": int(source.get("document_count", 0)),
            "pipeline_version": str(source.get("pipeline_version", "")),
            "physical_index": str(response.get("_index", "")),
        }

    def sync_title_index(
        self,
        title_summary: pd.DataFrame,
        *,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        fingerprint = title_corpus_fingerprint(title_summary)
        current = self.index_state()
        if (
            current.get("fingerprint") == fingerprint
            and current.get("pipeline_version") == REFINED_SEARCH_PIPELINE_VERSION
        ):
            return {**current, "created": False}

        physical_index = f"{self.settings.index_alias}-{fingerprint[:12]}"
        exists = self._request(
            "HEAD",
            f"/{physical_index}",
            allowed_statuses={404},
        )
        if exists.get("status") == 404:
            self._request("PUT", f"/{physical_index}", json_body=_index_definition())

        documents = list(_iter_title_documents(title_summary))
        for start in range(0, len(documents), max(1, int(batch_size))):
            batch = documents[start : start + max(1, int(batch_size))]
            bulk_body = _bulk_index_body(physical_index, batch)
            bulk_result = self._request(
                "POST",
                "/_bulk",
                data=bulk_body,
                headers={"Content-Type": "application/x-ndjson"},
            )
            if bulk_result.get("errors"):
                failures = _bulk_failures(bulk_result)
                raise OpenSearchRefinedError(
                    f"OpenSearch bulk indexing failed for {len(failures)} document(s): "
                    f"{failures[:3]}"
                )

        meta_document = {
            "document_type": "meta",
            "corpus_fingerprint": fingerprint,
            "document_count": len(documents),
            "pipeline_version": REFINED_SEARCH_PIPELINE_VERSION,
        }
        self._request(
            "PUT",
            f"/{physical_index}/_doc/{INDEX_META_DOCUMENT_ID}",
            params={"refresh": "true"},
            json_body=meta_document,
        )
        self._switch_alias(physical_index)
        return {
            "ready": True,
            "fingerprint": fingerprint,
            "document_count": len(documents),
            "pipeline_version": REFINED_SEARCH_PIPELINE_VERSION,
            "physical_index": physical_index,
            "created": True,
        }

    def search(
        self,
        normalized_query: str,
        match_mode: str,
        *,
        size: int = 1000,
    ) -> tuple[list[dict[str, Any]], int]:
        hits, total_hits, _ = self.search_page(
            normalized_query=normalized_query,
            match_mode=match_mode,
            size=size,
        )
        return hits, total_hits

    def search_page(
        self,
        normalized_query: str,
        match_mode: str,
        *,
        size: int = 1000,
        search_after: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], int, list[Any] | None]:
        """Return one stable ranked page and the cursor for the following page."""
        payload = self._request(
            "POST",
            f"/{self.settings.index_alias}/_search",
            json_body=build_refined_query(
                normalized_query,
                match_mode,
                size=size,
                search_after=search_after,
            ),
        )
        hits, total_hits = parse_search_hits(payload)
        raw_hits = payload.get("hits", {}).get("hits", [])
        next_search_after = None
        if raw_hits:
            raw_sort = raw_hits[-1].get("sort")
            if isinstance(raw_sort, list):
                next_search_after = raw_sort
        return hits, total_hits, next_search_after

    def _switch_alias(self, physical_index: str) -> None:
        aliases = self._request(
            "GET",
            f"/_alias/{self.settings.index_alias}",
            allowed_statuses={404},
        )
        actions = []
        if aliases.get("status") != 404:
            actions.extend(
                {
                    "remove": {
                        "index": index_name,
                        "alias": self.settings.index_alias,
                    }
                }
                for index_name in aliases
            )
        actions.append(
            {
                "add": {
                    "index": physical_index,
                    "alias": self.settings.index_alias,
                }
            }
        )
        self._request("POST", "/_aliases", json_body={"actions": actions})

    def _request(
        self,
        method: str,
        path: str,
        *,
        allowed_statuses: set[int] | None = None,
        json_body: dict[str, Any] | None = None,
        data: str | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.session.request(
                method,
                f"{self.settings.url}{path}",
                json=json_body,
                data=data,
                headers=headers,
                params=params,
                timeout=self.settings.timeout_seconds,
                verify=self.settings.verify_certs,
            )
        except requests.RequestException as exc:
            raise OpenSearchRefinedError(
                f"Unable to reach OpenSearch at {self.settings.url}: {exc}"
            ) from exc

        if allowed_statuses and response.status_code in allowed_statuses:
            payload = _response_json(response)
            payload["status"] = response.status_code
            return payload
        if not response.ok:
            detail = response.text.strip()[:1000]
            raise OpenSearchRefinedError(
                f"OpenSearch {method} {path} returned HTTP {response.status_code}: {detail}"
            )
        if method == "HEAD" or not response.content:
            return {"status": response.status_code}
        return _response_json(response)


def collect_all_refined_story_id_sets(
    client: OpenSearchRefinedClient,
    normalized_query: str,
    match_mode: str,
    *,
    page_size: int = 10_000,
) -> tuple[set[str], set[str]]:
    """Return complete displayed and primary-tier refined story-ID sets."""
    all_story_ids: set[str] = set()
    primary_story_ids: set[str] = set()
    search_after: list[Any] | None = None
    total_hits: int | None = None
    processed_hits = 0
    previous_cursor: list[Any] | None = None

    while total_hits is None or processed_hits < total_hits:
        hits, total_hits, next_cursor = client.search_page(
            normalized_query=normalized_query,
            match_mode=match_mode,
            size=max(1, min(int(page_size), 10_000)),
            search_after=search_after,
        )
        processed_hits += len(hits)
        all_story_ids.update(
            str(hit.get("story_id", "")).strip() for hit in hits
        )
        primary_story_ids.update(
            str(hit.get("story_id", "")).strip()
            for hit in hits
            if str(hit.get("refined_match_tier", "")) in PRIMARY_TRAFFIC_MATCH_TIERS
        )
        primary_story_ids.discard("")
        if not hits or next_cursor is None or next_cursor == previous_cursor:
            break
        previous_cursor = next_cursor
        search_after = next_cursor

    all_story_ids.discard("")
    return all_story_ids, primary_story_ids


def collect_all_refined_primary_story_ids(
    client: OpenSearchRefinedClient,
    normalized_query: str,
    match_mode: str,
    *,
    page_size: int = 10_000,
) -> set[str]:
    """Return every story ID that contributes to refined primary traffic."""
    _, primary_story_ids = collect_all_refined_story_id_sets(
        client=client,
        normalized_query=normalized_query,
        match_mode=match_mode,
        page_size=page_size,
    )
    return primary_story_ids


def title_corpus_fingerprint(title_summary: pd.DataFrame) -> str:
    columns = [
        column
        for column in (
            "story_id",
            "page_title",
            "clean_title",
            "total_views",
            "active_months",
            "first_month",
            "last_month",
        )
        if column in title_summary.columns
    ]
    records = (
        title_summary[columns]
        .fillna("")
        .astype(str)
        .sort_values(columns[0] if columns else title_summary.columns[0])
        .to_dict("records")
    )
    serialized = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    alias_rules = "|".join(opensearch_synonym_rules())
    return hashlib.sha256(
        f"{REFINED_SEARCH_PIPELINE_VERSION}|{alias_rules}|{serialized}".encode("utf-8")
    ).hexdigest()


def _iter_title_documents(title_summary: pd.DataFrame) -> Iterable[dict[str, Any]]:
    for row in title_summary.to_dict("records"):
        yield {
            "document_type": "story",
            "story_id": str(row.get("story_id", "")),
            "page_title": str(row.get("page_title", "")),
            "clean_title": str(row.get("clean_title", "")),
            "alias_title": refined_alias_title_text(str(row.get("page_title", ""))),
            "total_views": _safe_int(row.get("total_views")),
            "active_months": _safe_int(row.get("active_months")),
            "first_month": _safe_date(row.get("first_month")),
            "last_month": _safe_date(row.get("last_month")),
        }


def _index_definition() -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {
                    "refined_english_stemmer": {
                        "type": "stemmer",
                        "language": "light_english",
                    },
                    "refined_alias_synonyms": {
                        "type": "synonym_graph",
                        "lenient": False,
                        "synonyms": list(opensearch_synonym_rules()),
                    },
                },
                "analyzer": {
                    "refined_exact": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase"],
                    },
                    "refined_stemmed": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase", "refined_english_stemmer"],
                    },
                    "refined_alias_search": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase", "refined_alias_synonyms"],
                    },
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "_meta": {"pipeline_version": REFINED_SEARCH_PIPELINE_VERSION},
            "properties": {
                "document_type": {"type": "keyword"},
                "story_id": {"type": "keyword"},
                "page_title": {"type": "text", "analyzer": "standard"},
                "alias_title": {
                    "type": "text",
                    "analyzer": "refined_exact",
                    "search_analyzer": "refined_alias_search",
                },
                "clean_title": {
                    "type": "text",
                    "analyzer": "refined_exact",
                    "fields": {
                        "stemmed": {
                            "type": "text",
                            "analyzer": "refined_stemmed",
                        }
                    },
                },
                "total_views": {"type": "long"},
                "active_months": {"type": "integer"},
                "first_month": {"type": "date"},
                "last_month": {"type": "date"},
                "corpus_fingerprint": {"type": "keyword"},
                "document_count": {"type": "integer"},
                "pipeline_version": {"type": "keyword"},
            },
        },
    }


def _bulk_index_body(index_name: str, documents: list[dict[str, Any]]) -> str:
    lines = []
    for document in documents:
        lines.append(
            json.dumps(
                {
                    "index": {
                        "_index": index_name,
                        "_id": document["story_id"],
                    }
                },
                separators=(",", ":"),
            )
        )
        lines.append(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def _bulk_failures(payload: dict[str, Any]) -> list[str]:
    failures = []
    for item in payload.get("items", []):
        operation = item.get("index", {})
        if int(operation.get("status", 500)) >= 300:
            failures.append(str(operation.get("error", operation)))
    return failures


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {"value": payload}


def _safe_index_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-_")
    return cleaned or DEFAULT_INDEX_ALIAS


def _environment_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_date(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return None
