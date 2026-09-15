"""Wikidata relationship retrieval with cached, bounded title identity checks."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.data_processing import simple_query_tokens, simple_title_tokens
from src.search_aliases import normalize_refined_alias_text
from src.wikidata_identity import IDENTITY_VERSION, IdentityGate, identity_tokens, context_tokens, spans, CONTEXT_PREDICATES
from src.wikidata_identity import names as identity_names
from src.wikidata_relationship_policy import (
    POLICY_VERSION, TWO_HOP_POLICIES, RetrievalPolicy, RelationshipGate, compile_policy,
)

# Shared title matching and evidence types; no DBpedia client is instantiated.
from src.dbpedia_source_related import (
    EvidenceRoute, LinkedEntity, SourceProfile as BaseSourceProfile, SourceRelatedError,
    build_source_related_results,
    build_news_routes, join_news_headlines, strip_publisher_suffix, parse_news_date,
)

SOURCE_RELATED_PIPELINE_VERSION = "2026-09-14-wikidata-multi-entity-v1-" + POLICY_VERSION
MAX_SECOND_HOP_PER_ENTITY = 25
MAX_SECOND_HOP_ENTITIES = 200
MAX_SECOND_HOP_PATHS = 500
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "wikidata" / "entities.sqlite3"
PROPERTY_LABELS = {
    "P54": "member of sports team", "P118": "league", "P1344": "participant in",
    "P710": "participant", "P26": "spouse", "P22": "father", "P25": "mother",
    "P40": "child", "P3373": "sibling", "P108": "employer", "P69": "educated at",
    "P463": "member of", "P112": "founder", "P127": "owned by",
    "P749": "parent organization", "P355": "subsidiary", "P50": "author",
    "P57": "director", "P161": "cast member", "P58": "screenwriter",
    "P86": "composer", "P175": "performer", "P676": "lyrics by",
    "P162": "producer", "P264": "record label", "P123": "publisher",
    "P361": "part of", "P527": "has part", "P144": "based on",
    "P138": "named after", "P180": "depicts", "P921": "main subject",
    "P664": "organizer", "P276": "location", "P19": "birthplace",
    "P166": "award", "P793": "significant event",
    "P1830": "owner of", "P1416": "affiliation",
    "P169": "chief executive officer", "P488": "chairperson",
    "P1038": "relative", "P1327": "professional or sports partner",
    "P1269": "facet of", "P279": "subclass of", "P31": "instance of",
}
CONTEXT_PROPERTIES = {"P1269", "P279", "P31", "P19"}
# Generic types are not useful editorial subjects, and never expand to peers.
EXCLUDED_TARGETS = {"Q5", "Q35120", "Q488383"}


@dataclass
class SourceProfile(BaseSourceProfile):
    query_aliases: list[str] = field(default_factory=list)
    identity_entities: dict[str, dict] = field(default_factory=dict)
    checked_identity_names: set[str] = field(default_factory=set)
    query_entities: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class WikidataRoute(EvidenceRoute):
    target_qid: str = ""
    hop_count: int = 1
    path_qids: tuple[str, ...] = ()
    relationship_path: str = ""
    path_evidence_urls: tuple[str, ...] = ()
    path_properties: tuple[str, ...] = ()
    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    query_entity_qid: str = ""
    query_entity_label: str = ""
    query_entity_description: str = ""


@dataclass(frozen=True)
class SourceSettings:
    api_url: str = "https://www.wikidata.org/w/api.php"
    cache_path: str = str(DEFAULT_CACHE)
    timeout_seconds: float = 15.0
    user_agent: str = "TrafficPredictionSourceResearch/1.0 (local editorial research)"
    google_news_rss_url: str = "https://news.google.com/rss/search"


def load_source_settings() -> SourceSettings:
    return SourceSettings(
        api_url=os.getenv("WIKIDATA_API_URL", SourceSettings.api_url),
        cache_path=os.getenv("WIKIDATA_LOCAL_DB", str(DEFAULT_CACHE)),
        timeout_seconds=max(2.0, float(os.getenv("SOURCE_RELATED_TIMEOUT_SECONDS", "15"))),
        user_agent=os.getenv("SOURCE_RELATED_USER_AGENT", SourceSettings.user_agent),
        google_news_rss_url=os.getenv("GOOGLE_NEWS_RSS_URL", SourceSettings.google_news_rss_url),
    )


def name_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def entity_names(entity: dict) -> list[str]:
    return identity_names(entity)


def entity_label(entity: dict) -> str:
    return (entity.get("labels", {}).get("en", {}).get("value")
            or entity.get("sitelinks", {}).get("enwiki", {}).get("title")
            or next(iter(entity_names(entity)), entity["id"]))


def minor_name_correction(query: str, name: str) -> bool:
    """One edit in one word of a multiword name; never drop query terms."""
    left, right = identity_tokens(query), identity_tokens(name)
    if len(left) < 2 or len(left) != len(right):
        return False
    changed = [(a, b) for a, b in zip(left, right) if a != b]
    if len(changed) != 1:
        return False
    a, b = changed[0]
    if min(len(a), len(b)) < 4:
        return False
    if len(a) == len(b):
        indices = [i for i in range(len(a)) if a[i] != b[i]]
        return len(indices) == 1 or (len(indices) == 2 and indices[1] == indices[0] + 1
                                    and a[indices[0]] == b[indices[1]] and a[indices[1]] == b[indices[0]])
    if len(a) < len(b):
        a, b = b, a
    return len(a) == len(b) + 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a)))


def permitted_claims(entity: dict):
    for prop, claims in entity.get("claims", {}).items():
        if prop not in PROPERTY_LABELS:
            continue
        for claim in claims:
            snak = claim.get("mainsnak", {})
            if claim.get("rank") == "deprecated" or snak.get("snaktype") != "value":
                continue
            value = snak.get("datavalue", {}).get("value", {})
            target = value.get("id", "") if isinstance(value, dict) else ""
            if re.fullmatch(r"Q\d+", target) and target not in EXCLUDED_TARGETS:
                yield prop, target, claim


def two_hop_plan(root: dict, first_entities: dict[str, dict]):
    """Plan exactly one more outgoing step, bounded fairly across intermediates.

    Classification/birthplace routes are metadata only at either step. Cycles,
    duplicate statements, and endpoints already available directly add no route.
    No endpoint's claims are inspected here, so this cannot reach a third hop.
    """
    first = list(permitted_claims(root))
    direct_ids = {target for prop, target, _ in first if prop not in CONTEXT_PROPERTIES}
    branches, limited = [], 0
    seen_paths = set()
    for prop, middle, claim in sorted(first, key=lambda edge: (edge[1], edge[0])):
        if prop in CONTEXT_PROPERTIES or middle == root["id"]:
            continue
        entity = first_entities.get(middle)
        if not entity or not entity_names(entity):
            continue
        branch = []
        for next_prop, target, next_claim in sorted(permitted_claims(entity), key=lambda edge: (edge[0], edge[1])):
            if next_prop in CONTEXT_PROPERTIES or target in {root["id"], middle} or target in direct_ids:
                continue
            if (prop, next_prop) not in TWO_HOP_POLICIES:
                continue
            key = (prop, middle, next_prop, target, json.dumps(claim, sort_keys=True), json.dumps(next_claim, sort_keys=True))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            branch.append((prop, middle, claim, next_prop, target, next_claim))
        limited += max(0, len(branch) - MAX_SECOND_HOP_PER_ENTITY)
        branches.append(branch[:MAX_SECOND_HOP_PER_ENTITY])
    paths, endpoints = [], set()
    for offset in range(MAX_SECOND_HOP_PER_ENTITY):
        for branch in branches:
            if offset >= len(branch):
                continue
            path = branch[offset]
            target = path[4]
            if len(paths) >= MAX_SECOND_HOP_PATHS or (target not in endpoints and len(endpoints) >= MAX_SECOND_HOP_ENTITIES):
                limited += 1
                continue
            endpoints.add(target)
            paths.append(path)
    return paths, limited


def relationship_explanation(source: dict, prop: str, target: dict, claim: dict) -> str:
    explanation = f"Wikidata records {entity_label(source)} — {PROPERTY_LABELS[prop]} — {entity_label(target)}."
    if prop == "P54":
        explanation += " Team membership may be current or historical."
    dates = []
    for time_prop, label in (("P580", "start"), ("P582", "end"), ("P585", "point in time")):
        for snak in claim.get("qualifiers", {}).get(time_prop, []):
            value = snak.get("datavalue", {}).get("value", {})
            if isinstance(value, dict) and value.get("time"):
                dates.append(f"{label}: {value['time'].split('T')[0].lstrip('+')}")
    return explanation + (" " + "; ".join(dates) + "." if dates else "")


class WikidataLocalClient:
    def __init__(self, settings: SourceSettings | None = None, *, session=None):
        self.settings = settings or load_source_settings()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.settings.user_agent})

    def _connect(self):
        path = Path(self.settings.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=10)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                qid TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS names (
                name TEXT NOT NULL, qid TEXT NOT NULL, PRIMARY KEY(name, qid));
            CREATE TABLE IF NOT EXISTS profiles (
                name TEXT PRIMARY KEY, qid TEXT NOT NULL, fetched_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS query_resolutions (
                name TEXT PRIMARY KEY, qids TEXT NOT NULL, fetched_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS identity_entities (
                qid TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS identity_searches (
                name TEXT PRIMARY KEY, qids TEXT NOT NULL, fetched_at TEXT NOT NULL);
        """)
        return db

    def _api(self, **params):
        timeout = self.settings.timeout_seconds
        if getattr(self, "_identity_deadline", None) is not None:
            remaining = self._identity_deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Wikidata identity lookup time budget reached.")
            timeout = min(timeout, remaining)
        response = self.session.get(
            self.settings.api_url, params={"format": "json", "maxlag": 5, **params},
            timeout=timeout,
        )
        if response.status_code == 429:
            raise ValueError("Wikidata is rate-limiting downloads. Try updating later; local searches remain available.")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "error" in payload:
            raise ValueError("Wikidata returned an API error or invalid response.")
        return payload

    def cache_summary(self) -> str:
        db = self._connect()
        try:
            count, newest = db.execute("SELECT COUNT(*), MAX(fetched_at) FROM entities").fetchone()
            return f"{count:,} stored entities; latest download: {(newest or 'never')[:10]}. This is a subset of Wikidata."
        finally:
            db.close()

    def build_editorial_profile(self, query: str, *, refresh=False, include_news=True) -> SourceProfile:
        """Keep independent news evidence available when local knowledge is missing."""
        try:
            profile = self.build_profile(query, refresh=refresh)
        except SourceRelatedError as exc:
            profile = SourceProfile(query=query)
            profile.source_status["Wikidata local database"] = "No resolved local profile"
            profile.warnings.append(str(exc))
        if not include_news:
            profile.source_status["Google News"] = "Disabled"
            return profile
        try:
            response = self.session.get(
                self.settings.google_news_rss_url,
                params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = []
            for item in root.findall(".//item")[:40]:
                publisher = (item.findtext("source") or "Publisher").strip()
                title = strip_publisher_suffix((item.findtext("title") or "").strip(), publisher)
                url = (item.findtext("link") or "").strip()
                if title and url:
                    items.append(dict(title=title, url=url, publisher=publisher,
                                      published_at=parse_news_date(item.findtext("pubDate") or "")))
            _, spans = join_news_headlines(items)
            profile.routes.extend(build_news_routes(
                query=query, query_entities=[], news_items=items,
                headline_spans=spans, news_entities=[],
            ))
            profile.source_status["Google News"] = f"Available ({len(items)} sourced headlines)"
        except (requests.RequestException, ValueError, ET.ParseError) as exc:
            profile.source_status["Google News"] = "Unavailable"
            profile.warnings.append("Google News RSS could not be retrieved. Local Wikidata results remain available.")
            profile.diagnostics["Google News"] = str(exc)
        return profile

    def _fetch_entities(self, qids: list[str]) -> dict[str, dict]:
        result = {}
        for start in range(0, len(qids), 20):
            batch = qids[start:start + 20]
            payload = self._api(action="wbgetentities", ids="|".join(batch),
                                props="labels|aliases|descriptions|claims|sitelinks", languages="en|hi")
            entities = payload.get("entities")
            if not isinstance(entities, dict):
                raise ValueError("Wikidata returned no entity data.")
            for qid in batch:
                entity = entities.get(qid)
                if not isinstance(entity, dict) or "missing" in entity or entity.get("id") != qid:
                    raise ValueError(f"Wikidata entity {qid} could not be downloaded.")
                result[qid] = entity
        return result

    def refresh(self, query: str) -> tuple[str, ...]:
        """Download all matching roots and their separate paths atomically."""
        query = query.strip()
        if re.fullmatch(r"[Qq]\d+", query):
            query = query.upper()
            candidates = self._fetch_entities([query])
            matched = [candidates[query]]
        else:
            language = "hi" if re.search(r"[\u0900-\u097f]", query) else "en"
            payload = self._api(action="wbsearchentities", search=query, language=language, limit=10)
            hits = payload.get("search")
            if not isinstance(hits, list):
                raise ValueError("Wikidata returned an invalid entity search response.")
            qids = list(dict.fromkeys(h["id"] for h in hits if re.fullmatch(r"Q\d+", h.get("id", ""))))
            candidates = self._fetch_entities(qids)
            matched = [e for e in candidates.values() if name_key(query) in {name_key(n) for n in entity_names(e)}]
            if not matched:
                matched = [e for e in candidates.values()
                           if any(minor_name_correction(query, name) for name in entity_names(e))]
            if not matched and 2 <= len(identity_tokens(query)) <= 5:
                # wbsearchentities may return nothing for even a one-letter
                # typo. Discover candidates using at most two name components,
                # then validate every word of the original query against them.
                fallback_ids = set()
                words = identity_tokens(query)
                for word in dict.fromkeys((words[0], words[-1])):
                    if len(word) < 4:
                        continue
                    fallback = self._api(action="wbsearchentities", search=word, language=language, limit=10)
                    hits = fallback.get("search")
                    if not isinstance(hits, list):
                        raise ValueError("Wikidata returned an invalid entity search response.")
                    fallback_ids.update(hit["id"] for hit in hits if re.fullmatch(r"Q\d+", hit.get("id", "")))
                candidates.update(self._fetch_entities(sorted(fallback_ids - candidates.keys())))
                matched = [e for e in candidates.values()
                           if name_key(query) in {name_key(n) for n in entity_names(e)}]
                if not matched:
                    matched = [e for e in candidates.values()
                               if any(minor_name_correction(query, name) for name in entity_names(e))]
            if not matched:
                suggestions = "; ".join(dict.fromkeys(
                    name for e in candidates.values() for name in [entity_label(e)]
                    if name != e.get("id")))
                raise ValueError("No matching name or alias. Try the subject's full or official name. "
                                 + ("Possible matches: " + suggestions if suggestions else ""))
        roots = {item["id"]: item for item in matched}
        targets = set()
        for selected in roots.values():
            branch_targets = {target for _, target, _ in permitted_claims(selected)}
            if len(branch_targets) > 200:
                raise ValueError(f"{entity_label(selected)} has over 200 related entities; use a more specific subject.")
            targets.update(branch_targets)
        related = self._fetch_entities(sorted(targets - candidates.keys()))
        entities = {qid: candidates[qid] for qid in sorted(targets) if qid in candidates}
        entities.update(related)
        entities.update(roots)
        # Each path starts at one root; sharing fetched nodes does not combine
        # unrelated roots' claims into a synthetic relationship.
        second_targets = sorted({path[4] for selected in roots.values()
                                 for path in two_hop_plan(selected, entities)[0]} - entities.keys())
        if second_targets:
            entities.update(self._fetch_entities(second_targets))
        now = datetime.now(timezone.utc).isoformat()
        db = self._connect()
        try:
            with db:
                for qid, entity in entities.items():
                    db.execute("INSERT OR REPLACE INTO entities VALUES (?, ?, ?)", (qid, json.dumps(entity), now))
                    db.execute("DELETE FROM names WHERE qid = ?", (qid,))
                    db.executemany("INSERT OR IGNORE INTO names VALUES (?, ?)",
                                   [(name_key(n), qid) for n in entity_names(entity)])
                for qid in sorted(roots):
                    db.execute("INSERT OR REPLACE INTO profiles VALUES (?, ?, ?)", (name_key(qid), qid, now))
                db.execute("INSERT OR REPLACE INTO query_resolutions VALUES (?, ?, ?)",
                           (name_key(query), json.dumps(sorted(roots)), now))
        finally:
            db.close()
        return tuple(sorted(roots))

    def build_profile(self, query: str, *, refresh: bool = False, download_missing: bool = False) -> SourceProfile:
        """Resolve every matching root, then merge independently built profiles."""
        query = query.strip()
        if not query:
            raise ValueError("Enter a subject name.")
        if re.fullmatch(r"[Qq]\d+", query):
            return self._build_entity_profile(query.upper(), refresh=refresh, download_missing=download_missing)

        db = self._connect()
        try:
            resolution = db.execute("SELECT qids FROM query_resolutions WHERE name=?", (name_key(query),)).fetchone()
        finally:
            db.close()
        refresh_error = ""
        # Legacy single-root cache entries are not complete name-search results.
        # Upgrade them once when online lookup is enabled; future queries reuse
        # the persisted multi-root resolution, including spelling corrections.
        if refresh or (download_missing and resolution is None):
            try:
                self.refresh(query)
            except (requests.RequestException, ValueError) as exc:
                refresh_error = str(exc)
        db = self._connect()
        try:
            exact_ids = {row[0] for row in db.execute("SELECT qid FROM names WHERE name=?", (name_key(query),))}
            resolution = db.execute("SELECT qids FROM query_resolutions WHERE name=?", (name_key(query),)).fetchone()
            if exact_ids:
                root_ids = exact_ids
            elif resolution:
                root_ids = set(json.loads(resolution[0]))
            else:
                root_ids = {qid for stored_name, qid in db.execute("SELECT name, qid FROM names")
                            if minor_name_correction(query, stored_name)}
                if not root_ids:
                    # Preserve previously saved corrected spellings from the
                    # old schema, without choosing one of multiple exact names.
                    legacy = db.execute("SELECT qid FROM profiles WHERE name=?", (name_key(query),)).fetchone()
                    root_ids = {legacy[0]} if legacy else set()
        finally:
            db.close()
        if not root_ids:
            message = "This query has no downloaded name or alias match in the local Wikidata subset. Stored entities are retained."
            if refresh_error:
                message += " Download attempt failed: " + refresh_error
            else:
                message += " Download missing data online, or search a stored name or alias."
            raise SourceRelatedError(message)

        combined = SourceProfile(query=query)
        for qid in sorted(root_ids):
            branch = self._build_entity_profile(qid, download_missing=download_missing and not refresh_error)
            combined.routes.extend(branch.routes)
            combined.linked_entities.extend(branch.linked_entities)
            combined.query_entities.extend(branch.query_entities)
            combined.query_aliases.extend(branch.query_aliases)
            combined.identity_entities.update(branch.identity_entities)
            combined.checked_identity_names.update(branch.checked_identity_names)
            root = branch.identity_entities[qid]
            label = entity_label(root)
            description = root.get("descriptions", {}).get("en", {}).get("value", "")
            display_name = f"{label} ({description or qid})"
            combined.warnings.extend((f"{display_name}: {warning}" if len(root_ids) > 1 else warning)
                                     for warning in branch.warnings)
            combined.diagnostics.update({f"{display_name}: {key}": value for key, value in branch.diagnostics.items()})
            if len(root_ids) == 1:
                combined.source_status.update(branch.source_status)
            else:
                combined.source_status[display_name] = branch.source_status.get("Wikidata local database", "")
            if name_key(query) not in {name_key(n) for n in entity_names(root)}:
                combined.warnings.append(f"Resolved spelling correction: {query} → {label} ({qid}).")
        combined.query_aliases = list(dict.fromkeys(combined.query_aliases))
        combined.source_status["Query resolution"] = (
            f"{len(root_ids)} matching entities; independent paths combined. "
            "Coverage is limited to cached names and bounded Wikidata search results."
        )
        if len(root_ids) > 1:
            combined.source_status["Discovery depth"] = (
                f"Up to 2 hops per entity: {sum(r.hop_count == 1 for r in combined.routes)} one-hop routes, "
                f"{sum(r.hop_count == 2 for r in combined.routes)} two-hop routes."
            )
        if refresh_error:
            combined.warnings.append("Wikidata update failed. Using the previously downloaded local data.")
            combined.diagnostics["Wikidata update"] = refresh_error
        return combined

    def _build_entity_profile(self, query: str, *, refresh: bool = False, download_missing: bool = False) -> SourceProfile:
        """Build routes for one explicit root; public name searches combine roots."""
        if not query.strip():
            raise ValueError("Enter a subject name.")
        refresh_error = ""
        if refresh:
            try:
                self.refresh(query)
            except (requests.RequestException, ValueError) as exc:
                refresh_error = str(exc)
        db = self._connect()
        try:
            rows = db.execute("SELECT DISTINCT qid FROM names WHERE name = ?", (name_key(query),)).fetchall()
            profile_row = db.execute("SELECT qid, fetched_at FROM profiles WHERE name = ?", (name_key(query),)).fetchone()
            if not rows and not profile_row and not re.fullmatch(r"[Qq]\d+", query.strip()):
                # Resolve the same bounded spelling correction offline as well.
                # A missing exact query key must not hide an already saved subject.
                correction_ids = {stored_qid for stored_name, stored_qid in db.execute("SELECT name, qid FROM names")
                                  if minor_name_correction(query, stored_name)}
                rows = [(stored_qid,) for stored_qid in sorted(correction_ids)]
            if not profile_row:
                # Relationship targets also contain claims; reuse whatever is already local.
                local_qid = rows[0][0] if rows else query.strip().upper()
                profile_row = db.execute(
                    "SELECT qid, fetched_at FROM entities WHERE qid = ?", (local_qid,)
                ).fetchone()
            if not profile_row:
                if download_missing and not refresh:
                    # Release the read connection before the atomic download/write.
                    db.close()
                    return self.build_profile(query, refresh=True)
                message = "This query has no downloaded name or alias match in the local Wikidata subset. Stored entities are retained."
                if refresh_error:
                    message += " Download attempt failed: " + refresh_error
                else:
                    message += " Download missing data online, or search a stored name or alias."
                raise SourceRelatedError(message)
            qid, fetched_at = profile_row
            row = db.execute("SELECT payload FROM entities WHERE qid = ?", (qid,)).fetchone()
            if not row:
                raise SourceRelatedError("The local entity is missing. Download it again.")
            entity = json.loads(row[0])
            profile = SourceProfile(query=query)
            description = entity.get("descriptions", {}).get("en", {}).get("value", "")
            profile.query_entities = [{"qid": qid, "label": entity_label(entity), "description": description}]
            min_words = min(2, len(name_key(entity_label(entity)).split()))
            profile.query_aliases = [
                name for name in entity_names(entity)
                if len(name_key(name).split()) >= min_words
            ][:10]
            profile.linked_entities = [LinkedEntity(
                f"https://www.wikidata.org/wiki/{qid}", entity_label(entity), 1.0,
                resolution_source="Local Wikidata name/alias index",
            )]
            if (not re.fullmatch(r"[Qq]\d+", query.strip())
                    and name_key(query) not in {name_key(n) for n in entity_names(entity)}):
                profile.warnings.append(f"Resolved spelling correction: {query} → {entity_label(entity)} ({qid}).")
            missing = []
            needs_download = False
            first_entities = {}
            for prop, target, claim in permitted_claims(entity):
                if target == qid:
                    continue
                target_row = db.execute("SELECT payload FROM entities WHERE qid = ?", (target,)).fetchone()
                if not target_row:
                    missing.append(target)
                    needs_download = True
                    continue
                target_entity = json.loads(target_row[0])
                first_entities[target] = target_entity
                if not entity_names(target_entity):
                    missing.append(target)
                    continue
                label = entity_label(target_entity)
                relation = PROPERTY_LABELS[prop]
                explanation = relationship_explanation(entity, prop, target_entity, claim)
                profile.routes.append(WikidataRoute(
                    related_subject=label,
                    aliases=tuple(n for n in entity_names(target_entity) if n != label),
                    relationship_type=f"Wikidata: {relation}", why_related=explanation,
                    source_name="Wikidata", evidence_url=f"https://www.wikidata.org/wiki/{qid}#{prop}",
                    confidence=62 if prop in CONTEXT_PROPERTIES else 90,
                    source_kind="wikidata_context" if prop in CONTEXT_PROPERTIES else "wikidata_direct",
                    target_qid=target,
                    path_qids=(qid, target),
                    relationship_path=f"{entity_label(entity)} → {relation} → {label}",
                    path_evidence_urls=(f"https://www.wikidata.org/wiki/{qid}#{prop}",),
                    path_properties=(prop,),
                    retrieval_policy=compile_policy((entity, target_entity), (prop,), (claim,)),
                    query_entity_qid=qid, query_entity_label=entity_label(entity), query_entity_description=description,
                ))
            paths, limited = two_hop_plan(entity, first_entities)
            for prop, middle, claim, next_prop, target, next_claim in paths:
                target_row = db.execute("SELECT payload FROM entities WHERE qid=?", (target,)).fetchone()
                if not target_row:
                    missing.append(target)
                    needs_download = True
                    continue
                target_entity = json.loads(target_row[0])
                if not entity_names(target_entity):
                    missing.append(target)
                    continue
                intermediate = first_entities[middle]
                label = entity_label(target_entity)
                path_text = (f"{entity_label(entity)} → {PROPERTY_LABELS[prop]} → {entity_label(intermediate)} "
                             f"→ {PROPERTY_LABELS[next_prop]} → {label}")
                sources = (f"https://www.wikidata.org/wiki/{qid}#{prop}",
                           f"https://www.wikidata.org/wiki/{middle}#{next_prop}")
                profile.routes.append(WikidataRoute(
                    related_subject=label,
                    aliases=tuple(name for name in entity_names(target_entity) if name != label),
                    relationship_type=f"Wikidata two-hop: {PROPERTY_LABELS[prop]} → {PROPERTY_LABELS[next_prop]}",
                    why_related=(relationship_explanation(entity, prop, intermediate, claim) + " "
                                 + relationship_explanation(intermediate, next_prop, target_entity, next_claim)),
                    source_name="Wikidata", evidence_url=sources[0], confidence=80,
                    source_kind="wikidata_direct", target_qid=target, hop_count=2,
                    path_qids=(qid, middle, target), relationship_path=path_text,
                    path_evidence_urls=sources,
                    path_properties=(prop, next_prop),
                    retrieval_policy=compile_policy((entity, intermediate, target_entity),
                                                    (prop, next_prop), (claim, next_claim)),
                    query_entity_qid=qid, query_entity_label=entity_label(entity), query_entity_description=description,
                ))
            if limited:
                profile.warnings.append(f"Two-hop expansion limits omitted {limited} additional paths; coverage is partial.")
            profile.source_status["Discovery depth"] = (
                f"Up to 2 hops: {sum(r.hop_count == 1 and r.source_kind == 'wikidata_direct' for r in profile.routes)} "
                f"one-hop routes, {sum(r.hop_count == 2 for r in profile.routes)} two-hop routes."
            )
            profile.source_status["Relationship policy"] = (
                f"{POLICY_VERSION}; explicit paths only; bounded relationships require title context"
            )
            if needs_download and download_missing and not refresh:
                db.close()
                return self.build_profile(query, refresh=True)
            profile.source_status["Wikidata local database"] = f"Available ({len(profile.routes)} routes; downloaded {fetched_at[:10]})"
            if refresh_error:
                profile.warnings.append("Wikidata update failed. Using the previously downloaded local data.")
                profile.diagnostics["Wikidata update"] = refresh_error
            if missing:
                profile.warnings.append("Some relationship targets are missing or have no English/Hindi names; refresh this entity to check for updates.")
            if not any(route.source_kind == "wikidata_direct" for route in profile.routes):
                profile.warnings.append(
                    "This entity has no supported discovery relationships in the downloaded data. "
                    "Classification links alone do not establish related article subjects. "
                    "Increasing hop depth cannot expand a subject without an eligible first link."
                )
            profile.identity_entities = {
                stored_qid: json.loads(payload)
                for stored_qid, payload in db.execute("SELECT qid, payload FROM entities")
            }
            return profile
        finally:
            db.close()

    def prepare_identity_catalog(self, profile, candidate_titles, *, refresh=False):
        """Discover competing names once, with a seven-day cache and bounded work.

        Only names actually found in candidate titles trigger searches. No title
        or traffic data is sent to Wikidata. Failed lookups never become evidence
        of uniqueness and never overwrite a successful cached search.
        """
        titles = [identity_tokens(title) for title in candidate_titles]
        requested = {}
        matched_targets = set()
        for route in profile.routes:
            if route.source_kind != "wikidata_direct":
                continue
            for name in [route.related_subject, *route.aliases]:
                phrase = identity_tokens(name)
                support = sum(bool(spans(title, phrase)) for title in titles)
                if support:
                    requested[" ".join(phrase)] = (name, support, len(phrase))
                    matched_targets.add(getattr(route, "target_qid", ""))
        db = self._connect()
        self._identity_deadline = time.monotonic() + 30
        now = datetime.now(timezone.utc)
        searches_run = 0
        missing = []
        failures = []
        entity_stamps = {}

        def expired(qid):
            stamp = entity_stamps.get(qid)
            return bool(stamp and (now - datetime.fromisoformat(stamp)).total_seconds() >= 7 * 86400)

        def save_entities(entities):
            with db:
                db.executemany("INSERT OR REPLACE INTO identity_entities VALUES (?, ?, ?)",
                               [(qid, json.dumps(entity), now.isoformat()) for qid, entity in entities.items()])
            profile.identity_entities.update(entities)
            entity_stamps.update({qid: now.isoformat() for qid in entities})

        def fetch_metadata(qids):
            try:
                save_entities(self._fetch_entities(qids))
                return True
            except (requests.RequestException, ValueError) as exc:
                failures.append(str(exc))
                return False

        try:
            entity_stamps.update(dict(db.execute("SELECT qid, fetched_at FROM entities")))
            for qid, payload, stamp in db.execute("SELECT qid, payload, fetched_at FROM identity_entities"):
                if stamp >= entity_stamps.get(qid, ""):
                    profile.identity_entities[qid] = json.loads(payload)
                    entity_stamps[qid] = stamp
            # Spend the bounded identity budget on entities actually mentioned
            # in candidates; unrelated graph endpoints cannot help these titles.
            targets = matched_targets - {""}
            needs_metadata = sorted(qid for qid in targets if refresh or expired(qid) or
                                    "descriptions" not in profile.identity_entities.get(qid, {}))[:40]
            if needs_metadata and requested:
                fetch_metadata(needs_metadata)
            context_ids = set()
            for qid in targets:
                for prop in CONTEXT_PREDICATES:
                    for claim in profile.identity_entities.get(qid, {}).get("claims", {}).get(prop, []):
                        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                        if isinstance(value, dict) and re.fullmatch(r"Q\d+", value.get("id", "")):
                            context_ids.add(value["id"])
            absent = sorted(context_ids - profile.identity_entities.keys())[:40]
            if absent and requested:
                fetch_metadata(absent)
            # Check the shortest, most frequently matched names first.
            for key, (name, support, width) in sorted(requested.items(), key=lambda item: (item[1][2], -item[1][1], item[0])):
                cache_key = IDENTITY_VERSION + ":" + key
                cached = db.execute("SELECT qids, fetched_at FROM identity_searches WHERE name=?", (cache_key,)).fetchone()
                if cached and not refresh:
                    age = (now - datetime.fromisoformat(cached[1])).total_seconds()
                    ids = json.loads(cached[0])
                    if age < 7 * 86400 and all(qid in profile.identity_entities and not expired(qid) for qid in ids):
                        profile.checked_identity_names.add(key)
                        continue
                if searches_run >= 12 or time.monotonic() >= self._identity_deadline:
                    missing.append(key)
                    continue
                searches_run += 1
                language = "hi" if re.search(r"[\u0900-\u097f]", name) else "en"
                try:
                    payload = self._api(action="wbsearchentities", search=name, language=language, limit=10)
                    hits = payload.get("search")
                    if not isinstance(hits, list):
                        raise ValueError("Invalid Wikidata identity search response.")
                except (requests.RequestException, ValueError) as exc:
                    failures.append(f"{name}: {exc}")
                    continue
                qids = list(dict.fromkeys(hit["id"] for hit in hits
                                         if re.fullmatch(r"Q\d+", hit.get("id", ""))))
                # Fetch all candidates, not just the intended entity: competing
                # labels/aliases and descriptions are needed to reject collisions.
                needed = [qid for qid in qids if refresh or expired(qid) or qid not in profile.identity_entities
                          or "descriptions" not in profile.identity_entities[qid]]
                if needed and not fetch_metadata(needed):
                    continue
                with db:
                    db.execute("INSERT OR REPLACE INTO identity_searches VALUES (?, ?, ?)",
                               (cache_key, json.dumps(qids), now.isoformat()))
                profile.checked_identity_names.add(key)
        except (requests.RequestException, ValueError) as exc:
            failures.append(str(exc))
        finally:
            self._identity_deadline = None
            db.close()
        if failures:
            profile.warnings.append("Some Wikidata identity checks were unavailable. Weak alias matches without supporting context are withheld.")
            profile.diagnostics["Identity lookup"] = "; ".join(dict.fromkeys(failures))
        if missing:
            profile.warnings.append(f"Identity lookup limit reached for {len(missing)} names; unresolved aliases require supporting context.")
        profile.source_status["Identity checks"] = (
            f"{len(profile.checked_identity_names)} names checked; {searches_run} online searches. "
            "Search coverage is limited and does not establish name uniqueness."
        )
        return profile


def find_wikidata_related_stories(*, query, profile, title_summary, story_months,
                                 match_mode, index_fingerprint, primary_id_lookup,
                                 identity_client=None, refresh_identity=False):
    """Retrieve Wikidata-only coverage after complete primary and lexical exclusions.

    The supplied lookup must return primary IDs across ALL refined pages, never
    fuzzy-only review IDs. Keeping this boundary explicit also makes it testable
    without a running Streamlit or OpenSearch service.
    """
    def query_text(text):
        return " ".join(dict.fromkeys(simple_query_tokens(normalize_refined_alias_text(text))))

    searches = {query_text(query), *(query_text(alias) for alias in profile.query_aliases)} - {""}
    excluded = set()
    for search in sorted(searches):
        excluded.update(str(value).strip() for value in primary_id_lookup(
            normalized_query=search, match_mode=match_mode,
            index_fingerprint=index_fingerprint,
        ))
    primary_excluded_count = len(excluded)

    corpus = title_summary.copy()
    corpus["story_id"] = corpus["story_id"].astype(str).str.strip()
    tokens = corpus["page_title"].apply(
        lambda title: context_tokens(normalize_refined_alias_text(title))
    )
    direct_titles = set(tokens[corpus["story_id"].isin(excluded)]) - {()}
    searched_tokens = set(context_tokens(normalize_refined_alias_text(query)))
    alias_tokens = [context_tokens(normalize_refined_alias_text(alias))
                    for alias in [query, *profile.query_aliases]]
    routes = [route for route in profile.routes if route.source_kind == "wikidata_direct"]
    # Shared words inside another complete related name are not evidence of a
    # direct mention (e.g. Nita Ambani is not Mukesh Ambani). Admission still
    # requires the normal identity gate below, and primary IDs stay excluded.
    related_names = {context_tokens(normalize_refined_alias_text(name))
                     for route in routes for name in [route.related_subject, *route.aliases]}
    related_names = {name for name in related_names if len(name) >= 2 and searched_tokens.intersection(name)}

    def is_direct(title_tokens):
        if title_tokens in direct_titles:
            return True
        if any(spans(title_tokens, alias) for alias in alias_tokens if alias):
            return True
        if not searched_tokens.intersection(title_tokens):
            return False
        covered = {i for name in related_names for start, end in spans(title_tokens, name)
                   for i in range(start, end)}
        return any(token in searched_tokens and i not in covered for i, token in enumerate(title_tokens))

    direct_mask = tokens.apply(is_direct)
    excluded.update(corpus.loc[direct_mask, "story_id"])
    # Type, birthplace and broad classification edges are insufficient to admit
    # all stories about that category/location. Keep them inspectable in profile.
    candidate_gate = RelationshipGate(profile)
    candidates = build_source_related_results(
        title_summary=corpus.drop_duplicates("story_id"), story_months=story_months,
        excluded_story_ids=excluded, routes=routes,
        relationship_validator=candidate_gate,
        title_tokenizer=identity_tokens,
    )
    candidate_count = candidates.attrs["name_match_candidate_count"]
    if identity_client is not None and not candidates.empty:
        identity_client.prepare_identity_catalog(profile, candidates["page_title"].tolist(), refresh=refresh_identity)
    gate = IdentityGate(profile)
    relationship_gate = RelationshipGate(profile)
    relationship_eligible_ids = set()

    def validate_relationship(route, row):
        context = relationship_gate(route, row)
        if context is None:
            return None
        relationship_eligible_ids.add(str(row["story_id"]))
        return context

    results = build_source_related_results(
        title_summary=corpus.loc[corpus["story_id"].isin(candidates["story_id"])].drop_duplicates("story_id"),
        story_months=story_months, excluded_story_ids=excluded, routes=routes,
        identity_validator=gate,
        relationship_validator=validate_relationship,
        title_tokenizer=identity_tokens,
    )
    results.attrs["excluded_count"] = len(excluded)
    results.attrs["primary_excluded_count"] = primary_excluded_count
    results.attrs["lexical_excluded_count"] = len(excluded) - primary_excluded_count
    results.attrs["retrieval_route_count"] = len(routes)
    results.attrs["identity_audit"] = gate.audit
    results.attrs["identity_counts"] = dict(gate.counts)
    # Check context again after identity metadata refresh, which may reveal a
    # competing scope name. Deduplicate failures encountered at both stages.
    audit = {json.dumps(row, sort_keys=True): row for row in [*candidate_gate.audit, *relationship_gate.audit]}
    results.attrs["relationship_audit"] = list(audit.values())[:500]
    results.attrs["relationship_counts"] = {"candidate_stage": dict(candidate_gate.counts),
                                             "final_stage": dict(relationship_gate.counts)}
    results.attrs["relationship_withheld_story_count"] = candidate_count - len(relationship_eligible_ids)
    results.attrs["identity_withheld_story_count"] = len(relationship_eligible_ids) - len(results)
    results.attrs["candidate_count"] = candidate_count
    results.attrs["withheld_story_count"] = candidate_count - len(results)
    results.attrs["relationship_policy_version"] = POLICY_VERSION
    results.attrs["retrieval_pipeline_version"] = SOURCE_RELATED_PIPELINE_VERSION
    results.attrs["query_entity_count"] = len(profile.linked_entities)
    return results
