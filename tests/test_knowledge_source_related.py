import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import numpy as np

from src.knowledge_source_related import (
    _corpus_cooccurrence_relationships,
    _cross_entity_bridge_relationships,
    _analyze_query_structure,
    _estimate_relationship_coverage,
    _query_title_match_strength,
    _select_relationships_for_retrieval,
    _resolve_wikimedia_entity,
    _resolve_wikimedia_query,
    WIKIDATA_PROPERTY_RULES,
    WIKIDATA_PROPERTY_SPECIFICITY,
    discover_source_profile,
    retrieve_source_related_stories,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeKnowledgeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        params = dict(kwargs.get("params", {}))
        self.calls.append({"url": url, "params": params, **kwargs})
        action = params.get("action")
        if action == "wbsearchentities":
            return FakeResponse(
                {
                    "search": [
                        {
                            "id": "Q312",
                            "label": "Apple Inc.",
                            "description": "American technology company",
                        },
                        {
                            "id": "Q89",
                            "label": "apple",
                            "description": "fruit of the apple tree",
                        },
                    ]
                }
            )
        if action == "wbgetentities":
            ids = str(params.get("ids", "")).split("|")
            entities = {
                "Q312": {
                    "labels": {"en": {"value": "Apple Inc."}},
                    "aliases": {"en": [{"value": "Apple"}, {"value": "AAPL"}]},
                    "descriptions": {"en": {"value": "American technology company"}},
                    "sitelinks": {"enwiki": {"title": "Apple Inc."}},
                    "claims": {
                        "P1056": [self._statement("Q48493")],
                        "P169": [self._statement("Q265852")],
                        "P17": [self._statement("Q30")],
                    },
                },
                "Q89": {
                    "labels": {"en": {"value": "apple"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "fruit of the apple tree"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q48493": {
                    "labels": {"en": {"value": "iPhone"}},
                    "aliases": {"en": [{"value": "Apple iPhone"}]},
                    "descriptions": {"en": {"value": "line of smartphones"}},
                },
                "Q265852": {
                    "labels": {"en": {"value": "Tim Cook"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "American business executive"}},
                },
                "Q30": {
                    "labels": {"en": {"value": "United States"}},
                    "aliases": {"en": [{"value": "US"}]},
                    "descriptions": {"en": {"value": "country in North America"}},
                },
            }
            return FakeResponse({"entities": {qid: entities[qid] for qid in ids}})
        if action == "query":
            return FakeResponse(
                {
                    "query": {
                        "pages": [
                            {
                                "title": "Apple Inc.",
                                "redirects": [
                                    {"title": "Apple Computer"},
                                    {"title": "Apple Computer, Inc."},
                                ],
                            }
                        ]
                    }
                }
            )
        raise AssertionError(f"Unexpected request: {url} {params}")

    @staticmethod
    def _statement(qid: str) -> dict[str, object]:
        return {
            "rank": "normal",
            "mainsnak": {
                "datavalue": {
                    "value": {"id": qid},
                }
            },
            "references": [
                {
                    "snaks": {
                        "P854": [
                            {
                                "datavalue": {
                                    "value": "https://example.com/reference"
                                }
                            }
                        ]
                    }
                }
            ],
        }


class ResearchTaxonomyKnowledgeSession:
    """Wikidata fixture covering family, inverse governance, and path edges."""

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        params = dict(kwargs.get("params", {}))
        if "query.wikidata.org" in url:
            return FakeResponse(
                {
                    "results": {
                        "bindings": [
                            {
                                "queryEntity": {
                                    "value": "http://www.wikidata.org/entity/Q9001"
                                },
                                "propertyEntity": {
                                    "value": "http://www.wikidata.org/entity/P3320"
                                },
                                "subject": {
                                    "value": "http://www.wikidata.org/entity/Q9007"
                                },
                                "subjectLabel": {"value": "Jio"},
                                "subjectDescription": {
                                    "value": "Indian telecommunications company"
                                },
                            }
                        ]
                    }
                }
            )
        action = params.get("action")
        if action == "wbgetentities":
            ids = str(params.get("ids", "")).split("|")
            statement = FakeKnowledgeSession._statement
            entities = {
                "Q9001": {
                    "labels": {"en": {"value": "Isha Ambani"}},
                    "aliases": {"en": [{"value": "Isha Mukesh Ambani"}]},
                    "descriptions": {"en": {"value": "Indian heiress"}},
                    "sitelinks": {},
                    "claims": {
                        "P22": [statement("Q9002")],
                        "P25": [statement("Q9003")],
                        "P3373": [statement("Q9004")],
                        "P108": [statement("Q9005")],
                        "P31": [statement("Q5")],
                    },
                },
                "Q9002": {
                    "labels": {"en": {"value": "Mukesh Ambani"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "Indian businessperson"}},
                    "claims": {"P108": [statement("Q9006")]},
                },
                "Q9003": {
                    "labels": {"en": {"value": "Nita Ambani"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "Indian philanthropist"}},
                    "claims": {"P1416": [statement("Q9008")]},
                },
                "Q9004": {
                    "labels": {"en": {"value": "Akash Ambani"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "Indian businessperson"}},
                    "claims": {"P108": [statement("Q9007")]},
                },
                "Q9005": {
                    "labels": {"en": {"value": "Reliance Retail"}},
                    "aliases": {"en": [{"value": "Reliance Retail Limited"}]},
                    "descriptions": {"en": {"value": "Indian retail company"}},
                    "claims": {"P749": [statement("Q9006")]},
                },
                "Q9006": {
                    "labels": {"en": {"value": "Reliance Industries"}},
                    "aliases": {"en": [{"value": "Reliance Industries Limited"}]},
                    "descriptions": {"en": {"value": "Indian conglomerate"}},
                    "claims": {"P355": [statement("Q9007")]},
                },
                "Q9007": {
                    "labels": {"en": {"value": "Jio"}},
                    "aliases": {"en": [{"value": "Reliance Jio"}]},
                    "descriptions": {
                        "en": {"value": "Indian telecommunications company"}
                    },
                    "claims": {},
                },
                "Q9008": {
                    "labels": {"en": {"value": "Reliance Foundation"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "Indian foundation"}},
                    "claims": {},
                },
                "Q5": {
                    "labels": {"en": {"value": "human"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "human being"}},
                    "claims": {},
                },
            }
            return FakeResponse(
                {"entities": {qid: entities[qid] for qid in ids if qid in entities}}
            )
        raise AssertionError(f"Unexpected request: {url} {params}")


class KnowledgeSourceProfileTests(unittest.TestCase):
    def test_internally_created_session_ignores_ambient_proxy_settings(self) -> None:
        session = FakeKnowledgeSession()
        session.trust_env = True  # type: ignore[attr-defined]
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        try:
            with patch(
                "src.knowledge_source_related.requests.Session",
                return_value=session,
            ):
                discover_source_profile(
                    keyword_query="Apple",
                    selected_wikidata_qid="Q312",
                    include_wordnet=False,
                    include_conceptnet=False,
                    include_gdelt=False,
                    force_refresh=True,
                    cache_path=cache_path,
                )
        finally:
            cache_path.unlink(missing_ok=True)

        self.assertFalse(session.trust_env)  # type: ignore[attr-defined]

    def test_person_taxonomy_expands_family_inverse_and_corporate_paths(self) -> None:
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        titles = pd.DataFrame(
            [
                {"story_id": "direct", "page_title": "Isha Ambani attends an event"},
                {"story_id": "family", "page_title": "Mukesh Ambani announces succession plan"},
                {"story_id": "retail", "page_title": "Reliance Retail ranks among valuable firms"},
                {"story_id": "ril", "page_title": "Reliance Industries reports quarterly results"},
                {"story_id": "jio", "page_title": "Jio announces nationwide network upgrade"},
            ]
        )
        try:
            profile = discover_source_profile(
                keyword_query="Isha Ambani",
                title_summary=titles,
                selected_wikidata_qid="Q9001",
                include_wordnet=False,
                include_conceptnet=False,
                include_gdelt=False,
                force_refresh=True,
                cache_path=cache_path,
                session=ResearchTaxonomyKnowledgeSession(),  # type: ignore[arg-type]
            )
        finally:
            cache_path.unlink(missing_ok=True)

        relationships = profile["relationships"]
        subjects = {item["related_subject"] for item in relationships}
        self.assertTrue(
            {"Mukesh Ambani", "Nita Ambani", "Akash Ambani"}.issubset(subjects)
        )
        self.assertTrue(
            {"Reliance Retail", "Reliance Industries", "Jio"}.issubset(subjects)
        )
        jio_edges = [item for item in relationships if item["related_subject"] == "Jio"]
        self.assertTrue(any(item.get("direction") == "incoming_statement_to_query" for item in jio_edges))
        self.assertTrue(any(int(item.get("hop_count", 1)) >= 2 for item in relationships))

        coverage = profile["relationship_coverage_audit"]
        self.assertEqual(coverage["primary_type"], "person")
        covered_families = {
            item["relationship_family"]
            for item in coverage["branches"]
            if item["status"] == "covered"
        }
        self.assertIn("immediate family", covered_families)
        self.assertIn("roles and organizations", covered_families)

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
            excluded_story_ids={"direct"},
        )
        self.assertEqual(
            set(results["story_id"]),
            {"family", "retail", "ril", "jio"},
        )
        self.assertEqual(diagnostics["excluded_primary_count"], 1)
        self.assertGreaterEqual(
            int(results.set_index("story_id").loc["ril", "relationship_hops"]),
            2,
        )

    def test_shared_structured_neighbor_connects_multiple_query_mentions(self) -> None:
        entities = [
            {"qid": "Q1", "label": "Orchid Leader"},
            {"qid": "Q2", "label": "Nimbus"},
        ]
        relationships = [
            {
                "interpretation_id": "wikidata:Q1",
                "related_subject": "Cloud Group",
                "related_subject_id": "Q3",
                "related_subject_type": "company",
                "relationship_role": "INSTITUTIONAL_BRIDGE",
                "predicate_id": "P108",
                "predicate_label": "employer",
                "confidence": 0.78,
                "aliases": [],
                "evidence_urls": ["https://www.wikidata.org/wiki/Q1"],
            },
            {
                "interpretation_id": "wikidata:Q2",
                "related_subject": "Cloud Group",
                "related_subject_id": "Q3",
                "related_subject_type": "company",
                "relationship_role": "INSTITUTIONAL_BRIDGE",
                "predicate_id": "P749",
                "predicate_label": "parent organization",
                "confidence": 0.84,
                "aliases": [],
                "evidence_urls": ["https://www.wikidata.org/wiki/Q2"],
            },
        ]

        bridges = _cross_entity_bridge_relationships(
            resolved_entities=entities,
            relationships=relationships,
        )

        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0]["related_subject"], "Cloud Group")
        self.assertEqual(set(bridges[0]["bridge_source_qids"]), {"Q1", "Q2"})

    def test_automatic_provider_routing_reports_used_or_applicability_states(self) -> None:
        session = FakeKnowledgeSession()
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        gdelt_diagnostics = {
            "article_count": 0,
            "accepted_relationship_count": 0,
        }
        try:
            with patch(
                "src.knowledge_source_related.fetch_gdelt_relationships",
                return_value=([], gdelt_diagnostics),
            ) as gdelt:
                profile = discover_source_profile(
                    keyword_query="Apple",
                    selected_wikidata_qid="Q312",
                    include_wordnet=None,
                    include_conceptnet=None,
                    include_gdelt=None,
                    force_refresh=True,
                    cache_path=cache_path,
                    session=session,  # type: ignore[arg-type]
                )
        finally:
            cache_path.unlink(missing_ok=True)

        gdelt.assert_called_once()
        self.assertEqual(profile["source_status"]["GDELT"]["status"], "no_evidence")
        self.assertEqual(
            profile["source_status"]["ConceptNet"]["status"], "not_applicable"
        )
        self.assertEqual(
            profile["source_status"]["DBpedia"]["status"], "not_integrated"
        )

    def test_wikidata_profile_uses_approved_edges_and_wikipedia_aliases(self) -> None:
        session = FakeKnowledgeSession()
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        try:
            profile = discover_source_profile(
                keyword_query="Apple",
                selected_wikidata_qid="Q312",
                include_wikipedia=True,
                include_wordnet=True,
                include_conceptnet=False,
                include_gdelt=False,
                force_refresh=True,
                cache_path=cache_path,
                session=session,  # type: ignore[arg-type]
            )
        finally:
            cache_path.unlink(missing_ok=True)

        resolved = profile["resolved_entity"]
        self.assertEqual(resolved["qid"], "Q312")
        self.assertIn("Apple Computer", resolved["aliases"])
        relationships = profile["relationships"]
        relationships_by_subject = {
            item["related_subject"]: item for item in relationships
        }
        self.assertFalse(relationships_by_subject["iPhone"]["can_retrieve_standalone"])
        self.assertEqual(
            relationships_by_subject["iPhone"]["property_specificity"],
            "context_required",
        )
        self.assertTrue(relationships_by_subject["Tim Cook"]["can_retrieve_standalone"])
        self.assertEqual(
            relationships_by_subject["Tim Cook"]["property_specificity"],
            "standalone",
        )
        self.assertFalse(
            relationships_by_subject["United States"]["can_retrieve_standalone"]
        )
        self.assertEqual(
            profile["source_status"]["WordNet"]["status"], "not_applicable"
        )
        self.assertEqual(len(profile["interpretations"]), 1)
        self.assertTrue(profile["interpretations"][0]["selected"])
        self.assertIn("acceptance_condition", relationships_by_subject["iPhone"])
        self.assertIn("rejection_rule", relationships_by_subject["iPhone"])

    def test_sparse_profile_is_enriched_from_local_corpus(self) -> None:
        session = FakeKnowledgeSession()
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        title_rows = [
            {
                "story_id": "1",
                "page_title": "Apple partners with Quartz Collective",
                "last_month": "2026-06-01",
            },
            {
                "story_id": "2",
                "page_title": "Quartz Collective launches with Apple",
                "last_month": "2026-06-01",
            },
            *[
                {
                    "story_id": str(index),
                    "page_title": f"Unrelated bulletin {index}",
                    "last_month": "2026-04-01",
                }
                for index in range(3, 103)
            ],
        ]
        try:
            profile = discover_source_profile(
                keyword_query="Apple",
                title_summary=pd.DataFrame(title_rows),
                include_wordnet=False,
                include_conceptnet=False,
                include_gdelt=False,
                force_refresh=True,
                cache_path=cache_path,
                session=session,  # type: ignore[arg-type]
            )
        finally:
            cache_path.unlink(missing_ok=True)

        fallback = profile["density_fallback"]
        self.assertTrue(fallback["sparse"])
        self.assertTrue(fallback["activated"])
        self.assertGreaterEqual(fallback["added_relationship_count"], 1)
        self.assertEqual(profile["source_status"]["Local corpus"]["status"], "complete")
        self.assertIn(
            "Quartz Collective",
            {item["related_subject"] for item in profile["relationships"]},
        )
        corpus_relationships = [
            item
            for item in profile["relationships"]
            if "Local corpus" in item.get("source_names", [])
        ]
        self.assertTrue(corpus_relationships)
        self.assertTrue(
            all(
                item["can_retrieve_standalone"] is False
                for item in corpus_relationships
            )
        )

    def test_second_identical_request_uses_sqlite_cache(self) -> None:
        session = FakeKnowledgeSession()
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        try:
            first = discover_source_profile(
                keyword_query="Apple",
                selected_wikidata_qid="Q312",
                include_wordnet=False,
                include_conceptnet=False,
                include_gdelt=False,
                cache_path=cache_path,
                session=session,  # type: ignore[arg-type]
            )
            call_count = len(session.calls)
            second = discover_source_profile(
                keyword_query="Apple",
                selected_wikidata_qid="Q312",
                include_wordnet=False,
                include_conceptnet=False,
                include_gdelt=False,
                cache_path=cache_path,
                session=session,  # type: ignore[arg-type]
            )
        finally:
            cache_path.unlink(missing_ok=True)

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(len(session.calls), call_count)

    def test_optional_provider_outage_does_not_prevent_partial_profile_cache(self) -> None:
        session = FakeKnowledgeSession()
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        try:
            with patch(
                "src.knowledge_source_related.fetch_gdelt_relationships",
                side_effect=RuntimeError("temporary timeout"),
            ) as gdelt:
                first = discover_source_profile(
                    keyword_query="Apple",
                    selected_wikidata_qid="Q312",
                    include_wordnet=False,
                    include_conceptnet=False,
                    include_gdelt=None,
                    cache_path=cache_path,
                    session=session,  # type: ignore[arg-type]
                )
                second = discover_source_profile(
                    keyword_query="Apple",
                    selected_wikidata_qid="Q312",
                    include_wordnet=False,
                    include_conceptnet=False,
                    include_gdelt=None,
                    cache_path=cache_path,
                    session=session,  # type: ignore[arg-type]
                )
        finally:
            cache_path.unlink(missing_ok=True)

        self.assertEqual(first["source_status"]["GDELT"]["status"], "unavailable")
        self.assertTrue(second["cache_hit"])
        gdelt.assert_called_once()

    def test_normalized_corpus_seeding_accepts_inflection_and_token_gaps(self) -> None:
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Harbor workers begin protests over new rules",
                    "last_month": "2026-06-01",
                },
                {
                    "story_id": "2",
                    "page_title": "Harbor authority faces a prolonged worker protest",
                    "last_month": "2026-06-01",
                },
                {
                    "story_id": "3",
                    "page_title": "Unrelated city bulletin",
                    "last_month": "2026-06-01",
                },
            ]
        )

        _, diagnostics = _corpus_cooccurrence_relationships(
            query="Harbor protest",
            resolved_entity={},
            title_summary=titles,
        )

        self.assertEqual(diagnostics["query_seed_document_count"], 2)
        self.assertGreaterEqual(
            _query_title_match_strength(
                "Harbor authority faces prolonged protests",
                ["Harbor protest"],
            ),
            0.70,
        )

    def test_statistically_strong_corpus_relationships_generate_discovery_candidates(self) -> None:
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Harbor workers protest cargo fees and dock closure",
                    "last_month": "2026-06-01",
                },
                {
                    "story_id": "2",
                    "page_title": "Harbor protests continue over cargo fees after dock closure",
                    "last_month": "2026-06-01",
                },
                {
                    "story_id": "3",
                    "page_title": "Cargo fees and dock closure disrupt regional trade",
                    "last_month": "2026-06-01",
                },
                {
                    "story_id": "4",
                    "page_title": "Cargo fees explained for importers",
                    "last_month": "2026-06-01",
                },
                *[
                    {
                        "story_id": str(index),
                        "page_title": f"Unrelated bulletin number {index}",
                        "last_month": "2026-05-01",
                    }
                    for index in range(5, 105)
                ],
            ]
        )
        relationships, fallback = _corpus_cooccurrence_relationships(
            query="Harbor protest",
            resolved_entity={},
            title_summary=titles,
        )
        profile = {
            "query": "Harbor protest",
            "resolved_entity": {},
            "relationships": relationships,
            "density_fallback": fallback,
        }

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
            excluded_story_ids={"1", "2"},
        )

        self.assertEqual(set(results["story_id"]), {"3", "4"})
        self.assertEqual(
            set(results["result_tier"]), {"Corpus-supported discovery"}
        )
        self.assertEqual(
            diagnostics["corpus_supported_relationship_count"],
            len(relationships),
        )


class WikipediaFallbackSession:
    def get(self, url: str, **kwargs: object) -> FakeResponse:
        params = dict(kwargs.get("params", {}))
        action = params.get("action")
        if action == "wbsearchentities":
            return FakeResponse({"search": []})
        if action == "query" and params.get("list") == "search":
            return FakeResponse(
                {
                    "query": {
                        "search": [
                            {"pageid": 1, "title": "2026 Harbor dock strike"},
                            {"pageid": 2, "title": "Harbor strikes"},
                        ]
                    }
                }
            )
        if action == "wbgetentities" and params.get("sites") == "enwiki":
            return FakeResponse(
                {
                    "entities": {
                        "Q700": {
                            "labels": {"en": {"value": "2026 Harbor dock strike"}},
                            "aliases": {"en": [{"value": "Harbor dock strike"}]},
                            "descriptions": {"en": {"value": "labor strike event in Harbor"}},
                            "sitelinks": {"enwiki": {"title": "2026 Harbor dock strike"}},
                        },
                        "Q701": {
                            "labels": {"en": {"value": "Harbor strikes"}},
                            "aliases": {"en": []},
                            "descriptions": {"en": {"value": "Wikimedia disambiguation page"}},
                            "sitelinks": {"enwiki": {"title": "Harbor strikes"}},
                        },
                    }
                }
            )
        if action == "wbgetentities":
            ids = str(params.get("ids", "")).split("|")
            entities = {
                "Q700": {
                    "labels": {"en": {"value": "2026 Harbor dock strike"}},
                    "aliases": {"en": [{"value": "Harbor dock strike"}]},
                    "descriptions": {"en": {"value": "labor strike event in Harbor"}},
                    "sitelinks": {"enwiki": {"title": "2026 Harbor dock strike"}},
                    "claims": {},
                },
                "Q701": {
                    "labels": {"en": {"value": "Harbor strikes"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "Wikimedia disambiguation page"}},
                    "sitelinks": {"enwiki": {"title": "Harbor strikes"}},
                    "claims": {},
                },
            }
            return FakeResponse({"entities": {qid: entities[qid] for qid in ids}})
        raise AssertionError(f"Unexpected request: {url} {params}")


class EmptyKnowledgeSession:
    def get(self, url: str, **kwargs: object) -> FakeResponse:
        params = dict(kwargs.get("params", {}))
        if params.get("action") == "wbsearchentities":
            return FakeResponse({"search": []})
        if params.get("action") == "query" and params.get("list") == "search":
            return FakeResponse({"query": {"search": []}})
        raise AssertionError(f"Unexpected request: {url} {params}")


class MultiMentionKnowledgeSession:
    def get(self, url: str, **kwargs: object) -> FakeResponse:
        params = dict(kwargs.get("params", {}))
        action = params.get("action")
        if action == "wbsearchentities":
            query = str(params.get("search", "")).casefold()
            matches = {
                "orchid leader": {
                    "id": "Q8101",
                    "label": "Orchid Leader",
                    "description": "businessperson",
                },
                "nimbus": {
                    "id": "Q8102",
                    "label": "Nimbus",
                    "description": "technology company",
                },
                "cinema": {
                    "id": "Q8103",
                    "label": "Cinema",
                    "description": "motion-picture concept",
                },
                "aurora arts festival": {
                    "id": "Q8104",
                    "label": "Aurora Arts Festival",
                    "description": "annual arts festival event",
                },
                "riverland": {
                    "id": "Q8105",
                    "label": "Riverland",
                    "description": "geographic region",
                },
                "election": {
                    "id": "Q8106",
                    "label": "Election",
                    "description": "decision-making event",
                },
                "old port": {
                    "id": "Q8107",
                    "label": "Old Port",
                    "description": "city district",
                },
                "tourism": {
                    "id": "Q8108",
                    "label": "Tourism",
                    "description": "travel concept",
                },
                "moon day": {
                    "id": "Q8109",
                    "label": "Moon Day",
                    "description": "cultural observance",
                },
            }
            match = matches.get(query)
            return FakeResponse({"search": [match] if match else []})
        if action == "wbgetentities":
            ids = str(params.get("ids", "")).split("|")
            entities = {
                "Q8101": {
                    "labels": {"en": {"value": "Orchid Leader"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "businessperson"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8102": {
                    "labels": {"en": {"value": "Nimbus"}},
                    "aliases": {"en": [{"value": "Nimbus Technologies"}]},
                    "descriptions": {"en": {"value": "technology company"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8103": {
                    "labels": {"en": {"value": "Cinema"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "motion-picture concept"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8104": {
                    "labels": {"en": {"value": "Aurora Arts Festival"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "annual arts festival event"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8105": {
                    "labels": {"en": {"value": "Riverland"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "geographic region"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8106": {
                    "labels": {"en": {"value": "Election"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "decision-making event"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8107": {
                    "labels": {"en": {"value": "Old Port"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "city district"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8108": {
                    "labels": {"en": {"value": "Tourism"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "travel concept"}},
                    "sitelinks": {},
                    "claims": {},
                },
                "Q8109": {
                    "labels": {"en": {"value": "Moon Day"}},
                    "aliases": {"en": []},
                    "descriptions": {"en": {"value": "cultural observance"}},
                    "sitelinks": {},
                    "claims": {},
                },
            }
            return FakeResponse(
                {"entities": {qid: entities[qid] for qid in ids if qid in entities}}
            )
        raise AssertionError(f"Unexpected request: {url} {params}")


class EntityResolutionTests(unittest.TestCase):
    def test_query_structure_extracts_generic_spans_and_time_constraints(self) -> None:
        cases = {
            "Orchid Leader cinema": "orchid leader",
            "Riverland election 2028": "riverland election",
            "Old Port tourism": "old port",
        }
        for query, expected_span in cases.items():
            with self.subTest(query=query):
                structure = _analyze_query_structure(query)
                spans = {
                    item["text"] for item in structure["mention_candidates"]
                }
                self.assertIn(expected_span, spans)
        self.assertEqual(
            _analyze_query_structure("Riverland election 2028")[
                "temporal_constraints"
            ],
            ["2028"],
        )

    def test_compound_query_resolves_non_overlapping_mentions(self) -> None:
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Orchid Leader discusses Nimbus investment",
                },
                {
                    "story_id": "2",
                    "page_title": "Nimbus appoints a new technology chief",
                },
            ]
        )
        resolution = _resolve_wikimedia_query(
            MultiMentionKnowledgeSession(),  # type: ignore[arg-type]
            query="Orchid Leader Nimbus",
            title_summary=titles,
        )

        self.assertEqual(resolution["strategy"], "multi_mention")
        self.assertEqual(
            {item["qid"] for item in resolution["selected_mentions"]},
            {"Q8101", "Q8102"},
        )

    def test_resolution_covers_multiple_query_shapes(self) -> None:
        cases = {
            "Nimbus": {"Q8102"},
            "Aurora Arts Festival": {"Q8104"},
            "Orchid Leader cinema": {"Q8101", "Q8103"},
            "Riverland election": {"Q8105", "Q8106"},
            "Old Port tourism": {"Q8107", "Q8108"},
            "Moon Day 2028": {"Q8109"},
        }
        for query, expected_qids in cases.items():
            with self.subTest(query=query):
                resolution = _resolve_wikimedia_query(
                    MultiMentionKnowledgeSession(),  # type: ignore[arg-type]
                    query=query,
                    title_summary=pd.DataFrame(),
                )
                actual_qids = {
                    item["qid"] for item in resolution["selected_mentions"]
                }
                self.assertEqual(actual_qids, expected_qids)

    def test_english_wikipedia_fallback_is_disabled_by_default(self) -> None:
        resolution = _resolve_wikimedia_entity(
            WikipediaFallbackSession(),  # type: ignore[arg-type]
            query="Harbor strike",
            title_summary=pd.DataFrame(),
        )

        self.assertFalse(resolution["wikipedia_search_used"])
        self.assertEqual(resolution["wikipedia_candidate_count"], 0)
        self.assertEqual(resolution["selected_qid"], "")

    def test_wikipedia_full_text_fallback_maps_and_scores_qids(self) -> None:
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Harbor dock workers begin strike"},
                {"story_id": "2", "page_title": "Strike closes Harbor terminal"},
            ]
        )

        resolution = _resolve_wikimedia_entity(
            WikipediaFallbackSession(),  # type: ignore[arg-type]
            query="Harbor strike",
            title_summary=titles,
            include_wikipedia=True,
        )

        self.assertEqual(resolution["selected_qid"], "Q700")
        self.assertTrue(resolution["wikipedia_search_used"])
        candidates = resolution["candidates"]
        self.assertTrue(candidates[0]["selected"])
        self.assertTrue(candidates[1]["is_disambiguation"])

    def test_no_external_entity_degrades_to_local_corpus_profile(self) -> None:
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Harbor workers protest cargo fees and dock closure",
                    "last_month": "2026-06-01",
                },
                {
                    "story_id": "2",
                    "page_title": "Harbor protests continue over cargo fees after dock closure",
                    "last_month": "2026-06-01",
                },
                *[
                    {
                        "story_id": str(index),
                        "page_title": f"Unrelated bulletin number {index}",
                        "last_month": "2026-05-01",
                    }
                    for index in range(3, 103)
                ],
            ]
        )
        cache_path = Path.cwd() / ".tmp" / f"profiles-{uuid4().hex}.db"
        try:
            profile = discover_source_profile(
                keyword_query="Harbor protest",
                title_summary=titles,
                include_wordnet=False,
                include_conceptnet=False,
                include_gdelt=False,
                force_refresh=True,
                cache_path=cache_path,
                session=EmptyKnowledgeSession(),  # type: ignore[arg-type]
            )
        finally:
            cache_path.unlink(missing_ok=True)

        self.assertFalse(profile["resolved_entity"])
        self.assertEqual(profile["resolution_status"], "corpus_or_lexical_only")
        self.assertGreater(len(profile["relationships"]), 0)
        self.assertEqual(profile["source_status"]["Local corpus"]["status"], "complete")


class KnowledgeSourceRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = {
            "query": "Apple",
            "resolved_entity": {
                "label": "Apple Inc.",
                "aliases": ["Apple", "AAPL", "Apple Computer"],
            },
            "relationships": [
                {
                    "related_subject": "iPhone",
                    "aliases": ["Apple iPhone"],
                    "relationship_role": "QUERY_SPECIFIC_MANIFESTATION",
                    "factual_bridge": "iPhone is a product produced by Apple Inc.",
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": ["https://www.wikidata.org/wiki/Q312"],
                    "confidence": 0.92,
                },
                {
                    "related_subject": "Tim Cook",
                    "aliases": [],
                    "relationship_role": "DIRECT_PARTICIPANT",
                    "factual_bridge": "Tim Cook is an executive of Apple Inc.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Apple", "Apple Inc."],
                    "source_names": ["Wikidata"],
                    "evidence_urls": ["https://www.wikidata.org/wiki/Q312"],
                    "confidence": 0.84,
                },
                {
                    "related_subject": "United States",
                    "aliases": ["US"],
                    "relationship_role": "GEOGRAPHIC_CONTEXT",
                    "factual_bridge": "Apple Inc. is associated with the United States.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Apple"],
                    "source_names": ["Wikidata"],
                    "evidence_urls": ["https://www.wikidata.org/wiki/Q312"],
                    "confidence": 0.68,
                },
            ],
        }
        self.titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Apple reports record quarterly revenue",
                    "total_views": 100,
                },
                {
                    "story_id": "2",
                    "page_title": "iPhone demand rises sharply in India",
                    "total_views": 90,
                },
                {
                    "story_id": "3",
                    "page_title": "Tim Cook attends a charity dinner",
                    "total_views": 80,
                },
                {
                    "story_id": "4",
                    "page_title": "Tim Cook discusses iPhone supply constraints",
                    "total_views": 70,
                },
                {
                    "story_id": "5",
                    "page_title": "United States election campaigning intensifies",
                    "total_views": 60,
                },
                {
                    "story_id": "6",
                    "page_title": "AAPL shares rise after earnings",
                    "total_views": 50,
                },
            ]
        )

    def test_standalone_and_context_supported_titles_are_kept(self) -> None:
        results, diagnostics = retrieve_source_related_stories(
            profile=self.profile,
            title_summary=self.titles,
        )

        self.assertEqual(set(results["story_id"]), {"2", "4"})
        by_id = results.set_index("story_id")
        self.assertEqual(by_id.loc["2", "result_tier"], "High confidence")
        self.assertEqual(by_id.loc["4", "result_tier"], "High confidence")
        self.assertIn("iphone", by_id.loc["4", "matched_title_evidence"])
        self.assertEqual(diagnostics["candidate_count"], 2)
        self.assertEqual(diagnostics["retrieval_fusion"]["method"], "tiered_rrf")
        self.assertIn("lexical_score", results.columns)
        self.assertIn("rrf_score", results.columns)

    def test_query_identity_aliases_and_context_only_endpoints_are_rejected(self) -> None:
        results, diagnostics = retrieve_source_related_stories(
            profile=self.profile,
            title_summary=self.titles,
            excluded_story_ids={"2"},
        )

        self.assertEqual(set(results["story_id"]), {"4"})
        self.assertNotIn("3", set(results["story_id"]))
        self.assertNotIn("5", set(results["story_id"]))
        self.assertNotIn("6", set(results["story_id"]))
        self.assertGreaterEqual(diagnostics["primary_overlap_count"], 3)
        self.assertFalse(diagnostics["direct_fallback_used"])
        self.assertFalse(diagnostics["direct_evidence_included"])
        self.assertEqual(
            diagnostics["excluded_primary_count"],
            diagnostics["primary_overlap_count"],
        )

    def test_direct_evidence_is_not_returned_when_no_expanded_candidate_exists(self) -> None:
        profile = {
            "query": "Orchid Leader Nimbus",
            "resolved_entity": {"label": "Orchid Leader", "aliases": []},
            "resolved_entities": [
                {
                    "label": "Orchid Leader",
                    "aliases": [],
                    "matched_mention": "Orchid Leader",
                },
                {
                    "label": "Nimbus",
                    "aliases": ["Nimbus Technologies"],
                    "matched_mention": "Nimbus",
                },
            ],
            "relationships": [
                {
                    "related_subject": "Cloud Launch",
                    "aliases": [],
                    "relationship_role": "CURRENT_EVENTS_ASSOCIATION",
                    "factual_bridge": "Cloud Launch is linked to the query.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Orchid Leader", "Nimbus"],
                    "source_names": ["Local corpus"],
                    "evidence_urls": [],
                    "confidence": 0.82,
                    "evidence_kind": "current_events",
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Orchid Leader joins Nimbus Cloud Launch",
                    "total_views": 10,
                }
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
            excluded_story_ids={"1"},
        )

        self.assertTrue(results.empty)
        self.assertFalse(diagnostics["direct_fallback_used"])
        self.assertFalse(diagnostics["direct_evidence_included"])
        self.assertEqual(diagnostics["excluded_primary_count"], 1)
        self.assertIn("Search Results refined", diagnostics["reason"])

    def test_single_entity_direct_titles_are_excluded_but_related_subjects_remain(self) -> None:
        profile = {
            "query": "Haridwar",
            "resolved_entity": {"label": "Haridwar", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Somvati Amavasya",
                    "aliases": [],
                    "relationship_role": "NAMED_EVENT",
                    "factual_bridge": "Somvati Amavasya is associated with Haridwar.",
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.90,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "direct",
                    "page_title": "Haridwar special trains for Somvati Amavasya pilgrims",
                },
                {
                    "story_id": "related",
                    "page_title": "Somvati Amavasya bathing schedule announced",
                },
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["related"])
        self.assertEqual(diagnostics["excluded_primary_count"], 1)

    def test_every_refined_tier_id_is_authoritatively_excluded(self) -> None:
        profile = {
            "query": "Gujarat elections",
            "resolved_entity": {},
            "relationships": [
                {
                    "related_subject": "Assembly polls",
                    "aliases": [],
                    "relationship_role": "NAMED_EVENT",
                    "factual_bridge": "Assembly polls are related to the query.",
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.88,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {"story_id": "root", "page_title": "Assembly polls calendar"},
                {"story_id": "fuzzy", "page_title": "Assembly polls opinion survey"},
                {"story_id": "indirect", "page_title": "Assembly polls security plan"},
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
            excluded_story_ids={"root", "fuzzy"},
        )

        self.assertEqual(list(results["story_id"]), ["indirect"])
        self.assertEqual(diagnostics["excluded_primary_count"], 2)

    def test_broad_derived_association_cannot_generate_even_with_query_component(self) -> None:
        profile = {
            "query": "Orchid Leader Nimbus",
            "resolved_entities": [
                {"label": "Orchid Leader", "aliases": [], "matched_mention": "Orchid Leader"},
                {"label": "Nimbus", "aliases": [], "matched_mention": "Nimbus"},
            ],
            "relationships": [
                {
                    "related_subject": "Net Worth",
                    "aliases": [],
                    "relationship_role": "CURRENT_EVENTS_ASSOCIATION",
                    "factual_bridge": "Net Worth co-occurred with the query.",
                    "predicate_id": "corpus:recency_cooccurrence",
                    "corpus_high_specificity": False,
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Orchid Leader Nimbus"],
                    "source_names": ["Local corpus"],
                    "evidence_urls": [],
                    "confidence": 0.82,
                    "evidence_kind": "current_events",
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Rival billionaire net worth rises"},
                {"story_id": "2", "page_title": "Orchid Leader net worth disclosed"},
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)

    def test_symbolic_brand_alias_cannot_collapse_to_generic_word(self) -> None:
        profile = {
            "query": "Apple",
            "resolved_entity": {"label": "Apple Inc.", "aliases": ["Apple"]},
            "relationships": [
                {
                    "related_subject": "Apple Watch",
                    "aliases": ["iWatch", "AppleWatch", " Watch"],
                    "relationship_role": "QUERY_SPECIFIC_MANIFESTATION",
                    "factual_bridge": "Apple Watch is a product of Apple Inc.",
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.92,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Five films you must watch"},
                {"story_id": "2", "page_title": "AppleWatch gains a health feature"},
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["2"])

    def test_acronym_alias_cannot_qualify_standalone_relationship(self) -> None:
        profile = {
            "query": "Iran",
            "resolved_entity": {"label": "Iran", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Chemical Weapons Convention",
                    "aliases": ["CWC"],
                    "relationship_role": "NAMED_EVENT",
                    "factual_bridge": "Iran participated in the convention.",
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.88,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "CWC orders a new river-water assessment",
                },
                {
                    "story_id": "2",
                    "page_title": "Chemical Weapons Convention meeting opens",
                },
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["2"])

    def test_two_weak_contextual_neighbors_cannot_support_each_other(self) -> None:
        profile = {
            "query": "Apple",
            "resolved_entity": {"label": "Apple Inc.", "aliases": ["Apple"]},
            "relationships": [
                {
                    "related_subject": "Tim Cook",
                    "aliases": [],
                    "relationship_role": "DIRECT_PARTICIPANT",
                    "factual_bridge": "Tim Cook is an Apple executive.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Apple"],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.84,
                },
                {
                    "related_subject": "Foxconn",
                    "aliases": [],
                    "relationship_role": "INSTITUTIONAL_BRIDGE",
                    "factual_bridge": "Foxconn has an institutional bridge to Apple.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Apple"],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.80,
                },
            ],
        }
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Tim Cook attends a charity dinner"},
                {"story_id": "2", "page_title": "Tim Cook meets Foxconn leadership"},
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)

    def test_generic_classification_and_occupation_never_generate_candidates(self) -> None:
        profile = {
            "query": "Vaibhav Suryavanshi",
            "resolved_entity": {
                "label": "Vaibhav Suryavanshi",
                "aliases": ["Vaibhav Suryavanshi"],
            },
            "relationships": [
                {
                    "related_subject": "human",
                    "aliases": [],
                    "relationship_role": "GENERIC_TOPIC",
                    "predicate_id": "P31",
                    "factual_bridge": "Vaibhav Suryavanshi is classified as human.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "confidence": 0.72,
                },
                {
                    "related_subject": "cricketer",
                    "aliases": [],
                    "relationship_role": "INSTITUTIONAL_BRIDGE",
                    "predicate_id": "P106",
                    "factual_bridge": "Vaibhav Suryavanshi has occupation cricketer.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "confidence": 0.74,
                },
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "human",
                    "page_title": (
                        "Human footprint on Sri Lanka dates back 57 thousand years"
                    ),
                },
                {
                    "story_id": "occupation",
                    "page_title": "Veteran cricketer attends a film premiere",
                },
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)
        self.assertEqual(diagnostics["metadata_only_relationship_count"], 2)

    def test_corpus_associations_cannot_cross_validate_one_another(self) -> None:
        profile = {
            "query": "Vaibhav Suryavanshi",
            "resolved_entity": {"label": "Vaibhav Suryavanshi", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Sri Lanka",
                    "aliases": [],
                    "relationship_role": "CURRENT_EVENTS_ASSOCIATION",
                    "predicate_id": "corpus:recency_cooccurrence",
                    "factual_bridge": "Sri Lanka co-occurred with the query.",
                    "corpus_high_specificity": True,
                    "can_retrieve_standalone": False,
                    "required_title_cues": [],
                    "source_names": ["Local corpus"],
                    "evidence_kind": "current_events",
                    "confidence": 0.79,
                },
                {
                    "related_subject": "Ireland",
                    "aliases": [],
                    "relationship_role": "CURRENT_EVENTS_ASSOCIATION",
                    "predicate_id": "corpus:recency_cooccurrence",
                    "factual_bridge": "Ireland co-occurred with the query.",
                    "corpus_high_specificity": True,
                    "can_retrieve_standalone": False,
                    "required_title_cues": [],
                    "source_names": ["Local corpus"],
                    "evidence_kind": "current_events",
                    "confidence": 0.79,
                },
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Sri Lanka women beat Ireland in the series",
                }
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)
        self.assertEqual(diagnostics["corroboration_only_relationship_count"], 2)

    def test_named_team_relationship_retrieves_central_team_story(self) -> None:
        self.assertTrue(WIKIDATA_PROPERTY_RULES["P54"].can_retrieve_standalone)
        self.assertEqual(WIKIDATA_PROPERTY_SPECIFICITY["P54"], "standalone")
        profile = {
            "query": "Vaibhav Suryavanshi",
            "resolved_entity": {"label": "Vaibhav Suryavanshi", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Rajasthan Royals",
                    "aliases": ["Rajasthan Royals cricket team"],
                    "relationship_role": "INSTITUTIONAL_BRIDGE",
                    "predicate_id": "P54",
                    "predicate_label": "member of sports team",
                    "factual_bridge": (
                        "Vaibhav Suryavanshi is a member of Rajasthan Royals."
                    ),
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "confidence": 0.86,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "direct",
                    "page_title": "Vaibhav Suryavanshi stars for Rajasthan Royals",
                },
                {
                    "story_id": "team",
                    "page_title": "Rajasthan Royals announce their retained squad",
                },
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["team"])
        self.assertEqual(results.iloc[0]["related_subject"], "Rajasthan Royals")
        self.assertGreaterEqual(
            float(results.iloc[0]["title_subject_centrality"]),
            diagnostics["minimum_title_subject_centrality"],
        )
        self.assertEqual(diagnostics["excluded_primary_count"], 1)

    def test_late_incidental_team_mention_fails_title_centrality_gate(self) -> None:
        profile = {
            "query": "Vaibhav Suryavanshi",
            "resolved_entity": {"label": "Vaibhav Suryavanshi", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Rajasthan Royals",
                    "aliases": [],
                    "relationship_role": "INSTITUTIONAL_BRIDGE",
                    "predicate_id": "P54",
                    "factual_bridge": (
                        "Vaibhav Suryavanshi is a member of Rajasthan Royals."
                    ),
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "confidence": 0.86,
                },
                {
                    "related_subject": "cricketer",
                    "aliases": [],
                    "relationship_role": "INSTITUTIONAL_BRIDGE",
                    "predicate_id": "P106",
                    "factual_bridge": "Vaibhav Suryavanshi is a cricketer.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "confidence": 0.74,
                },
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "gossip",
                    "page_title": (
                        "Television celebrity makes allegation after private message "
                        "from Rajasthan Royals cricketer"
                    ),
                }
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)
        self.assertEqual(diagnostics["centrality_rejection_count"], 1)

    def test_semantic_subject_gate_rejects_early_incidental_affiliation(self) -> None:
        profile = {
            "query": "Vaibhav Suryavanshi",
            "resolved_entity": {"label": "Vaibhav Suryavanshi", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Rajasthan Royals",
                    "related_subject_type": "Indian Premier League cricket team",
                    "aliases": [],
                    "relationship_family": "team membership",
                    "relationship_role": "INSTITUTIONAL_BRIDGE",
                    "predicate_id": "P54",
                    "predicate_label": "member of sports team",
                    "factual_bridge": (
                        "Vaibhav Suryavanshi is a member of Rajasthan Royals."
                    ),
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "confidence": 0.86,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "gossip",
                    "page_title": (
                        "Nehal Vadoliya said Rajasthan Royals divorced cricketer "
                        "messaged her IPL 2026"
                    ),
                },
                {
                    "story_id": "team",
                    "page_title": (
                        "Rajasthan Royals captain Riyan Parag statement IPL 2026"
                    ),
                },
            ]
        )
        encoded = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.49, 0.871722],
                [0.56, 0.828493],
            ],
            dtype=np.float32,
        )

        with patch(
            "src.relationship_embeddings.encode_relationship_texts",
            return_value=encoded,
        ):
            results, diagnostics = retrieve_source_related_stories(
                profile=profile,
                title_summary=titles,
                use_semantic_centrality=True,
            )

        self.assertEqual(list(results["story_id"]), ["team"])
        self.assertEqual(diagnostics["semantic_centrality"]["processed"], 2)
        self.assertEqual(diagnostics["semantic_centrality"]["rejected"], 1)

    def test_similar_spelling_is_not_implicitly_expanded(self) -> None:
        profile = {
            "query": "Monsoon festivals",
            "resolved_entity": {},
            "relationships": [
                {
                    "related_subject": "Sawan",
                    "aliases": [],
                    "relationship_role": "NAMED_EVENT",
                    "factual_bridge": "Sawan is a monsoon observance period.",
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.90,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Saavan rituals begin this week"},
                {"story_id": "2", "page_title": "Unrelated monsoon bulletin"},
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)
        self.assertEqual(diagnostics["exact_match_count"], 0)

    def test_source_supplied_alias_can_match_without_phonetic_rules(self) -> None:
        profile = {
            "query": "Market topic",
            "resolved_entity": {},
            "relationships": [
                {
                    "related_subject": "Example Holdings",
                    "aliases": ["Example Group"],
                    "relationship_role": "ORGANIZATION",
                    "factual_bridge": "Example Holdings is linked to the market topic.",
                    "can_retrieve_standalone": True,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.90,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Example Group announces expansion"},
                {"story_id": "2", "page_title": "Unrelated market bulletin"},
            ]
        )

        results, diagnostics = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["1"])
        self.assertEqual(results.iloc[0]["retrieval_methods"], "exact")
        self.assertEqual(diagnostics["exact_match_count"], 1)

    def test_distant_corporate_path_requires_exact_entity_surface(self) -> None:
        profile = {
            "query": "Isha Ambani",
            "resolved_entity": {"label": "Isha Ambani", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Jio Platforms",
                    "aliases": ["Jio Platform"],
                    "relationship_family": "corporate and institutional network",
                    "relationship_role": "QUERY_SPECIFIC_MANIFESTATION",
                    "factual_bridge": (
                        "Isha Ambani -> Reliance Retail -> Reliance Industries "
                        "-> Jio Platforms."
                    ),
                    "can_retrieve_standalone": True,
                    "requires_exact_surface": True,
                    "hop_count": 3,
                    "required_title_cues": [],
                    "source_names": ["Wikidata"],
                    "evidence_urls": [],
                    "confidence": 0.68,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "generic",
                    "page_title": "Movie begins streaming on Jio platform",
                },
                {
                    "story_id": "entity",
                    "page_title": "Jio Platforms IPO preparations accelerate",
                },
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["entity"])
        self.assertEqual(int(results.iloc[0]["relationship_hops"]), 3)

    def test_effective_coverage_detects_dense_but_unusable_graph(self) -> None:
        relationships = [
            {
                "related_subject": f"Endpoint {index}",
                "aliases": [],
                "relationship_role": "DIRECT_PARTICIPANT",
                "can_retrieve_standalone": False,
                "required_title_cues": ["Balen Shah"],
            }
            for index in range(6)
        ]
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Balen Shah addresses residents"},
                {"story_id": "2", "page_title": "Kathmandu transport plan approved"},
            ]
        )

        coverage = _estimate_relationship_coverage(
            query="Balen Shah",
            resolved_entity={"label": "Balendra Shah", "aliases": []},
            relationships=relationships,
            title_summary=titles,
        )

        self.assertEqual(coverage["eligible_title_count"], 1)
        self.assertEqual(coverage["matched_non_primary_title_count"], 0)

    def test_adaptive_centrality_keeps_unambiguous_one_word_entity(self) -> None:
        profile = {
            "query": "Example country",
            "resolved_entity": {"label": "Example country", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Egypt",
                    "related_subject_type": "country",
                    "aliases": [],
                    "relationship_role": "GEOGRAPHIC_CONTEXT",
                    "predicate_id": "P47",
                    "predicate_label": "shares border with",
                    "factual_bridge": "Example country shares a border with Egypt.",
                    "can_retrieve_standalone": True,
                    "source_names": ["Wikidata"],
                    "confidence": 0.88,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": (
                        "Egypt announces new regional border security measures after "
                        "an emergency cabinet meeting today"
                    ),
                }
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["1"])

    def test_corpus_evidence_can_complete_a_multi_component_query(self) -> None:
        profile = {
            "query": "Metro weather",
            "resolved_entities": [
                {
                    "qid": "Q1",
                    "label": "Metro",
                    "description": "capital city in Exampleland",
                    "matched_mention": "Metro",
                },
                {
                    "qid": "Q2",
                    "label": "weather",
                    "description": "state of the atmosphere",
                    "matched_mention": "weather",
                },
            ],
            "relationships": [
                {
                    "related_subject": "Monsoon",
                    "related_subject_type": "local title-corpus association",
                    "aliases": [],
                    "relationship_role": "CURRENT_EVENTS_ASSOCIATION",
                    "predicate_id": "corpus:recency_cooccurrence",
                    "predicate_label": "recency-weighted co-occurrence",
                    "factual_bridge": "Monsoon repeatedly co-occurred with Metro weather.",
                    "can_retrieve_standalone": False,
                    "source_names": ["Local corpus"],
                    "evidence_kind": "current_events",
                    "corpus_high_specificity": False,
                    "corpus_support": 3,
                    "corpus_pmi": 5.0,
                    "recency_weighted_support": 2.5,
                    "corpus_association_precision": 0.05,
                    "confidence": 0.80,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "related",
                    "page_title": "Metro monsoon alert issued after heavy rain",
                },
                {"story_id": "broad", "page_title": "Monsoon reaches the coast"},
                {
                    "story_id": "topic-only",
                    "page_title": "Weather monsoon outlook across northern districts",
                },
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertEqual(list(results["story_id"]), ["related"])
        self.assertEqual(
            results.iloc[0]["result_tier"], "Corpus-supported discovery"
        )

    def test_place_graph_path_cannot_retrieve_unrelated_endpoint_story(self) -> None:
        profile = {
            "query": "Example country",
            "resolved_entities": [
                {
                    "qid": "Q1",
                    "label": "Example country",
                    "description": "country",
                    "matched_mention": "Example country",
                }
            ],
            "relationships": [
                {
                    "interpretation_id": "wikidata:Q1",
                    "related_subject": "Egypt",
                    "related_subject_type": "country",
                    "aliases": [],
                    "relationship_family": "corporate and institutional network",
                    "relationship_role": "QUERY_SPECIFIC_MANIFESTATION",
                    "predicate_id": "wikidata:path:P361>P527",
                    "predicate_label": "2-hop network",
                    "factual_bridge": "A distant graph path reaches Egypt.",
                    "can_retrieve_standalone": True,
                    "hop_count": 2,
                    "source_names": ["Wikidata"],
                    "confidence": 0.70,
                }
            ],
        }
        titles = pd.DataFrame(
            [{"story_id": "1", "page_title": "Egypt wins football qualifier"}]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)

    def test_same_component_context_does_not_fake_composite_coverage(self) -> None:
        profile = {
            "query": "Metro weather",
            "resolved_entities": [
                {"qid": "Q1", "label": "Metro", "matched_mention": "Metro"},
                {"qid": "Q2", "label": "weather", "matched_mention": "weather"},
            ],
            "relationships": [
                {
                    "interpretation_id": "wikidata:Q1",
                    "related_subject": "Exampleland",
                    "aliases": [],
                    "relationship_role": "GEOGRAPHIC_CONTEXT",
                    "predicate_id": "P17",
                    "factual_bridge": "Metro is in Exampleland.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Metro"],
                    "source_names": ["Wikidata"],
                    "confidence": 0.70,
                }
            ],
        }
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Metro Exampleland transport corridor approved",
                }
            ]
        )

        results, _ = retrieve_source_related_stories(
            profile=profile,
            title_summary=titles,
        )

        self.assertTrue(results.empty)

    def test_semantic_all_rejected_outcome_recovers_only_evidence_backed_rows(self) -> None:
        profile = {
            "query": "Example country",
            "resolved_entity": {"label": "Example country", "aliases": []},
            "relationships": [
                {
                    "related_subject": "Egypt",
                    "related_subject_type": "country",
                    "aliases": [],
                    "relationship_role": "GEOGRAPHIC_CONTEXT",
                    "predicate_id": "P47",
                    "predicate_label": "shares border with",
                    "factual_bridge": "Example country shares a border with Egypt.",
                    "can_retrieve_standalone": True,
                    "source_names": ["Wikidata"],
                    "confidence": 0.88,
                },
                {
                    "related_subject": "Jordan",
                    "related_subject_type": "country",
                    "aliases": [],
                    "relationship_role": "GEOGRAPHIC_CONTEXT",
                    "predicate_id": "P47",
                    "predicate_label": "shares border with",
                    "factual_bridge": "Example country shares a border with Jordan.",
                    "can_retrieve_standalone": True,
                    "source_names": ["Wikidata"],
                    "confidence": 0.88,
                },
            ],
        }
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Egypt announces border talks"},
                {"story_id": "2", "page_title": "Jordan announces border talks"},
            ]
        )
        encoded = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.40, 0.9165],
                [0.35, 0.9367],
            ],
            dtype=np.float32,
        )

        with patch(
            "src.relationship_embeddings.encode_relationship_texts",
            return_value=encoded,
        ):
            results, diagnostics = retrieve_source_related_stories(
                profile=profile,
                title_summary=titles,
                use_semantic_centrality=True,
            )

        self.assertEqual(set(results["story_id"]), {"1", "2"})
        self.assertTrue(results["semantic_recovery"].all())
        self.assertEqual(diagnostics["semantic_centrality"]["recovered"], 2)

    def test_relationship_selection_prioritizes_covered_direct_evidence(self) -> None:
        relationships = [
            {
                "related_subject": f"Peripheral company {index}",
                "aliases": [],
                "relationship_role": "DIRECT_PARTICIPANT",
                "predicate_id": "P112",
                "can_retrieve_standalone": True,
                "confidence": 0.90,
            }
            for index in range(12)
        ]
        relationships.append(
            {
                "related_subject": "Covered leader",
                "aliases": [],
                "relationship_role": "DIRECT_PARTICIPANT",
                "predicate_id": "P6",
                "can_retrieve_standalone": True,
                "confidence": 0.86,
            }
        )
        titles = pd.DataFrame(
            [{"story_id": "1", "page_title": "Covered leader addresses parliament"}]
        )

        selected, diagnostics = _select_relationships_for_retrieval(
            query="Example country",
            resolved_entities=[],
            relationships=relationships,
            title_summary=titles,
            limit=5,
        )

        self.assertIn("Covered leader", {item["related_subject"] for item in selected})
        self.assertGreaterEqual(diagnostics["selected_with_local_coverage"], 1)


if __name__ == "__main__":
    unittest.main()
