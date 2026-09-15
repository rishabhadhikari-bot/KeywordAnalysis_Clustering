import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.opensearch_semantic import (
    OpenSearchSemanticClient,
    OpenSearchSemanticSettings,
    _parse_semantic_hits,
    _semantic_bulk_index_body,
    _semantic_index_definition,
    build_semantic_performance_result,
    collect_all_refined_story_ids,
    semantic_entity_anchor_phrases,
    semantic_corpus_fingerprint,
)
from src.opensearch_refined import OpenSearchSettings
from src.relationship_embeddings import RelationshipRoute
from src.relationship_profile_store import load_latest_relationship_profile
from src.semantic_related import (
    build_fast_relationship_titles,
    build_relationship_performance_result,
)


class _PagedRefinedClient:
    def __init__(self) -> None:
        self.calls = []

    def search_page(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("search_after") is None:
            return (
                [{"story_id": "1"}, {"story_id": "2"}],
                3,
                [7.0, 10, "2"],
            )
        return ([{"story_id": "3"}], 3, [5.0, 5, "3"])


class OpenSearchSemanticTests(unittest.TestCase):
    def test_semantic_index_uses_isolated_knn_vector_mapping(self) -> None:
        definition = _semantic_index_definition(384)
        vector_mapping = definition["mappings"]["properties"]["title_embedding"]

        self.assertTrue(definition["settings"]["index.knn"])
        self.assertEqual(vector_mapping["type"], "knn_vector")
        self.assertEqual(vector_mapping["dimension"], 384)
        self.assertEqual(vector_mapping["method"]["space_type"], "cosinesimil")
        properties = definition["mappings"]["properties"]
        self.assertEqual(properties["title_entity_ids"]["type"], "keyword")
        self.assertEqual(properties["title_entity_names"]["type"], "text")

    def test_semantic_bulk_body_aligns_story_ids_and_vectors(self) -> None:
        body = _semantic_bulk_index_body(
            "semantic-test",
            ["2", "1"],
            np.asarray([[0.2, 0.8], [0.1, 0.9]], dtype=np.float32),
            {
                "1": {"story_id": "1", "page_title": "First", "total_views": 10},
                "2": {"story_id": "2", "page_title": "Second", "total_views": 20},
            },
            {
                "2": [{
                    "entity_id": "org_ongc",
                    "canonical_name": "Oil and Natural Gas Corporation",
                    "label": "organization",
                }]
            },
        )
        lines = [json.loads(line) for line in body.strip().splitlines()]

        self.assertEqual(lines[0]["index"]["_id"], "2")
        self.assertEqual(lines[1]["page_title"], "Second")
        self.assertAlmostEqual(lines[1]["title_embedding"][0], 0.2, places=5)
        self.assertEqual(lines[1]["title_entity_ids"], ["org_ongc"])
        self.assertEqual(
            lines[1]["title_entity_names"],
            ["Oil and Natural Gas Corporation"],
        )
        self.assertEqual(lines[2]["index"]["_id"], "1")
        self.assertEqual(lines[3]["page_title"], "First")

    def test_title_entity_model_participates_in_ingestion_fingerprint(self) -> None:
        titles = pd.DataFrame([{"story_id": "1", "page_title": "Oil rises"}])

        without_entities = semantic_corpus_fingerprint(titles, "bge-m3")
        with_entities = semantic_corpus_fingerprint(
            titles,
            "bge-m3",
            "gliner-medium",
        )

        self.assertNotEqual(without_entities, with_entities)

    def test_collect_refined_ids_reads_every_result_page(self) -> None:
        client = _PagedRefinedClient()

        excluded = collect_all_refined_story_ids(
            client,
            "delhi election",
            "All keywords",
            page_size=2,
        )

        self.assertEqual(excluded, {"1", "2", "3"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["search_after"], [7.0, 10, "2"])

    def test_semantic_result_excludes_all_refined_story_ids(self) -> None:
        semantic_hits = [
            {"story_id": "1", "semantic_score": 0.95},
            {"story_id": "2", "semantic_score": 0.90},
            {"story_id": "3", "semantic_score": 0.85},
        ]
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Direct", "total_views": 100},
                {"story_id": "2", "page_title": "Related A", "total_views": 80},
                {"story_id": "3", "page_title": "Related B", "total_views": 60},
            ]
        )
        months = pd.DataFrame(
            [
                {"story_id": "1", "month": "2026-01-01", "views": 100},
                {"story_id": "2", "month": "2026-01-01", "views": 50},
                {"story_id": "2", "month": "2026-02-01", "views": 30},
                {"story_id": "3", "month": "2026-01-01", "views": 60},
            ]
        )

        result = build_semantic_performance_result(
            semantic_hits,
            {"1"},
            months,
            titles,
            total_semantic_hits=3,
        )

        matched_ids = set(result["matched_titles"]["story_id"])
        self.assertEqual(matched_ids, {"2", "3"})
        self.assertFalse(matched_ids & result["refined_story_ids"])
        self.assertEqual(result["excluded_refined_titles"], 1)
        self.assertEqual(result["matched_story_months"]["views"].sum(), 140)
        self.assertEqual(result["matched_titles"]["semantic_rank"].tolist(), [1, 2])

    def test_semantic_hit_parser_preserves_score_and_total(self) -> None:
        hits, total = _parse_semantic_hits(
            {
                "hits": {
                    "total": {"value": 1, "relation": "eq"},
                    "hits": [
                        {
                            "_id": "42",
                            "_score": 0.87654321,
                            "_source": {
                                "story_id": "42",
                                "page_title": "Related title",
                            },
                        }
                    ],
                }
            }
        )

        self.assertEqual(total, 1)
        self.assertEqual(hits[0]["story_id"], "42")
        self.assertEqual(hits[0]["semantic_score"], 0.876543)

    def test_semantic_fingerprint_changes_with_model(self) -> None:
        titles = pd.DataFrame(
            [{"story_id": "1", "page_title": "Title", "total_views": 10}]
        )

        first = semantic_corpus_fingerprint(titles, "model-a")
        second = semantic_corpus_fingerprint(titles, "model-b")

        self.assertNotEqual(first, second)

    def test_semantic_entity_anchor_preserves_known_geography(self) -> None:
        anchors = semantic_entity_anchor_phrases("delhi election")

        self.assertIn("delhi", anchors)
        self.assertIn("dl", anchors)

    def test_semantic_entity_anchor_skips_ambiguous_alias(self) -> None:
        self.assertEqual(semantic_entity_anchor_phrases("mp election"), ())

    def test_relationship_result_keeps_validated_relationships_and_excludes_refined(self) -> None:
        related_titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Nita Ambani opens a new venue",
                    "ai_term": "Nita Ambani",
                    "ai_relationship": "Direct query title",
                    "ai_confidence": 0.99,
                    "ai_relevance_level": 3,
                    "total_views": 100,
                },
                {
                    "story_id": "2",
                    "page_title": "Isha Ambani expands Reliance Retail",
                    "ai_term": "Isha Ambani",
                    "ai_relationship": "Immediate family and Reliance leadership",
                    "ai_confidence": 0.94,
                    "ai_relevance_level": 3,
                    "total_views": 80,
                },
                {
                    "story_id": "3",
                    "page_title": "NMACC announces its cultural programme",
                    "ai_term": "NMACC",
                    "ai_relationship": "Institution founded by the query subject",
                    "ai_confidence": 0.91,
                    "ai_relevance_level": 2,
                    "total_views": 60,
                },
            ]
        )
        months = pd.DataFrame(
            [
                {"story_id": "1", "month": "2026-01-01", "views": 100},
                {"story_id": "2", "month": "2026-01-01", "views": 50},
                {"story_id": "2", "month": "2026-02-01", "views": 30},
                {"story_id": "3", "month": "2026-01-01", "views": 60},
            ]
        )

        result = build_relationship_performance_result(
            related_titles,
            {"1"},
            months,
        )

        matched = result["matched_titles"]
        self.assertEqual(set(matched["story_id"]), {"2", "3"})
        self.assertEqual(matched["ai_term"].tolist(), ["Isha Ambani", "NMACC"])
        self.assertEqual(matched["relationship_rank"].tolist(), [1, 2])
        self.assertEqual(result["excluded_refined_titles"], 1)
        self.assertFalse(set(matched["story_id"]) & result["refined_story_ids"])
        self.assertEqual(result["matched_story_months"]["views"].sum(), 140)

    def test_fast_relationship_filter_prefers_title_evidence_and_rejects_noise(self) -> None:
        candidates = [
            {
                "story_id": "2",
                "page_title": "NMACC announces its new cultural programme",
                "total_views": 80,
                "retrieval_evidence": [
                    {
                        "related_subject": "Nita Mukesh Ambani Cultural Centre (NMACC)",
                        "relationship_class": "CORE_RELATED",
                        "factual_bridge": "NMACC was founded by Nita Ambani.",
                        "semantic_score": 0.74,
                        "relationship_quality": 0.95,
                        "can_retrieve_standalone": True,
                        "required_title_cues": [],
                    }
                ],
            },
            {
                "story_id": "3",
                "page_title": "Government changes employee duty rules",
                "total_views": 60,
                "retrieval_evidence": [
                    {
                        "related_subject": "Customs duty",
                        "relationship_class": "CONTEXTUAL",
                        "factual_bridge": "Import taxation relationship.",
                        "semantic_score": 0.84,
                        "relationship_quality": 0.90,
                        "can_retrieve_standalone": True,
                        "required_title_cues": ["customs"],
                    }
                ],
            },
        ]

        result = build_fast_relationship_titles(candidates)

        self.assertEqual(result["story_id"].tolist(), ["2"])
        self.assertEqual(result.iloc[0]["title_relationship_evidence"], "NMACC")

    def test_bounded_relationship_requires_title_cue_even_with_high_score(self) -> None:
        result = build_fast_relationship_titles(
            [
                {
                    "story_id": "4",
                    "page_title": "Generic company market update",
                    "retrieval_evidence": [
                        {
                            "related_subject": "Reliance-Disney joint venture",
                            "semantic_score": 0.95,
                            "relationship_quality": 0.92,
                            "can_retrieve_standalone": False,
                            "required_title_cues": ["Reliance Disney"],
                        }
                    ],
                }
            ]
        )

        self.assertTrue(result.empty)

    def test_bounded_relationship_requires_semantic_support_with_title_evidence(self) -> None:
        result = build_fast_relationship_titles(
            [
                {
                    "story_id": "low",
                    "page_title": "Iran announces an unrelated domestic policy",
                    "retrieval_evidence": [
                        {
                            "related_subject": "Iran",
                            "semantic_score": 0.76,
                            "relationship_quality": 0.92,
                            "can_retrieve_standalone": False,
                            "required_title_cues": [],
                            "match_method": "embedding",
                        }
                    ],
                },
                {
                    "story_id": "supported",
                    "page_title": "Netanyahu and Iran clash over regional diplomacy",
                    "retrieval_evidence": [
                        {
                            "related_subject": "Iran",
                            "semantic_score": 0.82,
                            "relationship_quality": 0.92,
                            "can_retrieve_standalone": False,
                            "required_title_cues": [],
                            "match_method": "embedding",
                        }
                    ],
                },
            ]
        )

        self.assertEqual(result["story_id"].tolist(), ["supported"])
        self.assertEqual(
            result.iloc[0]["validation_status"],
            "Query-anchored route and relationship cue agree",
        )

    def test_short_intelligence_name_does_not_match_unrelated_person(self) -> None:
        result = build_fast_relationship_titles(
            [
                {
                    "story_id": "aman-person",
                    "page_title": "Aman Arora comments on the Chandigarh inquiry",
                    "retrieval_evidence": [
                        {
                            "related_subject": "Aman",
                            "relationship_role": "MILITARY INTELLIGENCE DIRECTORATE",
                            "semantic_score": 0.81,
                            "relationship_quality": 0.93,
                            "can_retrieve_standalone": True,
                            "required_title_cues": [],
                            "match_method": "embedding",
                        }
                    ],
                }
            ]
        )

        self.assertTrue(result.empty)

    @patch("src.opensearch_semantic.encode_relationship_texts")
    def test_relationship_vector_search_keeps_alternate_routes_for_same_relationship(
        self,
        encode_mock,
    ) -> None:
        encode_mock.return_value = np.asarray(
            [[0.2, 0.8], [0.3, 0.7]], dtype=np.float32
        )
        settings = OpenSearchSemanticSettings(
            connection=OpenSearchSettings(
                url="http://localhost:9200",
                index_alias="semantic-test",
            ),
            embedding_model="test-model",
        )
        client = OpenSearchSemanticClient(settings)
        client._request = Mock(
            return_value={
                "responses": [
                    {"hits": {"hits": []}},
                    {"hits": {"hits": []}},
                ]
            }
        )
        routes = [
            RelationshipRoute(
                relationship_id="iran",
                related_subject="Iran",
                relationship_class="CORE_RELATED",
                relationship_family="conflict",
                factual_bridge="Iran is a party to the Israel-Iran conflict.",
                acceptance_condition="The title concerns the conflict.",
                rejection_rule="Reject unrelated Iran coverage.",
                allowed_story_angles=(),
                excluded_story_angles=(),
                retrieval_text=retrieval_text,
                quality_score=0.92,
                can_retrieve_standalone=False,
            )
            for retrieval_text in (
                "Israel Iran conflict",
                "Iranian strikes affecting Israel",
            )
        ]

        _, diagnostics = client.search_relationship_routes(routes)

        self.assertEqual(encode_mock.call_args.args[0], [
            "Israel Iran conflict",
            "Iranian strikes affecting Israel",
        ])
        self.assertEqual(diagnostics["route_count"], 2)

    @patch("src.opensearch_semantic.encode_relationship_texts")
    def test_relationship_msearch_excludes_refined_ids_inside_opensearch(
        self,
        encode_mock,
    ) -> None:
        encode_mock.return_value = np.asarray([[0.2, 0.8]], dtype=np.float32)
        settings = OpenSearchSemanticSettings(
            connection=OpenSearchSettings(
                url="http://localhost:9200",
                index_alias="semantic-test",
            ),
            embedding_model="test-model",
        )
        client = OpenSearchSemanticClient(settings)
        client._request = Mock(
            return_value={
                "responses": [
                    {
                        "hits": {
                            "hits": [
                                {
                                    "_id": "2",
                                    "_score": 0.91,
                                    "_source": {
                                        "story_id": "2",
                                        "page_title": "NMACC programme",
                                        "total_views": 50,
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )
        route = RelationshipRoute(
            relationship_id="nmacc",
            related_subject="NMACC",
            relationship_class="CORE_RELATED",
            relationship_family="institution",
            factual_bridge="NMACC was founded by Nita Ambani.",
            acceptance_condition="The title concerns NMACC.",
            rejection_rule="Reject unrelated cultural centres.",
            allowed_story_angles=(),
            excluded_story_angles=(),
            retrieval_text="NMACC cultural centre",
            quality_score=0.95,
        )

        candidates, diagnostics = client.search_relationship_routes(
            [route],
            excluded_story_ids={"1"},
        )

        request_body = client._request.call_args.kwargs["data"]
        self.assertIn('"terms":{"story_id":["1"]}', request_body)
        self.assertEqual(candidates[0]["story_id"], "2")
        self.assertEqual(diagnostics["route_count"], 1)

    def test_fast_lexical_msearch_does_not_use_ambiguous_short_acronym(self) -> None:
        settings = OpenSearchSemanticSettings(
            connection=OpenSearchSettings(
                url="http://localhost:9200",
                index_alias="semantic-test",
            ),
            embedding_model="test-model",
        )
        client = OpenSearchSemanticClient(settings)
        client._request = Mock(return_value={"responses": [{"hits": {"hits": []}}]})
        route = RelationshipRoute(
            relationship_id="ioc",
            related_subject="International Olympic Committee (IOC)",
            relationship_class="CORE_RELATED",
            relationship_family="sport",
            factual_bridge="Nita Ambani is an IOC member.",
            acceptance_condition="The title concerns the Olympic committee.",
            rejection_rule="Reject Indian Oil Corporation references.",
            allowed_story_angles=(),
            excluded_story_angles=(),
            retrieval_text="International Olympic Committee",
            quality_score=0.95,
        )

        client.search_relationship_routes_lexically([route])

        request_body = client._request.call_args.kwargs["data"]
        self.assertIn("international olympic committee", request_body)
        self.assertNotIn('"query":"ioc"', request_body)

    def test_fast_lexical_msearch_skips_ambiguous_short_intelligence_name(self) -> None:
        settings = OpenSearchSemanticSettings(
            connection=OpenSearchSettings(
                url="http://localhost:9200",
                index_alias="semantic-test",
            ),
            embedding_model="test-model",
        )
        client = OpenSearchSemanticClient(settings)
        client._request = Mock(return_value={"responses": []})
        route = RelationshipRoute(
            relationship_id="aman",
            related_subject="Aman",
            relationship_class="DIRECT",
            relationship_family="Military & Security",
            factual_bridge="Aman is Israel's military intelligence directorate.",
            acceptance_condition="The title concerns Israeli military intelligence.",
            rejection_rule="Reject people whose given name is Aman.",
            allowed_story_angles=(),
            excluded_story_angles=(),
            retrieval_text="Israeli military intelligence Aman",
            quality_score=0.95,
            relationship_role="MILITARY INTELLIGENCE DIRECTORATE",
        )

        candidates, diagnostics = client.search_relationship_routes_lexically([route])

        self.assertEqual(candidates, [])
        self.assertEqual(diagnostics["route_count"], 0)
        client._request.assert_not_called()

    def test_cached_profile_is_reused_across_title_fingerprints(self) -> None:
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [
            (
                json.dumps({"research_text": "cached graph"}),
                "All keywords",
                "v1",
                "2026-01-02T00:00:00+00:00",
            )
        ]
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=connection)
        context_manager.__exit__ = Mock(return_value=False)
        with patch(
            "src.relationship_profile_store.sqlite3.connect",
            return_value=context_manager,
        ), patch.object(Path, "exists", return_value=True):
            cached = load_latest_relationship_profile(
                Path("query_profiles.db"),
                "  Nita   Ambani ",
                preferred_match_type="All keywords",
            )

        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["research_text"], "cached graph")

    def test_cached_profile_can_require_current_prompt_version(self) -> None:
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [
            (
                json.dumps({"research_text": "old graph"}),
                "All keywords",
                "v-old",
                "2026-01-03T00:00:00+00:00",
            ),
            (
                json.dumps({"research_text": "current graph"}),
                "All keywords",
                "v-current",
                "2026-01-02T00:00:00+00:00",
            ),
        ]
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=connection)
        context_manager.__exit__ = Mock(return_value=False)
        with patch(
            "src.relationship_profile_store.sqlite3.connect",
            return_value=context_manager,
        ), patch.object(Path, "exists", return_value=True):
            cached = load_latest_relationship_profile(
                Path("query_profiles.db"),
                "Nita Ambani",
                preferred_match_type="All keywords",
                required_profile_version="v-current",
            )

        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["research_text"], "current graph")
        self.assertEqual(cached[1]["profile_version"], "v-current")


if __name__ == "__main__":
    unittest.main()
