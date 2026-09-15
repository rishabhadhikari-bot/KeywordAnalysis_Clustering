import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.opensearch_refined import OpenSearchSettings
from src.semantic_seo_models import (
    SemanticSEOModelSettings,
    add_gliner_entity_compatibility,
    add_indexed_entity_compatibility,
    canonical_entity_ids_in_text,
    extract_and_link_title_entities,
    link_title_entities,
    score_deberta_title_relevance,
)
from src.semantic_seo_pipeline import (
    DEFAULT_SEMANTIC_SEO_INDEX_ALIAS,
    build_semantic_seo_client,
    build_direct_query_route,
    merge_gemini_validation,
)
from src.semantic_seo_routes import (
    SEMANTIC_SEO_ROUTE_SCHEMA_VERSION,
    SemanticSEORouteProfile,
    _merge_verified_routes,
    _retain_dual_grounded_routes,
    _resolve_controlled_source_ids,
    _route_response_schema,
    _validate_stage_payload,
    load_semantic_route_profile,
    save_semantic_route_profile,
    SemanticSEORouteError,
)


def relationship(route_id="route-01", subject="oil supply disruption"):
    return {
        "relationship_id": route_id,
        "interpretation_id": "primary",
        "related_subject": subject,
        "related_subject_type": "commodity consequence",
        "relationship_class": "CORE_RELATED",
        "relationship_family": "economic consequence",
        "relationship_role": "SPECIFIC_CONSEQUENCE",
        "factual_bridge": "The conflict can disrupt oil supply and shipping.",
        "direction": "query causes consequence",
        "durable_or_current": "current",
        "evidence_summary": "Independent reporting supports the route.",
        "evidence_source_urls": ["https://example.com/a"],
        "confidence": 0.9,
        "false_positive_risk": "low",
        "can_retrieve_standalone": False,
        "acceptance_condition": "The title must express an oil-market consequence.",
        "allowed_story_angles": ["oil supply disruption"],
        "excluded_story_angles": ["unrelated domestic fuel taxes"],
        "required_title_cues": ["oil supply", "crude prices"],
        "rejection_rule": "Reject unrelated oil-company or tax stories.",
        "editorial_manifestations": ["conflict-driven crude price increase"],
    }


class SemanticSEORouteContractTests(unittest.TestCase):
    def test_short_source_ids_resolve_to_exact_grounded_urls(self):
        profile = {
            "relationship_map": [
                {
                    "discovery_source_id": "D1",
                    "verification_source_id": "V1",
                }
            ]
        }
        _resolve_controlled_source_ids(
            profile,
            discovery_sources=[{"uri": "https://example.com/discovery"}],
            verification_sources=[{"uri": "https://example.com/verification"}],
        )

        self.assertEqual(
            profile["relationship_map"][0]["evidence_source_urls"],
            [
                "https://example.com/discovery",
                "https://example.com/verification",
            ],
        )

    def test_route_count_is_enforced_locally_not_in_serving_schema(self):
        schema = _route_response_schema(
            10,
            discovery_source_ids=["D1", "D2"],
            verification_source_ids=["V1", "V2"],
        )
        self.assertNotIn("maxItems", schema["properties"]["relationship_map"])
        relationship_properties = schema["properties"]["relationship_map"]["items"][
            "properties"
        ]
        self.assertEqual(
            relationship_properties["discovery_source_id"]["enum"],
            ["D1", "D2"],
        )
        self.assertEqual(
            relationship_properties["verification_source_id"]["enum"],
            ["V1", "V2"],
        )

        profile = {
            "query_resolution": {
                "canonical_query": "regional conflict",
                "primary_type": "conflict",
            },
            "relationship_map": [
                relationship(route_id=f"route-{index:02d}")
                for index in range(1, 4)
            ],
        }
        with self.assertRaises(SemanticSEORouteError):
            _validate_stage_payload(
                {
                    "profile": profile,
                    "sources": [{"uri": "https://example.com/a"}],
                },
                stage="test",
                max_routes=2,
            )

    def test_dual_grounding_requires_two_distinct_branch_sources(self):
        candidate = relationship()
        candidate["evidence_source_urls"] = [
            "https://example.com/discovery",
            "https://example.com/verification",
        ]

        retained = _retain_dual_grounded_routes(
            [candidate],
            discovery_sources=[{"uri": "https://example.com/discovery"}],
            verification_sources=[{"uri": "https://example.com/verification"}],
        )
        rejected = _retain_dual_grounded_routes(
            [candidate],
            discovery_sources=[{"uri": "https://example.com/discovery"}],
            verification_sources=[{"uri": "https://example.com/discovery"}],
        )

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["verification_branch_count"], 2)
        self.assertEqual(rejected, [])

    def test_merge_requires_matching_verifier_route_identity(self):
        discovery = {
            "query_resolution": {
                "canonical_query": "regional conflict",
                "primary_type": "conflict",
            },
            "relationship_map": [relationship()],
        }
        verifier_relationship = relationship()
        verifier_relationship["relationship_role"] = "GENERIC_TOPIC"
        verifier = {
            "query_resolution": discovery["query_resolution"],
            "relationship_map": [verifier_relationship],
        }

        merged = _merge_verified_routes(
            discovery,
            verifier,
            discovery_sources=[{"uri": "https://example.com/a"}],
            verification_sources=[{"uri": "https://example.com/a"}],
        )

        self.assertEqual(merged["relationship_map"], [])

    def test_merge_marks_route_as_independently_verified(self):
        discovery_relationship = relationship()
        verifier_relationship = relationship(subject="oil-market disruption")
        verifier_relationship["evidence_source_urls"] = ["https://example.com/b"]
        discovery = {
            "query_resolution": {
                "canonical_query": "regional conflict",
                "primary_type": "conflict",
            },
            "relationship_map": [discovery_relationship],
        }
        verifier = {
            "query_resolution": discovery["query_resolution"],
            "relationship_map": [verifier_relationship],
        }

        merged = _merge_verified_routes(
            discovery,
            verifier,
            discovery_sources=[{"uri": "https://example.com/a"}],
            verification_sources=[{"uri": "https://example.com/b"}],
        )

        self.assertEqual(len(merged["relationship_map"]), 1)
        accepted = merged["relationship_map"][0]
        self.assertEqual(accepted["verification_branch_count"], 2)
        self.assertEqual(
            accepted["evidence_source_urls"],
            ["https://example.com/a", "https://example.com/b"],
        )

    def test_only_validated_profile_is_loaded_from_isolated_store(self):
        validated_relationship = relationship()
        validated_relationship["verification_branch_count"] = 2
        research = {
            "research_text": json.dumps(
                {
                    "query_resolution": {
                        "canonical_query": "regional conflict",
                        "primary_type": "conflict",
                    },
                    "relationship_map": [validated_relationship],
                }
            ),
            "sources": [{"uri": "https://example.com/a"}],
            "web_search_queries": ["regional conflict oil impact"],
            "schema_version": SEMANTIC_SEO_ROUTE_SCHEMA_VERSION,
        }
        profile = SemanticSEORouteProfile(
            keyword_query="regional conflict",
            canonical_query="regional conflict",
            primary_type="conflict",
            research=research,
            sources=list(research["sources"]),
            web_search_queries=list(research["web_search_queries"]),
            model_name="test-model",
            created_at="2026-08-24T00:00:00+00:00",
        )
        workspace_tmp = Path(__file__).resolve().parents[1] / ".tmp"
        db_path = workspace_tmp / f"semantic-seo-{uuid.uuid4().hex}.db"
        try:
            save_semantic_route_profile(db_path, profile)
            loaded = load_semantic_route_profile(
                db_path,
                "regional conflict",
                "test-model",
            )
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(db_path) + suffix)
                if candidate.exists():
                    candidate.unlink()

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.relationship_count, 1)


class SemanticSEOIsolationTests(unittest.TestCase):
    def test_title_entity_linking_uses_curated_id_and_stable_fallback(self):
        linked = link_title_entities(
            [
                {"text": "ONGC", "label": "organization", "score": 0.96},
                {"text": "Brent crude", "label": "commodity", "score": 0.91},
            ]
        )
        by_surface = {item["surface"]: item for item in linked}

        self.assertEqual(by_surface["ONGC"]["entity_id"], "org_ongc")
        self.assertEqual(
            by_surface["Brent crude"]["entity_id"],
            link_title_entities(
                [{"text": "Brent crude", "label": "commodity", "score": 0.5}]
            )[0]["entity_id"],
        )
        self.assertEqual(
            canonical_entity_ids_in_text("ONGC production update"),
            {"org_ongc"},
        )

    def test_title_entity_ingestion_cache_avoids_repeated_gliner_inference(self):
        workspace_tmp = Path(__file__).resolve().parents[1] / ".tmp"
        cache_path = workspace_tmp / f"title-entities-{uuid.uuid4().hex}.db"
        model = Mock()
        model.inference.return_value = [
            [{"text": "ONGC", "label": "organization", "score": 0.95}]
        ]
        try:
            with patch("src.semantic_seo_models._load_gliner", return_value=model):
                first = extract_and_link_title_entities(
                    {"1": "ONGC increases production"},
                    model_name="test-gliner",
                    cache_path=cache_path,
                )
            with patch("src.semantic_seo_models._load_gliner") as loader:
                second = extract_and_link_title_entities(
                    {"1": "ONGC increases production"},
                    model_name="test-gliner",
                    cache_path=cache_path,
                )
                loader.assert_not_called()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(cache_path) + suffix)
                if candidate.exists():
                    candidate.unlink()

        self.assertEqual(first, second)
        self.assertEqual(first["1"][0]["entity_id"], "org_ongc")

    def test_indexed_gliner_entities_are_used_without_online_model(self):
        candidates = [
            {
                "story_id": "oil",
                "page_title": "Oil prices rise",
                "title_entity_names": ["oil"],
                "retrieval_evidence": [{
                    "related_subject": "oil supply",
                    "factual_bridge": "Shipping disruption affects oil supply.",
                    "semantic_score": 0.80,
                }],
            },
            {
                "story_id": "noise",
                "page_title": "Technology shares rise",
                "title_entity_names": ["technology"],
                "retrieval_evidence": [{
                    "related_subject": "oil supply",
                    "factual_bridge": "Shipping disruption affects oil supply.",
                    "semantic_score": 0.81,
                }],
            },
        ]

        ranked, diagnostic = add_indexed_entity_compatibility(candidates)

        self.assertEqual([item["story_id"] for item in ranked], ["oil", "noise"])
        self.assertFalse(diagnostic["online_model_inference"])
        self.assertEqual(ranked[0]["matched_indexed_entities"], ["oil"])

    def test_direct_route_allows_search_without_gemini_profile(self):
        route = build_direct_query_route("  Iran conflict  ")

        self.assertEqual(route.retrieval_text, "Iran conflict")
        self.assertEqual(route.relationship_role, "QUERY_BASELINE")

    def test_deberta_title_mode_does_not_require_article_body_or_filter(self):
        class TensorResult:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def numpy(self):
                return self.values

        class Model:
            config = type("Config", (), {"id2label": {0: "entailment", 1: "neutral"}})()

            def __call__(self, **_kwargs):
                return type("Output", (), {"logits": TensorResult([[2.0, 0.0]])})()

        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        torch = type("Torch", (), {"no_grad": staticmethod(lambda: NoGrad())})()
        tokenizer = Mock(return_value={"input_ids": [[1]]})
        candidates = [{
            "story_id": "1",
            "page_title": "Oil prices rise on supply concerns",
            "semantic_seo_local_score": 0.8,
            "retrieval_evidence": [{
                "related_subject": "oil supply",
                "factual_bridge": "Conflict can disrupt oil supply.",
                "semantic_score": 0.8,
            }],
        }]
        with patch(
            "src.semantic_seo_models._load_sequence_classifier",
            return_value=(tokenizer, Model(), torch),
        ):
            scored, diagnostic = score_deberta_title_relevance(
                "regional conflict",
                candidates,
                model_name="test-deberta",
            )

        self.assertEqual(len(scored), 1)
        self.assertTrue(diagnostic["applied"])
        self.assertEqual(diagnostic["evidence_mode"], "title_only")
        self.assertIn("deberta_title_relevance_score", scored[0])

    def test_gliner_compatibility_changes_local_ranking(self):
        candidates = [
            {
                "story_id": "1",
                "page_title": "Oil market update",
                "total_views": 10,
                "retrieval_evidence": [
                    {
                        "related_subject": "Red Sea disruption",
                        "factual_bridge": "Shipping disruption affects oil supply.",
                        "required_title_cues": ["oil supply"],
                        "semantic_score": 0.80,
                    }
                ],
            },
            {
                "story_id": "2",
                "page_title": "Crude prices rise",
                "total_views": 20,
                "retrieval_evidence": [
                    {
                        "related_subject": "Red Sea disruption",
                        "factual_bridge": "Shipping disruption affects oil supply.",
                        "required_title_cues": ["oil supply"],
                        "semantic_score": 0.81,
                    }
                ],
            },
        ]
        route_entities = [
            {"text": "oil supply", "label": "commodity", "score": 0.9}
        ]
        compatible_title = [
            {"text": "oil supply", "label": "commodity", "score": 0.9}
        ]
        incompatible_title = [
            {"text": "prices", "label": "financial instrument", "score": 0.9}
        ]
        with patch("src.semantic_seo_models._load_gliner", return_value=object()), patch(
            "src.semantic_seo_models._predict_gliner",
            side_effect=[
                route_entities,
                compatible_title,
                route_entities,
                incompatible_title,
            ],
        ):
            ranked, diagnostic = add_gliner_entity_compatibility(
                candidates,
                model_name="test-gliner",
            )

        self.assertTrue(diagnostic["applied"])
        self.assertEqual([item["story_id"] for item in ranked], ["1", "2"])
        self.assertGreater(
            ranked[0]["semantic_seo_local_score"],
            ranked[1]["semantic_seo_local_score"],
        )

    def test_lab_uses_distinct_opensearch_alias(self):
        settings = SemanticSEOModelSettings(
            embedding_model="BAAI/bge-m3",
            reranker_model="reranker",
            gliner_model="gliner",
            nli_model="nli",
            relation_model="mrebel",
            enable_mrebel=False,
        )
        base = OpenSearchSettings(
            url="http://localhost:9200",
            index_alias="traffic-title-refined",
        )
        with patch(
            "src.semantic_seo_pipeline.load_opensearch_settings",
            return_value=base,
        ):
            client = build_semantic_seo_client(settings)

        self.assertEqual(
            client.settings.connection.index_alias,
            DEFAULT_SEMANTIC_SEO_INDEX_ALIAS,
        )
        self.assertNotEqual(
            client.settings.connection.index_alias,
            base.index_alias,
        )
        self.assertTrue(client.settings.enable_title_entities)
        self.assertEqual(client.settings.entity_model, "gliner")

    def test_gemini_validation_filters_and_preserves_local_scores(self):
        candidates = [
            {
                "story_id": "1",
                "page_title": "Oil rises after shipping disruption",
                "semantic_seo_local_score": 0.8,
                "total_views": 10,
            },
            {
                "story_id": "2",
                "page_title": "Unrelated title",
                "semantic_seo_local_score": 0.9,
                "total_views": 20,
            },
        ]
        selected = [
            {
                "story_id": "1",
                "ai_relationship": "Conflict-related shipping disruption",
                "ai_audit_reason": "Title expresses the verified consequence",
                "ai_confidence": 0.95,
                "ai_relevance_level": 2,
                "ai_relevance_type": "core_related",
            }
        ]

        merged = merge_gemini_validation(candidates, selected)

        self.assertEqual([item["story_id"] for item in merged], ["1"])
        self.assertEqual(merged[0]["semantic_seo_local_score"], 0.8)
        self.assertEqual(merged[0]["gemini_confidence"], 0.95)


if __name__ == "__main__":
    unittest.main()
