import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.relationship_embeddings import (
    RelationshipRoute,
    extract_relationship_routes,
    passes_relationship_evidence_gate,
    retrieve_relationship_candidates,
)
from src.vertex_related import (
    _attach_claim_grounding_urls,
    _build_generative_retrieval_prompt,
    _build_generative_retrieval_response_schema,
    _create_vertex_context_cache,
    _extract_vertex_finish_reason,
    _group_candidates_by_primary_route,
    _merge_grounded_research_branches,
    _normalize_selected_existing_titles,
    _recover_truncated_grounded_research_profile,
    _run_grounded_research_branches,
    _select_generative_retrieval_batch_with_vertex,
    _text_is_contiguous_title_evidence,
)


def _selected_title(
    *,
    story_id: str = "1",
    confidence: float = 0.90,
    relevance_level: int = 2,
    relevance_type: str = "core_related",
    evidence_scope: str,
    context_match: str = "EXACT",
    non_replaceable: bool = True,
) -> dict[str, object]:
    return {
        "story_id": story_id,
        "confidence": confidence,
        "editorial_summary": "India's T20 captaincy transition connects directly to Shreyas Iyer.",
        "reason": "Title evidence and relationship edge were evaluated.",
        "relevance_level": relevance_level,
        "relevance_type": relevance_type,
        "central_subject": "India T20I captaincy",
        "query_specific_title_evidence": "India T20 captain",
        "matched_relationship_id": "R6",
        "relationship_edge_evidence": "Same India T20I succession context",
        "bridge_type": "CAPTAINCY_SUCCESSION",
        "context_match": context_match,
        "evidence_scope": evidence_scope,
        "query_is_non_replaceable": non_replaceable,
    }


class RelatedRelevanceCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {
                "story_id": "1",
                "page_title": "BCCI discusses the next India T20 captain",
                "retrieval_evidence": [
                    {
                        "relationship_id": "R6",
                        "factual_bridge": "Shreyas replaced Suryakumar as India T20I captain.",
                        "acceptance_condition": "The title expresses India captaincy.",
                        "rejection_rule": "Reject other competitions.",
                        "can_retrieve_standalone": False,
                        "required_title_cues": ["India T20 captain"],
                    }
                ],
            }
        ]

    def _normalize(self, item: dict[str, object]) -> list[dict[str, object]]:
        return _normalize_selected_existing_titles(
            {"matched_titles": [item]},
            self.candidates,
            max_related_titles=10,
        )

    def test_same_context_indirect_is_demoted_and_capped(self) -> None:
        normalized = self._normalize(
            _selected_title(
                evidence_scope="SAME_CONTEXT_INDIRECT",
                confidence=0.95,
                relevance_level=2,
            )
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["ai_confidence"], 0.74)
        self.assertEqual(normalized[0]["ai_relevance_level"], 1)
        self.assertEqual(normalized[0]["ai_relevance_type"], "contextual")

    def test_exact_edge_cannot_receive_direct_score(self) -> None:
        normalized = self._normalize(
            _selected_title(
                evidence_scope="EXACT_RELATIONSHIP_EDGE",
                confidence=0.99,
                relevance_level=3,
            )
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["ai_confidence"], 0.89)
        self.assertEqual(normalized[0]["ai_relevance_level"], 2)
        self.assertEqual(normalized[0]["ai_relevance_type"], "core_related")

    def test_replaceable_exact_edge_is_demoted_to_contextual(self) -> None:
        normalized = self._normalize(
            _selected_title(
                evidence_scope="EXACT_RELATIONSHIP_EDGE",
                confidence=0.90,
                relevance_level=2,
                non_replaceable=False,
            )
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["ai_confidence"], 0.74)
        self.assertEqual(normalized[0]["ai_relevance_level"], 1)
        self.assertEqual(normalized[0]["ai_evidence_scope"], "SAME_CONTEXT_INDIRECT")

    def test_endpoint_only_and_context_mismatch_are_rejected(self) -> None:
        endpoint_only = self._normalize(
            _selected_title(evidence_scope="RELATED_ENDPOINT_ONLY")
        )
        context_mismatch = self._normalize(
            _selected_title(
                evidence_scope="EXACT_RELATIONSHIP_EDGE",
                context_match="MISMATCH",
            )
        )

        self.assertEqual(endpoint_only, [])
        self.assertEqual(context_mismatch, [])

    def test_fabricated_route_id_is_rejected(self) -> None:
        item = _selected_title(evidence_scope="EXACT_RELATIONSHIP_EDGE")
        item["matched_relationship_id"] = "fabricated-route"

        self.assertEqual(self._normalize(item), [])

    def test_paraphrased_title_evidence_is_rejected(self) -> None:
        item = _selected_title(evidence_scope="EXACT_RELATIONSHIP_EDGE")
        item["query_specific_title_evidence"] = "national captaincy succession"

        self.assertEqual(self._normalize(item), [])

    def test_local_normalization_records_why_a_gemini_match_was_rejected(self) -> None:
        item = _selected_title(evidence_scope="EXACT_RELATIONSHIP_EDGE")
        item["query_specific_title_evidence"] = "invented title wording"
        rejection_reasons: dict[str, str] = {}

        normalized = _normalize_selected_existing_titles(
            {"matched_titles": [item]},
            self.candidates,
            max_related_titles=10,
            rejection_reasons=rejection_reasons,
        )

        self.assertEqual(normalized, [])
        self.assertIn("title evidence is not an exact quote", rejection_reasons["1"])

    def test_punctuation_collapsed_relationship_cue_is_accepted(self) -> None:
        self.candidates[0]["page_title"] = "USIran conflict enters ceasefire talks"
        self.candidates[0]["retrieval_evidence"][0]["required_title_cues"] = [
            "US-Iran conflict"
        ]
        item = _selected_title(
            evidence_scope="EXACT_RELATIONSHIP_EDGE",
            confidence=0.85,
        )
        item["query_specific_title_evidence"] = "USIran conflict"

        normalized = self._normalize(item)

        self.assertEqual(len(normalized), 1)
        self.assertTrue(
            _text_is_contiguous_title_evidence(
                "US-Iran conflict",
                "USIran conflict enters ceasefire talks",
            )
        )

    def test_direct_query_bypasses_indirect_route_cue_and_strict_embedding_gate(self) -> None:
        self.candidates[0]["page_title"] = "Iranian general responds to US threat"
        route = self.candidates[0]["retrieval_evidence"][0]
        route["required_title_cues"] = ["US-Iran conflict"]
        route.update({"match_method": "embedding", "similarity": 0.52})
        item = _selected_title(
            evidence_scope="DIRECT_QUERY",
            confidence=0.95,
            relevance_level=3,
            relevance_type="direct",
        )
        item["query_specific_title_evidence"] = "Iranian"

        normalized = self._normalize(item)
        self.assertEqual(len(normalized), 1)
        normalized[0]["retrieval_evidence"] = self.candidates[0]["retrieval_evidence"]
        self.assertTrue(passes_relationship_evidence_gate(normalized[0]))

    def test_response_schema_requires_auditable_relevance_fields(self) -> None:
        schema = _build_generative_retrieval_response_schema()
        item_schema = schema["properties"]["matched_titles"]["items"]

        for field in (
            "matched_relationship_id",
            "relationship_edge_evidence",
            "context_match",
            "evidence_scope",
            "query_is_non_replaceable",
            "editorial_summary",
        ):
            self.assertIn(field, item_schema["required"])

    def test_token_truncated_research_recovers_only_complete_relationships(self) -> None:
        truncated = """
        {
          "query_resolution": {"canonical_query": "Shreyas Iyer"},
          "identity_forms": {"aliases": ["Shreyas"]},
          "relationship_map": [
            {
              "relationship_id": "R1",
              "related_subject": "India cricket team",
              "evidence_source_urls": ["https://example.com/source"]
            },
            {"relationship_id": "R2", "related_subject": "unfinished"
        """

        recovered = _recover_truncated_grounded_research_profile(truncated)

        self.assertIsNotNone(recovered)
        self.assertEqual(
            recovered["query_resolution"]["canonical_query"],
            "Shreyas Iyer",
        )
        self.assertEqual(len(recovered["relationship_map"]), 1)
        self.assertEqual(recovered["relationship_map"][0]["relationship_id"], "R1")
        self.assertTrue(recovered["search_audit"]["quality_gate_passed"])

    def test_truncated_research_without_complete_relationship_is_rejected(self) -> None:
        self.assertIsNone(
            _recover_truncated_grounded_research_profile(
                '{"relationship_map":[{"relationship_id":"R1"'
            )
        )

    def test_vertex_finish_reason_is_extracted_for_diagnostics(self) -> None:
        payload = {"candidates": [{"finishReason": "MAX_TOKENS"}]}

        self.assertEqual(_extract_vertex_finish_reason(payload), "MAX_TOKENS")
        self.assertEqual(_extract_vertex_finish_reason({}), "UNKNOWN")

    def test_editorial_summary_is_separate_from_detailed_audit_reason(self) -> None:
        normalized = self._normalize(
            _selected_title(evidence_scope="EXACT_RELATIONSHIP_EDGE")
        )

        self.assertEqual(
            normalized[0]["ai_relationship"],
            "India's T20 captaincy transition connects directly to Shreyas Iyer.",
        )
        self.assertEqual(
            normalized[0]["ai_audit_reason"],
            "Title evidence and relationship edge were evaluated.",
        )

    def test_standard_prompt_contains_edge_gate_and_route_constraints(self) -> None:
        prompt = _build_generative_retrieval_prompt(
            keyword_query="Shreyas Iyer",
            candidate_titles=[
                {
                    **self.candidates[0],
                    "retrieval_evidence": [
                        {
                            "relationship_id": "R6",
                            "related_subject": "Suryakumar Yadav",
                            "relationship_class": "CORE_RELATED",
                            "factual_bridge": "Shreyas replaced Suryakumar as India T20I captain.",
                            "can_retrieve_standalone": False,
                            "required_title_cues": ["India", "T20I", "captaincy"],
                        }
                    ],
                }
            ],
            grounded_research={"research_text": "{}"},
        )

        self.assertIn("MANDATORY RELATIONSHIP-EDGE GATE", prompt)
        self.assertIn("EVIDENCE-BOUND STABILITY RULE", prompt)
        self.assertIn(
            "The same title and evidence must produce the same decision",
            prompt,
        )
        self.assertIn("EDITORIAL WORDING", prompt)
        self.assertIn("one plain-language sentence of 12 to 24 words", prompt)
        self.assertIn("CONTEXT_MISMATCH", prompt)
        self.assertIn("untrusted data", prompt)
        self.assertIn('"can_retrieve_standalone":false', prompt)
        self.assertIn('"required_title_cues":["India","T20I","captaincy"]', prompt)

    def test_route_catalog_deduplicates_full_route_across_candidates(self) -> None:
        bridge = "Shreyas replaced Suryakumar as India T20I captain."
        evidence = {
            "relationship_id": "R6",
            "related_subject": "Suryakumar Yadav",
            "relationship_class": "CORE_RELATED",
            "factual_bridge": bridge,
            "can_retrieve_standalone": False,
            "required_title_cues": ["India", "T20I", "captaincy"],
        }
        prompt = _build_generative_retrieval_prompt(
            keyword_query="Shreyas Iyer",
            candidate_titles=[
                {
                    "story_id": "1",
                    "page_title": "BCCI discusses the next India T20 captain",
                    "retrieval_evidence": [evidence],
                },
                {
                    "story_id": "2",
                    "page_title": "India T20 captaincy transition explained",
                    "retrieval_evidence": [evidence],
                },
            ],
            grounded_research={"research_text": "{}"},
        )

        self.assertEqual(prompt.count(bridge), 1)
        self.assertIn("RETRIEVAL ROUTE CATALOG", prompt)


class RelatedRetrievalPerformanceTests(unittest.TestCase):
    def test_grounding_supports_are_attached_to_the_exact_bridge(self) -> None:
        bridge = "Shreyas Iyer captained Punjab Kings."
        research_text = json.dumps(
            {"relationship_map": [{"factual_bridge": bridge}]},
            separators=(",", ":"),
        )
        start = research_text.index(bridge)
        profile = json.loads(research_text)
        payload = {
            "candidates": [
                {
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/source"}}
                        ],
                        "groundingSupports": [
                            {
                                "segment": {
                                    "startIndex": start,
                                    "endIndex": start + len(bridge),
                                },
                                "groundingChunkIndices": [0],
                            }
                        ],
                    }
                }
            ]
        }

        _attach_claim_grounding_urls(profile, payload, research_text)

        self.assertEqual(
            profile["relationship_map"][0]["evidence_source_urls"],
            ["https://example.com/source"],
        )

    def test_grounding_support_byte_offsets_survive_multilingual_profile_text(self) -> None:
        bridge = "Israel signed a treaty with Egypt."
        research_text = json.dumps(
            {
                "identity_forms": {"aliases": ["ישראל" * 200]},
                "relationship_map": [{"factual_bridge": bridge}],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        raw_response_text = f"\n  {research_text}\n"
        second_part_start = raw_response_text.index('"relationship_map"')
        response_parts = [
            raw_response_text[:second_part_start],
            raw_response_text[second_part_start:],
        ]
        bridge_start = len(
            response_parts[1][: response_parts[1].index(bridge)].encode("utf-8")
        )
        profile = json.loads(research_text)
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": part_text} for part_text in response_parts]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/treaty"}}
                        ],
                        "groundingSupports": [
                            {
                                "segment": {
                                    "partIndex": 1,
                                    "startIndex": bridge_start,
                                    "endIndex": bridge_start + len(bridge.encode("utf-8")),
                                },
                                "groundingChunkIndices": [0],
                            }
                        ],
                    }
                }
            ]
        }

        _attach_claim_grounding_urls(profile, payload, research_text)

        self.assertEqual(
            profile["relationship_map"][0]["evidence_source_urls"],
            ["https://example.com/treaty"],
        )

    def test_missing_profile_has_no_semantic_knowledge_fallback(self) -> None:
        selected, diagnostics = retrieve_relationship_candidates(
            keyword_query="Iran",
            candidate_titles=[{"story_id": "1", "page_title": "Oil prices rise"}],
            grounded_research={"research_text": ""},
        )

        self.assertEqual(selected, [])
        self.assertEqual(diagnostics["mode"], "bounded_retrieval_no_candidates")
    def test_direct_ids_are_excluded_before_per_route_and_global_limits(self) -> None:
        candidates = [
            {"story_id": "d1", "page_title": "Iran direct one"},
            {"story_id": "d2", "page_title": "Iran direct two"},
            {"story_id": "n1", "page_title": "Hormuz shipping disruption"},
            {"story_id": "n2", "page_title": "Regional oil supply pressure"},
            {"story_id": "n3", "page_title": "Diplomatic negotiations continue"},
        ]
        route = RelationshipRoute(
            relationship_id="R1",
            related_subject="Associated Subject",
            relationship_class="CORE_RELATED",
            relationship_family="regional consequence",
            factual_bridge="A specific consequence connected to Iran.",
            acceptance_condition="The title expresses the consequence.",
            rejection_rule="Reject generic regional coverage.",
            allowed_story_angles=(),
            excluded_story_angles=(),
            retrieval_text="Iran regional consequence",
            quality_score=0.9,
        )

        class FakeModel:
            @staticmethod
            def encode(*_args: object, **_kwargs: object) -> np.ndarray:
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        with (
            patch(
                "src.relationship_embeddings.extract_relationship_routes",
                return_value=[route],
            ),
            patch(
                "src.relationship_embeddings._load_or_build_title_index",
                return_value=(
                    np.zeros((5, 2), dtype=np.float32),
                    ["d1", "d2", "n1", "n2", "n3"],
                    {"embedding_dimension": 2, "corpus_fingerprint": "corpus"},
                ),
            ),
            patch(
                "src.relationship_embeddings._load_sentence_transformer",
                return_value=FakeModel(),
            ),
            patch(
                "src.relationship_embeddings._search_title_embeddings",
                return_value=(
                    np.asarray([[0.99, 0.98, 0.90, 0.80]], dtype=np.float32),
                    np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                    "test",
                ),
            ) as search_mock,
        ):
            selected, diagnostics = retrieve_relationship_candidates(
                keyword_query="Iran",
                candidate_titles=candidates,
                grounded_research={"research_text": "{}"},
                top_k_per_route=2,
                max_candidates=2,
                excluded_story_ids={"d1", "d2"},
            )

        self.assertEqual([item["story_id"] for item in selected], ["n1", "n2"])
        self.assertEqual(diagnostics["excluded_story_count"], 2)
        self.assertEqual(diagnostics["eligible_corpus_count"], 3)
        self.assertEqual(diagnostics["embedding_search_depth"], 4)
        self.assertEqual(search_mock.call_args.kwargs["top_k"], 4)

    def test_parallel_branch_merge_requires_independent_corroboration(self) -> None:
        duplicate_relationship = {
            "relationship_id": "R1",
            "interpretation_id": "primary",
            "related_subject": "Punjab Kings",
            "related_subject_type": "TEAM",
            "relationship_class": "CORE_RELATED",
            "relationship_family": "team",
            "relationship_role": "FORMAL_AFFILIATION",
            "factual_bridge": "Shreyas Iyer captained Punjab Kings.",
            "direction": "Shreyas Iyer -> captained -> Punjab Kings",
            "durable_or_current": "CURRENT",
            "evidence_summary": "The team announcement confirms the role.",
            "evidence_source_urls": ["https://example.com/a"],
            "can_retrieve_standalone": True,
            "acceptance_condition": "The title is centrally about the captaincy.",
            "allowed_story_angles": ["Punjab Kings captaincy"],
            "excluded_story_angles": ["Unrelated team fixtures"],
            "required_title_cues": [],
            "rejection_rule": "Reject unrelated fixtures.",
        }
        merged = _merge_grounded_research_branches(
            [
                {
                    "branch_id": "identity_current",
                    "profile": {
                        "query_resolution": {"canonical_query": "Shreyas Iyer"},
                        "identity_forms": {"aliases": ["Shreyas"]},
                        "relationship_map": [duplicate_relationship],
                        "coverage_audit": {"families": ["team"]},
                        "search_audit": {"quality_gate_passed": True},
                    },
                    "sources": [{"uri": "https://example.com/a", "title": "A"}],
                    "web_search_queries": ["Shreyas Iyer current team"],
                },
                {
                    "branch_id": "historical_timeline",
                    "profile": {
                        "query_resolution": {"primary_type": "PERSON"},
                        "identity_forms": {"aliases": ["Shreyas", "Iyer"]},
                        "relationship_map": [
                            duplicate_relationship,
                            {
                                "relationship_id": "R2",
                                "interpretation_id": "primary",
                                "related_subject": "Kolkata Knight Riders",
                                "related_subject_type": "TEAM",
                                "relationship_class": "CORE_RELATED",
                                "relationship_family": "former team",
                                "relationship_role": "FORMER_AFFILIATION",
                                "factual_bridge": "Shreyas Iyer formerly captained KKR.",
                                "direction": "Shreyas Iyer -> formerly captained -> KKR",
                                "durable_or_current": "HISTORICAL",
                                "evidence_summary": "A source confirms the former role.",
                                "evidence_source_urls": ["https://example.com/b"],
                                "can_retrieve_standalone": True,
                                "acceptance_condition": "Title concerns the former role.",
                                "allowed_story_angles": ["Former KKR captaincy"],
                                "excluded_story_angles": ["Unrelated KKR fixtures"],
                                "required_title_cues": [],
                                "rejection_rule": "Reject unrelated fixtures.",
                            },
                        ],
                        "coverage_audit": {"families": ["former team"]},
                        "search_audit": {"quality_gate_passed": False},
                    },
                    "sources": [
                        {"uri": "https://example.com/a", "title": "Duplicate"},
                        {"uri": "https://example.com/b", "title": "B"},
                    ],
                    "web_search_queries": [
                        "shreyas iyer current team",
                        "Shreyas Iyer former teams",
                    ],
                },
            ]
        )

        profile = json.loads(str(merged["research_text"]))
        self.assertEqual(len(profile["relationship_map"]), 1)
        self.assertEqual(profile["relationship_map"][0]["verification_branch_count"], 2)
        self.assertEqual(
            profile["identity_forms"]["aliases"], ["Shreyas", "Iyer"]
        )
        self.assertEqual(profile["search_audit"]["quality_gate_passed"], False)
        self.assertEqual(len(merged["sources"]), 2)
        self.assertEqual(len(merged["web_search_queries"]), 2)
        self.assertTrue(
            all(
                relationship["relationship_id"].startswith("verified:")
                for relationship in profile["relationship_map"]
            )
        )

    def test_branch_corroboration_tolerates_free_text_taxonomy_variation(self) -> None:
        relationship = {
            "relationship_id": "R1",
            "interpretation_id": "primary",
            "related_subject": "Punjab Kings",
            "related_subject_type": "TEAM",
            "relationship_class": "CORE_RELATED",
            "relationship_family": "current team",
            "relationship_role": "FORMAL_AFFILIATION",
            "factual_bridge": "Shreyas Iyer captains Punjab Kings.",
            "direction": "Shreyas Iyer -> captains -> Punjab Kings",
            "durable_or_current": "CURRENT",
            "evidence_summary": "The official team source confirms the role.",
            "evidence_source_urls": ["https://example.com/source"],
            "can_retrieve_standalone": True,
            "acceptance_condition": "The title concerns the captaincy.",
            "allowed_story_angles": ["Punjab Kings captaincy"],
            "excluded_story_angles": ["Unrelated fixtures"],
            "required_title_cues": [],
            "rejection_rule": "Reject unrelated fixtures.",
        }
        variant = copy.deepcopy(relationship)
        variant["interpretation_id"] = "resolved-person"
        variant["relationship_family"] = "franchise affiliation"
        variant["relationship_role"] = "DIRECT_PARTICIPANT"
        variant["related_subject_type"] = "SPORTS_FRANCHISE"
        variant["relationship_class"] = "CONTEXTUAL"
        variant["can_retrieve_standalone"] = False
        variant["required_title_cues"] = ["Punjab Kings captaincy"]
        variant["durable_or_current"] = "current role"
        branches = [
            {
                "branch_id": branch_id,
                "profile": {"relationship_map": [profile_relationship]},
                "sources": [{"uri": "https://example.com/source"}],
                "web_search_queries": ["Shreyas Iyer Punjab Kings captain"],
            }
            for branch_id, profile_relationship in (
                ("evidence_primary", relationship),
                ("independent_verifier", variant),
            )
        ]

        merged = _merge_grounded_research_branches(branches)
        profile = json.loads(str(merged["research_text"]))

        self.assertEqual(len(profile["relationship_map"]), 1)
        self.assertEqual(profile["relationship_map"][0]["verification_branch_count"], 2)
        self.assertEqual(profile["relationship_map"][0]["relationship_class"], "CONTEXTUAL")
        self.assertFalse(profile["relationship_map"][0]["can_retrieve_standalone"])
        self.assertEqual(
            profile["relationship_map"][0]["required_title_cues"],
            ["Punjab Kings captaincy"],
        )

    def test_failed_research_branch_is_retried_without_rerunning_successes(self) -> None:
        calls: dict[str, int] = {}

        def fake_branch(
            _prompt: str,
            branch_id: str,
            _branch_focus: str,
            _endpoint: str,
            _access_token: str,
            _model_name: str,
        ) -> dict[str, object]:
            calls[branch_id] = calls.get(branch_id, 0) + 1
            if branch_id == "independent_verifier" and calls[branch_id] == 1:
                raise ValueError("temporary missing grounding metadata")
            return {
                "branch_id": branch_id,
                "profile": {"relationship_map": []},
                "sources": [],
                "web_search_queries": [],
            }

        with patch(
            "src.vertex_related._execute_grounded_research_branch",
            side_effect=fake_branch,
        ):
            results, errors = _run_grounded_research_branches(
                prompt="prompt",
                branch_specs=(
                    ("evidence_primary", "primary"),
                    ("independent_verifier", "verify"),
                    ("ambiguity_temporal_verifier", "ambiguity"),
                ),
                endpoint="endpoint",
                access_token="token",
                model_name="model",
                max_attempts=2,
            )

        self.assertEqual(
            [result["branch_id"] for result in results],
            [
                "evidence_primary",
                "independent_verifier",
                "ambiguity_temporal_verifier",
            ],
        )
        self.assertEqual(errors, [])
        self.assertEqual(calls["evidence_primary"], 1)
        self.assertEqual(calls["independent_verifier"], 2)
        self.assertEqual(calls["ambiguity_temporal_verifier"], 1)

    def test_candidate_grouping_is_stable_by_primary_route(self) -> None:
        candidates = [
            {"story_id": "1", "retrieval_evidence": [{"relationship_id": "B"}]},
            {"story_id": "2", "retrieval_evidence": [{"relationship_id": "A"}]},
            {"story_id": "3", "retrieval_evidence": [{"relationship_id": "B"}]},
            {"story_id": "4", "retrieval_evidence": []},
        ]

        grouped = _group_candidates_by_primary_route(candidates)

        self.assertEqual([item["story_id"] for item in grouped], ["2", "1", "3", "4"])

    def test_context_cache_contains_complete_profile_and_ttl(self) -> None:
        class FakeResponse:
            status_code = 200
            text = '{"name":"projects/p/locations/global/cachedContents/cache-1"}'

            @staticmethod
            def json() -> dict[str, str]:
                return {"name": "projects/p/locations/global/cachedContents/cache-1"}

            @staticmethod
            def raise_for_status() -> None:
                return None

        class FakeSession:
            def __init__(self) -> None:
                self.call: dict[str, object] = {}

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                self.call = {"url": url, **kwargs}
                return FakeResponse()

        session = FakeSession()
        profile = json.dumps({"relationship_map": [{"relationship_id": "R1"}]})

        cache_name = _create_vertex_context_cache(
            keyword_query="Shreyas Iyer",
            research_text=profile,
            session=session,  # type: ignore[arg-type]
            project_id="p",
            location="global",
            model_name="gemini-2.5-flash",
            access_token="token",
        )

        request_json = session.call["json"]
        self.assertIsInstance(request_json, dict)
        assert isinstance(request_json, dict)
        cached_text = request_json["contents"][0]["parts"][0]["text"]
        self.assertEqual(cache_name, "projects/p/locations/global/cachedContents/cache-1")
        self.assertEqual(request_json["ttl"], "3600s")
        self.assertIn(profile, cached_text)

    def test_validation_retries_with_full_profile_when_cache_is_rejected(self) -> None:
        class FakeResponse:
            reason = ""

            def __init__(self, status_code: int, payload: dict[str, object]) -> None:
                self.status_code = status_code
                self._payload = payload
                self.text = json.dumps(payload)

            def json(self) -> dict[str, object]:
                return self._payload

            def raise_for_status(self) -> None:
                return None

        successful_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "batch_complete": True,
                                        "evaluated_count": 1,
                                        "evaluated_story_ids": ["1"],
                                        "matched_titles": [],
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }

        class FakeSession:
            def __init__(self) -> None:
                self.trust_env = True
                self.calls: list[dict[str, object]] = []

            def post(self, _url: str, **kwargs: object) -> FakeResponse:
                self.calls.append(copy.deepcopy(kwargs))
                if len(self.calls) == 1:
                    return FakeResponse(400, {"error": "cache unsupported"})
                return FakeResponse(200, successful_payload)

        class FakeCredentials:
            token = "token"

        session = FakeSession()
        research_text = json.dumps({"relationship_map": [{"relationship_id": "R1"}]})
        candidate = {
            "story_id": "1",
            "page_title": "India T20 captaincy transition",
            "retrieval_evidence": [{"relationship_id": "R1"}],
        }

        with patch("src.vertex_related.requests.Session", return_value=session):
            parsed = _select_generative_retrieval_batch_with_vertex(
                keyword_query="Shreyas Iyer",
                candidate_titles=[candidate],
                grounded_research={"research_text": research_text},
                credentials=FakeCredentials(),  # type: ignore[arg-type]
                project_id="p",
                location="global",
                model_name="gemini-2.5-flash",
                cached_content_name="projects/p/locations/global/cachedContents/cache-1",
            )

        self.assertTrue(parsed["batch_complete"])
        self.assertEqual(len(session.calls), 2)
        self.assertIn("cachedContent", session.calls[0]["json"])
        self.assertNotIn("cachedContent", session.calls[1]["json"])
        fallback_prompt = session.calls[1]["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn(research_text, fallback_prompt)


class RelationshipRouteConstraintTests(unittest.TestCase):
    def test_default_route_extraction_does_not_truncate_full_profile(self) -> None:
        relationships = [
            {
                "relationship_id": f"R{index}",
                "related_subject": f"Subject {index}",
                "relationship_class": "CORE_RELATED",
                "relationship_family": f"family {index}",
                "factual_bridge": f"Shreyas Iyer has a verified relationship {index}.",
                "confidence": 0.95,
                "evidence_summary": "Verified evidence.",
                "interpretation_id": "primary",
                "verification_branch_count": 2,
                "evidence_source_urls": ["https://example.com/source"],
                "false_positive_risk": "low",
                "can_retrieve_standalone": True,
                "acceptance_condition": "The title expresses the verified relationship.",
                "rejection_rule": "Reject unrelated subjects.",
                "editorial_manifestations": [f"Manifestation {index}"],
            }
            for index in range(25)
        ]

        routes = extract_relationship_routes(
            keyword_query="Shreyas Iyer",
            grounded_research={
                "research_text": json.dumps({"relationship_map": relationships})
            },
            max_texts_per_relationship=1,
        )

        self.assertEqual(len({route.relationship_id for route in routes}), 25)

    def test_profile_can_disable_standalone_endpoint_retrieval(self) -> None:
        relationship = {
            "relationship_id": "R6",
            "related_subject": "Suryakumar Yadav",
            "relationship_class": "CORE_RELATED",
            "relationship_family": "Captaincy succession",
            "factual_bridge": "Shreyas replaced Suryakumar as India T20I captain.",
            "confidence": 0.95,
            "evidence_summary": "BCCI announced the succession.",
            "interpretation_id": "primary",
            "verification_branch_count": 2,
            "evidence_source_urls": ["https://example.com/source"],
            "false_positive_risk": "low",
            "can_retrieve_standalone": False,
            "acceptance_condition": "Title expresses the India T20I succession.",
            "allowed_story_angles": ["India T20I captaincy transition"],
            "excluded_story_angles": ["Mumbai Indians captaincy"],
            "required_title_cues": ["India", "T20I", "captaincy"],
            "rejection_rule": "Reject endpoint-only or franchise captaincy stories.",
            "editorial_manifestations": ["India T20I captaincy transition"],
        }
        routes = extract_relationship_routes(
            keyword_query="Shreyas Iyer",
            grounded_research={"research_text": json.dumps({"relationship_map": [relationship]})},
        )

        self.assertTrue(routes)
        self.assertTrue(all(not route.can_retrieve_standalone for route in routes))
        self.assertTrue(all(route.retrieval_text != "Suryakumar Yadav" for route in routes))
        self.assertTrue(all("Shreyas Iyer" in route.retrieval_text for route in routes))


if __name__ == "__main__":
    unittest.main()
