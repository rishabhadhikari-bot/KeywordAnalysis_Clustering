import os
import unittest
from unittest.mock import patch

import pandas as pd

from src.opensearch_refined import (
    OpenSearchRefinedError,
    OpenSearchSettings,
    _index_definition,
    _iter_title_documents,
    build_refined_primary_analysis_result,
    build_refined_query,
    classify_matched_queries,
    collect_all_refined_primary_story_ids,
    collect_all_refined_story_id_sets,
    load_opensearch_settings,
    parse_search_hits,
    start_local_opensearch,
    title_corpus_fingerprint,
)


class RefinedOpenSearchQueryTests(unittest.TestCase):
    def test_complete_refined_ids_include_every_displayed_tier(self) -> None:
        class FakeClient:
            def search_page(self, **kwargs):
                return (
                    [
                        {"story_id": "exact", "refined_match_tier": "Exact phrase"},
                        {"story_id": "alias", "refined_match_tier": "Known alias"},
                        {"story_id": "fuzzy", "refined_match_tier": "Similar spelling"},
                    ],
                    3,
                    None,
                )

        all_ids, primary_ids = collect_all_refined_story_id_sets(
            FakeClient(),
            normalized_query="haridwar",
            match_mode="All keywords",
        )

        self.assertEqual(all_ids, {"exact", "alias", "fuzzy"})
        self.assertEqual(primary_ids, {"exact", "alias"})

    def test_complete_primary_ids_include_root_variants_but_not_similar_spelling(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def search_page(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return (
                        [
                            {"story_id": "exact", "refined_match_tier": "Exact keywords"},
                            {"story_id": "root", "refined_match_tier": "Root variant"},
                        ],
                        3,
                        [3.0, 10, "root"],
                    )
                return (
                    [{"story_id": "fuzzy", "refined_match_tier": "Similar spelling"}],
                    3,
                    None,
                )

        client = FakeClient()
        story_ids = collect_all_refined_primary_story_ids(
            client,
            normalized_query="israel",
            match_mode="All keywords",
            page_size=2,
        )

        self.assertEqual(story_ids, {"exact", "root"})
        self.assertEqual(client.calls, 2)

    def test_load_settings_reads_opt_in_auto_start_options(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENSEARCH_AUTO_START": "true",
                "OPENSEARCH_HOME": "C:/OpenSearch",
                "OPENSEARCH_STARTUP_TIMEOUT_SECONDS": "45",
            },
        ):
            settings = load_opensearch_settings()

        self.assertTrue(settings.auto_start)
        self.assertEqual(settings.home, "C:/OpenSearch")
        self.assertEqual(settings.startup_timeout_seconds, 45.0)

    def test_auto_start_rejects_remote_opensearch_url(self) -> None:
        settings = OpenSearchSettings(
            url="https://search.example.com",
            auto_start=True,
            home="C:/OpenSearch",
        )

        with self.assertRaisesRegex(OpenSearchRefinedError, "local loopback URL"):
            start_local_opensearch(settings)

    @patch("src.opensearch_refined._opensearch_is_reachable", return_value=True)
    def test_auto_start_returns_when_local_service_is_already_reachable(
        self,
        reachable,
    ) -> None:
        settings = OpenSearchSettings(
            url="http://127.0.0.1:9200",
            auto_start=True,
        )

        start_local_opensearch(settings)

        reachable.assert_called_once_with(settings)

    def test_all_keywords_builds_ranked_automatic_tiers(self) -> None:
        payload = build_refined_query("government policy", "All keywords")
        clauses = payload["query"]["bool"]["should"]
        serialized = str(clauses)

        self.assertIn("exact_phrase", serialized)
        self.assertIn("exact_keywords", serialized)
        self.assertIn("root_variant", serialized)
        self.assertIn("similar_spelling", serialized)
        self.assertIn("AUTO", serialized)
        self.assertEqual(payload["query"]["bool"]["minimum_should_match"], 1)

    def test_exact_phrase_mode_does_not_add_out_of_order_exact_keyword_clause(self) -> None:
        payload = build_refined_query("lok sabha", "Exact cleaned phrase")
        serialized = str(payload["query"]["bool"]["should"])

        self.assertIn("exact_phrase", serialized)
        self.assertIn("root_variant", serialized)
        self.assertIn("similar_spelling", serialized)
        self.assertNotIn("exact_keywords", serialized)

    def test_any_keyword_uses_or_operator(self) -> None:
        payload = build_refined_query("custom duty", "Any keyword")
        clauses = payload["query"]["bool"]["should"]
        keyword_clause = next(
            clause for clause in clauses if "match" in clause and "clean_title" in clause["match"]
        )

        self.assertEqual(keyword_clause["match"]["clean_title"]["operator"], "or")

    def test_search_after_adds_stable_pagination_cursor(self) -> None:
        payload = build_refined_query(
            "delhi",
            "All keywords",
            size=1000,
            search_after=[12.5, 100, "42"],
        )

        self.assertEqual(payload["search_after"], [12.5, 100, "42"])
        self.assertEqual(payload["sort"][-1], {"story_id": {"order": "asc"}})

    def test_match_classification_uses_highest_precision_tier(self) -> None:
        self.assertEqual(
            classify_matched_queries(["similar_spelling", "root_variant", "exact_keywords"]),
            "Exact keywords",
        )
        self.assertEqual(classify_matched_queries(["root_variant"]), "Root variant")
        self.assertEqual(classify_matched_queries(["known_alias"]), "Known alias")

    def test_safe_alias_adds_separate_ranked_clause(self) -> None:
        payload = build_refined_query("up election", "All keywords")
        serialized = str(payload["query"]["bool"]["should"])

        self.assertIn("known_alias", serialized)
        self.assertIn("alias_title", serialized)
        self.assertIn("refined_alias_search", serialized)

    def test_ambiguous_alias_does_not_expand(self) -> None:
        payload = build_refined_query("ap election", "All keywords")

        self.assertNotIn("known_alias", str(payload["query"]["bool"]["should"]))

    def test_index_uses_query_time_synonym_graph(self) -> None:
        definition = _index_definition()
        analysis = definition["settings"]["analysis"]
        alias_mapping = definition["mappings"]["properties"]["alias_title"]

        self.assertEqual(analysis["filter"]["refined_alias_synonyms"]["type"], "synonym_graph")
        self.assertEqual(alias_mapping["search_analyzer"], "refined_alias_search")

    def test_index_document_preserves_ampersand_initialism(self) -> None:
        titles = pd.DataFrame(
            [{"story_id": "1", "page_title": "J&K election update", "clean_title": "election update"}]
        )

        document = next(iter(_iter_title_documents(titles)))

        self.assertEqual(document["alias_title"], "jk election")

    def test_search_hit_parsing_preserves_rank_evidence(self) -> None:
        rows, total = parse_search_hits(
            {
                "hits": {
                    "total": {"value": 1, "relation": "eq"},
                    "hits": [
                        {
                            "_id": "42",
                            "_score": 7.12567,
                            "_source": {
                                "story_id": "42",
                                "page_title": "Government policies explained",
                            },
                            "matched_queries": ["root_variant", "similar_spelling"],
                        }
                    ],
                }
            }
        )

        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["story_id"], "42")
        self.assertEqual(rows[0]["refined_match_tier"], "Root variant")
        self.assertEqual(rows[0]["opensearch_score"], 7.1257)

    def test_primary_analysis_excludes_only_similar_spelling_tier(self) -> None:
        refined_result = {
            "matched_titles": pd.DataFrame(
                [
                    {"story_id": "1", "refined_match_tier": "Exact phrase"},
                    {"story_id": "2", "refined_match_tier": "Exact keywords"},
                    {"story_id": "3", "refined_match_tier": "Known alias"},
                    {"story_id": "4", "refined_match_tier": "Root variant"},
                    {"story_id": "5", "refined_match_tier": "Similar spelling"},
                ]
            ),
            "matched_story_months": pd.DataFrame(
                [
                    {
                        "story_id": str(story_id),
                        "month": pd.Timestamp("2026-01-01"),
                        "views": story_id * 100,
                        "refined_match_tier": tier,
                    }
                    for story_id, tier in enumerate(
                        [
                            "Exact phrase",
                            "Exact keywords",
                            "Known alias",
                            "Root variant",
                            "Similar spelling",
                        ],
                        start=1,
                    )
                ]
            ),
        }

        analysis_result = build_refined_primary_analysis_result(refined_result)

        self.assertEqual(
            analysis_result["matched_title_summary"]["story_id"].tolist(),
            ["1", "2", "3", "4"],
        )
        self.assertEqual(
            analysis_result["matched_story_months"]["story_id"].tolist(),
            ["1", "2", "3", "4"],
        )
        self.assertEqual(analysis_result["monthly_summary"].iloc[0]["views"], 1000)
        self.assertEqual(
            analysis_result["excluded_refined_match_tiers"],
            ("Similar spelling",),
        )

    def test_corpus_fingerprint_changes_with_traffic_data(self) -> None:
        titles = pd.DataFrame(
            [
                {
                    "story_id": "1",
                    "page_title": "Policy explained",
                    "clean_title": "policy explain",
                    "total_views": 100,
                }
            ]
        )
        first = title_corpus_fingerprint(titles)
        titles.loc[0, "total_views"] = 101

        self.assertNotEqual(first, title_corpus_fingerprint(titles))


if __name__ == "__main__":
    unittest.main()
