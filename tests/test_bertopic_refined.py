import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src import topic_retrieval_refined as refined


class RefinedTopicTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"story_id": "1", "page_title": "school reform announced"},
            {"story_id": "2", "page_title": "school reform annouced"},
            {"story_id": "3", "page_title": "school reform announced"},
            {"story_id": "4", "page_title": "UP election results"},
            {"story_id": "5", "page_title": "uttar pradesh assembly"},
            {"story_id": "6", "page_title": "update education policy"},
            {"story_id": "7", "page_title": "sports tournament"},
        ]
        self.artifact = {
            "story_ids": np.array([row["story_id"] for row in self.rows]),
            "assignments": np.array([0, 0, 0, 0, 0, 0, -1]),
            "topic_ids": np.array([0]), "centroids": np.array([[1., 0.]]),
            "members": {0: np.arange(6)},
            "manifest": {"labels": {"0": "education"}, "outlier_count": 1,
                         "title_count": 7, "fingerprint": "fixture"},
        }

    def retrieve(self, query, **kwargs):
        with patch.object(refined, "_encode", return_value=np.array([[1., 0.]])):
            return refined.retrieve_bertopic_refined_candidates(
                query, self.rows, artifact=self.artifact, **kwargs)

    def test_identity_only_exclusions_and_token_boundaries(self):
        results, diagnostics = self.retrieve("UP", excluded_story_ids=("1",))
        self.assertEqual({r["story_id"] for r in results}, {"2", "3", "6"})
        self.assertEqual(diagnostics["removed_excluded_ids"], 1)
        self.assertEqual(diagnostics["removed_keyword_titles"], 2)
        forbidden = refined.keyword_variant_tokens("UP")
        self.assertTrue(all(not forbidden.intersection(refined._tokens(r["page_title"])) for r in results))

    def test_multi_word_query_removes_any_token(self):
        results, _ = self.retrieve("election education")
        self.assertNotIn("4", {r["story_id"] for r in results})
        self.assertNotIn("6", {r["story_id"] for r in results})

    def test_transliteration_and_punctuation_aliases(self):
        self.assertIn("bangalore", refined.keyword_variant_tokens("Bengaluru"))
        self.assertEqual(refined.keyword_variant_tokens("U.P."), refined.keyword_variant_tokens("UP"))
        self.assertIn("kashmir", refined.keyword_variant_tokens("J&K"))
        self.assertNotIn("madhya", refined.keyword_variant_tokens("MP"))

    def test_unrelated_query_and_all_outliers_have_no_fallback(self):
        with patch.object(refined, "_encode", return_value=np.array([[0., 1.]])):
            results, diagnostics = refined.retrieve_bertopic_refined_candidates(
                "astronomy", self.rows, artifact=self.artifact)
        self.assertEqual(results, [])
        self.assertEqual(diagnostics["selected_topics"], [])
        self.artifact.update(topic_ids=np.array([], dtype=int), centroids=np.empty((0, 2)), members={})
        self.assertEqual(self.retrieve("UP")[0], [])

    def test_empty_titles_and_fingerprint_changes(self):
        extra = [{"story_id": "8", "page_title": " "}, {"story_id": "9", "page_title": np.nan}]
        self.assertEqual(refined._rows(self.rows + extra), self.rows)
        fingerprint = refined.refined_corpus_fingerprint(self.rows)
        self.assertEqual(fingerprint, refined.refined_corpus_fingerprint(list(reversed(self.rows))))
        changed = [*self.rows, {"story_id": "10", "page_title": "new article"}]
        self.assertNotEqual(fingerprint, refined.refined_corpus_fingerprint(changed))
        with self.assertRaisesRegex(ValueError, "16 non-empty"):
            refined.fit_bertopic_refined_model(extra)

    def test_persistence_reuses_fit_and_refreshes_on_title_change(self):
        rows = [{"story_id": str(i), "page_title": f"school reform {i}"} for i in range(16)]
        model = Mock()
        trained = (model, np.zeros(16, dtype=int), np.array([0]), np.array([[1., 0.]]), {"0": "schools"})
        test_root = Path(__file__).resolve().parents[1] / ".tmp"
        directory = test_root / f"refined-test-{uuid.uuid4().hex}"
        directory.mkdir()
        self.addCleanup(lambda: shutil.rmtree(directory) if directory.resolve().parent == test_root.resolve() else None)
        with patch.object(refined, "_train", return_value=trained) as train:
            first = refined.fit_bertopic_refined_model(rows, index_dir=directory)
            refined._load_artifact.cache_clear()
            second = refined.fit_bertopic_refined_model(rows, index_dir=directory)
            self.assertEqual(train.call_count, 1)
            np.testing.assert_array_equal(first["assignments"], second["assignments"])
            self.assertEqual(len(second["members"][0]), 16)
            rows[0]["page_title"] = "new education legislation"
            refined.fit_bertopic_refined_model(rows, index_dir=directory)
            self.assertEqual(train.call_count, 2)
            self.assertEqual(len(list(Path(directory).glob("*/manifest.json"))), 2)
        refined._load_artifact.cache_clear()

    def test_app_flow_uses_independent_model_and_preserves_duplicate_titles(self):
        import app2
        with patch.object(app2, "fit_bertopic_refined_model", return_value=self.artifact), \
             patch.object(refined, "_encode", return_value=np.array([[1., 0.]])), \
             patch.object(app2, "get_bertopic_view_data", side_effect=AssertionError("Legacy flow called")):
            results, _, catalog = app2.run_bertopic_refined_tab_flow(
                "UP", pd.DataFrame(self.rows), ("1",), 5, 0.45)
        self.assertEqual({r["story_id"] for r in results}, {"2", "3", "6"})
        self.assertEqual(catalog, [{"topic_id": 0, "label": "education", "size": 6}])


if __name__ == "__main__":
    unittest.main()
