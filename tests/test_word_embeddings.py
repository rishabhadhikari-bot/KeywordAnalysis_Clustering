import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.word_embeddings import build_vocabulary, index_key, load_or_build_index, query_texts, rank_keywords, title_evidence


class WordEmbeddingTests(unittest.TestCase):
    def test_vocabulary_deduplicates_stories_and_removes_stopwords(self):
        rows = pd.DataFrame({"story_id": ["1", "1", "2", "3"],
                             "page_title": ["The Election election 2026", "The Election election 2026",
                                            "Voting और चुनाव", None]})
        original = rows.copy(deep=True)
        vocabulary = build_vocabulary(rows, "voting")
        self.assertEqual(dict(vocabulary.values), {"election": 1, "चुनाव": 1})
        pd.testing.assert_frame_equal(rows, original)

    def test_cosine_ranking_filters_and_queries(self):
        vocabulary = pd.DataFrame({"keyword": ["election", "polls", "sport"], "title_occurrences": [3, 2, 1]})
        vectors = np.array([[3, 0], [2, 1], [0, 4]])
        result = rank_keywords(vocabulary, vectors, "election", np.array([2, 0]), min_similarity=.5)
        self.assertEqual(result.keyword.tolist(), ["polls"])
        self.assertAlmostEqual(result.similarity.iloc[0], 2 / np.sqrt(5), places=6)
        self.assertTrue(rank_keywords(vocabulary, vectors, "election", np.array([2, 0]), min_occurrences=3).empty)
        self.assertEqual(query_texts("Electric vehicle", False), ["electric vehicle"])
        self.assertEqual(query_texts("Polls, voting and polls", True), ["polls", "voting"])
        self.assertEqual(query_texts("!!!", False), [])

    def test_index_cache_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch("src.word_embeddings.encode_words", return_value=np.array([[1., 0.], [0., 1.]])) as encode:
                first = load_or_build_index(("polls", "voting"), "model", directory)
                second = load_or_build_index(("polls", "voting"), "model", directory)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(encode.call_count, 1)
                load_or_build_index(("ballot", "polls"), "model", directory)
                self.assertEqual(encode.call_count, 2)
            self.assertNotEqual(index_key(("polls",), "a"), index_key(("polls",), "b"))

    def test_evidence_rejects_high_scoring_noise_and_counts_distinct_stories(self):
        rows = pd.DataFrame({"story_id": ["1", "1", "2", "3"],
                             "page_title": ["Ekadashi vrat vrat", "Ekadashi vrat vrat",
                                            "Ekadashi vrat", "Ekadashis ogai"]})
        original = rows.copy(deep=True)
        evidence, matched = title_evidence(rows, "ekadashi")
        self.assertEqual(matched, 2)
        self.assertEqual(evidence.set_index("keyword").loc["vrat", "shared_titles"], 2)
        vocabulary = pd.DataFrame({"keyword": ["ogai", "vrat"], "title_occurrences": [1, 2]})
        vectors = np.array([[1., 0.], [.6, .8]])
        result = rank_keywords(vocabulary, vectors, "ekadashi", np.array([1., 0.]),
                               top_k=1, evidence=evidence)
        self.assertEqual(result.keyword.tolist(), ["vrat"])
        self.assertIn("Ekadashi", result.example_title.iloc[0])
        absent, matched = title_evidence(rows, "unknown")
        self.assertEqual(matched, 0)
        self.assertTrue(rank_keywords(vocabulary, vectors, "unknown", np.array([1., 0.]), evidence=absent).empty)
        self.assertEqual(title_evidence(rows, "ekadashi and vrat")[1], 2)
        self.assertEqual(title_evidence(rows, "ekadashi ogai")[1], 0)
        self.assertEqual(title_evidence(rows, "the")[1], 0)
        pd.testing.assert_frame_equal(rows, original)


if __name__ == "__main__":
    unittest.main()
