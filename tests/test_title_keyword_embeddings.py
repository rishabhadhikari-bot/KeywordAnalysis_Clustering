import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.title_keyword_embeddings import prepare_titles, retrieve_keywords, load_title_vectors, INDEX_DIR


class TitleKeywordTests(unittest.TestCase):
    def test_titles_retain_context_and_deduplicate(self):
        rows = pd.DataFrame({"story_id": ["1", "1", "2", "3"],
                             "page_title": ["old", " The  Election: 2026! ", None, " "]})
        original = rows.copy(deep=True)
        titles = prepare_titles(rows)
        self.assertEqual(titles.page_title.tolist(), ["The Election: 2026!"])
        pd.testing.assert_frame_equal(rows, original)
        with patch("src.title_keyword_embeddings.load_or_build_index") as build:
            load_title_vectors(tuple(titles.page_title), "model")
            build.assert_called_once_with(("The Election: 2026!",), "model", index_dir=INDEX_DIR)

    def test_semantic_titles_without_literal_query_and_keyword_evidence(self):
        titles = prepare_titles(pd.DataFrame({"story_id": ["1", "2", "3"],
            "page_title": ["The ballot voting voting", "Voting ballot 2026", "Cricket scores"]}))
        original = titles.copy(deep=True)
        vectors = np.array([[1., 0.], [.8, .6], [0., 1.]])
        result, matched = retrieve_keywords(titles, vectors, "election", np.array([1., 0.]), min_similarity=.5)
        self.assertEqual(matched.story_id.tolist(), ["1", "2"])
        self.assertEqual(result.keyword.tolist(), ["ballot", "voting"])
        self.assertEqual(result.matched_title_occurrences.tolist(), [2, 2])
        self.assertAlmostEqual(result.relevance.iloc[0], .9 * np.log1p(3 / 2), places=6)
        self.assertEqual(result.best_title_similarity.iloc[0], 1.)
        empty, _ = retrieve_keywords(titles, vectors, "election", np.array([-1., 0.]), min_similarity=.5)
        self.assertTrue(empty.empty)
        excluded, _ = retrieve_keywords(titles, vectors, "ballot", np.array([1., 0.]), exclusions="voting")
        self.assertTrue(excluded.empty)
        pd.testing.assert_frame_equal(titles, original)

    def test_ui_uses_title_vectors_and_exclusions_without_rebuild(self):
        script = '''
import pandas as pd
from src.embeddings_tab import show_embeddings_tab
show_embeddings_tab(pd.DataFrame({"story_id": ["1", "2", "3"],
    "page_title": ["Ballot voting", "Voting ballot", "Cricket scores"]}))
'''
        with patch("src.embeddings_tab._index", return_value=np.array([[1., 0.], [.8, .6], [0., 1.]])) as build:
            with patch("src.embeddings_tab.encode_texts", return_value=np.array([[1., 0.]])) as encode:
                app = AppTest.from_string(script).run()
                self.assertFalse(app.exception)
                build.assert_not_called()
                app.button(key="emb_build").click().run()
                self.assertEqual(build.call_args.args[0], ("Ballot voting", "Voting ballot", "Cricket scores"))
                app.text_input(key="emb_query").set_value("election")
                app.button[-1].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(encode.call_args.args[0], ["election"])
                self.assertEqual(app.dataframe[0].value.keyword.tolist(), ["ballot", "voting"])
                app.text_input(key="emb_extra_stopwords").set_value("ballot")
                app.button[-1].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.dataframe[0].value.keyword.tolist(), ["voting"])


if __name__ == "__main__":
    unittest.main()
