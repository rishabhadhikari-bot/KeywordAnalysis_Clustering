import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.daily_keywords import build_daily_keyword_dataset, title_tokens


class DailyKeywordsTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({"story_id": ["01", "02"],
                                      "page_title": ["The traffic traffic forecast", "Traffic news"]})

    def test_daily_denominator_repetitions_zero_views_and_no_mutation(self):
        rows = pd.DataFrame({"date": ["2026-09-08"] * 3 + ["20260909"],
                             "story_id": ["01", "01", "02", "02"], "views": [40, 60, 0, 200]})
        original = rows.copy(deep=True)
        metadata = self.metadata.copy(deep=True)
        result, audit = build_daily_keyword_dataset(rows, self.metadata)
        traffic = result.loc[result.token.eq("traffic")].set_index("date")
        self.assertEqual(traffic.loc["2026-09-08", "occurrences"], 2)
        self.assertEqual(traffic.loc["2026-09-08", "total_token_occurrences"], 3)
        self.assertEqual(traffic.loc["2026-09-08", "total_views"], 100)
        self.assertEqual(traffic.loc["2026-09-08", "views_per_occurrence"], 50)
        self.assertEqual(traffic.loc["2026-09-09", "views_per_occurrence"], 200)
        self.assertNotIn("the", result.token.tolist())
        self.assertEqual(audit["page_days"], 3)
        assert_frame_equal(rows, original)
        assert_frame_equal(self.metadata, metadata)

    def test_historical_titles_missing_titles_and_aliases(self):
        rows = pd.DataFrame({"Date": ["2026-09-08"] * 2 + ["2026-09-09"],
                             "story_id": ["01", "missing", "01"], "Event count": [10, 7, 20],
                             "Page title": ["Old title", None, "New title"]})
        result, audit = build_daily_keyword_dataset(rows, self.metadata)
        self.assertEqual(audit["missing_title_page_days"], 1)
        self.assertEqual(audit["missing_title_views"], 7)
        self.assertIn("old", result.token.tolist())
        self.assertNotIn("traffic", result.token.tolist())

    def test_bad_dates_views_ids_and_monthly_data_are_rejected(self):
        base = {"date": ["2026-09-08"], "story_id": ["01"], "views": [10]}
        for column, value in [("date", "Sep-2026"), ("date", "2026-02-30"),
                              ("views", -1), ("views", "bad"), ("views", float("inf")),
                              ("story_id", None), ("story_id", " ")]:
            with self.subTest(column=column, value=value):
                rows = pd.DataFrame({**base, column: [value]})
                with self.assertRaises(ValueError):
                    build_daily_keyword_dataset(rows, self.metadata)
        with self.assertRaisesRegex(ValueError, "date"):
            build_daily_keyword_dataset(pd.DataFrame({"Month": ["Sep-2026"], "story_id": ["01"],
                                                      "Event count": [10]}), self.metadata)

    def test_conflicting_titles_rejected(self):
        rows = pd.DataFrame({"date": ["2026-09-08"] * 2, "story_id": ["01"] * 2,
                             "views": [10, 20], "page_title": ["Old title", "New title"]})
        with self.assertRaisesRegex(ValueError, "Conflicting titles"):
            build_daily_keyword_dataset(rows, self.metadata)

    def test_unicode_punctuation_numbers_and_optional_exclusions(self):
        self.assertEqual(title_tokens("AI-driven हिंदी 2026!"), ["ai", "driven", "हिंदी", "2026"])
        rows = pd.DataFrame({"date": ["2026-09-08"], "story_id": ["01"], "views": [10]})
        result, _ = build_daily_keyword_dataset(rows, self.metadata, frozenset())
        self.assertIn("the", result.token.tolist())
        empty, _ = build_daily_keyword_dataset(rows, self.metadata, frozenset({"the", "traffic", "forecast"}))
        self.assertTrue(empty.empty)


if __name__ == "__main__":
    unittest.main()
