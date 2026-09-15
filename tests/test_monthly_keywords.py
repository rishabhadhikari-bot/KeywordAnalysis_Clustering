import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.monthly_keywords import build_monthly_keyword_dataset


class MonthlyKeywordsTests(unittest.TestCase):
    def test_month_grouping_duplicates_repetitions_and_zero_views(self):
        rows = pd.DataFrame({
            "month": pd.to_datetime(["2026-08-01", "2026-08-15", "2026-08-01", "2026-09-01"]),
            "story_id": ["01", "01", "02", "01"], "views": [40, 60, 0, 200],
            "page_title": ["Traffic traffic", "Traffic traffic", "Traffic news", "Traffic traffic"],
            "above_monthly_threshold": [False] * 4,
        })
        original = rows.copy(deep=True)
        result, audit = build_monthly_keyword_dataset(rows)
        traffic = result.loc[result.token.eq("traffic")].set_index("month")
        self.assertEqual(traffic.loc["2026-08", "occurrences"], 2)
        self.assertEqual(traffic.loc["2026-08", "total_token_occurrences"], 3)
        self.assertEqual(traffic.loc["2026-08", "total_views"], 100)
        self.assertEqual(traffic.loc["2026-08", "views_per_occurrence"], 50)
        self.assertEqual(traffic.loc["2026-09", "views_per_occurrence"], 200)
        self.assertEqual(audit["page_months"], 3)
        assert_frame_equal(rows, original)

    def test_missing_titles_empty_data_and_exclusions(self):
        rows = pd.DataFrame({"month": ["2026-08-01"], "story_id": ["01"],
                             "page_title": [None], "views": [10]})
        result, audit = build_monthly_keyword_dataset(rows)
        self.assertTrue(result.empty)
        self.assertEqual(audit["missing_title_views"], 10)
        self.assertEqual(audit["missing_title_page_months"], 1)
        result, _ = build_monthly_keyword_dataset(rows.iloc[:0])
        self.assertTrue(result.empty)
        rows["page_title"] = "Traffic"
        result, _ = build_monthly_keyword_dataset(rows, frozenset({"traffic"}))
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
