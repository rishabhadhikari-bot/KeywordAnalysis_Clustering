import unittest

from streamlit.testing.v1 import AppTest


SCRIPT = '''
import pandas as pd
from src.miscellaneous_tab import show_miscellaneous_tab
rows = pd.DataFrame({"story_id": ["01", "01"], "page_title": ["Traffic traffic news"] * 2,
                     "month": pd.to_datetime(["2026-08-01", "2026-09-01"]), "views": [100, 200]})
show_miscellaneous_tab(rows)
'''


class MiscellaneousTabTests(unittest.TestCase):
    def test_build_filter_and_invalidate_results(self):
        app = AppTest.from_string(SCRIPT).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertEqual(len(app.dataframe), 0)
        app.button(key="misc_build").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 4)
        app.text_input(key="misc_token_filter").set_value("traffic").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 2)
        self.assertEqual(app.dataframe[0].value.iloc[0]["views_per_occurrence"], 100)
        app.multiselect[0].set_value(["2026-09"]).run()
        self.assertEqual(len(app.dataframe[0].value), 1)
        self.assertEqual(app.dataframe[0].value.iloc[0]["total_views"], 200)
        app.checkbox(key="misc_stopwords").uncheck().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 0)


if __name__ == "__main__":
    unittest.main()
