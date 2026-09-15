"""Exercise the app2 Wikidata controls and real filtering with mocked services."""
import unittest

from streamlit.testing.v1 import AppTest


SCRIPT = '''
from unittest.mock import Mock, patch
import pandas as pd
import streamlit as st
import app2
from src.wikidata_source_related import SourceProfile, WikidataRoute
from src.wikidata_relationship_policy import compile_policy

titles = pd.DataFrame([
    {"story_id": "direct", "page_title": "Acme Brandworks Motors launch", "total_views": 10},
    {"story_id": "primary", "page_title": "Brandworks Motors results", "total_views": 20},
    {"story_id": "fuzzy", "page_title": "Acmee Brandworks Motors expansion", "total_views": 30},
    {"story_id": "related", "page_title": "Brandworks Motors new factory", "total_views": 40},
    {"story_id": "noise", "page_title": "Maruti Suzuki launches SUV", "total_views": 50000},
])
months = pd.DataFrame([{
    "story_id": "related", "month": pd.Timestamp("2026-05-01"),
    "views": st.session_state.get("test_monthly_views", 40),
    "title_tokens": st.session_state.get("test_title_tokens", ["brandworks", "factory"]),
}])
profile = SourceProfile(query="Acme", query_aliases=["Acme"], routes=[WikidataRoute(
    related_subject="Brandworks Motors", aliases=("Maruti",), relationship_type="Wikidata: subsidiary",
    why_related="Wikidata records Acme has subsidiary Brandworks.",
    source_name="Wikidata", evidence_url="https://www.wikidata.org/wiki/Q1#P355",
    confidence=90, source_kind="wikidata_direct", target_qid="Q2",
    hop_count=2, path_qids=("Q1", "Q4", "Q2"),
    relationship_path="Acme → subsidiary → Parent Group → subsidiary → Brandworks Motors",
    path_evidence_urls=("https://www.wikidata.org/wiki/Q1#P355", "https://www.wikidata.org/wiki/Q4#P355"),
    retrieval_policy=compile_policy(({}, {}, {}), ("P355", "P355")),
)], identity_entities={
    "Q2": {"labels": {"en": {"value": "Brandworks Motors"}}, "aliases": {"en": [{"value": "Maruti"}]}},
    "Q3": {"labels": {"en": {"value": "Maruti Suzuki"}}},
})
client = Mock()
client.index_state.return_value = {
    "ready": True, "fingerprint": app2.title_corpus_fingerprint(titles), "physical_index": "test-v1",
}
with patch.object(app2, "load_opensearch_settings", return_value=Mock(configured=True)), \
     patch.object(app2, "OpenSearchRefinedClient", return_value=client), \
     patch.object(app2, "WikidataLocalClient"), \
     patch.object(app2, "load_wikidata_source_profile", return_value=profile) as profile_loader, \
     patch.object(app2, "get_complete_refined_primary_story_ids", return_value=("direct", "primary")):
    app2.show_wikidata_source_related_tab(st.session_state.get("test_query", "Acme"), "all", months, titles)
    if profile_loader.called:
        st.session_state["test_resolved_query"] = profile_loader.call_args.kwargs["query"]
'''


class WikidataTabTests(unittest.TestCase):
    def test_multiple_query_entities_share_one_article_row_and_preserve_origins(self):
        setup = '''
from dataclasses import replace
profile.routes[0] = replace(profile.routes[0], query_entity_qid="Q1",
    query_entity_label="Acme", query_entity_description="First company")
profile.routes.append(replace(profile.routes[0], query_entity_qid="Q5",
    query_entity_description="Second company", path_qids=("Q5", "Q4", "Q2")))
profile.query_entities = [
    {"qid": "Q1", "label": "Acme", "description": "First company"},
    {"qid": "Q5", "label": "Acme", "description": "Second company"},
]
client = Mock()
'''
        app = AppTest.from_string(SCRIPT.replace("client = Mock()", setup), default_timeout=30).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        results = app.dataframe[-1].value
        self.assertEqual(set(results["Story ID"]), {"fuzzy", "related"})
        self.assertEqual(len(results), 2)
        self.assertEqual(results["Views (total)"].sum(), 70)
        self.assertTrue(results["Matching query entity IDs"].str.contains("Q1").all())
        self.assertTrue(results["Matching query entity IDs"].str.contains("Q5").all())
        self.assertTrue(results["Matching query subjects"].str.contains("Second company").all())
        self.assertTrue(results["All supporting relationships"].str.contains('"Q5"').all())

    def test_saved_single_entity_pipeline_results_are_rebuilt(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        app.button[0].click().run()
        state_key = next(key for key in app.session_state.filtered_state
                         if key.startswith("wikidata_source_related_"))
        state = app.session_state[state_key]
        state["results"].attrs.pop("retrieval_pipeline_version")
        state["results"].loc[0, "page_title"] = "STALE SINGLE ENTITY TITLE"
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("STALE SINGLE ENTITY TITLE", str(app.dataframe[-1].value))
        self.assertTrue(app.session_state[state_key]["results"].attrs["retrieval_pipeline_version"])

    def test_saved_route_missing_policy_is_rebuilt_before_display(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        app.button[0].click().run()
        state_key = next(key for key in app.session_state.filtered_state
                         if key.startswith("wikidata_source_related_"))
        state = app.session_state[state_key]
        object.__delattr__(state["profile"].routes[0], "retrieval_policy")
        state["results"].loc[0, "page_title"] = "STALE UNFILTERED TITLE"
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Updating saved results" in message.value for message in app.info))
        self.assertNotIn("STALE UNFILTERED TITLE", str(app.dataframe[-1].value))
        self.assertEqual(len(app.get("download_button")), 1)
        self.assertTrue(hasattr(app.session_state[state_key]["profile"].routes[0], "retrieval_policy"))

    def test_saved_results_without_policy_version_are_rebuilt(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        app.button[0].click().run()
        state_key = next(key for key in app.session_state.filtered_state
                         if key.startswith("wikidata_source_related_"))
        state = app.session_state[state_key]
        state["results"].attrs.pop("relationship_policy_version")
        state["results"].loc[0, "page_title"] = "STALE UNFILTERED TITLE"
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("STALE UNFILTERED TITLE", str(app.dataframe[-1].value))
        self.assertTrue(app.session_state[state_key]["results"].attrs["relationship_policy_version"])

    def test_outdated_loaded_definition_shows_recovery_message_not_attribute_error(self):
        script = SCRIPT.replace("client = Mock()", 'object.__delattr__(profile.routes[0], "retrieval_policy")\nclient = Mock()')
        app = AppTest.from_string(script, default_timeout=30).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(any("Restart the app" in message.value for message in app.caption))

    def test_relationship_rejection_is_visible_and_has_no_export(self):
        script = SCRIPT.replace('compile_policy(({}, {}, {}), ("P355", "P355"))',
                                'compile_policy(({}, {}, {}), ("P69", "P112"))')
        app = AppTest.from_string(script, default_timeout=30).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(any("No permitted relationship policy" in str(frame.value) for frame in app.dataframe))
        self.assertTrue(any("relationship context" in message.value for message in app.info))

    def test_empty_results_explain_missing_relationships(self):
        script = SCRIPT.replace('source_kind="wikidata_direct"', 'source_kind="wikidata_context"')
        app = AppTest.from_string(script, default_timeout=30).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("knowledge-coverage gap" in message.value for message in app.info))
        self.assertEqual(len(app.get("download_button")), 0)

    def test_search_displays_only_related_titles_and_invalidates_when_query_changes(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.checkbox), 1)
        self.assertEqual(app.button[0].label, "Find Wikidata Related Stories")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        results = app.dataframe[-1].value
        self.assertEqual(set(results["Story ID"]), {"fuzzy", "related"})
        self.assertEqual(results["Views (total)"].sum(), 70)
        self.assertEqual(set(results["Related entity Q-ID"]), {"Q2"})
        self.assertEqual(set(results["Hops"]), {2})
        self.assertTrue(results["Relationship path"].str.contains("Parent Group").all())
        self.assertTrue(results["Path source URLs"].str.contains("Q4#P355").all())
        self.assertTrue(any("Competing longer name" in str(frame.value) for frame in app.dataframe))
        self.assertEqual(len(app.get("download_button")), 1)
        self.assertEqual(app.session_state["test_resolved_query"], "Acme")
        app.session_state["test_query"] = "Another Company"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 0)

    def test_search_needs_no_qid_input_and_ignores_old_override_state(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        self.assertEqual(len(app.text_input), 0)
        import hashlib
        app.session_state["wikidata_qid_" + hashlib.sha256(b"Acme").hexdigest()[:12]] = "Q42"
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["test_resolved_query"], "Acme")
        self.assertEqual(len(app.get("download_button")), 1)

    def test_list_metadata_is_ignored_but_monthly_traffic_changes_invalidate_results(self):
        app = AppTest.from_string(SCRIPT, default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        results = app.dataframe[-1].value.set_index("Story ID")
        self.assertEqual(results.loc["related", "Highest views number"], 40)

        app.session_state["test_title_tokens"] = ["different", "tokens"]
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("download_button")), 1)

        app.session_state["test_monthly_views"] = 80
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        results = app.dataframe[-1].value.set_index("Story ID")
        self.assertEqual(results.loc["related", "Highest views number"], 80)


if __name__ == "__main__":
    unittest.main()
