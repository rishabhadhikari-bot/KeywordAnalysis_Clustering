import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from dataclasses import replace

import pandas as pd
import requests

from src.wikidata_source_related import (
    SourceSettings, SourceRelatedError, WikidataLocalClient, build_source_related_results,
    find_wikidata_related_stories,
)


def entity(qid, label, claims=None, aliases=()):
    return {"id": qid, "labels": {"en": {"value": label}},
            "aliases": {"en": [{"value": a} for a in aliases]}, "claims": claims or {}}


def claim(qid, rank="normal", qualifiers=None):
    return {"rank": rank, "mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": qid}}},
            "qualifiers": qualifiers or {}}


class WikidataLocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = SourceSettings(cache_path=str(Path(self.tmp.name) / "knowledge.sqlite3"))
        self.client = WikidataLocalClient(self.settings, session=Mock())
        self.root = entity("Q1", "Vaibhav Suryavanshi", {"P54": [claim("Q2")]})
        self.root["sitelinks"] = {"enwiki": {"title": "Vaibhav Sooryavanshi"}}
        self.team = entity("Q2", "Rajasthan Royals", aliases=["Rajasthan cricket franchise"])
        self.data = {"Q1": self.root, "Q2": self.team}
        self.client._api = Mock(return_value={"search": [{"id": "Q1"}]})
        self.client._fetch_entities = Mock(side_effect=lambda qids: {q: self.data[q] for q in qids})

    def tearDown(self):
        self.tmp.cleanup()

    def test_case_and_spelling_aliases_resolve_offline_after_refresh(self):
        self.client.refresh("Vaibhav Sooryavanshi")
        offline = WikidataLocalClient(self.settings, session=Mock())
        offline.session.get.side_effect = AssertionError("Offline search called network")
        for query in ("vaibhav suryavanshi", "VAIBHAV SOORYAVANSHI", "Q1"):
            profile = offline.build_profile(query)
            self.assertEqual(profile.routes[0].related_subject, "Rajasthan Royals")
            self.assertIn("historical", profile.routes[0].why_related)
        offline.session.get.assert_not_called()

    def test_exclusion_aliases_include_full_name_variants_not_bare_surname(self):
        self.root["aliases"]["en"] = [{"value": "Suryavanshi"}]
        self.client.refresh("Q1")
        names = self.client.build_profile("Q1").query_aliases
        self.assertIn("Vaibhav Suryavanshi", names)
        self.assertIn("Vaibhav Sooryavanshi", names)
        self.assertNotIn("Suryavanshi", names)

    def test_unknown_local_query_does_not_call_network(self):
        with self.assertRaisesRegex(SourceRelatedError, "no downloaded"):
            self.client.build_profile("New person")
        self.client._api.assert_not_called()

    def test_download_missing_resolves_once_then_reuses_cache_without_news(self):
        first = self.client.build_profile("Vaibhav Suryavanshi", download_missing=True)
        self.assertEqual(first.routes[0].related_subject, "Rajasthan Royals")
        self.assertEqual(self.client._api.call_count, 1)
        self.client._api.reset_mock()
        self.client.build_profile("Vaibhav Suryavanshi", download_missing=True)
        self.client._api.assert_not_called()
        self.client.session.get.assert_not_called()

    def test_failed_automatic_download_is_retryable(self):
        self.client._api.side_effect = requests.Timeout("offline")
        with self.assertRaisesRegex(SourceRelatedError, "Download attempt failed"):
            self.client.build_profile("Vaibhav Suryavanshi", download_missing=True)
        self.client._api.side_effect = None
        self.assertTrue(self.client.build_profile("Vaibhav Suryavanshi", download_missing=True).routes)

    def test_composite_query_is_not_silently_reduced_to_one_entity(self):
        with self.assertRaisesRegex(SourceRelatedError, "No matching name or alias"):
            self.client.build_profile("Vaibhav Suryavanshi election", download_missing=True)

    def test_hindi_query_uses_hindi_search(self):
        self.root["labels"]["hi"] = {"value": "वैभव सूर्यवंशी"}
        self.client.build_profile("वैभव सूर्यवंशी", download_missing=True)
        self.assertEqual(self.client._api.call_args.kwargs["language"], "hi")

    def test_english_sitelink_is_used_when_english_label_is_missing(self):
        self.root["labels"] = {"hi": {"value": "वैभव सूर्यवंशी"}}
        self.client.refresh("Q1")
        profile = self.client.build_profile("Q1")
        self.assertEqual(profile.linked_entities[0].anchor, "Vaibhav Sooryavanshi")

    def test_unlabelled_target_does_not_become_a_qid_retrieval_term(self):
        self.team["labels"] = {}
        self.team["aliases"] = {}
        self.client.refresh("Q1")
        profile = self.client.build_profile("Q1")
        self.assertFalse(profile.routes)
        self.assertIn("no English/Hindi names", profile.warnings[0])

    def test_searching_cached_target_downloads_its_missing_relationship_targets(self):
        self.team["claims"] = {"P118": [claim("Q3")]}
        self.data["Q3"] = entity("Q3", "Cricket Premier League")
        self.client.refresh("Q1")
        profile = self.client.build_profile("Q2", download_missing=True)
        self.assertEqual(profile.routes[0].related_subject, "Cricket Premier League")
        self.assertFalse(profile.warnings)

    def test_wikidata_results_keep_fuzzy_only_but_remove_primary_query_and_alias_titles(self):
        self.client.refresh("Q1")
        profile = self.client.build_profile("Q1")
        profile.checked_identity_names.add("rajasthan cricket franchise")
        titles = pd.DataFrame([
            {"story_id": "primary", "page_title": "Rajasthan Royals captain profile", "total_views": 10},
            {"story_id": "copy", "page_title": "RAJASTHAN ROYALS CAPTAIN PROFILE!", "total_views": 11},
            {"story_id": "fuzzy", "page_title": "Suryavanshee joins Rajasthan Royals", "total_views": 20},
            {"story_id": "query_word", "page_title": "Vaibhav joins Rajasthan Royals", "total_views": 30},
            {"story_id": "alias", "page_title": "Vaibhav Sooryavanshi Rajasthan Royals debut", "total_views": 40},
            {"story_id": "related", "page_title": "Rajasthan cricket franchise announces squad", "total_views": 50},
            {"story_id": "other", "page_title": "Unrelated cricket news", "total_views": 60},
        ])
        lookup = Mock(return_value=("primary",))
        months = pd.DataFrame([
            {"story_id": "related", "month": "2026-04-01", "views": 15},
            {"story_id": "related", "month": "2026-05-01", "views": 35},
        ])
        results = find_wikidata_related_stories(
            query="Vaibhav Suryavanshi", profile=profile, title_summary=titles,
            story_months=months, match_mode="all", index_fingerprint="current",
            primary_id_lookup=lookup,
        )
        self.assertEqual(set(results.story_id), {"fuzzy", "related"})
        self.assertEqual(results.total_views.sum(), 70)
        self.assertEqual(results.set_index("story_id").loc["related", "highest_views"], 35)
        self.assertEqual(results.attrs["excluded_count"], 4)
        searches = {c.kwargs["normalized_query"] for c in lookup.call_args_list}
        self.assertIn("vaibhav sooryavanshi", searches)
        self.assertNotIn("primary", results.to_csv(index=False))

    def test_qid_search_still_excludes_resolved_identity_aliases(self):
        self.client.refresh("Q1")
        titles = pd.DataFrame([
            {"story_id": "direct", "page_title": "Vaibhav Suryavanshi Rajasthan Royals debut", "total_views": 10},
            {"story_id": "related", "page_title": "Rajasthan Royals debut", "total_views": 20},
        ])
        results = find_wikidata_related_stories(
            query="Q1", profile=self.client.build_profile("Q1"), title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="current",
            primary_id_lookup=Mock(return_value=()),
        )
        self.assertEqual(list(results.story_id), ["related"])

    def test_wikidata_retrieval_rejects_generic_context_and_news_routes(self):
        self.client.refresh("Q1")
        profile = self.client.build_profile("Q1")
        route = profile.routes[0]
        profile.routes = [replace(route, source_kind="wikidata_context"),
                          replace(route, source_kind="news_headline")]
        titles = pd.DataFrame([{"story_id": "related", "page_title": "Rajasthan Royals news", "total_views": 20}])
        results = find_wikidata_related_stories(
            query="Q1", profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="current",
            primary_id_lookup=Mock(return_value=()),
        )
        self.assertTrue(results.empty)
        self.assertEqual(results.attrs["retrieval_route_count"], 0)

    def test_exclusion_lookup_failure_does_not_return_unfiltered_results(self):
        self.client.refresh("Q1")
        with self.assertRaisesRegex(RuntimeError, "index unavailable"):
            find_wikidata_related_stories(
                query="Q1", profile=self.client.build_profile("Q1"), title_summary=pd.DataFrame(),
                story_months=pd.DataFrame(), match_mode="all", index_fingerprint="current",
                primary_id_lookup=Mock(side_effect=RuntimeError("index unavailable")),
            )

    def test_stored_target_can_be_searched_without_download(self):
        self.client.refresh("Q1")
        self.client._api.reset_mock()
        profile = self.client.build_profile("Rajasthan Royals")
        self.assertEqual(profile.linked_entities[0].anchor, "Rajasthan Royals")
        self.client._api.assert_not_called()

    def test_news_survives_missing_wikidata(self):
        self.client.session.get.return_value.content = b'''<rss><channel>
        <item><title>Indian cricket Rajasthan Royals victory - A</title><link>https://example.org/a</link><source>A</source></item>
        <item><title>Indian cricket Rajasthan Royals victory - B</title><link>https://example.org/b</link><source>B</source></item>
        </channel></rss>'''
        profile = self.client.build_editorial_profile("indian cricket")
        self.assertTrue(any(r.source_kind.startswith("news") for r in profile.routes))
        self.assertIn("Available", profile.source_status["Google News"])
        self.client._api.assert_not_called()
        self.assertEqual(self.client.session.get.call_args.args[0], self.settings.google_news_rss_url)

    def test_news_failure_preserves_wikidata(self):
        self.client.refresh("Q1")
        self.client.session.get.side_effect = requests.Timeout("offline")
        profile = self.client.build_editorial_profile("Q1")
        self.assertEqual(profile.routes[0].related_subject, "Rajasthan Royals")
        self.assertEqual(profile.source_status["Google News"], "Unavailable")

    def test_both_online_options_off_makes_no_network_requests(self):
        self.client.refresh("Q1")
        self.client._api.reset_mock()
        self.client.build_editorial_profile("Q1", include_news=False)
        self.client.session.get.assert_not_called()
        self.client._api.assert_not_called()

    def test_failed_initial_download_does_not_ask_to_repeat_enabled_checkbox(self):
        self.client._api.side_effect = requests.Timeout("offline")
        profile = self.client.build_editorial_profile("unknown", refresh=True, include_news=False)
        self.assertIn("Download attempt failed", profile.warnings[0])
        self.assertNotIn("Select Download", profile.warnings[0])

    def test_failed_refresh_preserves_last_complete_data(self):
        self.client.refresh("Q1")
        self.client._fetch_entities = Mock(side_effect=requests.Timeout("offline"))
        profile = self.client.build_profile("Q1", refresh=True)
        self.assertEqual(profile.routes[0].related_subject, "Rajasthan Royals")
        self.assertIn("previously downloaded", profile.warnings[0])

    def test_target_download_failure_does_not_partially_overwrite_cache(self):
        self.client.refresh("Q1")
        changed = entity("Q1", "Changed name", {"P54": [claim("Q3")]})
        self.client._fetch_entities = Mock(side_effect=[{"Q1": changed}, requests.Timeout("failed")])
        with self.assertRaises(requests.Timeout):
            self.client.refresh("Q1")
        self.assertEqual(self.client.build_profile("Q1").linked_entities[0].anchor, "Vaibhav Suryavanshi")

    def test_deprecated_claims_and_generic_human_type_are_excluded(self):
        self.root["claims"]["P54"].append(claim("Q3", rank="deprecated"))
        self.root["claims"]["P31"] = [claim("Q5")]
        self.client.refresh("Q1")
        self.assertEqual(len(self.client.build_profile("Q1").routes), 1)

    def test_context_is_lower_confidence_and_does_not_expand_to_peers(self):
        self.root["claims"] = {"P1269": [claim("Q2")]}
        self.team["claims"] = {"P527": [claim("Q3")]}
        self.client.refresh("Q1")
        routes = self.client.build_profile("Q1").routes
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].source_kind, "wikidata_context")
        self.assertLess(routes[0].confidence, 70)

    def test_membership_dates_are_preserved(self):
        self.root["claims"]["P54"][0]["qualifiers"] = {
            "P582": [{"datavalue": {"value": {"time": "+2025-01-01T00:00:00Z"}}}]
        }
        self.client.refresh("Q1")
        self.assertIn("end: 2025-01-01", self.client.build_profile("Q1").routes[0].why_related)

    def test_same_name_entities_are_all_included_offline(self):
        self.client.refresh("Q1")
        self.data["Q3"] = entity("Q3", "Vaibhav Suryavanshi")
        self.client.refresh("Q3")
        profile = self.client.build_profile("Vaibhav Suryavanshi")
        self.assertEqual({item.uri.rsplit('/', 1)[-1] for item in profile.linked_entities}, {"Q1", "Q3"})
        self.assertTrue(self.client.build_profile("Q1").routes)

    def test_corpus_matching_uses_team_alias_and_excludes_direct_story(self):
        self.client.refresh("Q1")
        titles = pd.DataFrame([
            {"story_id": "direct", "page_title": "Vaibhav Rajasthan Royals debut", "total_views": 10},
            {"story_id": "related", "page_title": "Rajasthan cricket franchise latest", "total_views": 20},
            {"story_id": "other", "page_title": "Unrelated cricketer scores century", "total_views": 30},
        ])
        results = build_source_related_results(title_summary=titles, story_months=pd.DataFrame(),
            excluded_story_ids={"direct"}, routes=self.client.build_profile("Q1").routes)
        self.assertEqual(set(results.story_id), {"related"})
        self.assertEqual(results.iloc[0].evidence_source, "Wikidata")

    def test_api_error_is_not_silently_cached(self):
        client = WikidataLocalClient(self.settings, session=Mock())
        client.session.get.return_value.json.return_value = {"error": {"code": "maxlag"}}
        with self.assertRaises(ValueError):
            client._api(action="wbsearchentities")

    def test_rate_limit_produces_actionable_download_error(self):
        client = WikidataLocalClient(self.settings, session=Mock())
        client.session.get.return_value.status_code = 429
        with self.assertRaisesRegex(ValueError, "rate-limiting downloads"):
            client._api(action="wbsearchentities")


if __name__ == "__main__":
    unittest.main()
