"""Synthetic graph fixtures for namesake discovery, isolation and aggregation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd
import requests

from src.wikidata_source_related import SourceRelatedError, SourceSettings, WikidataLocalClient, find_wikidata_related_stories
from tests.test_wikidata_source_related import claim, entity


class MultiEntityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = WikidataLocalClient(SourceSettings(cache_path=str(Path(self.tmp.name) / "cache.db")), session=Mock())
        # The names reproduce the collision; the relationships below are test
        # fixtures, not statements about any actual temple or organization.
        self.data = {
            "Q1": entity("Q1", "Ram Mandir", {"P361": [claim("Q11"), claim("Q13")]}),
            "Q2": entity("Q2", "Ram Mandir", {"P361": [claim("Q12"), claim("Q13")]}),
            "Q3": entity("Q3", "Ram Mandir"),
            "Q4": entity("Q4", "Ram Mandir railway station"),
            "Q11": entity("Q11", "Ayodhya Heritage Trust"),
            "Q12": entity("Q12", "Bhubaneswar Cultural Centre"),
            "Q13": entity("Q13", "Shared Cultural Network"),
        }
        self.data["Q1"]["descriptions"] = {"en": {"value": "temple in Ayodhya"}}
        self.data["Q2"]["descriptions"] = {"en": {"value": "temple in Bhubaneswar"}}
        self.client._api = Mock(return_value={"search": [{"id": qid} for qid in ("Q1", "Q2", "Q3", "Q4")]})
        self.client._fetch_entities = Mock(side_effect=lambda ids: {qid: self.data[qid] for qid in ids})

    def tearDown(self):
        self.tmp.cleanup()

    def retrieve(self, profile, rows, excluded=()):
        return find_wikidata_related_stories(query="ram mandir", profile=profile,
            title_summary=pd.DataFrame([{"story_id": key, "page_title": title, "total_views": views}
                                        for key, title, views in rows]),
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test",
            primary_id_lookup=lambda **kw: excluded)

    def test_all_exact_namesakes_are_downloaded_not_all_raw_search_hits(self):
        profile = self.client.build_profile("ram mandir", download_missing=True)
        self.assertEqual({item["qid"] for item in profile.query_entities}, {"Q1", "Q2", "Q3"})
        self.assertEqual({route.query_entity_qid for route in profile.routes}, {"Q1", "Q2"})
        self.assertNotIn("Q4", profile.identity_entities)
        self.assertIn("3 matching entities", profile.source_status["Query resolution"])
        self.assertTrue(any("no supported discovery relationships" in warning for warning in profile.warnings))

    def test_complete_resolution_reuses_cache_without_network(self):
        first = self.client.build_profile("ram mandir", download_missing=True)
        self.client._api.reset_mock()
        self.client._fetch_entities.reset_mock()
        second = self.client.build_profile("RAM MANDIR", download_missing=True)
        self.assertEqual(first.routes, second.routes)
        self.assertEqual(first.query_entities, second.query_entities)
        self.client._api.assert_not_called()
        self.client._fetch_entities.assert_not_called()
        with self.client._connect() as db:
            row = db.execute("SELECT qids FROM query_resolutions WHERE name='ram mandir'").fetchone()
        db.close()
        self.assertEqual(json.loads(row[0]), ["Q1", "Q2", "Q3"])

    def test_results_union_preserves_origins_and_counts_shared_story_once(self):
        profile = self.client.build_profile("ram mandir", download_missing=True)
        result = self.retrieve(profile, [
            ("ayodhya", "Ayodhya Heritage Trust announces exhibition", 10),
            ("bhubaneswar", "Bhubaneswar Cultural Centre opens museum", 20),
            ("shared", "Shared Cultural Network launches programme", 30),
            ("direct", "Ram Mandir joins Shared Cultural Network", 999),
            ("primary", "Ayodhya Heritage Trust announces plans", 999),
            ("copy", "AYODHYA HERITAGE TRUST ANNOUNCES PLANS!", 999),
        ], excluded=("primary",))
        self.assertEqual(set(result.story_id), {"ayodhya", "bhubaneswar", "shared"})
        self.assertEqual(result.total_views.sum(), 60)
        shared = result.set_index("story_id").loc["shared"]
        details = json.loads(shared.supporting_relationships)
        self.assertEqual({item["query_entity"]["qid"] for item in details}, {"Q1", "Q2"})
        self.assertEqual(shared.supporting_evidence_count, 2)
        self.assertIn("temple in Ayodhya", shared.matched_query_entities)
        self.assertIn("temple in Bhubaneswar", shared.matched_query_entities)
        self.assertEqual(result.attrs["excluded_count"], 3)
        self.assertNotIn("joins Shared", result.to_csv(index=False))

    def test_legacy_single_root_cache_upgrades_on_first_online_name_search(self):
        self.client.refresh("Q1")
        with self.client._connect() as db:
            db.execute("INSERT OR REPLACE INTO profiles VALUES ('ram mandir', 'Q1', '2020-01-01')")
        db.close()
        self.client._api.reset_mock()
        profile = self.client.build_profile("ram mandir", download_missing=True)
        self.assertEqual(len(profile.query_entities), 3)
        self.client._api.assert_called_once()

    def test_one_persisted_resolution_cannot_hide_another_known_exact_name(self):
        self.client._api.return_value = {"search": [{"id": "Q1"}]}
        self.client.build_profile("ram mandir", download_missing=True)
        self.client.refresh("Q2")
        self.client._api.reset_mock()
        profile = self.client.build_profile("ram mandir")
        self.assertEqual({item["qid"] for item in profile.query_entities}, {"Q1", "Q2"})
        self.client._api.assert_not_called()

    def test_explicit_internal_id_still_selects_only_that_root(self):
        self.client.build_profile("ram mandir", download_missing=True)
        profile = self.client.build_profile("Q2")
        self.assertEqual([item["qid"] for item in profile.query_entities], ["Q2"])

    def test_shared_alias_with_distinct_canonical_names_resolves_all(self):
        for qid, city in (("Q1", "Ayodhya"), ("Q2", "Bhubaneswar")):
            self.data[qid]["labels"]["en"]["value"] = f"Temple in {city}"
            self.data[qid]["aliases"]["en"] = [{"value": "Ram Mandir"}]
        profile = self.client.build_profile("ram mandir", download_missing=True)
        self.assertEqual(len(profile.query_entities), 3)
        self.assertIn("Temple in Ayodhya", profile.query_aliases)
        self.assertIn("Temple in Bhubaneswar", profile.query_aliases)

    def test_failed_group_refresh_preserves_all_previous_facts_and_resolution(self):
        self.client.build_profile("ram mandir", download_missing=True)
        self.data["Q1"]["descriptions"]["en"]["value"] = "changed description"
        self.data["Q2"]["claims"]["P361"].append(claim("Q99"))
        def fetch(ids):
            if "Q99" in ids:
                raise requests.Timeout("one branch unavailable")
            return {qid: self.data[qid] for qid in ids}
        self.client._fetch_entities.side_effect = fetch
        profile = self.client.build_profile("ram mandir", refresh=True)
        self.assertEqual(len(profile.query_entities), 3)
        self.assertEqual(profile.query_entities[0]["description"], "temple in Ayodhya")
        self.assertTrue(any("previously downloaded" in warning for warning in profile.warnings))
        self.assertFalse(any(route.target_qid == "Q99" for route in profile.routes))

    def test_two_hop_paths_never_cross_between_namesake_roots(self):
        self.data["Q1"]["claims"] = {"P355": [claim("Q11")]}
        self.data["Q2"]["claims"] = {"P355": [claim("Q12")]}
        self.data["Q11"]["claims"] = {"P355": [claim("Q13")]}
        self.data["Q12"]["claims"] = {"P355": [claim("Q14")]}
        self.data["Q14"] = entity("Q14", "Regional Culture Network")
        profile = self.client.build_profile("ram mandir", download_missing=True)
        paths = {route.path_qids for route in profile.routes if route.hop_count == 2}
        self.assertEqual(paths, {("Q1", "Q11", "Q13"), ("Q2", "Q12", "Q14")})

    def test_wrong_event_branch_is_not_included_in_shared_article_provenance(self):
        self.data["Q1"]["claims"] = {"P1344": [claim("Q11")]}
        self.data["Q2"]["claims"] = {"P1344": [claim("Q12")]}
        self.data["Q11"] = entity("Q11", "Culture Forum 2018", {"P710": [claim("Q13")]})
        self.data["Q12"] = entity("Q12", "Culture Forum 2026", {"P710": [claim("Q13")]})
        self.data["Q13"] = entity("Q13", "Example Speaker")
        profile = self.client.build_profile("ram mandir", download_missing=True)
        profile.routes = [route for route in profile.routes if route.hop_count == 2]
        result = self.retrieve(profile, [("good", "Example Speaker attends Culture Forum 2026", 10)])
        self.assertEqual(list(result.story_id), ["good"])
        details = json.loads(result.iloc[0].supporting_relationships)
        self.assertEqual({item["query_entity"]["qid"] for item in details}, {"Q2"})

    def test_zero_matching_names_still_produces_clear_error(self):
        self.client._api.return_value = {"search": [{"id": "Q4"}]}
        with self.assertRaisesRegex(SourceRelatedError, "No matching name or alias"):
            self.client.build_profile("ram mandir", download_missing=True)


if __name__ == "__main__":
    unittest.main()
