import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.wikidata_source_related import (
    SourceSettings, WikidataLocalClient, find_wikidata_related_stories, two_hop_plan,
)
from tests.test_wikidata_source_related import entity, claim


class TwoHopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = WikidataLocalClient(SourceSettings(cache_path=str(Path(self.tmp.name) / "cache.db")), session=Mock())
        self.root = entity("Q1", "Example Player", {"P54": [claim("Q2")]})
        self.team = entity("Q2", "Example Team", {"P127": [claim("Q3")]})
        self.owner = entity("Q3", "Example Holdings", {"P112": [claim("Q4")]})
        self.data = {"Q1": self.root, "Q2": self.team, "Q3": self.owner}
        self.client._fetch_entities = Mock(side_effect=lambda ids: {qid: self.data[qid] for qid in ids})

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_downloads_two_hops_but_never_third(self):
        profile = self.client.build_profile("Q1", download_missing=True)
        self.assertEqual([(r.target_qid, r.hop_count) for r in profile.routes], [("Q2", 1), ("Q3", 2)])
        self.assertEqual(profile.routes[1].path_qids, ("Q1", "Q2", "Q3"))
        self.assertEqual(profile.routes[1].path_evidence_urls, (
            "https://www.wikidata.org/wiki/Q1#P54", "https://www.wikidata.org/wiki/Q2#P127"))
        self.assertIn("Example Team", profile.routes[1].relationship_path)
        self.assertNotIn("Q4", [qid for call in self.client._fetch_entities.call_args_list for qid in call.args[0]])
        self.client._fetch_entities.reset_mock()
        self.assertEqual(len(self.client.build_profile("Q1", download_missing=True).routes), 2)
        self.client._fetch_entities.assert_not_called()

    def test_old_one_hop_cache_upgrades_automatically(self):
        self.client.refresh("Q1")
        db = self.client._connect()
        with db:
            db.execute("DELETE FROM entities WHERE qid='Q3'")
            db.execute("DELETE FROM names WHERE qid='Q3'")
        db.close()
        self.client._fetch_entities.reset_mock()
        offline = self.client.build_profile("Q1")
        self.assertEqual(len(offline.routes), 1)
        self.assertTrue(offline.warnings)
        self.client._fetch_entities.assert_not_called()
        updated = self.client.build_profile("Q1", download_missing=True)
        self.assertEqual(len(updated.routes), 2)
        self.assertFalse(updated.warnings)

    def test_failed_second_hop_refresh_preserves_complete_previous_snapshot(self):
        self.client.refresh("Q1")
        self.root["labels"]["en"]["value"] = "Changed Player"
        def fetch(ids):
            if "Q3" in ids:
                raise requests.Timeout("second hop unavailable")
            return {qid: self.data[qid] for qid in ids}
        self.client._fetch_entities.side_effect = fetch
        profile = self.client.build_profile("Q1", refresh=True)
        self.assertEqual(profile.linked_entities[0].anchor, "Example Player")
        self.assertEqual(len(profile.routes), 2)
        self.assertTrue(any("previously downloaded" in warning for warning in profile.warnings))

    def test_cycles_and_duplicate_statements_do_not_create_routes(self):
        self.team["claims"] = {"P127": [claim("Q1"), claim("Q2"), claim("Q3"), claim("Q3")]}
        self.client.refresh("Q1")
        self.assertEqual([(r.target_qid, r.hop_count) for r in self.client.build_profile("Q1").routes], [("Q2", 1), ("Q3", 2)])

    def test_direct_endpoint_is_not_repeated_via_longer_path(self):
        self.root["claims"]["P108"] = [claim("Q3")]
        self.owner["claims"] = {}
        self.client.refresh("Q1")
        routes = self.client.build_profile("Q1").routes
        self.assertEqual({r.target_qid for r in routes}, {"Q2", "Q3"})
        self.assertTrue(all(r.hop_count == 1 for r in routes))

    def test_context_relations_cannot_be_used_at_either_hop(self):
        for first_prop, second_prop in (("P31", "P127"), ("P54", "P31"), ("P19", "P127")):
            with self.subTest(first_prop=first_prop, second_prop=second_prop):
                root = entity("Q1", "Example Player", {first_prop: [claim("Q2")]})
                middle = entity("Q2", "Example Team", {second_prop: [claim("Q3")]})
                self.assertEqual(two_hop_plan(root, {"Q2": middle})[0], [])

    def test_deprecated_second_edge_is_ignored(self):
        self.team["claims"]["P127"] = [claim("Q3", rank="deprecated")]
        self.client.refresh("Q1")
        self.assertEqual(len(self.client.build_profile("Q1").routes), 1)

    def test_each_edges_time_qualifiers_are_preserved(self):
        self.root["claims"]["P54"][0]["qualifiers"] = {"P582": [{"datavalue": {"value": {"time": "+2020-01-01T00:00:00Z"}}}]}
        self.team["claims"]["P127"][0]["qualifiers"] = {"P580": [{"datavalue": {"value": {"time": "+2010-01-01T00:00:00Z"}}}]}
        self.client.refresh("Q1")
        explanation = self.client.build_profile("Q1").routes[1].why_related
        self.assertIn("end: 2020-01-01", explanation)
        self.assertIn("start: 2010-01-01", explanation)
        self.assertIn("Example Player", explanation)
        self.assertIn("Example Team", explanation)

    def test_limits_are_deterministic_and_share_budget_across_intermediates(self):
        root = entity("Q1", "Root", {"P54": [claim("Q2"), claim("Q3")]})
        middles = {"Q2": entity("Q2", "First Team", {"P127": [claim(f"Q{i}") for i in (10, 11, 12)]}),
                   "Q3": entity("Q3", "Second Team", {"P127": [claim(f"Q{i}") for i in (20, 21, 22)]})}
        with patch("src.wikidata_source_related.MAX_SECOND_HOP_PER_ENTITY", 2), \
             patch("src.wikidata_source_related.MAX_SECOND_HOP_ENTITIES", 3):
            paths, limited = two_hop_plan(root, middles)
        self.assertEqual([p[4] for p in paths], ["Q10", "Q20", "Q11"])
        self.assertEqual(limited, 3)

    def test_second_hop_retrieval_preserves_exclusions_and_path_in_csv(self):
        self.client.refresh("Q1")
        profile = self.client.build_profile("Q1")
        profile.routes = [route for route in profile.routes if route.hop_count == 2]
        titles = pd.DataFrame([
            {"story_id": "first", "page_title": "Example Team update", "total_views": 10},
            {"story_id": "second", "page_title": "Example Team owner Example Holdings expands", "total_views": 20},
            {"story_id": "direct", "page_title": "Example Player joins Example Holdings", "total_views": 999},
            {"story_id": "excluded", "page_title": "Example Holdings report", "total_views": 999},
        ])
        # Use the root Q-ID so common synthetic 'Example' words are not excluded.
        result = find_wikidata_related_stories(query="Q1", profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test",
            primary_id_lookup=lambda **kwargs: ("excluded",))
        self.assertEqual(list(result.story_id), ["second"])
        self.assertEqual(list(result.hop_count), [2])
        self.assertEqual(result.total_views.sum(), 20)
        self.assertIn("Q2#P127", result.to_csv(index=False))
        self.assertIn("relationship_title_evidence", result.to_csv(index=False))

    def test_two_hop_alias_matches_still_reject_maruti_suzuki(self):
        from tests.test_wikidata_identity import fixture
        from dataclasses import replace
        from src.wikidata_identity import IdentityGate
        profile, route = fixture()
        route = replace(route, hop_count=2, path_qids=("Q1", "Q2", "Q188618"))
        gate = IdentityGate(profile)
        self.assertIsNone(gate(route, {"story_id": "car", "page_title": "Maruti Suzuki launches SUV"}))
        accepted = gate(route, {"story_id": "good", "page_title": "Maruti Hindu deity celebrations"})
        self.assertEqual(accepted["hop_count"], 2)

    def test_downloaded_shared_event_path_has_enforced_title_policy(self):
        self.root["labels"]["en"]["value"] = "Mukesh Ambani"
        self.root["claims"] = {"P1344": [claim("Q2")]}
        self.team["labels"]["en"]["value"] = "World Economic Forum Annual Meeting 2018"
        self.team["claims"] = {"P710": [claim("Q3")]}
        self.owner["labels"]["en"]["value"] = "Donald Trump"
        profile = self.client.build_profile("Q1", download_missing=True)
        profile.routes = [route for route in profile.routes if route.hop_count == 2]
        self.assertEqual(profile.routes[0].retrieval_policy.family, "shared_event")
        self.assertEqual(profile.routes[0].path_properties, ("P1344", "P710"))
        titles = pd.DataFrame([
            {"story_id": "bad", "page_title": "Donald Trump threatens Iran over negotiations", "total_views": 9999},
            {"story_id": "good", "page_title": "Donald Trump addresses World Economic Forum Annual Meeting 2018", "total_views": 10},
        ])
        result = find_wikidata_related_stories(query="Mukesh Ambani", profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test",
            primary_id_lookup=lambda **kw: ())
        self.assertEqual(list(result.story_id), ["good"])
        self.assertEqual(result.iloc[0].relationship_family, "shared_event")


if __name__ == "__main__":
    unittest.main()
