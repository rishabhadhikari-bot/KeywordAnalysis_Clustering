import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import requests

from src.wikidata_identity import IdentityGate, identity_tokens
from src.wikidata_relationship_policy import compile_policy
from src.wikidata_source_related import (
    SourceProfile, SourceSettings, WikidataRoute, WikidataLocalClient, find_wikidata_related_stories,
)


def item(qid, label, description="", aliases=()):
    return {"id": qid, "labels": {"en": {"value": label}},
            "aliases": {"en": [{"value": name} for name in aliases]},
            "descriptions": {"en": {"value": description}}, "claims": {}}


def fixture():
    hanuman = item("Q188618", "Hanuman", "Hindu deity and divine companion of Rama", ["Maruti", "Bajrang Bali"])
    company = item("Q963718", "Maruti Suzuki", "automobile manufacturer producing passenger vehicles", ["Maruti"])
    route = WikidataRoute("Hanuman", ("Maruti", "Bajrang Bali"), "Wikidata: main subject",
                          "Hanuman Chalisa has main subject Hanuman.", "Wikidata",
                          "https://www.wikidata.org/wiki/Q2663608#P921", 90, "wikidata_direct",
                          target_qid="Q188618", retrieval_policy=compile_policy(
                              (item("Q2663608", "Hanuman Chalisa"), hanuman), ("P921",)))
    profile = SourceProfile(query="Hanuman Chalisa", routes=[route],
                            identity_entities={e["id"]: e for e in [hanuman, company]})
    return profile, route


class IdentityGateTests(unittest.TestCase):
    def check_title(self, title, profile=None):
        default_profile, route = fixture()
        gate = IdentityGate(profile or default_profile)
        result = gate(route, {"story_id": "1", "page_title": title})
        return result, gate

    def test_maruti_suzuki_is_rejected_even_with_religious_context(self):
        result, gate = self.check_title("Maruti Suzuki supports Hindu deity temple festival")
        self.assertIsNone(result)
        self.assertIn("Competing longer name", gate.audit[0]["reason"])

    def test_uninformative_alias_is_withheld_even_if_lookup_found_no_collision(self):
        profile, _ = fixture()
        profile.identity_entities.pop("Q963718")
        profile.checked_identity_names.add("maruti")
        result, gate = self.check_title("Maruti announces major development", profile)
        self.assertIsNone(result)
        self.assertIn("Insufficient identity context", gate.audit[0]["reason"])

    def test_alias_with_two_independent_context_terms_is_accepted(self):
        result, _ = self.check_title("Maruti worship: Hindu deity celebrations")
        self.assertEqual(result["related_entity_qid"], "Q188618")
        self.assertEqual(result["matched_name"], "Maruti")
        self.assertIn("Supporting Wikidata context", result["identity_reason"])

    def test_one_context_word_is_not_enough(self):
        result, _ = self.check_title("Maruti deity celebration")
        self.assertIsNone(result)

    def test_other_entity_context_wins(self):
        result, gate = self.check_title("Maruti automobile manufacturer producing passenger vehicles")
        self.assertIsNone(result)
        self.assertIn("Context favors another entity", gate.audit[0]["reason"])

    def test_separate_valid_mention_is_not_blocked_by_longer_competitor(self):
        result, _ = self.check_title("Maruti Suzuki sponsors worship of Maruti Hindu deity")
        self.assertIsNotNone(result)

    def test_multiword_alias_requires_context_or_search_coverage(self):
        profile, _ = fixture()
        result, _ = self.check_title("Bajrang Bali celebrations", profile)
        self.assertIsNone(result)
        profile.checked_identity_names.add("bajrang bali")
        result, _ = self.check_title("Bajrang Bali celebrations", profile)
        self.assertIsNotNone(result)

    def test_known_multiword_name_collision_requires_context(self):
        profile, _ = fixture()
        profile.identity_entities["Q88"] = item("Q88", "Bajrang Bali", "Indian movie film")
        profile.checked_identity_names.add("bajrang bali")
        result, _ = self.check_title("Bajrang Bali celebrations", profile)
        self.assertIsNone(result)

    def test_noncontiguous_alias_words_are_not_an_entity_mention(self):
        profile, _ = fixture()
        profile.checked_identity_names.add("bajrang bali")
        result, _ = self.check_title("Bajrang travels to Bali", profile)
        self.assertIsNone(result)

    def test_hindi_alias_keeps_combining_marks_and_requires_context(self):
        profile, route = fixture()
        profile.identity_entities["Q188618"]["aliases"]["hi"] = [{"value": "मारुति"}]
        profile.identity_entities["Q188618"]["descriptions"]["hi"] = {"value": "हिन्दू देवता"}
        from dataclasses import replace
        route = replace(route, aliases=("मारुति",))
        self.assertEqual(identity_tokens("मारुति"), ("मारुति",))
        gate = IdentityGate(profile)
        result = gate(route, {"story_id": "1", "page_title": "मारुति हिन्दू देवता"})
        self.assertIsNotNone(result)

    def test_hindi_titles_pass_candidate_retrieval_and_identity_together(self):
        profile, route = fixture()
        from dataclasses import replace
        profile.routes = [replace(route, aliases=("मारुति",))]
        profile.identity_entities["Q188618"]["aliases"]["hi"] = [{"value": "मारुति"}]
        profile.identity_entities["Q188618"]["descriptions"]["hi"] = {"value": "हिन्दू देवता"}
        profile.identity_entities["Q963718"]["labels"]["hi"] = {"value": "मारुति सुजुकी"}
        titles = pd.DataFrame([
            {"story_id": "good", "page_title": "मारुति हिन्दू देवता", "total_views": 20},
            {"story_id": "bad", "page_title": "मारुति सुजुकी नई कार", "total_views": 999},
        ])
        result = find_wikidata_related_stories(query="Hanuman Chalisa", profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test", primary_id_lookup=lambda **kw: ())
        self.assertEqual(list(result.story_id), ["good"])

    def test_rejected_route_does_not_poison_independent_accepted_route(self):
        profile, route = fixture()
        company_route = WikidataRoute("Maruti Suzuki", (), "Wikidata: subsidiary", "Sample other relationship",
                                      "Wikidata", "https://www.wikidata.org/wiki/Q1", 90,
                                      "wikidata_direct", target_qid="Q963718", retrieval_policy=compile_policy(
                                          (item("Q1", "Sample Parent"), profile.identity_entities["Q963718"]), ("P355",)))
        profile.routes.append(company_route)
        titles = pd.DataFrame([{"story_id": "car", "page_title": "Maruti Suzuki launches SUV", "total_views": 9999}])
        result = find_wikidata_related_stories(query="sample query", profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test", primary_id_lookup=lambda **kw: ())
        self.assertEqual(list(result["related_entity_qid"]), ["Q963718"])
        self.assertEqual(result.iloc[0]["supporting_evidence_count"], 1)
        self.assertNotIn("Hanuman", result.iloc[0]["related_subject"])

    def test_rejected_high_traffic_stories_never_reach_export_or_totals(self):
        profile, _ = fixture()
        titles = pd.DataFrame([
            {"story_id": "car", "page_title": "Maruti Suzuki launches SUV", "total_views": 999999},
            {"story_id": "religion", "page_title": "Maruti Hindu deity celebrations", "total_views": 20},
            {"story_id": "unknown", "page_title": "Maruti development", "total_views": 999999},
        ])
        result = find_wikidata_related_stories(query="Hanuman Chalisa", profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test", primary_id_lookup=lambda **kw: ())
        self.assertEqual(list(result.story_id), ["religion"])
        self.assertEqual(result.total_views.sum(), 20)
        self.assertNotIn("Suzuki", result.to_csv(index=False))
        self.assertEqual(result.attrs["withheld_story_count"], 2)


class IdentityKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = WikidataLocalClient(SourceSettings(cache_path=str(Path(self.tmp.name) / "cache.sqlite3")), session=Mock())
        self.profile, _ = fixture()
        self.company = self.profile.identity_entities.pop("Q963718")
        self.client._api = Mock(return_value={"search": [{"id": "Q963718"}]})
        self.client._fetch_entities = Mock(return_value={"Q963718": self.company})

    def tearDown(self):
        self.tmp.cleanup()

    def test_competitor_discovery_and_cache_reuse(self):
        self.client.prepare_identity_catalog(self.profile, ["Maruti Suzuki launches SUV"])
        self.assertIn("Q963718", self.profile.identity_entities)
        self.assertIn("maruti", self.profile.checked_identity_names)
        self.assertEqual(self.client._api.call_args.kwargs["search"], "Maruti")
        self.client._api.reset_mock()
        self.client._fetch_entities.reset_mock()
        fresh, _ = fixture()
        fresh.identity_entities.pop("Q963718")
        self.client.prepare_identity_catalog(fresh, ["Maruti Suzuki launches SUV"])
        self.client._api.assert_not_called()
        self.client._fetch_entities.assert_not_called()
        self.assertIn("Q963718", fresh.identity_entities)
        self.assertIsNone(IdentityGate(fresh)(fresh.routes[0], {"story_id": "1", "page_title": "Maruti Suzuki launches SUV"}))

    def test_failed_search_is_not_cached_as_unique_and_can_retry(self):
        self.client._api.side_effect = requests.Timeout("offline")
        self.client.prepare_identity_catalog(self.profile, ["Maruti announces development"])
        self.assertNotIn("maruti", self.profile.checked_identity_names)
        self.assertTrue(self.profile.warnings)
        self.client._api.side_effect = None
        self.client.prepare_identity_catalog(self.profile, ["Maruti announces development"])
        self.assertIn("Q963718", self.profile.identity_entities)

    def test_failed_candidate_download_does_not_cache_success(self):
        self.client._fetch_entities.side_effect = requests.Timeout("offline")
        self.client.prepare_identity_catalog(self.profile, ["Maruti Suzuki launches SUV"])
        self.assertNotIn("maruti", self.profile.checked_identity_names)
        self.client._fetch_entities.side_effect = None
        self.client.prepare_identity_catalog(self.profile, ["Maruti Suzuki launches SUV"])
        self.assertEqual(self.client._api.call_count, 2)

    def test_no_candidates_does_not_call_network(self):
        self.client.prepare_identity_catalog(self.profile, [])
        self.client._api.assert_not_called()
        self.client._fetch_entities.assert_not_called()

    def test_unmatched_graph_endpoints_do_not_consume_identity_budget(self):
        from dataclasses import replace
        self.profile.routes.append(replace(self.profile.routes[0], target_qid="Q999",
                                           related_subject="Unmentioned University", aliases=()))
        self.profile.identity_entities["Q999"] = {"id": "Q999", "labels": {"en": {"value": "Unmentioned University"}}}
        self.client.prepare_identity_catalog(self.profile, ["Maruti Suzuki launches SUV"])
        fetched = {qid for call in self.client._fetch_entities.call_args_list for qid in call.args[0]}
        self.assertNotIn("Q999", fetched)

    def test_one_failed_name_lookup_does_not_skip_other_names(self):
        from dataclasses import replace
        self.profile.routes.append(replace(self.profile.routes[0], target_qid="Q999",
                                           related_subject="Zeta Entity", aliases=()))
        self.profile.identity_entities["Q999"] = item("Q999", "Zeta Entity")
        self.client._api.side_effect = [requests.Timeout("first lookup unavailable"), {"search": []}]
        self.client.prepare_identity_catalog(self.profile, ["Maruti update", "Zeta Entity news"])
        self.assertNotIn("maruti", self.profile.checked_identity_names)
        self.assertIn("zeta entity", self.profile.checked_identity_names)
        self.assertTrue(self.profile.warnings)

    def test_expired_search_and_metadata_are_refreshed(self):
        self.client.prepare_identity_catalog(self.profile, ["Maruti Suzuki launches SUV"])
        db = self.client._connect()
        with db:
            db.execute("UPDATE identity_searches SET fetched_at='2020-01-01T00:00:00+00:00'")
            db.execute("UPDATE identity_entities SET fetched_at='2020-01-01T00:00:00+00:00'")
        db.close()
        self.client._api.reset_mock()
        self.client._fetch_entities.reset_mock()
        fresh, _ = fixture()
        fresh.identity_entities.pop("Q963718")
        self.client.prepare_identity_catalog(fresh, ["Maruti Suzuki launches SUV"])
        self.client._api.assert_called_once()
        self.client._fetch_entities.assert_called_once_with(["Q963718"])

    def test_request_count_is_bounded_and_unchecked_names_stay_unchecked(self):
        from dataclasses import replace
        aliases = tuple(f"aliasnumber{i}" for i in range(15))
        self.profile.routes = [replace(self.profile.routes[0], aliases=aliases)]
        self.client.prepare_identity_catalog(self.profile, [" ".join(aliases)])
        self.assertEqual(self.client._api.call_count, 12)
        self.assertEqual(len(self.profile.checked_identity_names), 12)
        self.assertTrue(any("limit reached" in warning for warning in self.profile.warnings))


if __name__ == "__main__":
    unittest.main()
