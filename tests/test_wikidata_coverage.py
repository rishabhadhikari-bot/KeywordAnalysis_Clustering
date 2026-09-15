"""Regressions for missing business/family coverage and sparse concept data."""
import unittest
from unittest.mock import Mock

import pandas as pd

from tests import test_wikidata_source_related as local_tests
from tests.test_wikidata_source_related import entity, claim
from tests.test_wikidata_identity import item
from src.wikidata_identity import IdentityGate
from src.wikidata_relationship_policy import compile_policy
from src.wikidata_source_related import (
    SourceProfile, WikidataRoute, find_wikidata_related_stories, minor_name_correction,
)


def route(qid, label):
    return WikidataRoute(label, (), "Wikidata: child", "Recorded relationship", "Wikidata",
                         "https://www.wikidata.org/wiki/Q1#P40", 90, "wikidata_direct", target_qid=qid,
                         retrieval_policy=compile_policy((item("Q1", "Query Person"), item(qid, label)), ("P40",)))


class CoverageTests(unittest.TestCase):
    def test_shared_surname_is_allowed_only_inside_complete_related_name(self):
        profile = SourceProfile(query="Mukesh Ambani", query_aliases=["Mukesh Ambani"],
            routes=[route("Q2", "Nita Ambani"), route("Q3", "Anant Ambani")],
            identity_entities={"Q2": item("Q2", "Nita Ambani"), "Q3": item("Q3", "Anant Ambani")})
        titles = pd.DataFrame([
            {"story_id": key, "page_title": title, "total_views": 1}
            for key, title in [
                ("nita", "Nita Ambani inaugurates cultural centre"),
                ("anant", "Anant Ambani celebrates birthday"),
                ("primary", "Nita Ambani attends event"),
                ("direct", "Mukesh Ambani and Nita Ambani attend event"),
                ("partial", "Mukesh accompanies Nita Ambani"),
                ("surname", "Ambani attends event"),
            ]])
        result = find_wikidata_related_stories(query=profile.query, profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="All keywords", index_fingerprint="test",
            primary_id_lookup=lambda **kw: {"primary"})
        self.assertEqual(set(result.story_id), {"nita", "anant"})
        self.assertEqual(result.attrs["primary_excluded_count"], 1)
        self.assertEqual(result.attrs["lexical_excluded_count"], 3)

    def test_unique_checked_short_label_can_use_one_product_context_term(self):
        jio = item("Q2", "Jio", "telecommunications network in India")
        jio["claims"] = {"P1056": [claim("Q3")]}
        profile = SourceProfile(query="Company owner", routes=[route("Q2", "Jio")],
            identity_entities={"Q2": jio, "Q3": item("Q3", "broadband")})
        row = {"story_id": "1", "page_title": "Jio launches broadband plan"}
        self.assertIsNone(IdentityGate(profile)(profile.routes[0], row))
        profile.checked_identity_names.add("jio")
        self.assertIsNotNone(IdentityGate(profile)(profile.routes[0], row))
        self.assertIsNone(IdentityGate(profile)(profile.routes[0], dict(row, page_title="Jio launches plan")))
        profile.identity_entities["Q4"] = item("Q4", "Jio", "film")
        self.assertIsNone(IdentityGate(profile)(profile.routes[0], row))

    def test_longer_competing_company_name_still_blocks_short_label(self):
        profile = SourceProfile(query="Company owner", routes=[route("Q2", "Jio")],
            checked_identity_names={"jio"}, identity_entities={
                "Q2": item("Q2", "Jio", "telecommunications network"),
                "Q3": item("Q3", "Jio Financial Services", "financial services company")})
        self.assertIsNone(IdentityGate(profile)(profile.routes[0],
            {"story_id": "1", "page_title": "Jio Financial Services funds telecommunications network"}))

    def test_legal_suffix_omission_uses_complete_company_identity(self):
        from src.wikidata_source_related import entity_names
        company = item("Q2", "Reliance Industries Limited")
        related_route = route("Q2", "Reliance Industries Limited")
        from dataclasses import replace
        related_route = replace(related_route, aliases=tuple(entity_names(company)))
        profile = SourceProfile(query="Company owner", routes=[related_route], identity_entities={"Q2": company})
        self.assertIn("Reliance Industries", entity_names(company))
        self.assertNotIn("Reliance", entity_names(company))
        result = IdentityGate(profile)(related_route, {"story_id": "1", "page_title": "Reliance Industries posts results"})
        self.assertEqual(result['related_entity_qid'], "Q2")
        profile.identity_entities["Q3"] = item("Q3", "Reliance Industries Foundation")
        self.assertIsNone(IdentityGate(profile)(related_route,
            {"story_id": "1", "page_title": "Reliance Industries Foundation opens centre"}))

    def test_minor_correction_is_bounded_to_one_edit_in_a_complete_name(self):
        for typo in ("Mukesh Amabani", "Mukesh Ambnai", "Mukesh Amban"):
            self.assertTrue(minor_name_correction(typo, "Mukesh Ambani"))
        for query in ("Mukesh", "Mukesh Ambani election", "Mokesh Amabani", "Reliance Jio", "Nita Ambani"):
            self.assertFalse(minor_name_correction(query, "Mukesh Ambani"))


# Reuse the isolated SQLite fixture without inheriting its test methods.
class CoverageCacheTests(unittest.TestCase):
    setUp = local_tests.WikidataLocalTests.setUp
    tearDown = local_tests.WikidataLocalTests.tearDown

    def test_ownership_affiliation_and_company_leadership_are_traversed(self):
        self.root["claims"] = {"P1830": [claim("Q2")], "P1416": [claim("Q2")]}
        self.team = self.data["Q2"] = entity("Q2", "Example Industries", {"P355": [claim("Q3")]})
        self.data["Q3"] = entity("Q3", "Example Telecom")
        profile = self.client.build_profile("Q1", download_missing=True)
        self.assertEqual({r.related_subject for r in profile.routes}, {"Example Industries", "Example Telecom"})
        self.assertTrue(any(r.hop_count == 2 and "owner of" in r.relationship_path for r in profile.routes))

    def test_old_cache_downloads_newly_supported_ownership_target(self):
        self.client.refresh("Q1")
        self.root["claims"]["P1830"] = [claim("Q3")]
        self.data["Q3"] = entity("Q3", "Example Telecom")
        import json
        with self.client._connect() as db:
            db.execute("UPDATE entities SET payload=? WHERE qid='Q1'", (json.dumps(self.root),))
        db.close()
        self.assertTrue(any(r.target_qid == "Q3" for r in self.client.build_profile("Q1", download_missing=True).routes))

    def test_unique_typo_resolves_and_reuses_cache_with_visible_correction(self):
        self.root["labels"]["en"]["value"] = "Mukesh Ambani"
        profile = self.client.build_profile("Mukesh Amabani", download_missing=True)
        self.assertEqual(profile.linked_entities[0].anchor, "Mukesh Ambani")
        self.assertTrue(any("spelling correction" in warning for warning in profile.warnings))
        self.client._api.reset_mock()
        self.client.build_profile("Mukesh Amabani", download_missing=True)
        self.client._api.assert_not_called()

    def test_multiple_bounded_corrections_are_all_included(self):
        self.root["labels"]["en"]["value"] = "Mukesh Ambani"
        self.data["Q3"] = entity("Q3", "Mukesh Amabani", aliases=["Mukesh Amaboni"])
        self.client._api = Mock(return_value={"search": [{"id": "Q1"}, {"id": "Q3"}]})
        profile = self.client.build_profile("Mukesh Amabni", download_missing=True)
        self.assertEqual({item.uri.rsplit('/', 1)[-1] for item in profile.linked_entities}, {"Q1", "Q3"})

    def test_typo_candidate_fallback_still_requires_complete_name(self):
        self.root["labels"]["en"]["value"] = "Mukesh Ambani"
        self.client._api.side_effect = [
            {"search": []}, {"search": [{"id": "Q1"}]}, {"search": []},
        ]
        profile = self.client.build_profile("Mukesh Amabani", download_missing=True)
        self.assertEqual(profile.linked_entities[0].anchor, "Mukesh Ambani")
        self.assertEqual([call.kwargs['search'] for call in self.client._api.call_args_list],
                         ["Mukesh Amabani", "mukesh", "amabani"])

    def test_previously_saved_subject_resolves_new_minor_typo_offline(self):
        self.root["labels"]["en"]["value"] = "Mukesh Ambani"
        self.client.refresh("Q1")
        self.client._api.reset_mock()
        profile = self.client.build_profile("Mukesh Amabani")
        self.assertEqual(profile.linked_entities[0].anchor, "Mukesh Ambani")
        self.client._api.assert_not_called()

    def test_sparse_classification_only_concept_reports_knowledge_gap(self):
        self.root["claims"] = {"P31": [claim("Q2")]}
        profile = self.client.build_profile("Q1", download_missing=True)
        self.assertFalse(any(r.source_kind == "wikidata_direct" for r in profile.routes))
        self.assertTrue(any("no supported discovery relationships" in warning for warning in profile.warnings))


if __name__ == "__main__":
    unittest.main()
