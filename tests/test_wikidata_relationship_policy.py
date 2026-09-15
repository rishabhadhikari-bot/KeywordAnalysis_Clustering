"""Article-level regressions: graph reachability must not imply relevance."""
from dataclasses import replace
import unittest
from unittest.mock import Mock

import pandas as pd

from src.wikidata_relationship_policy import RelationshipGate, RetrievalPolicy, compile_policy
from src.wikidata_source_related import SourceProfile, WikidataRoute, find_wikidata_related_stories, two_hop_plan
from tests.test_wikidata_source_related import claim, entity


def make_route(nodes, props, claims=None):
    claims = claims or tuple(claim(node["id"]) for node in nodes[1:])
    endpoint = nodes[-1]
    return WikidataRoute(
        endpoint["labels"]["en"]["value"], tuple(a["value"] for a in endpoint.get("aliases", {}).get("en", [])),
        "Wikidata test path", "Recorded path", "Wikidata", "https://www.wikidata.org/wiki/" + nodes[0]["id"],
        80, "wikidata_direct", target_qid=endpoint["id"], hop_count=len(props),
        path_qids=tuple(n["id"] for n in nodes), path_properties=props,
        relationship_path=" -> ".join(n["labels"]["en"]["value"] for n in nodes),
        retrieval_policy=compile_policy(nodes, props, claims))


def event_fixture():
    root = entity("Q1", "Mukesh Ambani", {"P1344": [claim("Q2")]})
    event = entity("Q2", "World Economic Forum Annual Meeting 2018", {"P710": [claim("Q3")]},
                   aliases=("WEF", "Davos", "World Economic Forum Annual Meeting"))
    person = entity("Q3", "Donald Trump")
    route = make_route((root, event, person), ("P1344", "P710"))
    return SourceProfile(query="Mukesh Ambani", query_aliases=["Mukesh Ambani"], routes=[route],
                         identity_entities={n["id"]: n for n in (root, event, person)}), route


def retrieve(profile, rows, excluded=()):
    titles = pd.DataFrame([{"story_id": key, "page_title": title, "total_views": views} for key, title, views in rows])
    return find_wikidata_related_stories(query=profile.query, profile=profile, title_summary=titles,
        story_months=pd.DataFrame(), match_mode="all", index_fingerprint="scope-test",
        primary_id_lookup=lambda **kw: excluded)


class SharedEventTests(unittest.TestCase):
    def test_rejected_context_does_not_trigger_identity_network_lookups(self):
        profile, _ = event_fixture()
        client = Mock()
        titles = pd.DataFrame([{"story_id": "bad", "page_title": "Donald Trump threatens Iran", "total_views": 99}])
        results = find_wikidata_related_stories(query=profile.query, profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test",
            primary_id_lookup=lambda **kw: (), identity_client=client)
        client.prepare_identity_catalog.assert_not_called()
        self.assertTrue(results.empty)
        self.assertEqual(results.attrs["candidate_count"], 1)

    def test_newly_discovered_scope_collision_is_rechecked_after_metadata_lookup(self):
        profile, _ = event_fixture()
        client = Mock()
        def add_collision(*args, **kwargs):
            profile.identity_entities["Q4"] = entity("Q4", "WEF")
        client.prepare_identity_catalog.side_effect = add_collision
        titles = pd.DataFrame([{"story_id": "ambiguous", "page_title": "Donald Trump addresses WEF 2018", "total_views": 99}])
        results = find_wikidata_related_stories(query=profile.query, profile=profile, title_summary=titles,
            story_months=pd.DataFrame(), match_mode="all", index_fingerprint="test",
            primary_id_lookup=lambda **kw: (), identity_client=client)
        client.prepare_identity_catalog.assert_called_once()
        self.assertTrue(results.empty)
        self.assertEqual(results.attrs["relationship_withheld_story_count"], 1)

    def test_unrelated_endpoint_and_wrong_edition_never_enter_results_or_totals(self):
        profile, _ = event_fixture()
        results = retrieve(profile, [
            ("noise", "Donald Trump threatens Iran over negotiations", 999999),
            ("wrong_year", "Donald Trump addresses World Economic Forum Annual Meeting 2026", 999999),
            ("undated", "Donald Trump addresses World Economic Forum Annual Meeting", 999999),
            ("good", "Donald Trump addresses World Economic Forum Annual Meeting 2018", 25),
            ("alias", "Donald Trump addresses Davos 2018", 10),
        ])
        self.assertEqual(set(results.story_id), {"good", "alias"})
        self.assertEqual(results.total_views.sum(), 35)
        self.assertTrue((results.relationship_family == "shared_event").all())
        self.assertFalse(results.can_retrieve_standalone.any())
        self.assertTrue(results.relationship_title_evidence.str.contains("2018").all())
        self.assertNotIn("Iran", results.to_csv(index=False))
        self.assertEqual(results.attrs["relationship_withheld_story_count"], 3)
        self.assertEqual(results.attrs["identity_withheld_story_count"], 0)

    def test_year_elsewhere_and_noncontiguous_event_name_are_insufficient(self):
        profile, _ = event_fixture()
        results = retrieve(profile, [
            ("distant", "Donald Trump recalls 2018 trade policy at WEF 2026", 1),
            ("split", "Donald Trump addresses World leaders at Economic Forum Annual Meeting 2018", 1),
            ("year_only", "Donald Trump outlines 2018 policy", 1),
            ("conflicting", "Donald Trump discusses 2018 WEF 2026", 1),
        ])
        self.assertTrue(results.empty)

    def test_direct_ids_full_aliases_and_duplicate_direct_titles_stay_excluded(self):
        profile, _ = event_fixture()
        results = retrieve(profile, [
            ("direct", "Mukesh Ambani and Donald Trump attend WEF 2018", 100),
            ("primary", "Donald Trump addresses WEF 2018", 100),
            ("copy", "DONALD TRUMP ADDRESSES WEF 2018!", 100),
            ("good", "Donald Trump speaks at Davos 2018", 5),
        ], excluded=("primary",))
        self.assertEqual(list(results.story_id), ["good"])
        self.assertEqual(results.attrs["excluded_count"], 3)

    def test_context_pass_does_not_override_wrong_person_identity(self):
        profile, _ = event_fixture()
        profile.identity_entities["Q4"] = entity("Q4", "Donald Trump Junior")
        results = retrieve(profile, [("bad", "Donald Trump Junior attends WEF 2018", 99)])
        self.assertTrue(results.empty)
        self.assertEqual(results.attrs["relationship_withheld_story_count"], 0)
        self.assertEqual(results.attrs["identity_withheld_story_count"], 1)

    def test_rejected_path_does_not_poison_independent_accepted_path(self):
        profile, route = event_fixture()
        root, person = profile.identity_entities["Q1"], profile.identity_entities["Q3"]
        # A separate hypothetical supported immediate-family route proves
        # per-route rejection, not a factual claim about these real people.
        profile.routes.append(make_route((root, person), ("P40",)))
        results = retrieve(profile, [("independent", "Donald Trump discusses policy", 10)])
        self.assertEqual(list(results.story_id), ["independent"])
        self.assertEqual(results.iloc[0].supporting_evidence_count, 1)
        self.assertEqual(results.iloc[0].relationship_family, "immediate_family")
        self.assertEqual(results.attrs["relationship_withheld_story_count"], 0)
        self.assertEqual(len(results.attrs["relationship_audit"]), 1)

    def test_event_route_itself_also_requires_correct_edition(self):
        profile, _ = event_fixture()
        profile.routes = [make_route((profile.identity_entities["Q1"], profile.identity_entities["Q2"]), ("P1344",))]
        profile.checked_identity_names.update({"world economic forum annual meeting"})
        results = retrieve(profile, [
            ("good", "World Economic Forum Annual Meeting 2018 highlights", 2),
            ("bad", "World Economic Forum Annual Meeting 2026 highlights", 99),
        ])
        self.assertEqual(list(results.story_id), ["good"])

    def test_event_date_from_claim_can_bind_an_undated_alias(self):
        profile, _ = event_fixture()
        event = profile.identity_entities["Q2"]
        event["labels"]["en"]["value"] = "World Economic Forum Annual Meeting"
        event["claims"]["P585"] = [{"mainsnak": {"datavalue": {"value": {
            "time": "+2018-01-23T00:00:00Z", "precision": 11}}}}]
        profile.routes = [make_route(tuple(profile.identity_entities.values()), ("P1344", "P710"))]
        results = retrieve(profile, [("good", "Donald Trump addresses WEF 2018", 2),
                                     ("bad", "Donald Trump addresses WEF 2026", 99)])
        self.assertEqual(list(results.story_id), ["good"])

    def test_event_with_missing_edition_evidence_is_withheld(self):
        profile, _ = event_fixture()
        profile.identity_entities["Q2"]["labels"]["en"]["value"] = "World Economic Forum Annual Meeting"
        profile.routes = [make_route(tuple(profile.identity_entities.values()), ("P1344", "P710"))]
        results = retrieve(profile, [("unknown", "Donald Trump addresses WEF 2026", 99)])
        self.assertTrue(results.empty)

    def test_scope_alias_collision_is_withheld(self):
        profile, _ = event_fixture()
        profile.identity_entities["Q4"] = entity("Q4", "Davos", aliases=("Davos Regional Forum",))
        results = retrieve(profile, [("ambiguous", "Donald Trump addresses Davos Regional Forum 2018", 99)])
        self.assertTrue(results.empty)

    def test_results_are_repeatable_for_same_facts_and_titles(self):
        profile, _ = event_fixture()
        rows = [("good", "Donald Trump addresses WEF 2018", 1), ("bad", "Donald Trump addresses Iran", 99)]
        first, second = retrieve(profile, rows), retrieve(profile, rows)
        self.assertEqual(first.to_csv(index=False), second.to_csv(index=False))
        self.assertEqual(first.attrs, second.attrs)


class OtherRelationshipTests(unittest.TestCase):
    def test_institutional_bridge_requires_specific_directional_expression(self):
        player = entity("Q10", "Arun Shah")
        team = entity("Q11", "Coastal Club")
        owner = entity("Q12", "Delta Holdings")
        route = make_route((player, team, owner), ("P54", "P127"))
        profile = SourceProfile(query="Arun Shah", routes=[route], identity_entities={n["id"]: n for n in (player, team, owner)})
        results = retrieve(profile, [
            ("good", "Coastal Club owner Delta Holdings announces investment", 10),
            ("reverse", "Delta Holdings owner Coastal Club announces investment", 99),
            ("cooccur", "Coastal Club and Delta Holdings in the news", 99),
            ("endpoint", "Delta Holdings launches housing project", 99),
        ])
        self.assertEqual(list(results.story_id), ["good"])
        self.assertIn("owner", results.iloc[0].relationship_title_evidence)

    def test_former_affiliation_cannot_retrieve_arbitrary_current_team_news(self):
        person, team = entity("Q10", "Arun Shah"), entity("Q11", "Coastal Club")
        past = claim("Q11", qualifiers={"P582": [{"datavalue": {"value": {"time": "+2020-01-01T00:00:00Z"}}}]})
        route = make_route((person, team), ("P54",), (past,))
        profile = SourceProfile(query="Arun Shah", routes=[route], identity_entities={"Q10": person, "Q11": team})
        self.assertFalse(route.retrieval_policy.can_retrieve_standalone)
        self.assertTrue(retrieve(profile, [("bad", "Coastal Club announces 2026 captain", 99)]).empty)

    def test_work_specific_path_requires_connecting_work(self):
        film, book, writer = entity("Q10", "Harbor Movie"), entity("Q11", "Silent Coast"), entity("Q12", "Mira Sen")
        route = make_route((film, book, writer), ("P144", "P50"))
        profile = SourceProfile(query="Harbor Movie", routes=[route], identity_entities={n["id"]: n for n in (film, book, writer)})
        results = retrieve(profile, [("good", "Mira Sen discusses Silent Coast", 5), ("bad", "Mira Sen buys new home", 99)])
        self.assertEqual(list(results.story_id), ["good"])

    def test_corporate_chain_is_explicitly_permitted_standalone(self):
        nodes = tuple(entity(f"Q{i}", label) for i, label in enumerate(("Alpha Group", "Beta Industries", "Gamma Telecom"), 10))
        route = make_route(nodes, ("P355", "P355"))
        profile = SourceProfile(query="Alpha Group", routes=[route], identity_entities={n["id"]: n for n in nodes})
        results = retrieve(profile, [("good", "Gamma Telecom launches broadband", 5)])
        self.assertEqual(list(results.story_id), ["good"])
        self.assertTrue(results.iloc[0].can_retrieve_standalone)

    def test_unapproved_property_pair_is_not_discovered(self):
        root = entity("Q10", "Sample Person", {"P69": [claim("Q11")]})
        university = entity("Q11", "Sample University", {"P112": [claim("Q12")]})
        self.assertEqual(two_hop_plan(root, {"Q11": university})[0], [])

    def test_all_supported_discovery_properties_have_a_family_policy(self):
        from src.wikidata_source_related import PROPERTY_LABELS
        from src.wikidata_relationship_policy import PROPERTY_FAMILY
        self.assertEqual(set(PROPERTY_LABELS), set(PROPERTY_FAMILY))

    def test_legacy_or_unconfigured_route_does_not_bypass_policy(self):
        profile, route = event_fixture()
        profile.routes = [replace(route, retrieval_policy=RetrievalPolicy())]
        results = retrieve(profile, [("bad", "Donald Trump threatens Iran", 99)])
        self.assertTrue(results.empty)
        self.assertIn("No permitted", results.attrs["relationship_audit"][0]["reason"])

    def test_route_missing_new_dataclass_fields_is_withheld_without_hashing_it(self):
        profile, route = event_fixture()
        # Reproduce an object deserialized from the schema before these fields.
        object.__delattr__(route, "retrieval_policy")
        object.__delattr__(route, "path_properties")
        results = retrieve(profile, [("bad", "Donald Trump threatens Iran", 99)])
        self.assertTrue(results.empty)
        self.assertIn("No permitted", results.attrs["relationship_audit"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
