import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

import pandas as pd
import requests

from src.dbpedia_source_related import (
    DBO,
    DBPEDIA,
    EvidenceRoute,
    LinkedEntity,
    SourceSettings,
    DBpediaSpotlightClient,
    build_news_routes,
    build_source_related_results,
    join_news_headlines,
    parse_spotlight_json,
    parse_lookup_json,
)


class StubDBpediaClient(DBpediaSpotlightClient):
    def __init__(self) -> None:
        super().__init__(
            SourceSettings(
                spotlight_url="https://example.com/annotate",
                dbpedia_sparql_url="https://example.com/sparql",
                google_news_rss_url="https://example.com/news",
                timeout_seconds=5,
                user_agent="test",
            )
        )

    def lookup_entities(self, query):
        return []

    def _map_labels_to_dbpedia(self, labels):
        return {}

    def _load_dbpedia_facts(self, entity_uri):
        if entity_uri.endswith("Kajra_Re"):
            return [
                {
                    "predicate": f"{DBO}artist",
                    "related": f"{DBPEDIA}Alisha_Chinai",
                    "related_label": "Alisha Chinai",
                    "direction": "outgoing",
                },
                {
                    "predicate": f"{DBO}partOf",
                    "related": f"{DBPEDIA}Bunty_Aur_Babli",
                    "related_label": "Bunty Aur Babli",
                    "direction": "outgoing",
                },
                {
                    "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    "related": f"{DBPEDIA}Song",
                    "related_label": "song",
                    "direction": "outgoing",
                },
            ]
        if entity_uri.endswith("Bunty_Aur_Babli"):
            return [
                {
                    "predicate": f"{DBO}starring",
                    "related": f"{DBPEDIA}Aishwarya_Rai_Bachchan",
                    "related_label": "Aishwarya Rai Bachchan",
                    "direction": "outgoing",
                },
                {
                    "predicate": f"{DBO}starring",
                    "related": f"{DBPEDIA}Amitabh_Bachchan",
                    "related_label": "Amitabh Bachchan",
                    "direction": "outgoing",
                },
            ]
        return []


class DBpediaSourceRelatedTests(unittest.TestCase):
    def test_sparse_entity_uses_labelled_reciprocal_context_without_category_expansion(self) -> None:
        client = StubDBpediaClient()
        client._sparql = Mock(return_value=[
            {"related": {"value": DBPEDIA + uri}, "label": {"value": label}}
            for uri, label in (("Vishnu", "Vishnu"), ("Category:Hindu_festivals", "Hindu festivals"))
        ])
        routes = client._build_dbpedia_routes([LinkedEntity(DBPEDIA + "Ekadashi", "Ekadashi", 0.7)])
        self.assertEqual([r.related_subject for r in routes], ["Vishnu"])
        self.assertEqual(routes[0].source_kind, "dbpedia_context")
        self.assertIn("contextual association", routes[0].why_related)
        self.assertLess(routes[0].confidence, 70)
        query = client._sparql.call_args.args[0]
        self.assertIn("<" + DBPEDIA + "Ekadashi> dbo:wikiPageWikiLink ?related", query)
        self.assertIn("?related dbo:wikiPageWikiLink <" + DBPEDIA + "Ekadashi>", query)
        client._sparql.assert_called_once()
        results = build_source_related_results(
            title_summary=pd.DataFrame([
                {"story_id": "direct", "page_title": "Ekadashi Vishnu puja", "total_views": 30},
                {"story_id": "related", "page_title": "Vishnu avatars explained", "total_views": 20},
                {"story_id": "unrelated", "page_title": "Hindu festivals date timings", "total_views": 50},
            ]), story_months=pd.DataFrame(), excluded_story_ids={"direct"}, routes=routes,
        )
        self.assertEqual(set(results["story_id"]), {"related"})
        self.assertEqual(results.iloc[0]["evidence_source"], "DBpedia")

    def test_typed_facts_do_not_trigger_context_expansion(self) -> None:
        client = StubDBpediaClient()
        client._build_context_routes = Mock(return_value=[])
        client._build_dbpedia_routes([LinkedEntity(DBPEDIA + "Kajra_Re", "Kajra Re", 0.9)])
        client._build_context_routes.assert_not_called()

    def test_lookup_rejects_fuzzy_top_hit_and_matches_highlighted_name(self) -> None:
        payload = {"docs": [
            {"resource": [DBPEDIA + "Kara_Region"], "label": ["<B>Kara</B> <B>Region</B>"]},
            {"resource": [DBPEDIA + "Kajra_Re"], "label": ["<B>Kajra</B> <B>Re</B>"]},
        ]}
        entities = parse_lookup_json(payload, "kajra re")
        self.assertEqual([e.uri for e in entities], [DBPEDIA + "Kajra_Re"])
        self.assertIn("Lookup", entities[0].resolution_source)

    def test_lookup_matches_redirect_alias_and_rejects_ambiguity(self) -> None:
        doc = {"resource": [DBPEDIA + "United_States"], "redirectlabel": ["<B>USA</B>"]}
        self.assertEqual(parse_lookup_json({"docs": [doc]}, "usa")[0].uri, DBPEDIA + "United_States")
        other = {"resource": [DBPEDIA + "USA_(film)"], "label": ["USA"]}
        self.assertEqual(parse_lookup_json({"docs": [doc, other]}, "usa"), [])
        self.assertEqual(parse_lookup_json({"docs": [doc]}, "unrelated"), [])
        with self.assertRaises(ValueError):
            parse_lookup_json({"error": "failed"}, "usa")

    def test_lookup_recovers_graph_without_retrying_failed_spotlight_for_news(self) -> None:
        client = StubDBpediaClient()
        client.link_entities = Mock(side_effect=requests.ConnectionError("connection refused"))
        client.lookup_entities = Mock(return_value=[LinkedEntity(
            DBPEDIA + "Kajra_Re", "kajra re", 0.7, resolution_source="DBpedia Lookup"
        )])
        client.load_google_news = Mock(return_value=[{
            "title": "Kajra Re singer Alisha Chinai recalls the song",
            "url": "https://example.com/story", "publisher": "Example",
        }])
        profile = client.build_profile("kajra re")
        client.link_entities.assert_called_once_with("kajra re")
        self.assertTrue(any(r.source_kind == "dbpedia_direct" for r in profile.routes))
        self.assertEqual(profile.source_status["DBpedia Lookup"], "Available (1 matched entities)")
        self.assertEqual(len(profile.warnings), 1)
        self.assertEqual(profile.diagnostics["DBpedia Spotlight"], "connection refused")

    def test_lookup_failure_keeps_exact_name_fallback(self) -> None:
        client = StubDBpediaClient()
        client.link_entities = Mock(return_value=[])
        client.lookup_entities = Mock(side_effect=requests.Timeout("lookup timed out"))
        client._map_labels_to_dbpedia = Mock(return_value={"kajra re": [(DBPEDIA + "Kajra_Re", "Kajra Re")]})
        client.load_google_news = Mock(return_value=[])
        profile = client.build_profile("Kajra Re")
        self.assertTrue(profile.routes)
        self.assertEqual(profile.source_status["DBpedia Lookup"], "Unavailable")

    def test_exact_name_fallback_does_not_expand_ambiguous_matches(self) -> None:
        client = DBpediaSpotlightClient()
        client._sparql = Mock(return_value=[
            {"entity": {"value": DBPEDIA + name}, "label": {"value": "Mercury"}}
            for name in ("Mercury_(planet)", "Mercury_(element)")
        ])
        self.assertEqual(client._map_labels_to_dbpedia(["Mercury"]), {})

    def test_lookup_request_uses_configured_endpoint(self) -> None:
        session = Mock()
        session.get.return_value.json.return_value = {"docs": []}
        client = DBpediaSpotlightClient(session=session)
        self.assertEqual(client.lookup_entities("kajra re"), [])
        args, kwargs = session.get.call_args
        self.assertEqual(args[0], client.settings.lookup_url)
        self.assertEqual(kwargs["params"], {"query": "kajra re", "format": "JSON", "maxResults": 20})

    def test_spotlight_request_and_repeated_mention_offsets(self) -> None:
        session = Mock()
        session.post.return_value.json.return_value = {"Resources": [
            {"@URI": DBPEDIA + "Iran", "@surfaceForm": "Iran", "@offset": str(offset),
             "@similarityScore": "0.9"} for offset in (0, 5)
        ]}
        client = DBpediaSpotlightClient(session=session)
        entities = client.link_entities("Iran\nIran")
        self.assertEqual([(e.begin_index, e.end_index) for e in entities], [(0, 4), (5, 9)])
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], client.settings.spotlight_url)
        self.assertEqual(kwargs["data"]["text"], "Iran\nIran")
        self.assertEqual(kwargs["headers"]["Accept"], "application/json")

    def test_empty_and_malformed_spotlight_responses(self) -> None:
        self.assertEqual(parse_spotlight_json({"@text": "no entities"}), [])
        for payload in ([], {"Resources": {}}):
            with self.assertRaises(ValueError):
                parse_spotlight_json(payload)
        self.assertEqual(parse_spotlight_json({"Resources": [
            {"@URI": DBPEDIA + "Iran> . ?s ?p ?o", "@surfaceForm": "Iran",
             "@offset": "0", "@similarityScore": "0.9"},
            {"@URI": DBPEDIA + "Iran", "@surfaceForm": "Iran",
             "@offset": "0", "@similarityScore": "nan"}
        ]}), [])

    def test_spotlight_failure_uses_dbpedia_label_fallback(self) -> None:
        client = StubDBpediaClient()
        client.link_entities = Mock(side_effect=requests.ConnectionError("unavailable"))
        client.load_google_news = Mock(return_value=[])
        client._map_labels_to_dbpedia = Mock(return_value={
            "kajra re": [(DBPEDIA + "Kajra_Re", "Kajra Re")]
        })
        profile = client.build_profile("Kajra Re")
        self.assertEqual(profile.source_status["DBpedia Spotlight"], "Unavailable")
        self.assertTrue(profile.source_status["DBpedia"].startswith("Available"))
        self.assertTrue(profile.routes)
        client._map_labels_to_dbpedia.assert_called_once_with(["Kajra Re"])

    def test_linked_uri_does_not_expand_to_label_homonyms(self) -> None:
        client = StubDBpediaClient()
        client._map_labels_to_dbpedia = Mock(return_value={})
        client._build_dbpedia_routes([LinkedEntity(DBPEDIA + "Kajra_Re", "Kajra Re", 0.9)])
        client._map_labels_to_dbpedia.assert_called_once_with([])

    def test_sparql_errors_are_not_reported_as_empty_results(self) -> None:
        session = Mock()
        client = DBpediaSpotlightClient(session=session)
        for payload in ({"error": "query failed"}, [], {"results": {}}):
            session.post.return_value.json.return_value = payload
            with self.assertRaises(ValueError):
                client._sparql("SELECT * WHERE { ?s ?p ?o } LIMIT 1")

    def test_fact_query_uses_dbpedia_predicates_and_english_labels(self) -> None:
        client = DBpediaSpotlightClient()
        client._sparql = Mock(return_value=[{
            "predicate": {"value": DBO + "artist"},
            "related": {"value": DBPEDIA + "Alisha_Chinai"},
            "relatedLabel": {"value": "Alisha Chinai"},
            "direction": {"value": "outgoing"},
        }])
        facts = client._load_dbpedia_facts(DBPEDIA + "Kajra_Re")
        query = client._sparql.call_args.args[0]
        self.assertIn("<" + DBO + "starring>", query)
        self.assertIn('LANG(?relatedLabel) = "en"', query)
        self.assertEqual(facts[0]["related_label"], "Alisha Chinai")
        client._sparql.reset_mock()
        self.assertEqual(client._load_dbpedia_facts(DBPEDIA + "bad> . ?s ?p ?o"), [])
        client._sparql.assert_not_called()

    def test_label_fallback_excludes_wikipedia_categories(self) -> None:
        client = DBpediaSpotlightClient()
        client._sparql = Mock(return_value=[
            {"entity": {"value": DBPEDIA + name}, "label": {"value": "Albert Einstein"}}
            for name in ("Albert_Einstein", "Category:Albert_Einstein")
        ])
        self.assertEqual(client._map_labels_to_dbpedia(["Albert Einstein"]), {
            "albert einstein": [(DBPEDIA + "Albert_Einstein", "Albert Einstein")]
        })

    def test_parses_spotlight_json_entities(self) -> None:
        payload = {"Resources": [{"@URI": DBPEDIA + "Kajra_Re", "@surfaceForm": "Kajra Re",
                                  "@offset": "0", "@similarityScore": "0.94"}]}
        entities = parse_spotlight_json(payload)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].uri, DBPEDIA + "Kajra_Re")
        self.assertEqual(entities[0].anchor, "Kajra Re")
        self.assertEqual(entities[0].end_index, 8)
        self.assertAlmostEqual(entities[0].confidence, 0.94)

    def test_dbpedia_uses_allowlisted_direct_and_controlled_two_hop_routes(self) -> None:
        client = StubDBpediaClient()
        routes = client._build_dbpedia_routes(
            [LinkedEntity(uri=f"{DBPEDIA}Kajra_Re", anchor="Kajra Re", confidence=0.95)]
        )
        subjects = {route.related_subject for route in routes}

        self.assertIn("Alisha Chinai", subjects)
        self.assertIn("Bunty Aur Babli", subjects)
        self.assertIn("Aishwarya Rai Bachchan", subjects)
        self.assertIn("Amitabh Bachchan", subjects)
        self.assertNotIn("song", subjects)
        cast_route = next(
            route for route in routes if route.related_subject == "Aishwarya Rai Bachchan"
        )
        self.assertIn("two-hop", cast_route.relationship_type.lower())

    def test_song_does_not_expand_to_unrelated_singers(self) -> None:
        routes = StubDBpediaClient()._build_dbpedia_routes(
            [LinkedEntity(uri=f"{DBPEDIA}Kajra_Re", anchor="Kajra Re", confidence=0.95)]
        )
        titles = pd.DataFrame(
            [
                {"story_id": "1", "page_title": "Alisha Chinai career highlights", "total_views": 10},
                {"story_id": "2", "page_title": "Bunty Aur Babli cast reunion", "total_views": 20},
                {"story_id": "3", "page_title": "Asha Bhosle birthday special", "total_views": 30},
                {"story_id": "4", "page_title": "Lata Mangeshkar remembered", "total_views": 40},
            ]
        )
        result = build_source_related_results(
            title_summary=titles,
            story_months=pd.DataFrame(),
            excluded_story_ids=set(),
            routes=routes,
        )

        self.assertIn("1", set(result["story_id"]))
        self.assertIn("2", set(result["story_id"]))
        self.assertNotIn("3", set(result["story_id"]))
        self.assertNotIn("4", set(result["story_id"]))

    def test_single_name_credit_does_not_match_someone_elses_full_name(self) -> None:
        titles = pd.DataFrame(
            [
                {"story_id": "subject", "page_title": "Gulzar remembers writing iconic lyrics", "total_views": 20},
                {"story_id": "other", "page_title": "Rakhee Gulzar reacts to recent criticism", "total_views": 30},
            ]
        )
        routes = [
            EvidenceRoute(
                related_subject="Gulzar",
                aliases=(),
                relationship_type="DBpedia: lyricist",
                why_related="DBpedia records Gulzar as lyricist for Kajra Re.",
                source_name="DBpedia",
                evidence_url=f"{DBPEDIA}Kajra_Re",
                confidence=95,
                source_kind="dbpedia_direct",
            )
        ]
        result = build_source_related_results(
            title_summary=titles,
            story_months=pd.DataFrame(),
            excluded_story_ids=set(),
            routes=routes,
        )
        self.assertEqual(set(result["story_id"]), {"subject"})

    def test_news_routes_use_linked_entities_and_corroborated_topics(self) -> None:
        items = [
            {
                "title": "Israel conflict with Iran threatens oil supplies",
                "url": "https://example.com/one",
                "publisher": "Publisher One",
                "published_at": datetime.now(timezone.utc),
            },
            {
                "title": "Israel tensions push oil prices near Hormuz",
                "url": "https://example.com/two",
                "publisher": "Publisher Two",
                "published_at": datetime.now(timezone.utc),
            },
            {
                "title": "Asha Bhosle discusses her musical career",
                "url": "https://example.com/three",
                "publisher": "Publisher Three",
                "published_at": datetime.now(timezone.utc),
            },
        ]
        text, spans = join_news_headlines(items)
        entities = []
        uris = {"Israel": DBPEDIA + "Israel", "Iran": DBPEDIA + "Iran", "Hormuz": DBPEDIA + "Strait_of_Hormuz"}
        for anchor, uri in uris.items():
            start = text.find(anchor)
            while start >= 0:
                entities.append(
                    LinkedEntity(
                        uri=uri,
                        anchor=anchor,
                        confidence=0.9,
                        begin_index=start,
                        end_index=start + len(anchor),
                    )
                )
                start = text.find(anchor, start + 1)

        routes = build_news_routes(
            query="Israel",
            query_entities=[LinkedEntity(DBPEDIA + "Israel", "Israel", 0.99)],
            news_items=items,
            headline_spans=spans,
            news_entities=entities,
        )
        subjects = {route.related_subject.lower() for route in routes}

        self.assertIn("iran", subjects)
        self.assertIn("hormuz", subjects)
        self.assertIn("oil", subjects)
        oil_route = next(route for route in routes if route.related_subject == "oil")
        self.assertEqual(oil_route.supporting_mentions, 2)
        self.assertNotIn("asha bhosle", subjects)

    def test_refined_ids_are_removed_before_result_building(self) -> None:
        titles = pd.DataFrame(
            [
                {"story_id": "direct", "page_title": "Israel conflict latest", "total_views": 900},
                {"story_id": "iran", "page_title": "Iran geopolitical crisis", "total_views": 700},
                {"story_id": "hormuz", "page_title": "Strait of Hormuz oil crisis", "total_views": 800},
            ]
        )
        months = pd.DataFrame(
            [
                {"story_id": "iran", "month": "2026-04-01", "views": 200},
                {"story_id": "iran", "month": "2026-05-01", "views": 500},
                {"story_id": "hormuz", "month": "2026-04-01", "views": 800},
            ]
        )
        routes = [
            EvidenceRoute(
                related_subject="Iran",
                aliases=(),
                relationship_type="Current news: DBpedia Spotlight-linked entity",
                why_related="Sourced Israel coverage identifies Iran.",
                source_name="Google News / Publisher",
                evidence_url="https://example.com/evidence",
                confidence=88,
                source_kind="news_entity",
            ),
            EvidenceRoute(
                related_subject="Israel",
                aliases=(),
                relationship_type="DBpedia: subject",
                why_related="Direct query entity.",
                source_name="DBpedia",
                evidence_url="http://dbpedia.org/resource/Israel",
                confidence=95,
                source_kind="dbpedia_direct",
            ),
        ]
        result = build_source_related_results(
            title_summary=titles,
            story_months=months,
            excluded_story_ids={"direct"},
            routes=routes,
        )

        self.assertEqual(set(result["story_id"]), {"iran"})
        iran = result.set_index("story_id").loc["iran"]
        self.assertEqual(iran["highest_view_month"], "May 2026")
        self.assertEqual(iran["highest_views"], 500)


if __name__ == "__main__":
    unittest.main()
