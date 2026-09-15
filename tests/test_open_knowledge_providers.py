import unittest

from src.open_knowledge_providers import (
    CONCEPTNET_API_URL,
    GDELT_DOC_API_URL,
    fetch_conceptnet_relationships,
    fetch_gdelt_relationships,
)


class OpenKnowledgeProviderTests(unittest.TestCase):
    def test_conceptnet_keeps_only_approved_grounded_edges(self) -> None:
        def fetch_json(url: str, params: dict[str, object]) -> dict[str, object]:
            self.assertEqual(url, CONCEPTNET_API_URL)
            self.assertEqual(params["node"], "/c/en/mehndi")
            return {
                "edges": [
                    {
                        "@id": "/a/[/r/Synonym/,/c/en/mehndi/,/c/en/henna/]",
                        "rel": {"@id": "/r/Synonym"},
                        "start": {"@id": "/c/en/mehndi", "language": "en", "label": "mehndi"},
                        "end": {"@id": "/c/en/henna", "language": "en", "label": "henna"},
                        "weight": 2.5,
                    },
                    {
                        "@id": "/a/unsafe",
                        "rel": {"@id": "/r/Antonym"},
                        "start": {"@id": "/c/en/mehndi", "language": "en", "label": "mehndi"},
                        "end": {"@id": "/c/en/plain", "language": "en", "label": "plain"},
                        "weight": 5.0,
                    },
                ]
            }

        relationships, diagnostics = fetch_conceptnet_relationships(
            query="Mehndi",
            fetch_json=fetch_json,
        )

        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0]["related_subject"], "henna")
        self.assertTrue(relationships[0]["can_retrieve_standalone"])
        self.assertEqual(relationships[0]["source_family"], "commonsense_lexical")
        self.assertEqual(diagnostics["language_namespace"], "en")

    def test_conceptnet_uses_official_web_fallback_and_follows_form_root(self) -> None:
        requested_urls: list[str] = []

        def fetch_json(_url: str, _params: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("502 Bad Gateway")

        def fetch_text(url: str, params: dict[str, object]) -> str:
            requested_urls.append(url)
            self.assertEqual(params["limit"], 50)
            if url.endswith("/c/en/etfs"):
                return (
                    '<a href="/a/[/r/FormOf/,/c/en/etfs/n/,/c/en/etf/]" '
                    'class="edge-link">edge</a>'
                )
            if url.endswith("/c/en/etf"):
                return (
                    '<a href="/a/[/r/Synonym/,/c/en/etf/n/wn/possession/,'
                    '/c/en/exchange_traded_fund/n/wn/possession/]" '
                    'class="edge-link">edge</a>'
                    '<a href="/a/[/r/HasContext/,/c/en/etf/n/,'
                    '/c/en/finance/]" class="edge-link">edge</a>'
                )
            self.fail(f"Unexpected ConceptNet URL: {url}")

        relationships, diagnostics = fetch_conceptnet_relationships(
            query="ETFs",
            fetch_json=fetch_json,
            fetch_text=fetch_text,
        )

        self.assertEqual(
            requested_urls,
            ["https://conceptnet.io/c/en/etfs", "https://conceptnet.io/c/en/etf"],
        )
        self.assertEqual(
            {item["related_subject"] for item in relationships},
            {"exchange traded fund", "finance"},
        )
        self.assertEqual(diagnostics["retrieval_mode"], "official_web_fallback")
        self.assertEqual(diagnostics["canonical_form_count"], 1)

    def test_gdelt_requires_repetition_across_publishers(self) -> None:
        def fetch_json(url: str, params: dict[str, object]) -> dict[str, object]:
            self.assertEqual(url, GDELT_DOC_API_URL)
            self.assertEqual(params["mode"], "artlist")
            return {
                "articles": [
                    {
                        "title": "Balen Shah meets Kathmandu transport officials",
                        "domain": "publisher-one.example",
                        "url": "https://publisher-one.example/a",
                    },
                    {
                        "title": "Kathmandu mayor Balen Shah unveils transport plan",
                        "domain": "publisher-two.example",
                        "url": "https://publisher-two.example/b",
                    },
                    {
                        "title": "Balen Shah discusses a unique riverside project",
                        "domain": "publisher-one.example",
                        "url": "https://publisher-one.example/c",
                    },
                ]
            }

        relationships, diagnostics = fetch_gdelt_relationships(
            query="Balen Shah",
            fetch_json=fetch_json,
        )

        subjects = {item["related_subject"] for item in relationships}
        self.assertIn("Kathmandu", subjects)
        self.assertIn("Transport", subjects)
        self.assertNotIn("Riverside", subjects)
        self.assertTrue(all(not item["can_retrieve_standalone"] for item in relationships))
        self.assertEqual(diagnostics["article_count"], 3)

    def test_gdelt_removes_exact_query_tokens_from_phrases(self) -> None:
        def fetch_json(_url: str, _params: dict[str, object]) -> dict[str, object]:
            return {
                "articles": [
                    {"title": "Sawan festival calendar released", "domain": "one.example"},
                    {"title": "Sawan festival dates announced", "domain": "two.example"},
                ]
            }

        relationships, _ = fetch_gdelt_relationships(
            query="Sawan",
            fetch_json=fetch_json,
        )

        self.assertTrue(
            all("sawan" not in str(item["related_subject"]).casefold() for item in relationships)
        )


if __name__ == "__main__":
    unittest.main()
