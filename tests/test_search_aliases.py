import unittest

from src.search_aliases import (
    SEARCH_ALIAS_ENTITIES,
    find_ambiguous_aliases,
    normalize_refined_alias_text,
    opensearch_synonym_rules,
    query_uses_alias_expansion,
)


class SearchAliasRegistryTests(unittest.TestCase):
    def test_registry_covers_all_editorial_categories(self) -> None:
        categories = {entity.category for entity in SEARCH_ALIAS_ENTITIES}

        self.assertEqual(
            categories,
            {
                "geographic",
                "government",
                "political",
                "public_office",
                "news_acronym",
                "spelling_transliteration",
                "organization_company",
            },
        )

    def test_safe_multiword_aliases_generate_synonym_rules(self) -> None:
        rules = opensearch_synonym_rules()

        self.assertIn("uttar pradesh, up", rules)
        self.assertIn("reserve bank of india, rbi", rules)
        self.assertIn("bengaluru, bangalore", rules)

    def test_ambiguous_aliases_are_reported_and_not_expanded(self) -> None:
        ap = find_ambiguous_aliases("AP election")
        mp = find_ambiguous_aliases("MP election")
        serialized_rules = " | ".join(opensearch_synonym_rules())

        self.assertEqual(ap[0].candidates, ("Andhra Pradesh", "Arunachal Pradesh"))
        self.assertEqual(mp[0].candidates, ("Madhya Pradesh", "Member of Parliament"))
        self.assertNotIn(", ap", serialized_rules)
        self.assertNotIn(", mp", serialized_rules)

    def test_ampersand_initialism_is_preserved(self) -> None:
        self.assertEqual(normalize_refined_alias_text("J&K election"), "JK election")
        self.assertTrue(query_uses_alias_expansion("jk election"))

    def test_non_alias_query_does_not_request_expansion(self) -> None:
        self.assertFalse(query_uses_alias_expansion("government policy"))


if __name__ == "__main__":
    unittest.main()
