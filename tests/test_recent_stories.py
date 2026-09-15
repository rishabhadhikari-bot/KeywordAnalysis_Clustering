import unittest
from unittest.mock import Mock

from src.recent_stories import fetch_recent_stories, parse_publisher_domains


class RecentStoriesTests(unittest.TestCase):
    def test_both_editions_merge_duplicates_and_sort(self):
        def feed(items):
            return Mock(content=("<rss><channel>" + items + "</channel></rss>").encode())
        def item(article, language, date):
            return (f'<item><title>Story {article}</title><link>https://news.google.com/rss/articles/{article}?hl={language}</link>'
                    f'<source url="https://www.bhaskar.com">Bhaskar</source><pubDate>{date}</pubDate></item>')
        session = Mock()
        session.get.side_effect = [
            feed(item("same", "en", "Sun, 06 Sep 2026 08:00:00 GMT")),
            feed(item("same", "hi", "Sun, 06 Sep 2026 08:00:00 GMT") +
                 item("hindi", "hi", "Mon, 07 Sep 2026 08:00:00 GMT")),
        ]
        rows = fetch_recent_stories("satyaniketan building collapse", rss_url="https://news.google.com/rss/search",
                                    session=session, domains=("bhaskar.com",))
        self.assertEqual([r["News article title"] for r in rows], ["Story hindi", "Story same"])
        self.assertEqual([call.kwargs["params"]["ceid"] for call in session.get.call_args_list], ["IN:en", "IN:hi"])

    def test_single_edition_and_empty_english_feed(self):
        empty = Mock(content=b"<rss><channel/></rss>")
        hindi = Mock(content='<rss><channel><item><title>Hindi story</title><link>https://news.google.com/rss/articles/1</link><source url="https://www.bhaskar.com">Bhaskar</source></item></channel></rss>'.encode())
        session = Mock()
        session.get.side_effect = [empty, hindi]
        self.assertEqual(len(fetch_recent_stories("collapse", rss_url="https://news.google.com/rss/search", session=session)), 1)
        session = Mock()
        session.get.return_value = hindi
        self.assertEqual(len(fetch_recent_stories("collapse", rss_url="https://news.google.com/rss/search", session=session, languages=("hi",))), 1)
        self.assertEqual(session.get.call_count, 1)
        self.assertEqual(session.get.call_args.kwargs["params"]["hl"], "hi-IN")

    def test_domain_normalization(self):
        self.assertEqual(parse_publisher_domains(
            "https://www.NDTV.com/news, ndtv.com\nindianexpress.com; thehindu.com"
        ), ("indianexpress.com", "ndtv.com", "thehindu.com"))
        self.assertEqual(parse_publisher_domains(""), ())

    def test_invalid_domains(self):
        for value in ("ndtv", "*.ndtv.com", "https://user@ndtv.com", "bad_domain.com", "ftp://ndtv.com"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_publisher_domains(value)

    def test_filter_checks_publisher_host_and_preserves_unrestricted_results(self):
        session = Mock()
        hosts = ["ndtv.com", "sports.ndtv.com", "ndtv.com.other.org", "fakendtv.com", "", "thehindu.com"]
        items = [f'<item><title>Story {i}</title><link>https://news.google.com/{i}</link>'
                 f'<source url="https://{host}">Publisher</source></item>' for i, host in enumerate(hosts)]
        session.get.return_value.content = ("<rss><channel>" + "".join(items) + "</channel></rss>").encode()
        rows = fetch_recent_stories("cricket", rss_url="https://news.google.com/rss/search",
                                    session=session, domains=("ndtv.com", "thehindu.com"))
        self.assertEqual([r["Publisher domain"] for r in rows], ["ndtv.com", "sports.ndtv.com", "thehindu.com"])
        self.assertEqual(session.get.call_args.kwargs["params"]["q"],
                         "(cricket) (site:ndtv.com OR site:thehindu.com) when:7d")
        rows = fetch_recent_stories("cricket", rss_url="https://news.google.com/rss/search", session=session)
        self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
