"""Retrieve publisher headlines from Google News RSS independently of the corpus."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
import re
from urllib.parse import urlsplit

import requests


def parse_publisher_domains(value):
    """Normalize editor-entered domains or URLs, rejecting invalid host names."""
    domains = set()
    for entry in re.split(r"[,;\s]+", value.strip()):
        if not entry:
            continue
        try:
            parsed = urlsplit(entry if "://" in entry else "https://" + entry)
            if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.port:
                raise ValueError
            host = (parsed.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
            host = host.removeprefix("www.")
            labels = host.split(".")
            if len(host) > 253 or len(labels) < 2 or not all(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels
            ) or labels[-1].isdigit():
                raise ValueError
        except (ValueError, UnicodeError):
            raise ValueError(f"Invalid publisher domain: {entry}. Enter a domain such as ndtv.com.") from None
        domains.add(host)
    return tuple(sorted(domains))


def fetch_recent_stories(query, *, rss_url, timeout=15, user_agent="TrafficPredictionSourceResearch/1.0", days=7, session=None, domains=(), languages=("en", "hi")):
    query = query.strip()
    if not query:
        return []
    domains = parse_publisher_domains(" ".join(domains))
    search = f"({query}) ({' OR '.join('site:' + domain for domain in domains)})" if domains else query
    if not languages or any(language not in ("en", "hi") for language in languages):
        raise ValueError("Choose English, Hindi, or both news editions.")
    items = []
    for language in dict.fromkeys(languages):
        response = (session or requests).get(
            rss_url,
            params={"q": f"{search} when:{days}d", "hl": f"{language}-IN", "gl": "IN", "ceid": f"IN:{language}"},
            headers={"User-Agent": user_agent}, timeout=timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        if root.tag != "rss":
            raise ValueError("The news service returned an unexpected feed format.")
        items.extend(root.findall("./channel/item"))
    stories, seen = [], set()
    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        publisher = (item.findtext("source") or "Unknown publisher").strip()
        source = item.find("source")
        try:
            source_host = (urlsplit(source.get("url", "") if source is not None else "").hostname or "").lower().rstrip(".").encode("idna").decode("ascii")
        except (ValueError, UnicodeError):
            source_host = ""
        if domains and not any(source_host == domain or source_host.endswith("." + domain) for domain in domains):
            continue
        # Google may attach edition-specific query parameters to the same article.
        parsed_link = urlsplit(link)
        identity = (parsed_link.netloc, parsed_link.path) if parsed_link.hostname == "news.google.com" else link
        if not title or not link.startswith(("https://", "http://")) or identity in seen:
            continue
        seen.add(identity)
        suffix = f" - {publisher}"
        if title.endswith(suffix):
            title = title[:-len(suffix)]
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
            published = published.replace(tzinfo=timezone.utc) if published.tzinfo is None else published
            published = published.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            published = None
        stories.append({"Publisher": publisher, "News article title": title,
                        "Published": published, "Article link": link,
                        "Publisher domain": source_host or "Not provided"})
    stories.sort(key=lambda story: story["Published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return stories
