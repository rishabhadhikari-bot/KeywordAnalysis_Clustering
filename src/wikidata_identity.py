"""Conservative title identity checks; no generated text or per-query rules."""
from __future__ import annotations

import unicodedata
import re
from collections import Counter

from src.data_processing import simple_query_tokens

IDENTITY_VERSION = "wikidata-identity-v2"
GENERIC_CONTEXT = frozenset("human person man woman people entity thing object company organization institution indian india english hindi known also born former current new latest news update famous renowned based located world international के की का को में से पर और है हैं एक यह वह भारतीय".split())
CONTEXT_PREDICATES = {"P31", "P279", "P106", "P54", "P108", "P463", "P69", "P361", "P452", "P1056"}


def identity_tokens(text):
    # Keep Indic combining marks and exact phrase order. Stopword removal must
    # not turn separated headline words into an apparently exact entity name.
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple("".join(char if unicodedata.category(char)[0] in "LNM" else " "
                         for char in text).split())


def context_tokens(text):
    # The older English keyword cleaner drops Devanagari entirely. Preserve
    # those words here, while retaining English stopword removal/lemmatization.
    return tuple(token for word in identity_tokens(text)
                 for token in ([word] if any("\u0900" <= c <= "\u097f" for c in word)
                               else simple_query_tokens(word)))


def label_variants(label):
    """Standard legal suffixes can be omitted; retain a multiword identity."""
    shortened = re.sub(r",?\s+(?:(?:private|pvt\.?)\s+)?(?:limited|ltd|incorporated|inc|llc|llp|plc)\.?$",
                       "", label, flags=re.IGNORECASE).strip()
    return [label, shortened] if shortened != label and len(identity_tokens(shortened)) >= 2 else [label]


def names(entity):
    values = []
    for lang in ("en", "hi"):
        values.extend(label_variants(entity.get("labels", {}).get(lang, {}).get("value", "")))
        values.extend(item.get("value", "") for item in entity.get("aliases", {}).get(lang, []))
        values.append(entity.get("sitelinks", {}).get(f"{lang}wiki", {}).get("title", ""))
    return list(dict.fromkeys(value for value in values if value))


def spans(title, phrase):
    return [(i, i + len(phrase)) for i in range(len(title) - len(phrase) + 1)
            if phrase and title[i:i + len(phrase)] == phrase]


def context_words(entity, catalog):
    text = [item.get("value", "") for lang, item in entity.get("descriptions", {}).items()
            if lang in {"en", "hi"}]
    for prop in CONTEXT_PREDICATES:
        for statement in entity.get("claims", {}).get(prop, []):
            if statement.get("rank") == "deprecated":
                continue
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            target = catalog.get(value.get("id"), {}) if isinstance(value, dict) else {}
            text.extend(item.get("value", "") for lang, item in target.get("labels", {}).items()
                        if lang in {"en", "hi"})
    return set(context_tokens(" ".join(text))) - GENERIC_CONTEXT


class IdentityGate:
    def __init__(self, profile):
        self.catalog = profile.identity_entities
        self.checked_names = profile.checked_identity_names
        self.contexts = {qid: context_words(entity, self.catalog) for qid, entity in self.catalog.items()}
        self.name_index = {}
        for qid, entity in self.catalog.items():
            for name in names(entity):
                token_name = identity_tokens(name)
                if token_name:
                    self.name_index.setdefault(token_name[0], []).append((token_name, qid, name))
        self.audit = []
        self.counts = Counter()

    def __call__(self, route, row):
        qid = getattr(route, "target_qid", "")
        title = identity_tokens(row["page_title"])
        title_words = set(context_tokens(row["page_title"]))
        mentions = []
        for token in set(title):
            for phrase, entity_qid, name in self.name_index.get(token, []):
                for start, end in spans(title, phrase):
                    mentions.append((start, end, entity_qid, name))
        accepted, reasons = [], []
        for name in dict.fromkeys([route.related_subject, *route.aliases]):
            phrase = identity_tokens(name)
            occurrences = spans(title, phrase)
            if not occurrences:
                continue
            if not qid or qid not in self.catalog:
                reasons.append("Missing related-entity identity data")
                continue
            supporting = (title_words & self.contexts.get(qid, set())) - set(context_tokens(name))
            # A longer overlapping full name wins at that occurrence only. A
            # separate valid mention of the intended entity can still qualify.
            competitors = set()
            viable = False
            for start, end in occurrences:
                longer = [(other_qid, other_name) for left, right, other_qid, other_name in mentions
                          if other_qid != qid and left <= start and right >= end
                          and right - left > end - start]
                if longer:
                    reasons.append("Competing longer name: " + "; ".join(sorted({n for _, n in longer})))
                    continue
                viable = True
                competitors.update(other_qid for left, right, other_qid, _ in mentions
                                   if other_qid != qid and left == start and right == end)
            if not viable:
                continue
            competitor_support = max((len((title_words & self.contexts.get(other, set()))
                                          - set(context_tokens(name))) for other in competitors), default=0)
            if competitor_support >= max(2, len(supporting)):
                reasons.append("Context favors another entity or is tied")
                continue
            is_label = any(phrase == identity_tokens(variant)
                           for lang, item in self.catalog[qid].get("labels", {}).items()
                           if lang in {"en", "hi"} for variant in label_variants(item.get("value", "")))
            # Full labels offer stronger evidence than arbitrary aliases. Alias
            # search coverage is never a uniqueness guarantee, especially for
            # one-word names, which always require independent context.
            checked = " ".join(phrase) in self.checked_names
            strong_full_name = len(phrase) >= 2 and not competitors and (is_label or checked)
            context_pass = len(supporting) >= 2 and len(supporting) > competitor_support
            checked_label_pass = len(phrase) == 1 and is_label and checked and not competitors and bool(supporting)
            if not strong_full_name and not context_pass and not checked_label_pass:
                reasons.append("Insufficient identity context for alias or ambiguous name")
                continue
            reason = ("Full multiword name; no competing match in available knowledge"
                      if strong_full_name else
                      ("Checked single-word label with Wikidata context: " if checked_label_pass else
                       "Supporting Wikidata context: ") + ", ".join(sorted(supporting)))
            accepted.append(dict(matched_name=name, related_entity_qid=qid,
                                 identity_reason=reason, identity_score=96 if strong_full_name else 86,
                                 hop_count=getattr(route, "hop_count", 1),
                                 relationship_path=getattr(route, "relationship_path", "") or route.why_related,
                                 path_entity_ids=" → ".join(getattr(route, "path_qids", ())),
                                 path_evidence_urls=" ; ".join(getattr(route, "path_evidence_urls", ()) or (route.evidence_url,))))
        if accepted:
            self.counts["accepted_route_matches"] += 1
            return max(accepted, key=lambda match: (match["identity_score"], len(match["matched_name"])))
        reason = "; ".join(dict.fromkeys(reasons)) or "No contiguous entity-name match"
        self.counts["withheld_route_matches"] += 1
        if len(self.audit) < 500:
            self.audit.append({"story_id": str(row["story_id"]), "page_title": row["page_title"],
                               "related_entity_qid": qid, "related_subject": route.related_subject,
                               "hop_count": getattr(route, "hop_count", 1),
                               "relationship_path": getattr(route, "relationship_path", ""),
                               "matched_names": "; ".join(name for name in dict.fromkeys([route.related_subject, *route.aliases])
                                                             if spans(title, identity_tokens(name))),
                               "reason": reason})
        return None
