"""Deterministic article-scope policies derived from vertex_related's taxonomy.

Only cached, explicit Wikidata edges supply facts. This is a title-evidence
filter, not causal inference or a claim that matching headlines are verified.
Unknown paths fail closed. No network or model calls are made here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import re

from src.wikidata_identity import identity_tokens, names, spans


POLICY_VERSION = "relationship-scope-v1"
FAMILY_PROPERTIES = {
    "immediate_family": {"P26", "P22", "P25", "P40", "P3373"},
    "family_network": {"P1038"},
    "affiliation": {"P54", "P108", "P463", "P1416", "P69"},
    "corporate_structure": {"P112", "P127", "P749", "P355", "P1830", "P169", "P488"},
    "creative_work": {"P50", "P57", "P161", "P58", "P86", "P175", "P676", "P162", "P264", "P123", "P144"},
    "event": {"P1344", "P710", "P664", "P793"},
    "sport": {"P118", "P1327"},
    "structural_topic": {"P361", "P527", "P138", "P180", "P921"},
    "award": {"P166"},
    "geographic_context": {"P276", "P19"},
    "classification": {"P1269", "P279", "P31"},
}
PROPERTY_FAMILY = {prop: family for family, props in FAMILY_PROPERTIES.items() for prop in props}
METADATA_PROPERTIES = {"P1269", "P279", "P31", "P19"}
# Product decisions, not probabilities: these first-order subjects may qualify
# on their own identity. Event/work participants and broad places may not.
STANDALONE_PROPERTIES = (
    FAMILY_PROPERTIES["immediate_family"] | FAMILY_PROPERTIES["corporate_structure"]
    | {"P54", "P108", "P463", "P1416", "P144", "P361", "P527", "P921"}
)
EVENT_TARGET_PROPERTIES = {"P1344", "P793"}

# Explicit directional path templates. All other property pairs are withheld
# before downloading second-hop endpoints. No unrestricted graph expansion.
TWO_HOP_POLICIES = {
    ("P1344", "P710"): "shared_event",
    ("P1344", "P664"): "shared_event",
    ("P1344", "P276"): "shared_event",
    ("P793", "P710"): "named_event",
    ("P793", "P664"): "named_event",
    ("P793", "P276"): "named_event",
    ("P355", "P355"): "corporate_chain",
    ("P1830", "P355"): "corporate_chain",
    ("P749", "P749"): "corporate_chain",
    ("P127", "P127"): "corporate_chain",
}
for first in ("P54", "P108", "P463", "P1416"):
    for second in ("P127", "P112", "P169", "P488", "P749", "P355"):
        TWO_HOP_POLICIES[first, second] = "institutional_bridge"
for second in ("P50", "P57", "P161", "P58", "P86", "P175", "P162"):
    TWO_HOP_POLICIES["P144", second] = "work_specific"
for second in ("P22", "P25", "P40", "P3373"):
    TWO_HOP_POLICIES["P26", second] = "family_network"

# Directional headline expressions: {middle} and {endpoint} refer to the exact
# entities on the second edge. Deliberately narrow; an unsupported paraphrase
# is withheld rather than treated as a proven relationship.
EDGE_EXPRESSIONS = {
    "P127": ("{middle} owner {endpoint}", "{middle} owned by {endpoint}", "{endpoint} owns {middle}"),
    "P112": ("{middle} founder {endpoint}", "{endpoint} founded {middle}"),
    "P169": ("{middle} ceo {endpoint}", "{endpoint} ceo of {middle}"),
    "P488": ("{middle} chairperson {endpoint}", "{middle} chairman {endpoint}", "{endpoint} chairs {middle}"),
    "P749": ("{middle} parent {endpoint}", "{middle} parent company {endpoint}", "{endpoint} subsidiary {middle}"),
    "P355": ("{middle} subsidiary {endpoint}", "{endpoint} subsidiary of {middle}"),
}
GENERIC_SCOPE_NAMES = {"event", "summit", "meeting", "conference", "forum", "team", "company", "group", "annual meeting"}


@dataclass(frozen=True)
class TitleScope:
    qid: str
    label: str
    aliases: tuple[str, ...]
    event: bool = False
    years: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalPolicy:
    family: str = "unsupported"
    enabled: bool = False
    can_retrieve_standalone: bool = False
    scopes: tuple[TitleScope, ...] = ()
    edge_property: str = ""
    acceptance_condition: str = "No supported relationship policy."
    rejection_rule: str = "Withhold relationships without an explicit policy."


def _time_years(statements):
    years = set()
    for statement in statements:
        snak = statement.get("mainsnak", statement)
        value = snak.get("datavalue", {}).get("value", {})
        if statement.get("rank") == "deprecated" or not isinstance(value, dict):
            continue
        # A year is usable only at year precision or finer (Wikidata precision 9).
        if value.get("precision", 9) >= 9:
            match = re.match(r"\+(\d{4})-", value.get("time", ""))
            if match:
                years.add(match[1])
    return years


def _scope(entity, *, event=False):
    aliases = tuple(names(entity))
    label = entity.get("labels", {}).get("en", {}).get("value") or next(iter(aliases), entity.get("id", ""))
    # Only labels identify an edition; an alias for a different edition must
    # never broaden its years. Claim dates are a fallback for undated labels.
    years = set(re.findall(r"\b(?:1|2)\d{3}\b", " ".join(
        item.get("value", "") for item in entity.get("labels", {}).values()))) if event else set()
    if event and not years:
        for prop in ("P585", "P580", "P582"):
            years.update(_time_years(entity.get("claims", {}).get(prop, [])))
    return TitleScope(entity.get("id", ""), label, aliases, event, tuple(sorted(years)))


def compile_policy(entities, properties, claims=()):
    """Compile policy from ordered root/intermediate/endpoint facts."""
    if len(entities) != len(properties) + 1 or not properties:
        return RetrievalPolicy()
    first = properties[0]
    if len(properties) == 1:
        family = PROPERTY_FAMILY.get(first, "unsupported")
        if family == "unsupported" or first in METADATA_PROPERTIES:
            return RetrievalPolicy(family=family)
        if first in EVENT_TARGET_PROPERTIES:
            scopes = [_scope(entities[-1], event=True)]
        elif first in STANDALONE_PROPERTIES:
            scopes = []
        else:
            # A work's creator, event's participant, educational institution,
            # award or location needs the source work/event/subject in title.
            scopes = [_scope(entities[0], event=first in {"P710", "P664"})]
        edge = ""
    elif len(properties) == 2:
        family = TWO_HOP_POLICIES.get(tuple(properties))
        if not family:
            return RetrievalPolicy()
        scopes = [] if family == "corporate_chain" else [
            _scope(entities[1], event=family in {"shared_event", "named_event"})]
        edge = properties[1] if family == "institutional_bridge" else ""
    else:
        return RetrievalPolicy()
    # Time-qualified connections cannot grant arbitrary endpoint coverage.
    # A source mention is deliberately required: title-only data may not
    # identify whose former role is meant. Never use traffic month as tenure.
    for index, claim in enumerate(claims):
        if any(claim.get("qualifiers", {}).get(prop) for prop in ("P582", "P585")):
            if properties[index] not in EVENT_TARGET_PROPERTIES and not (
                index == 1 and family in {"shared_event", "named_event"}
            ):
                qualifiers = claim.get("qualifiers", {})
                years = set().union(*(_time_years(qualifiers.get(prop, [])) for prop in ("P580", "P582", "P585")))
                if not years:
                    return RetrievalPolicy(family=family, rejection_rule="Time-qualified edge has no usable year evidence.")
                scope = replace(_scope(entities[index]), years=tuple(sorted(years)))
                if scope not in scopes:
                    scopes.append(scope)
    standalone = not scopes
    condition = "Related subject identity must be established in the title."
    if scopes:
        condition += " Also require: " + "; ".join(s.label + (" (exact event edition)" if s.event else "")
                                                     + (" [" + ", ".join(s.years) + "]" if s.years else "")
                                                     for s in scopes) + "."
    if edge:
        condition += " Require a supported directional relationship expression."
    return RetrievalPolicy(family, True, standalone, tuple(scopes), edge, condition,
                           "Reject endpoint-only mentions, missing or ambiguous scope, and mismatched event editions."
                           if scopes else "Reject unresolved or ambiguous related-subject identity.")


def _name_forms(scope):
    for name in scope.aliases:
        phrase = identity_tokens(name)
        words = " ".join(phrase)
        if not phrase or words in GENERIC_SCOPE_NAMES or all(token.isdigit() for token in phrase):
            continue
        if len(phrase) == 1 and len(phrase[0]) < 3:
            continue
        if scope.years:
            explicit_years = set(re.findall(r"\b(?:1|2)\d{3}\b", words))
            if explicit_years:
                if explicit_years.issubset(scope.years):
                    yield phrase, phrase
            else:
                # The year must be adjacent to the verified event alias; a
                # matching year elsewhere in the headline is insufficient.
                for year in scope.years:
                    yield (*phrase, year), phrase
                    yield (year, *phrase), phrase
        else:
            yield phrase, phrase


class RelationshipGate:
    """Apply scope before identity scoring; audit every rejected route match."""
    def __init__(self, profile):
        self.audit = []
        self.counts = Counter()
        self.name_owners = {}
        for qid, entity in profile.identity_entities.items():
            for name in names(entity):
                self.name_owners.setdefault(identity_tokens(name), set()).add(qid)
        self.scope_blockers = {}
        self.compiled = {}
        for route in profile.routes:
            policy = getattr(route, "retrieval_policy", None) or RetrievalPolicy()
            # Dataclass hashing reads every field. Older unpickled/session
            # routes may lack newly added fields, so key by object identity.
            self.compiled[id(route)] = (policy, tuple(tuple(_name_forms(s)) for s in policy.scopes))
            for scope, forms in zip(policy.scopes, self.compiled[id(route)][1]):
                for _, original in forms:
                    key = (scope.qid, original)
                    if key not in self.scope_blockers:
                        self.scope_blockers[key] = tuple(other for other, owners in self.name_owners.items()
                            if owners - {scope.qid} and len(other) > len(original) and spans(other, original))

    def _scope_match(self, scope, forms, title):
        if scope.event and not scope.years:
            # Without dated evidence we cannot bind participation to an
            # edition. Do not silently widen a shared-event route to all years.
            return ""
        for phrase, original in forms:
            for start, end in spans(title, phrase):
                adjacent = title[max(0, start - 1):start] + title[end:end + 1]
                if scope.years and any(re.fullmatch(r"(?:1|2)\d{3}", token) and token not in scope.years
                                       for token in adjacent):
                    continue
                # A longer colliding name cannot masquerade as this scope.
                collision = self.name_owners.get(original, set()) - {scope.qid}
                if collision:
                    continue
                if any(any(left < end and right > start for left, right in spans(title, other))
                       for other in self.scope_blockers.get((scope.qid, original), ())):
                    continue
                return " ".join(phrase)
        return ""

    def __call__(self, route, row):
        policy, forms = self.compiled.get(id(route), (RetrievalPolicy(), ()))
        title = identity_tokens(row["page_title"])
        evidence = []
        reason = "" if policy.enabled else "No permitted relationship policy for this path"
        if not reason:
            for scope, scope_forms in zip(policy.scopes, forms):
                matched = self._scope_match(scope, scope_forms, title)
                if not matched:
                    reason = "Missing or ambiguous connecting " + ("event/edition: " if scope.event else "subject: ") + scope.label
                    break
                evidence.append(matched)
        if not reason and policy.edge_property:
            # Templates operate on already-matched, normalized entity names.
            expressions = EDGE_EXPRESSIONS.get(policy.edge_property, ())
            matched_edge = next((pattern.format(middle=middle, endpoint=" ".join(identity_tokens(endpoint)))
                for middle in evidence[:1] for endpoint in (route.related_subject, *route.aliases)
                for pattern in expressions
                if spans(title, identity_tokens(pattern.format(middle=middle, endpoint=endpoint)))), "")
            if not matched_edge:
                reason = "Title does not express the required directional relationship"
            else:
                evidence.append(matched_edge)
        if reason:
            self.counts["withheld_route_matches"] += 1
            if len(self.audit) < 500:
                self.audit.append({"story_id": str(row["story_id"]), "page_title": row["page_title"],
                                   "relationship_family": policy.family, "related_subject": route.related_subject,
                                   "relationship_path": getattr(route, "relationship_path", ""), "reason": reason})
            return None
        self.counts["accepted_route_matches"] += 1
        return {"relationship_family": policy.family, "can_retrieve_standalone": policy.can_retrieve_standalone,
                "relationship_title_evidence": "; ".join(evidence) or "Standalone subject permitted by policy",
                "acceptance_condition": policy.acceptance_condition, "rejection_rule": policy.rejection_rule}
