# Keyword Analysis and Clustering

## Run app2.py

Create a Python virtual environment, activate it, and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place your local data exports in the repository root:

- `PageTitle2026.csv` and `Stories2026.csv` for title and traffic analysis.
- `StoryID-Category.csv`, `StoryID-WordCount.csv`, `StoryID-Region_withViews.csv`,
  `StoryID-planned_trending.csv`, and `StoryID- AudienceType.csv` for the corresponding metadata features.

Datasets, credentials, and generated indexes/caches are not included in this repository.

For OpenSearch features, copy `.env.opensearch.example` to `.env` and start the local service:

```powershell
Copy-Item .env.opensearch.example .env
docker compose -f compose.opensearch.yml up -d
python -m streamlit run app2.py
```

Alternatively, on Windows with a native OpenSearch installation:

```powershell
.\start-native-opensearch.ps1 -OpenSearchHome 'C:\path\to\opensearch-3.8.0'
```

The native launcher uses the repository's `.venv` and launches `app2.py`.
Vertex AI features require your own `service_account.json` and configured Google Cloud access.
Embedding and NLP features may download models on first use; generated artifacts are stored locally under `data/`.

## Tests

```powershell
python -m unittest discover -s tests
```

## Feature documentation

# Keyword Traffic Aggregator MVP

### Embeddings: keywords from full-title retrieval in app2.py

Click **Build / load title embeddings** to embed all loaded story titles in full.
Titles are deduplicated by story ID, missing/blank titles are skipped, and Unicode
and whitespace are normalized. Stop words and punctuation remain in the embedded
text to preserve context. The query uses the same locally cached multilingual model.
Combined meaning embeds the complete query; Separate keywords embeds each query word.

Nearest titles are selected by cosine similarity and the title count/score filters,
without requiring literal query matches. Keywords are extracted from those titles,
excluding common English/Hindi stop words, numbers, custom exclusions and (optionally)
query words. Repeated words count once per story. Minimum occurrences refer to the
retrieved titles. Relevance = sum of supporting positive title similarities times
log(1 + corpus story count / keyword story count), divided by retrieved title count.
This weights recurring terms and reduces common corpus words; it is not keyword
cosine similarity or a probability. Results include supporting counts, best title
similarity, example titles, CSV export, and an expandable retrieved-title table.

Vectors are built on demand under `data/title_keyword_embeddings`, keyed by full
ordered title texts and model. Changing keyword exclusions needs no re-embedding.
The old individual-word index is not used by this tab. Other tabs and their indexes
are unchanged. The model can truncate titles beyond its token limit; semantic
retrieval and keyword extraction do not guarantee synonymy.

### Miscellaneous: monthly title keyword dataset

Both `app.py` and `app2.py` include a **Miscellaneous** tab. Click **Build monthly
keyword dataset** to aggregate the existing monthly traffic and title data.
No upload is required or provided. The full, unfiltered story-month model is
used, independently of the main search and high-traffic cutoff.

The output has `month` (YYYY-MM), `token`, `occurrences` (pages containing the
token), `total_token_occurrences` (including repetitions within titles),
`total_views`, and `views_per_occurrence` (`total_views / occurrences`). Traffic
rows are summed per story/month before counting tokens. Explicit zero-view
records are included; pages absent from a month are not inferred. Missing
titles and their excluded views are reported. Existing titles apply to all months.

Tokens are Unicode words/numbers normalized to lowercase; punctuation and
hyphens separate tokens. Common English words and custom words can be excluded.
Each page contributes views to every included token, so token views overlap.
Month, token, and minimum-occurrence filters control the preview and filtered
CSV. The full CSV contains the complete generated monthly dataset.

Generation is on demand with separate session state. Changes to source data or
exclusions invalidate results. The tab does not modify source CSVs, shared lookup
data, or other tabs' search filters.

### BERTopic fixes in app2.py

The BERTopic tab collects the complete current OpenSearch refined result set,
including similar-spelling matches and results beyond the displayed page. It
does not run when that exclusion set is unavailable or the index is stale.
Excluded stories and duplicate normalized titles are suppressed from candidates,
catalog examples, and direct-title previews. Primary refined matches may still
provide internal query context. Catalog counts describe the remaining corpus,
including an explicit Unassigned group; they are not full-corpus traffic totals.

The shared BERTopic engine now retains topic-specific vocabulary (`min_df=1`),
excludes direct results before neighbour limits, handles zero learned topics,
and rejects invalid membership strengths. Training configuration and library
versions invalidate the disk cache. Topic artifacts use immutable generations
with an atomic manifest switch and an in-process build lock.

Optional settings are `BERTOPIC_MIN_CLUSTER_SIZE` (20), `BERTOPIC_MIN_SAMPLES`
(20), `BERTOPIC_N_NEIGHBORS` (15), and `BERTOPIC_N_COMPONENTS` (5). Defaults
preserve the previous clustering density while allowing independent experiments.
The next analysis rebuilds automatically when configuration or titles change.
Older artifact generations are retained. These shared engine fixes also apply
to app.py; the complete refined-result UI exclusion is wired specifically in app2.py.

Editorial label review, measured parameter/model comparisons, stable editorial
topic IDs, and a topic-by-content-type performance dashboard remain subsequent
work. Membership strength is not a probability of editorial correctness.

Phase 1 module for editorial keyword traffic exploration using local/static CSV files.

## Current MVP scope

- Reads local/static CSV files
- Calculates monthly total page views
- Joins title metadata with monthly story views by `story_id`
- Cleans each page title into stopword-filtered tokens
- Lets users search for a keyword or group of keywords
- Supports uppercase/dotted two-letter abbreviations and configured lowercase
  forms against lowercase title data
- Shows titles containing the search and the views those titles generated
- Shows month-level rows so repeated stories across months remain visible

## Code organization

| path | responsibility |
| --- | --- |
| `app2.py` | Streamlit dashboard, search controls, charts, and tables |
| `src/data_processing.py` | CSV loading, title-view joining, simple token lookup, view aggregation |
| `src/nlp/normalization.py` | Text cleaning, token normalization, stopword filtering, lemmatization handoff, n-gram creation |
| `src/nlp/lemmatization.py` | spaCy-backed English lemmatization with a small fallback for offline runs |
| `src/nlp/keyword_extraction.py` | Advanced keyword extraction kept for the later NLP pipeline |
| `src/nlp/semantic_expansion.py` | Advanced related-keyword context kept for the later NLP pipeline |
| `src/nlp/search.py` | Advanced exact, contains, fuzzy, and semantic search kept for the later NLP pipeline |
| `src/opensearch_refined.py` | Isolated OpenSearch indexing and tiered retrieval for the Search Results refined tab |
| `src/opensearch_semantic.py` | Isolated related-title index, batched relationship-route retrieval, and refined-result exclusion |
| `src/relationship_profile_store.py` | Reusable cached grounded relationship profiles across match modes and title-data fingerprints |
| `src/semantic_related.py` | Conservative title-evidence checks and relationship-only traffic aggregation |
| `src/semantic_search_tab.py` | Separate low-latency relationship-aware title and traffic tab |
| `src/semantic_seo_tab.py` | Isolated Semantic SEO Lab UI and orchestration |
| `src/semantic_seo_routes.py` | Strict, source-grounded Gemini route discovery and independent verification |
| `src/semantic_seo_models.py` | Ingestion-time GLiNER linking/cache, BGE reranker, optional DeBERTa title scoring, and dormant mREBEL adapter |
| `src/semantic_seo_pipeline.py` | OpenSearch-first retrieval and bounded title-only scoring |
| `src/knowledge_source_related.py` | Public Wikidata/Wikipedia adapters, research-taxonomy family/corporate graph expansion, optional local WordNet expansion, SQLite profile caching, and deterministic source-based title matching |

## Simple MVP flow

```text
Stories2026.csv + PageTitle2026.csv
-> join by story_id
-> aggregate repeated story-month views
-> clean each page title
-> tokenize after stopword removal
-> user searches keyword/group of keywords
-> match against cleaned title tokens
-> show matching page titles and associated views
```

## Search modes

| mode | behavior |
| --- | --- |
| `All keywords` | Every cleaned query token must exist in the cleaned title |
| `Exact cleaned phrase` | Cleaned query tokens must appear together in the same order |
| `Any keyword` | At least one cleaned query token must exist in the cleaned title |

## Expected CSV files

### `PageTitle2026.csv`

| column | description |
| --- | --- |
| `story_id` / `StoryID` | Unique story identifier |
| `PageTtile` / `PageTitle` / `Page Titel` | Story/page title used for rule-based keyword extraction |

### `Stories2026.csv`

| column | description |
| --- | --- |
| `story_id` | Unique story identifier |
| `Month` | Month label/date, for example `Apr-2026` |
| `Event count` | Monthly view events for the story; used as views |

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app2.py
```

Replace the local CSV files with updated exports when ready. The app does not use a CSV uploader flow.

### Source-Based Related Stories

The separate `Source-Based Related Stories` tab is an experimental, non-Gemini
flow. It normalizes the query, scores merged Wikidata candidates, uses bounded
English Wikipedia full-text search when label search is weak, enriches accepted
identities with redirects, optionally adds context-bounded local WordNet
synonyms, and applies deterministic subject/cue rules against the local title
corpus. Ambiguous candidates are retained for editorial review rather than
silently selecting the first result. Existing refined primary/direct IDs and
source-derived identity aliases are excluded before results are displayed.

Wikidata and Wikipedia use public read-only HTTP APIs and do not require an API
key. Set a meaningful application User-Agent, preferably including a project URL
or contact address:

```powershell
$env:KNOWLEDGE_SOURCE_USER_AGENT='TrafficPrediction/1.0 (https://example.com/contact)'
```

WordNet runs locally. Install its data once in the same Python environment:

```powershell
.\.venv\Scripts\python.exe -m nltk.downloader -d data\nltk_data wordnet omw-1.4
```

Profiles are cached for seven days in
`data/knowledge_sources/source_profiles.db`. The local fallback accepts
lemmatized and bounded all-token seed matches, so inflection and intervening
headline words do not prevent corpus discovery. Generic classifications and
occupations remain metadata-only. A local-corpus association can generate a
clearly labeled discovery candidate only when its complete statistical evidence
contract passes support, recency, PMI, concentration, and phrase-specificity
requirements. Weaker associations remain corroborating evidence only.

Relationship selection is coverage- and diversity-aware: usable one-hop facts
are preferred over repetitive or distant graph paths before the bounded profile
limit is applied. Multi-part queries must bridge distinct query components, and
named-entity-plus-topic queries retain the named-entity anchor. Exact canonical
subjects use phrase-aware title centrality; ambiguous aliases and multi-hop paths
remain stricter. BGE-M3 checks all deterministic candidates. If every candidate
misses the normal semantic threshold, only strongly grounded one-hop or
statistically supported candidates above a lower recovery floor may appear in an
`Evidence-backed exploration` tier. Confirmed primary refined-search matches are
excluded, while similar-spelling review matches are not automatically removed.
The diagnostics expose the rejection funnel and never treat semantic similarity
as factual authentication. The tab uses no Neo4j service and does not alter the
existing Gemini Related Stories flow.

## Optional OpenSearch refined search

The `Search Results refined` tab is the primary title-search view. It automatically
ranks exact phrases, exact keywords, stemmed/root variants, curated aliases, and
fuzzy spelling matches. Known aliases use a query-time
OpenSearch `synonym_graph` and are included in primary traffic totals.
Similar-spelling matches are shown for review but are not included in the primary
traffic totals.

The reviewed alias registry is in `src/search_aliases.py` and is organized by
geography, government, politics, public offices, news acronyms, alternate
spellings/transliterations, and organizations/companies. Aliases shared by more
than one entity, such as `AP` and `MP`, are not expanded automatically; the tab
asks the user to enter the intended full form instead.

For a local development service:

```powershell
docker compose -f compose.opensearch.yml up -d
streamlit run app2.py
```

Open the refined tab and select `Sync OpenSearch refined index` once. The sync
creates a versioned physical index and switches only the refined-search alias;
it does not modify the existing search data or behavior. The app automatically
loads local connection options from the ignored `.env` file. Supported options
are listed in `.env.opensearch.example`; explicitly exported environment
variables take precedence.

### Native Windows installation

For the local archive at `C:\Users\rishabh.adhikari_jag\opensearch-3.8.0`,
start OpenSearch and the configured Streamlit app together with:

```powershell
.\start-native-opensearch.ps1
```

The launcher binds OpenSearch only to `127.0.0.1`, uses its bundled JDK, and
disables the Security plugin for local development. It does not modify the
OpenSearch installation's `opensearch.yml` file.

Alternatively, set `OPENSEARCH_AUTO_START=true` and `OPENSEARCH_HOME` in the
project `.env` file. With those local-only options enabled, the refined-search
tab starts OpenSearch in the background when needed, so the normal command is
enough:

```powershell
streamlit run app2.py
```

### Semantic related titles

The separate `Semantic Related Titles` tab uses a cached factual relationship
graph and a single batched OpenSearch request over related subjects and required
title cues. Normal searches make no Gemini calls and do not load an embedding
model. Select `Sync relationship title index` once and use `Fast OpenSearch
relationship search` for the interactive path.

If no graph exists, `Build/refresh relationship graph` explicitly runs Gemini
with Google Search grounding and saves the result for subsequent searches. It
does not perform the old candidate-by-candidate Gemini sweep. The complete
`Search Results refined` story-ID set is excluded inside the OpenSearch request
and checked again before traffic is calculated, so the two tabs cannot overlap.

### Semantic SEO Lab

The separate `Semantic SEO Lab` tab is an opt-in title-only semantic pipeline. It
does not change the indexes, caches, session state, or controls used by the
existing tabs. It uses its own OpenSearch alias (`traffic-semantic-seo-lab`) and
SQLite route store (`data/semantic_seo/semantic_seo_profiles.db`).

The one-time/incremental ingestion flow is:

```text
title -> GLiNER entity/event extraction -> deterministic canonical entity IDs
      -> BGE-M3 vector -> isolated OpenSearch index
```

Normal searches use the low-latency path:

```text
OpenSearch direct-result IDs (exclusion only)
-> direct BGE-M3 query route + any already cached indirect routes
-> ingestion-time GLiNER entity compatibility (no online GLiNER inference)
-> BGE cross-encoder reranking of at most 20 titles
-> results
```

Gemini route enrichment and final validation are both explicit optional controls;
neither blocks a normal search. DeBERTa is available as an optional soft
title-relevance score over at most ten candidates. It does not claim to validate
article evidence and never hard-filters an indirect title.

Vertex does not support Google Search and controlled generation in the same
request. Route creation therefore runs two grounded evidence calls concurrently,
then a separate schema-controlled synthesis call without Search. Local validation
requires URLs from both grounding branches and two distinct sources before the
profile is saved; incomplete responses never replace the cached valid profile.
The controlled profile is capped at four complete routes. Gemini selects one
schema-constrained source ID from each grounding branch, and the application
resolves those IDs to the exact Vertex grounding URLs before validation.

All model loads are lazy and local-only. Stage these checkpoints once before
using the lab:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS='1'
.\.venv\Scripts\hf.exe download BAAI/bge-m3
.\.venv\Scripts\hf.exe download urchade/gliner_medium-v2.1
.\.venv\Scripts\hf.exe download microsoft/deberta-v3-base
.\.venv\Scripts\hf.exe download BAAI/bge-reranker-v2-m3
.\.venv\Scripts\hf.exe download MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

On the current CPU-only Windows host, the earlier BGE-M3-only build for 27,235
valid titles took about 56 minutes. The enriched build also performs GLiNER, so
it must be treated as an offline ingestion job. Extracted entities are cached by
model and normalized title in `data/semantic_seo/title_entities.db`; unchanged
titles are not sent through GLiNER again. GLiNER uses its
PyTorch checkpoint in this process; its unused ONNX path is disabled on Windows
to avoid an ONNX Runtime/PyTorch native DLL load-order conflict.

The exports are title-only. DeBERTa therefore runs only in optional title-
relevance mode, not as an evidence gate. mREBEL remains disabled because
relation extraction from headline fragments is unreliable and its published
checkpoint is non-commercial.

Optional overrides are `SEMANTIC_SEO_OPENSEARCH_INDEX`,
`SEMANTIC_SEO_EMBEDDING_MODEL`, `SEMANTIC_SEO_GLINER_MODEL`,
`SEMANTIC_SEO_RERANKER_MODEL`, `SEMANTIC_SEO_NLI_MODEL`, and
`SEMANTIC_SEO_RELATION_MODEL`.


## Wikidata related stories in app2.py

Open **Wikidata** in `app2.py` and click **Find Wikidata Related Stories**.
The query resolves to a Wikidata subject using English/Hindi names and aliases.
Missing subjects and their supported relationships up to two hops are downloaded automatically
through the public Wikidata API and saved in `data/wikidata/entities.sqlite3`.
Subsequent searches reuse local data; **Refresh saved Wikidata facts** explicitly
updates it. No LLM, Google News, paid service, or API key is used by this tab.

Discovery includes both `query → related entity` and
`query → intermediate entity → related entity` paths. It follows only outgoing
allowed Wikidata properties and explicit two-hop policy templates, never a third
hop or an inferred reverse edge. Unsupported property combinations are withheld
before downloading second-hop endpoints.
Business discovery includes ownership (P1830), affiliation (P1416), company
leadership (P169/P488), relatives (P1038), and recorded professional/sports
partners (P1327), alongside the existing family and company properties.
These statements do not imply personal friendship.
The same identity checks apply to the final subject at either depth. The result
table and CSV include **Hops**, the full relationship path, its entity IDs, and
source URLs for every edge. Equally strong identity matches prefer shorter paths.
Dates remain attached to their respective edges; a two-hop path does not imply
that both relationships were current simultaneously or that a new direct fact
can be inferred between the endpoints.

Article admission is governed by
[`src/wikidata_relationship_policy.py`](src/wikidata_relationship_policy.py),
using the relationship families and scope rules in the `vertex_related.py`
research prompt. No prompt or LLM is executed. The initial policies cover the
existing Wikidata properties; they do not add causes, consequences, current-news
facts, or missing query-type coverage to the local knowledge source.

- Shared-event paths require a verified event name/alias **and its edition year**
  in the title. A year must be part of, or immediately beside, the matched event
  phrase. Event dates come from cached labels or explicit date claims. Missing
  date evidence, unknown aliases, and ambiguous scope names are withheld.
- Institutional bridges require the connecting institution and a supported
  directional expression such as `Coastal Club owner Delta Holdings`. Mentioning
  only `Delta Holdings`, or merely mentioning both names, does not qualify.
- Work-specific and family-network paths require the intermediate work or person.
- Four explicit corporate-chain patterns permit standalone endpoint coverage:
  subsidiary/subsidiary, owner-of/subsidiary, parent/parent, and owned-by/owned-by.
- First-order standalone subjects are explicitly configured. Event participants,
  creative-work contributors, education, awards, and geographical context require
  the source subject in the title. When that subject is the searched query, the
  direct-match exclusion still wins; many such routes will have no related titles.
- End-date or point-in-time qualifiers on non-event edges additionally require
  the source entity and an evidenced adjacent year. Traffic months are never
  used as relationship dates. This conservative rule can withhold valid historical
  coverage, and absent qualifiers do not prove a relationship is current.

For `Mukesh Ambani -> WEF Annual Meeting 2018 -> Donald Trump`, a Trump/Iran
headline is rejected. A title naming Trump and the exact 2018 event can qualify;
the path does not establish that the two people met or collaborated. The result
table and CSV expose the relationship family, standalone policy, matched title
context, acceptance condition, and rejection rule. Relationship failures and
identity failures have separate counts and inspectable reasons. Only routes
passing both gates contribute to results, view totals, or export.

These are deterministic title heuristics, not semantic proofs or measured
precision guarantees. Known names/aliases remain limited by source coverage;
institutional expressions currently cover narrow English phrasing. The pipeline
version invalidates existing Streamlit result state when this policy changes;
saved raw Wikidata facts are reused. See
[`WIKIDATA_RELATIONSHIP_POLICY.md`](WIKIDATA_RELATIONSHIP_POLICY.md) for maintenance
and validation details.

Two-hop expansion excludes classification/birthplace edges at both steps,
self-links, cycles back to the query, duplicate statements, and endpoints already
reachable directly. It is bounded to 25 eligible paths per first-hop route,
200 distinct second-hop endpoints, and 500 second-hop paths, with round-robin
allocation across intermediate routes. Limits are deterministic and partial
coverage is reported. Missing second-hop entities in older caches download on
the next search; failed updates retain the previous complete stored data.

Sync **Search Results refined** first. The Wikidata tab excludes its complete
primary result set across all pages, including confirmed aliases and root variants.
Similar-spelling-only review IDs are not automatically excluded; they still need
a Wikidata-supported subject match. Titles containing any meaningful normalized
query word, the resolved subject's full identity aliases, or the same normalized
title as an excluded primary story are removed before matching and traffic reporting.
Shared words inside a complete related name are exempt from the query-word check:
`Nita Ambani` can qualify for `Mukesh Ambani`, subject to identity validation.
Confirmed refined primary IDs remain excluded even in this case. With **Any keyword**,
the refined tab can classify shared-surname titles as primary; use **All keywords**
when the intended direct search is the complete name.
The result table and CSV export use the same filtered story set.

Related-name matches now pass an entity identity check before ranking. Each
accepted match retains the target Wikidata Q-ID, the exact matched name/alias,
and its identity evidence. A longer overlapping competing name (for example,
`Maruti Suzuki` instead of Hanuman's alias `Maruti`) blocks that interpretation.
Single-word aliases and known ambiguous names require at least two distinct
supporting context terms from Wikidata descriptions or selected entity-type and
relationship labels, with more support than competing interpretations. The
matched name itself does not count as context. Full multiword labels can qualify
without extra context when no competing match is known; multiword aliases need
a completed identity lookup or sufficient supporting context. These are
conservative heuristics, not proof of uniqueness or article-body meaning.
A single-word canonical label can also qualify with one independent context term
after a successful identity lookup when no competing name match is known.
Standard company legal suffixes may be omitted while retaining a multiword name,
so `Reliance Industries Limited` can match `Reliance Industries`.
Industry (P452) and product/service (P1056) labels enrich identity context;
these properties do not expand discovery to all products or industry peers.

Identity lookups search only names that occur in remaining candidates and cache
candidate entity metadata and search results in separate SQLite tables. Names,
not corpus headlines or traffic, are sent to Wikidata. Searches return at most
10 entity candidates each, with at most 12 uncached name searches and a 30-second
network budget per run. Available English/Hindi descriptions and up to 40
context targets enrich identity evidence. Identity metadata/search caches are
limited to entities actually matched in candidate titles. These caches are
rechecked after seven days; the explicit refresh control also refreshes them.
Unavailable or incomplete lookups never establish that a name is unique.
This bounded discovery cannot enumerate every possible competing entity.
Failures for one name do not skip independent name checks; failed lookups are
never cached as successful, and the same total time/request limits still apply.

The tab reports withheld-story counts and up to 500 rejected relationship
matches with reasons. A story rejected through one route can still qualify
through a different accepted route. Only accepted routes contribute to its
explanation and score; relationship confidence and popularity cannot rescue
a failed identity check. Results, CSV exports, and traffic use the accepted set.
Weak titles may therefore disappear even when their underlying subject really
is related; title-only evidence limits recall.

Multiword queries are resolved as a complete name or concept, not independently
expanded into unrelated entities. Users enter ordinary names; there is no Q-ID
override field. All matching namesake entities are searched independently, and
their accepted articles are combined. An article's views are counted once even
when several entities support it; results and CSV exports retain all accepted
originating entities and relationship paths. A namesake with no eligible routes
does not suppress results from the others.
Unresolved compound queries need a more specific
subject; the flow does not invent event or causal connections. Downloaded targets
can also be searched locally; missing relationship targets are reported as incomplete
coverage. The source details show the stored entity count and download date.
When exact name/alias resolution fails, candidates differing by only
one character edit in one word of a multiword name can resolve automatically.
All other query words must match, and the correction is displayed. This does not
enable fuzzy article-title retrieval. Empty results distinguish missing eligible
relationships, no remaining name matches after exclusions, and failed relationship
or identity checks. A classification-only subject such as the inspected Amavasya item cannot
gain discovery routes merely by increasing hop depth.

Name lookups include every exact locally cached name/alias match and every match
validated in the bounded API search (10 initial hits, with up to two additional
10-hit component searches for spelling recovery). This is not exhaustive Wikidata
enumeration, and a raw search suggestion alone does not establish a name match.
The additive `query_resolutions` SQLite table stores the full resolved entity set;
legacy single-entity caches are upgraded on their first online name lookup.

This is a growing local subset, not a complete offline Wikidata mirror. Initial
downloads and explicit refreshes require connectivity and can encounter API rate
limits. Updates commit atomically after all required requests succeed. Failed
updates preserve the previous complete data; the UI shows its download date.
Relationship profiles have no automatic expiry or background refresh. The separate
identity caches follow the seven-day policy described above.

Sports-team memberships, film/music credits, family and organizational relations
are supported. Deprecated statements are ignored; team memberships are described
as current or historical and available time qualifiers are displayed. Direct
concept classifications and birthplace context remain inspectable but do not admit
articles into this tab's results. No reverse
class or team-roster inference is performed; only explicitly stored allowed
outgoing relationships can form the second step. Relationship target
aliases are matched against corpus titles. Scores are heuristics, not probabilities.

Results show the matched related subject, relationship explanation, Wikidata source
link, total historical views, and peak traffic month. The graph establishes a subject
relationship, not the contents of the unavailable article body. Shared title matching
helpers are reused without initializing a DBpedia network client. Existing other
research tabs and their exclusion policies are unchanged.

Optional environment settings:

- `WIKIDATA_LOCAL_DB`: path to the local SQLite database.
- `WIKIDATA_API_URL`: update endpoint (default `https://www.wikidata.org/w/api.php`).
- `SOURCE_RELATED_TIMEOUT_SECONDS`: update request timeout, default 15 seconds.
- `SOURCE_RELATED_USER_AGENT`: identifying User-Agent for update requests.

Wikidata structured data is CC0. See the official
[Wikidata API documentation](https://www.mediawiki.org/wiki/Wikibase/API) and
[data licensing](https://www.wikidata.org/wiki/Wikidata:Licensing).
# BERTopic refined

In `app2.py`, open **BERTopic refined** and click **Run BERTopic refined** after syncing
the refined OpenSearch index to the current title corpus. The first run fits an
independent BGE-M3 / UMAP / HDBSCAN topic model in the background. Subsequent queries
reuse artifacts in `data/bertopic_refined/<corpus-and-model-fingerprint>/`; changed
story IDs or titles trigger a fresh fit. BGE-M3 must be cached locally or downloadable
on the first run.

Results exclude every story ID matched by refined search, including its similar
spelling tier. Distinct IDs with identical or similarly spelled titles remain
eligible. Titles containing **any** normalized query or safe known-alias token are
removed before ranking. The tab shows removal counts, selected-topic cosine scores,
the outlier ratio, and a catalog of topic labels and sizes.

Topic selection settings expose maximum topics (default 5) and minimum cosine
similarity (default 0.45). This threshold is provisional: full-corpus coherence,
manual editorial topic review, and unrelated-query calibration remain necessary
before treating these settings as validated. No fallback bypasses the filters.

Run focused verification with:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_bertopic_refined.py
```
