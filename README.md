# Football News Similarity and Storyline Pipeline

This repository contains a compact NLP workflow for football news:

1. retrieve top-k similar articles
2. decide whether incoming news should be published or skipped
3. build storyline clusters, summarize updates, extract entities, and generate a quick markdown overview

## Data Files

- `data/news.csv`: base published news dataset
- `data/incoming_news.json`: incoming batch used for publish/skip and clustering
- `data/incoming_news_added.json`: additional incoming batch used for incremental ingest demo
- `data/entity_name_db.json`: curated club/player names with aliases for regex-based entity extraction
- `example_output/`: reproducible sample outputs from pipeline runs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For LLM-based scripts:

```bash
export OPENAI_API_KEY="your_openai_key"
```

## Scripts, Outputs, and Design Decisions

### `task_1.py` (Retrieval)

Example run:

```bash
python task_1.py --query "transfer news about PSG" --k 1 --method bm25
```

Saves:

- `output/task1_results.json`
- per run: `query`, `method`, `k`, `results`
- per result row: `index`, `score`, `text`

Logic:

- scores all news articles with selected method (`dense`, `tfidf`, or `bm25`)
- returns top `k` rows sorted by score

Reasoning:

- provides multiple retrieval baselines in one interface

### `task_2.py` (Publish/Skip via Similarity + Novelty)

Run:

```bash
python task_2.py
```

Saves:

- `output/task2_results.json`
- per row: `incoming_text`, `similarity`, `novelty`, `decision`, `best_match_index`, `best_match_text`

Logic:

- for each incoming article, find the closest already-indexed article (dense similarity)
- compute novelty using informative tokens (not stopwords, length > 2, not numeric)
- decision rule:
  - `skip` if `similarity >= duplicate_thresh` and `novelty < novelty_thresh`
  - else `publish`
- if `publish`, append article to references for next comparisons

Reasoning:

- simple, deterministic baseline with no external model call

Tradeoff:

- limited semantic sensitivity (token novelty can stay low while meaning still changes)

### `task_2_alt_LLM.py` (LLM as Judge)

Run:

```bash
python task_2_alt_LLM.py
```

Saves:

- `output/task2_alt_llm_results.json`
- per row: `incoming_text`, `best_match_index`, `best_match_text`, `llm_decision`, `reason`, `llm_raw_output`

Logic:

- same retrieval step as task 2
- prompt an LLM with:
  - closest article
  - incoming article
  - similarity + novelty scores
  - rule-based judging instructions
- expects JSON output with `decision` and `reason`

Reasoning:

- added to better capture semantic progression where the baseline can fail

### Task 3 Feature: Storyline Tracking

Feature goals:

- Storylines: group related transfer/injury/club articles into a single thread
- Change tracking: keep update history as new items arrive
- Entity context: attach clubs/players per story
- Quick digest: generate readable summaries and overview markdown

### `task_3.py` (Storyline Clustering)

Run:

```bash
python task_3.py
```

Saves:

- `output/task3_clusters.json`
- per story: `story_id`, `size`, `articles`
- per article in story: `index`, `source`, `text`

Logic:

- process articles in order
- find best dense match against already-clustered articles
- if best score `< cluster_threshold`, create new story
- otherwise, assign to matched story

Reasoning:

- lightweight online clustering aligned with ingestion flow

### `task_3_summarize.py` (Story Summarization + Update Tracking)

Run:

```bash
python task_3_summarize.py
```

Saves:

- `output/task3_clusters_summarized.json`
- per story adds:
  - `story_summary`
  - `whats_new_by_article`
- per `whats_new_by_article` row:
  - `news_id`, `source`, `whats_new`, `updated_story_summary`

Logic:

- first article initializes story summary
- each next article produces a `whats_new` assessment
- summary is then updated with the new information

Reasoning:

- provides a concise evolving digest with traceability of updates

### `task_3_db_NER.py` (DB + Regex Entity Extraction)

Run:

```bash
python task_3_db_NER.py
```

Saves:

- `output/task3_db_NER.json`
- `output/task3_db_NER_as_df.csv`
- per story: `story_id`, `entities`
- per entity: `entity_type`, `entity_name`, `news_ids`

Logic:

- build regex matchers from `data/entity_name_db.json` canonical names + aliases
- scan story articles and aggregate matched entities per story

Reasoning:

- deterministic and easy to debug for known football entities

### `task_3_overview.py` (Markdown Overview)

Run:

```bash
python task_3_overview.py
```

Output:

- `output/task3_overview.md`
- `output/task3_overview_together.md` (when used on combined state)

Logic:

- merges summarized stories with NER output
- ranks stories by update signal and size
- writes a scan-friendly markdown table with metrics and previews

### `task_3_together.py` (End-to-End Incremental Ingest)

Run:

```bash
python task_3_together.py --incoming_json data/incoming_news_added.json
```

Saves:

- `output/task3_all_results.json`
- `output/task3_overview_together.md`
- per story includes clustered `articles`, `story_summary`, `whats_new_by_article`, `entities`

Logic:

- combines base files (`news.csv` + `incoming_news.json`) with additional incoming file
- runs clustering + summarization + DB/regex entities in one flow
- generates final json state + markdown overview

Reasoning:

- demonstrates an operational "ingest and update" mode

## Practical Run Order

```bash
python task_1.py
python task_2.py
python task_2_alt_LLM.py
python task_3.py
python task_3_summarize.py
python task_3_db_NER.py
python task_3_overview.py
python task_3_together.py --incoming_json data/incoming_news_added.json
```

## Assumptions and Tradeoffs

- dense retrieval is reused across clustering/judging flows; lexical methods are mainly exposed in task 1
- entity extraction is DB-driven and depends on alias coverage
- LLM steps improve semantic judgment but add cost and can introduce output variability
- generated outputs are written to `output/` (ignored by git); reproducible examples are in `example_output/`
