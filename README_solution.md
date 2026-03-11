# README Solution

## Data Files Added
- `data/incoming_news_added.json`: extra batch of incoming news used in (`task_3_together.py`).
- `data/entity_name_db.json`: curated clubs and players with aliases for regex-based entity extraction.

## Setup
Within venv
```bash
pip install -r requirements.txt
```

For LLM-based scripts, set OpenAI token:
```bash
export OPENAI_API_KEY="your_openai_key"
```

## How to Run & Design Decisions

### `task_1.py` (Task 1 retrieval)

Example run:
```bash
python task_1.py --query "transfer news about PSG" --k 1 --method bm25
```
Saves:
- `output/task1_results.json` 
- Per run entry fields: `query`, `method`, `k`, `results`
- `results`: `index`, `score`, `text`

Method:
- Scores all articles with selected method (`dense`, `tfidf` or `bm25`) and keeps top `k`.

### `task_2.py` (Task 2 publish/skip via similarity & novelty scores)
```bash
python task_2.py
```
Saves:
- `output/task2_results.json`
- `results`: `incoming_text`, `similarity`, `novelty`, `decision`, `best_match_index`, `best_match_text`

Logic:
- For each incoming article, finds closest already-indexed article
- Computes novelty using informative tokens (not stopword, length > 2, not numeric).
- `skip` if `similarity >= duplicate_thresh` and `novelty < novelty_thresh`, else `publish`.
- If `publish`, article is appended to references for next comparisons.

Baseline
- In practice it is limited for semantic meaning, because token novelty can stay low while meaning changes.

### `task_2_alt_LLM.py` (Task 2 alternative: LLM-as-a-judge)
```bash
python task_2_alt_LLM.py
```
Saves:
- `output/task2_alt_llm_results.json`
- `results`: `incoming_text`, `best_match_index`, `best_match_text`, `llm_decision`, `reason`, `llm_raw_output`

Logic:
- As before, retrieves closest reference article.
- Sends incoming article + closest article to GPT-4o.
- Expects strict JSON: `decision` and `reason`.

Reasoning:
- Added because Task 2 baseline struggled with semantic update that is important (transfer status change).

## Task 3 Feature
- Storylines: related transfer/injury/club articles are grouped into one thread.
- Change tracking: new incoming items are added as updates to existing stories or as new stories.
- Entity context: clubs and players are attached per story for quick filtering.
- Quick digest: summaries + overview markdown make stories easier to scan than full article text.

### `task_3.py` (Storyline Clustering)
```bash
python task_3.py
```
Saves:
- `output/task3_clusters.json`
- `story_id`, `size`, `articles`
- `articles` includes `index`, `source`, `text`

Logic:
- For each article, gets best dense match against already-clustered articles.
- If best score `< cluster_threshold`, creates a new story. Otherwise, assigns to matched story.

Reasoning:
- Reused the existing similarity approach from task 1/2.
- Simple clustering, with ingestion-like flow.

### `task_3_summarize.py` (Task 3 story summarization)
```bash
python task_3_summarize.py
```
Saves:
- `output/task3_clusters_summarized.json`
- Keeps cluster structure and adds:
- Per story: `story_summary`, `whats_new_by_article`
- Per `whats_new_by_article`: `news_id`, `source`, `whats_new`, `updated_story_summary`

Logic:
- For each story:
- First article creates initial `story_summary`.
- Each next article generates `whats_new` against current summary
- Then updates story summary with that new information.

Reasoning:
- Quick Digest: one evolving story summary instead of reading all articles.
- Change tracking: `whats_new` keeps traceability of story updates.

### `task_3_db_NER.py` (Task 3 entities)
```bash
python task_3_db_NER.py
```
Saves:
- `output/task3_db_NER.json`
- `output/task3_db_NER_as_df.csv`
- Per story: `story_id`, `entities`
- Per entity: `entity_type`, `entity_name`, `news_ids`
- CSV columns: `story_id`, `entity_type`, `entity_name`, `entity_name_lower`, `news_ids` 

Logic:
- Builds regex matches from `data/entity_name_db.json`.
- Scans each story article text for club/player matches.

Reasoning:
- Chose DB+regex NER for simplicity and accuracy, assumes a DB with club and player names exists.
- Also tried NER model `dslim/bert-base-NER` via huggingface, but with disappointing results

### `task_3_overview.py` (Generates a markdown overview)
```bash
python task_3_overview.py
```
Output:
- `output/task3_overview.md` / `output/task3_overview_together.md` 

Logic:
- Selects stories with "meaningful" updates first:
- Sorted by `updated_articles_count`, then `article_count`, then `entity_count`.

Reasoning:
- Built for an easy overview of the Storyline results.

### `task_3_together.py` (Incremental ingest from new incoming news)
```bash
python task_3_together.py
```
Saves:
- `output/task3_all_results.json`
- Per story contains clustered `articles`, `story_summary`, `whats_new_by_article`, `entities`
- `output/task3_overview_together.md`

Logic:
- Combines base files (`news.csv` + `incoming_news.json`) with additional incoming file (`data/incoming_news_added.json`).
- Runs clustering with summarization and regex based NER

Reasoning:
- End-to-end operational mode: ingest, add to groups, summarize, retrieve entities, and present.

## Practical Run Order
If you want all outputs listed in `example_outputs`:
1. `python task_1.py`
2. `python task_2.py`
3. `python task_2_alt_LLM.py`
4. `python task_3.py`
5. `python task_3_summarize.py`
6. `python task_3_db_NER.py`
7. `python task_3_overview.py`
8. `python task_3_together.py --incoming_json data/incoming_news_added.json`

## Assumptions and Tradeoffs
- Dense retrieval is used in clustering/judging flows; lexical methods only in Task 1.
- Entity extraction is database-driven rather than model-driven (assumes that DBs is correct and complete).
- LLM calls introduce chance of hallucination
