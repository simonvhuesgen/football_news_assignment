# Football News Similarity and Storyline Pipeline

This repository contains a compact NLP pipeline for football news:

1. retrieve similar articles (`task_1.py`)
2. decide publish vs skip for incoming items (`task_2.py`, `task_2_alt_LLM.py`)
3. build story clusters, summarize updates, extract entities, and generate a markdown overview (`task_3*.py`)

## Project Structure

- `data/news.csv`: base published news dataset
- `data/incoming_news.json`: incoming articles
- `data/incoming_news_added.json`: extra incoming batch for incremental ingest demo
- `data/entity_name_db.json`: canonical club/player names and aliases
- `task_1.py` ... `task_3_together.py`: runnable scripts
- `helper.py`, `helper_llm.py`: shared utilities
- `example_output/`: sample result artifacts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For scripts using LLM calls:

```bash
export OPENAI_API_KEY="your_openai_key"
```

## Run

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

## Notes

- Retrieval options in task 1: `dense`, `tfidf`, `bm25`
- LLM scripts use environment-based key loading (no key is stored in repo)
- Output files are written to `output/` (ignored by git); reproducible examples are in `example_output/`
