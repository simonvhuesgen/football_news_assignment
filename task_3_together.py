

import argparse
from pathlib import Path

from helper import load_incoming_articles, load_news_articles, load_json, resolve_path, save_json
from helper_llm import get_token, make_llm_call
from task_3 import build_cluster_doc, build_story_clusters, combine_articles
from task_3_db_NER import build_matchers, extract_story_entities
from task_3_overview import write_markdown
from task_3_summarize import process_story


def ingest_results(articles, assignments, news_count, base_incoming_count):
    start_idx = int(news_count + base_incoming_count)
    story_by_article = {int(row.get("article_index", -1)): int(row.get("story_id", -1)) for row in assignments}

    story_has_old = {}
    for row in assignments:
        story_id = int(row.get("story_id", -1))
        i = int(row.get("article_index", -1))
        if story_id not in story_has_old:
            story_has_old[story_id] = False
        if i < start_idx:
            story_has_old[story_id] = True

    out = []
    for pos, _ in enumerate(articles[start_idx:]):
        article_i = start_idx + pos
        story_id = story_by_article.get(article_i, -1)
        action = "added_to_existing_story" if story_has_old.get(story_id, False) else "created_new_story"

        out.append(
            {
                "incoming_position": pos,
                "already_in_cluster": False,
                "action": action,
                "story_id": story_id,
                "news_id": article_i,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Ingest new news articles and update story clusters")
    parser.add_argument("--news_csv", type=Path, default=Path(__file__).resolve().parent / "data/news.csv")
    parser.add_argument("--base_incoming_json", type=Path, default=Path(__file__).resolve().parent / "data/incoming_news.json")
    parser.add_argument("--incoming_json", type=Path, default=Path(__file__).resolve().parent / "data/incoming_news_added.json")
    parser.add_argument("--entity_db", type=Path, default=Path(__file__).resolve().parent / "data/entity_name_db.json")
    parser.add_argument("--cluster_threshold", type=float, default=0.75)
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_token_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output/task3_all_results.json")
    parser.add_argument("--markdown_output", type=Path, default=Path(__file__).resolve().parent / "output/task3_overview_together.md")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    news_path = resolve_path(args.news_csv, here)
    base_incoming_path = resolve_path(args.base_incoming_json, here)
    incoming_path = resolve_path(args.incoming_json, here)
    entity_db_path = resolve_path(args.entity_db, here)
    output_path = resolve_path(args.output, here)
    markdown_path = resolve_path(args.markdown_output, here)

    openai_token = get_token(args.openai_token_env)
    llm_call, selected_model = make_llm_call(args.openai_model, openai_token)

    news_articles = load_news_articles(news_path)
    base_incoming_articles = load_incoming_articles(base_incoming_path)
    new_incoming_articles = load_incoming_articles(incoming_path)

    articles = combine_articles(news_articles, base_incoming_articles) + [
        {"source": incoming_path.name, "text": str(text)} for text in new_incoming_articles
    ]

    raw_stories, assignments = build_story_clusters(articles, args.cluster_threshold)
    cluster_state = build_cluster_doc(articles, raw_stories, assignments, cluster_thresh=args.cluster_threshold)
    stories = [process_story(story, llm_call) for story in cluster_state.get("stories", [])]

    entity_db = load_json(entity_db_path)
    club_matchers = build_matchers(entity_db.get("clubs", []))
    player_matchers = build_matchers(entity_db.get("players", []))
    total_entities = 0
    for story in stories:
        story["entities"] = extract_story_entities(story, club_matchers, player_matchers).get("entities", [])
        total_entities += len(story["entities"])

    out_state = {
        "mode": "update_with_new_incoming",
        "incoming_ingest_file": str(incoming_path),
        "cluster_threshold": float(args.cluster_threshold),
        "summary_provider": "openai",
        "summary_model": selected_model,
        "total_articles": len(articles),
        "total_stories": len(stories),
        "total_entities": total_entities,
        "ingest_results": ingest_results(
            articles=articles,
            assignments=cluster_state.get("assignments", []),
            news_count=len(news_articles),
            base_incoming_count=len(base_incoming_articles),
        ),
        "stories": stories,
    }

    save_json(out_state, output_path)
    write_markdown(out_state, output_path, markdown_path)

    print("Ingest new news articles and update story clusters")
    print("Total Stories:", out_state["total_stories"])
    print("Total Articles:", out_state["total_articles"])
    print("Total Entities:", out_state["total_entities"])
    print("See JSON with all results at:", output_path)
    print("See Markdown with quick overview at:", markdown_path)


if __name__ == "__main__":
    main()
