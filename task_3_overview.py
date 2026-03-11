import argparse
from pathlib import Path

from helper import load_json, resolve_path


def clean_text(text, limit=260):
    return " ".join(str(text or "").split())[:limit]


def meaningful_updates(story):
    #get valuable updates filtering out "no materially new information"
    updates = []
    for row in story.get("whats_new_by_article", []):
        text = clean_text(row.get("whats_new", ""), 500)
        if text != "" and "no materially new information" not in text.lower():
            updates.append(text)
    return updates


def merge_summarized_and_ner(summarized_state, ner_state):
    ner_by_story = {int(row.get("story_id", -1)): row.get("entities", []) for row in ner_state.get("stories", [])}

    stories = []
    total_articles = 0
    total_entities = 0
    for story in summarized_state.get("stories", []):
        row = dict(story)
        story_id = int(row.get("story_id", -1))
        row["entities"] = ner_by_story.get(story_id, [])
        total_articles += len(row.get("articles", []))
        total_entities += len(row["entities"])
        stories.append(row)

    return {
        "total_stories": len(stories),
        "total_articles": int(summarized_state.get("total_articles", total_articles)),
        "total_entities": total_entities,
        "stories": stories,
    }


def prepare_md(state, top_k):
    updated_rows = []
    fallback_rows = []

    for story in state.get("stories", []):
        updates = meaningful_updates(story)
        names = []
        seen = set()
        for entity in story.get("entities", []):
            name = str(entity.get("entity_name", "")).strip()
            if name == "":
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
        if len(names) == 0:
            entities = "-"
        else:
            entities = ", ".join(names[:8])
            if len(names) > 8:
                entities += f" (+{len(names) - 8} more)"

        summary = clean_text(story.get("story_summary", ""), 260)
        if summary == "":
            articles = story.get("articles", [])
            if len(articles) > 0:
                summary = clean_text(articles[0].get("text", ""), 260)

        row = {
            "story_id": int(story.get("story_id", -1)),
            "meaningful_updates": len(updates),
            "article_count": len(story.get("articles", [])),
            "entity_count": len(story.get("entities", [])),
            "entities_preview": clean_text(entities, 220),
            "summary_preview": summary,
            "latest_whats_new": clean_text(updates[-1], 200) if len(updates) > 0 else "-",
            "selection_reason": "updated_story" if len(updates) > 0 else "new_story",
        }

        updated_rows.append(row)
        

    updated_rows.sort(
        key=lambda row: (
            -row["meaningful_updates"],
            -row["article_count"],
            -row["entity_count"],
            row["story_id"],
        )
    )
    fallback_rows.sort(
        key=lambda row: (
            -row["article_count"],
            -row["entity_count"],
            row["story_id"],
        )
    )

    selected = updated_rows[:top_k]
    if len(selected) < top_k:
        selected += fallback_rows[: top_k - len(selected)]

    return selected, len(updated_rows)



def write_markdown(state, source_label, md_path, top_k=5):
    top_rows, stories_with_updates = prepare_md(state, top_k)

    lines = ["# Storyline Overview", "", "## Overview Metrics", ""]
    lines.append(f"- total_stories: **{int(state.get('total_stories', 0))}**")
    lines.append(f"- total_articles: **{int(state.get('total_articles', 0))}**")
    lines.append(f"- total_entities: **{int(state.get('total_entities', 0))}**")
    lines.append(f"- stories with meaningful updates: **{stories_with_updates}**")
    lines.append(f"- source_data: `{source_label}`")
    lines.extend(["", "## Selected Stories sorted by meaningful updates", ""])

    if len(top_rows) == 0:
        lines.append("No stories with meaningful updates.")
    else:
        lines.append(
            "| story_id | selection_reason | meaningful_updates | article_count | entity_count | entities | summary_preview | latest_whats_new |"
        )
        lines.append("|---:|---|---:|---:|---:|---|---|---|")
        for row in top_rows:
            entities = row["entities_preview"].replace("|", " ")
            summary = row["summary_preview"].replace("|", " ")
            latest = row["latest_whats_new"].replace("|", " ")
            lines.append(
                f"| {row['story_id']} | {row['selection_reason']} | {row['meaningful_updates']} | "
                f"{row['article_count']} | {row['entity_count']} | {entities} | {summary} | {latest} |"
            )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Task 3 Markdown Overview")
    parser.add_argument("--input_json", type=Path, default=None)
    parser.add_argument(
        "--summarized_state",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task3_clusters_summarized.json",
    )
    parser.add_argument(
        "--ner_state",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task3_db_NER.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task3_overview.md",
    )
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    output_path = resolve_path(args.output, here)

    if args.input_json is not None:
        input_path = resolve_path(args.input_json, here)
        state = load_json(input_path)
        source_label = str(input_path)
    else:
        summarized_path = resolve_path(args.summarized_state, here)
        ner_path = resolve_path(args.ner_state, here)
        summarized_state = load_json(summarized_path)
        ner_state = load_json(ner_path)
        state = merge_summarized_and_ner(summarized_state, ner_state)
        source_label = f"summarized={summarized_path}, ner={ner_path}"

    write_markdown(state, source_label, output_path, top_k=max(1, int(args.top_k)))

    print("Storyline Overview")
    print("See markdown at:", output_path)


if __name__ == "__main__":
    main()
