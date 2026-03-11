
import argparse
import re
from pathlib import Path
import pandas as pd

from helper import load_json, resolve_path, save_json


def normalize_spaces(text):
    value = str(text or "")
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def build_matchers(entries):
    #creating regex patterns
    out = []

    for entry in entries:
        canonical = normalize_spaces(entry.get("canonical", ""))
        aliases = entry.get("aliases", [])

        values = [canonical]
        for alias in aliases:
            values.append(normalize_spaces(alias))

        seen = set()
        patterns = []
        for value in values:
            if not value:
                continue

            key = value.lower()
            if key in seen:
                continue
            seen.add(key)

            pattern = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", flags=re.IGNORECASE)
            patterns.append(pattern)

        if canonical and len(patterns) > 0:
            out.append({"canonical": canonical, "patterns": patterns})

    return out


def find_matches(text, matchers):
    source = normalize_spaces(text)
    found = set()

    for item in matchers:
        for pattern in item["patterns"]:
            if pattern.search(source):
                found.add(item["canonical"])
                break

    return found


def extract_story_entities(story, club_matchers, player_matchers):
    #exctracting entities via re
    story_id = int(story.get("story_id", -1))
    articles = story.get("articles", [])

    entity_news_ids = {}

    for article in articles:
        news_id = int(article.get("index", -1))
        text = str(article.get("text", ""))

        clubs = find_matches(text, club_matchers)
        players = find_matches(text, player_matchers)

        for name in clubs:
            key = ("club", name)
            if key not in entity_news_ids:
                entity_news_ids[key] = set()
            entity_news_ids[key].add(news_id)

        for name in players:
            key = ("player", name)
            if key not in entity_news_ids:
                entity_news_ids[key] = set()
            entity_news_ids[key].add(news_id)

    entities = []
    for (entity_type, entity_name), ids in sorted(
        entity_news_ids.items(),
        key=lambda item: (item[0][0], item[0][1].lower()),
    ):
        entities.append(
            {
                "entity_type": entity_type,
                "entity_name": entity_name,
                "news_ids": sorted(ids),
            }
        )

    return {"story_id": story_id, "entities": entities}


def entity_overview(cluster_state, club_matchers, player_matchers):
    stories = cluster_state.get("stories", [])

    out_stories = []
    total_entities = 0

    for story in stories:
        row = extract_story_entities(story, club_matchers, player_matchers)
        out_stories.append(row)
        total_entities += len(row["entities"])

    return {
        "mode": "story_entities_db",
        "source_mode": cluster_state.get("mode", "story_clustering"),
        "total_stories": len(out_stories),
        "total_entities": total_entities,
        "stories": out_stories,
    }


def as_df(state):
    rows = []

    for story in state.get("stories", []):
        story_id = int(story.get("story_id", -1))
        for entity in story.get("entities", []):
            entity_type = str(entity.get("entity_type", ""))
            entity_name = str(entity.get("entity_name", ""))
            news_ids = entity.get("news_ids", [])

            rows.append(
                {
                    "story_id": story_id,
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "entity_name_lower": entity_name.lower(),
                    "news_ids": ",".join(str(x) for x in news_ids),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    return df.sort_values(["entity_type", "entity_name_lower", "story_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="NER from DB")
    parser.add_argument(
        "--cluster_state",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task3_clusters.json",
    )
    parser.add_argument(
        "--entity_db",
        type=Path,
        default=Path(__file__).resolve().parent / "data/entity_name_db.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task3_db_NER.json",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cluster_state_path = resolve_path(args.cluster_state, here)
    entity_db_path = resolve_path(args.entity_db, here)
    output_path = resolve_path(args.output, here)

    cluster_state = load_json(cluster_state_path)
    entity_db = load_json(entity_db_path)

    club_matchers = build_matchers(entity_db.get("clubs", []))
    player_matchers = build_matchers(entity_db.get("players", []))

    overview = entity_overview(cluster_state, club_matchers, player_matchers)

    save_json(overview, output_path)

    df = as_df(overview)
    df_path = output_path.with_name(output_path.stem + "_as_df.csv")
    df.to_csv(df_path, index=False, encoding="utf-8")

    print("NER from DB")
    print("Stories:", overview["total_stories"])
    print("Entities:", overview["total_entities"])
    print("See full results at:", output_path)
    print("and as csv at:", df_path)


if __name__ == "__main__":
    main()
