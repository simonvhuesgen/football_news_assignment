

from helper import load_news_articles, load_incoming_articles, best_dense_match, resolve_path, save_json
import argparse
from pathlib import Path

def combine_articles(news, incoming):
    out = []
    for text in news:
        out.append({"source": "news.csv", "text": str(text)})
    for text in incoming:
        out.append({"source": "incoming_news.json", "text": str(text)})
    return out


def add_new_story(stories, index):
    story_id = len(stories)
    stories.append({"story_id": story_id, "article_indices": [index]})
    return story_id


def assign_story(indexed_articles, indexed_story_ids, article_text, cluster_thresh):
    match = best_dense_match(indexed_articles, article_text)

    if match["best_match_index"] < 0:
        matched_story_id = -1
    else:
        matched_story_id = int(indexed_story_ids[match["best_match_index"]])

    return {
        "best_match_index": match["best_match_index"],
        "best_similarity_score": match["best_similarity_score"],
        "matched_story_id": matched_story_id,
        "create_new_story": match["best_similarity_score"] < cluster_thresh,
    }


def build_story_clusters(articles, cluster_thresh):
    stories = []
    indexed_articles = []
    indexed_story_ids = []
    assignments = []

    for article_i, article in enumerate(articles):
        info = assign_story(
            indexed_articles=indexed_articles,
            indexed_story_ids=indexed_story_ids,
            article_text=article["text"],
            cluster_thresh=cluster_thresh,
        )

        if info["create_new_story"]:
            story_id = add_new_story(stories, article_i)
        else:
            story_id = info["matched_story_id"]
            stories[story_id]["article_indices"].append(article_i)

        indexed_articles.append(article["text"])
        indexed_story_ids.append(story_id)

        assignments.append(
            {
                "article_index": article_i,
                "source": article["source"],
                "story_id": story_id,
                "best_match_index": info["best_match_index"],
                "best_similarity_score": info["best_similarity_score"],
            }
        )

    return stories, assignments


def build_cluster_doc(articles, stories, assignments, cluster_thresh):
    out_stories = []
    for story in stories:
        rows = []
        for idx in story["article_indices"]:
            rows.append(
                {
                    "index": idx,
                    "source": articles[idx]["source"],
                    "text": articles[idx]["text"],
                }
            )

        out_stories.append(
            {
                "story_id": story["story_id"],
                "size": len(story["article_indices"]),
                "articles": rows,
            }
        )

    return {
        "mode": "story_clustering",
        "method": "dense",
        "cluster_threshold": cluster_thresh,
        "total_articles": len(articles),
        "total_stories": len(out_stories),
        "stories": out_stories,
        #"assignments": assignments,
    }


def main():
    parser = argparse.ArgumentParser(description="Generating clusters (storylines) from news articles")
    parser.add_argument("--news_csv", type=str, default=Path("./data/news.csv"))
    parser.add_argument("--incoming_json", type=str, default=Path("./data/incoming_news.json"))
    parser.add_argument("--cluster_threshold", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output/task3_clusters.json")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    news_path = resolve_path(args.news_csv, here)
    incoming_path = resolve_path(args.incoming_json, here)
    output_path = resolve_path(args.output, here)

    news_articles = load_news_articles(news_path)
    incoming_articles = load_incoming_articles(incoming_path)
    articles = combine_articles(news_articles, incoming_articles)

    stories, assignments = build_story_clusters(articles, args.cluster_threshold)
    state = build_cluster_doc(
        articles=articles,
        stories=stories,
        assignments=assignments,
        cluster_thresh=args.cluster_threshold,
    )

    save_json(state, output_path)

    print("Generating clusters (storylines) from news articles")
    print("Stories:", state["total_stories"])
    print("See full results at:", output_path)


if __name__ == "__main__":
    main()
