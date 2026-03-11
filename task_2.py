

from helper import best_dense_match, load_news_articles, load_incoming_articles, novelty_score, resolve_path, save_json
import argparse
from pathlib import Path


def decide(similarity, novelty, duplicate_thresh, novelty_thresh):
    if similarity >= duplicate_thresh and novelty < novelty_thresh:
        return "skip"
    return "publish"


def main():
    
    parser = argparse.ArgumentParser(description="Task 2 using Similarity and Novelty")
    parser.add_argument("--news_csv", type=str, default=Path("./data/news.csv"))
    parser.add_argument("--incoming_json", type=str, default=Path("./data/incoming_news.json"))
    parser.add_argument("--duplicate_thresh", type=float, default=0.75)
    parser.add_argument("--novelty_thresh", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output/task2_results.json")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    news_path = resolve_path(args.news_csv, here)
    incoming_path = resolve_path(args.incoming_json, here)
    output_path = resolve_path(args.output, here)

    references = load_news_articles(news_path)
    incoming = load_incoming_articles(incoming_path)

    decisions = []
    for article in incoming:
        match = best_dense_match(references, article)
        best_index = int(match.get("best_match_index", -1))
        best_similarity = float(match.get("best_similarity_score", 0.0))
        best_match = str(match.get("best_match_text", ""))

        novelty = novelty_score(article, best_match)
        action = decide(
            similarity=best_similarity,
            novelty=novelty,
            duplicate_thresh=args.duplicate_thresh,
            novelty_thresh=args.novelty_thresh,
        )

        decisions.append(
            {
                "incoming_text": article,
                "similarity": best_similarity,
                "novelty": novelty,
                "decision": action,
                "best_match_index": best_index,
                "best_match_text": best_match,
            }
        )

        if action == "publish":
            references.append(article)

    result = {
        "method": "dense",
        "duplicate_thresh": args.duplicate_thresh,
        "novelty_thresh": args.novelty_thresh,
        "results": decisions,
    }

    save_json(result, output_path)

    print("Decisions using Similarity and Novelty")
    publish_count = sum(1 for row in decisions if row["decision"] == "publish")
    skip_count = sum(1 for row in decisions if row["decision"] == "skip")
    print("Publish:", publish_count)
    print("Skip:", skip_count)
    print("See full results at:", output_path)


if __name__ == "__main__":
    main()
