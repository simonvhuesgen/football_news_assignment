


from helper import Retriever, load_news_articles, resolve_path, save_json
import argparse
from pathlib import Path


def get_topk(news_csv, query, k=1, method="dense"):
    ret = Retriever(news_csv)

    if method == "tfidf":
        scores = ret.score_tfidf(query)
    elif method == "bm25":
        scores = ret.score_bm25(query)
    else:
        scores = ret.score_dense(query)

    results = ret.top_k_rows(scores, k)

    ret_dict = {
        "query": query,
        "method": method,
        "k": k,
        "results": results
        }
    return ret_dict


def main():
    parser = argparse.ArgumentParser(description="Task 1")
    parser.add_argument("--news_csv", type=str, default=Path("./data/news.csv"))
    parser.add_argument("--query", type=str, default="transfer news about PSG")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--method", type=str, choices=["dense", "tfidf", "bm25"], default="bm25")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output/task1_results.json")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    news_path = resolve_path(args.news_csv, here)
    output_path = resolve_path(args.output, here)
    articles = load_news_articles(news_path)

    result = get_topk(
        news_csv=articles,
        query=args.query,
        k=args.k,
        method=args.method,
    )

    save_json(result, output_path, append=True)

    print("Query used:", result["query"], " with method: ", result["method"])
    print("Best found match:", result["results"][0]["text"])
    print("See full results at:", output_path)


if __name__ == "__main__":
    main()
