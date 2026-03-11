

import argparse
from pathlib import Path

from helper_llm import get_token, make_llm_call
from helper import load_json, resolve_path, save_json


def build_context(articles):
    rows = []
    for idx, article in enumerate(articles, start=1):
        rows.append(
            f"Article {idx} (news_id={int(article.get('index', -1))}, source={str(article.get('source', ''))}):\n"
            f"{str(article.get('text', ''))}"
        )
    context = "\n\n".join(rows).strip()
    if len(context) > 16000:
        context = context[:16000]
    return context


def ask_llm(kind, llm_call, story_context="", current_summary="", previous_context="", article_text="", whats_new="", full_context=""):
    if kind == "summary":
        prompt = (
            "You are an editor summarizing one football storyline.\n"
            "Write a factual story summary from the full context.\n"
            "Rules:\n"
            "- Use only facts present in the context.\n"
            "- Focus on outcomes, timeline changes, and transfer/injury status if mentioned.\n"
            "- Avoid repeating the same fact in different wording.\n"
            "- Keep it concise and readable.\n"
            "- Output exactly 3-5 bullet points.\n\n"
            f"Story context:\n{story_context}\n"
        )
    elif kind == "whats_new":
        prompt = (
            "You are an editor deciding what is NEW in a football article.\n"
            "Compare the new article against both current summary and previous context.\n"
            "Rules:\n"
            "- Return only facts that are materially new.\n"
            "- Do not restate old facts.\n"
            "- State progression counts as new (example: rumored -> talks -> agreement -> official).\n"
            "- Changes in fee, contract, timeline, injury return date, involved clubs/players also count as new.\n"
            "- If there is no material update, return exactly: No materially new information.\n"
            "- Otherwise output 1-3 bullet points only.\n\n"
            f"Current summary:\n{current_summary}\n\n"
            f"Previous story context:\n{previous_context}\n\n"
            f"New article:\n{article_text}\n"
        )
    else:
        prompt = (
            "You are updating a football storyline summary.\n"
            "Merge the current summary with the new information.\n"
            "Rules:\n"
            "- Use only facts from the provided inputs.\n"
            "- Keep the story coherent and remove duplicates.\n"
            "- Reflect the latest state when status changes happened.\n"
            "- Output exactly 3-6 bullet points.\n\n"
            f"Current summary:\n{current_summary}\n\n"
            f"New information:\n{whats_new}\n\n"
            f"Full updated context:\n{full_context}\n"
        )

    text = str(llm_call(prompt)).strip()
    return text


def process_story(story, llm_call):
    out_story = dict(story)
    articles = list(story.get("articles", []))

    if len(articles) == 0:
        out_story["story_summary"] = ""
        out_story["whats_new_by_article"] = []
        return out_story

    processed_articles = [articles[0]]
    current_summary = ask_llm("summary", llm_call, story_context=build_context(processed_articles))
    updates = []

    for article in articles[1:]:
        previous_context = build_context(processed_articles)
        article_text = str(article.get("text", ""))

        whats_new = ask_llm(
            "whats_new",
            llm_call,
            current_summary=current_summary,
            previous_context=previous_context,
            article_text=article_text,
        )

        processed_articles.append(article)
        full_context = build_context(processed_articles)
        current_summary = ask_llm(
            "update",
            llm_call,
            current_summary=current_summary,
            whats_new=whats_new,
            full_context=full_context,
        )

        updates.append(
            {
                "news_id": int(article.get("index", -1)),
                "source": str(article.get("source", "")),
                "whats_new": whats_new,
                "updated_story_summary": current_summary,
            }
        )

    out_story["story_summary"] = current_summary
    out_story["whats_new_by_article"] = updates
    return out_story


def main():
    parser = argparse.ArgumentParser(description="Summarization of storylines")
    parser.add_argument(
        "--cluster_state",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task3_clusters.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task3_clusters_summarized.json",
    )
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_token_env", type=str, default="OPENAI_API_KEY")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    cluster_path = resolve_path(args.cluster_state, here)
    output_path = resolve_path(args.output, here)

    openai_token = get_token(args.openai_token_env)
    llm_call, selected_model = make_llm_call(args.openai_model, openai_token)

    clusters = load_json(cluster_path)
    out = dict(clusters)

    summarized_stories = []
    for story in clusters.get("stories", []):
        summarized_stories.append(process_story(story, llm_call))

    #out["summary_mode"] = "iterative_story_summary"
    out["summary_provider"] = "openai"
    out["summary_model"] = selected_model
    out["stories"] = summarized_stories
    out.pop("assignments", None)

    save_json(out, output_path)

    print("Summarization of storylines")
    print("Stories processed:", len(summarized_stories))
    print("See full results at:", output_path)


if __name__ == "__main__":
    main()
