


import argparse
import json
from pathlib import Path

from helper import best_dense_match, load_incoming_articles, load_news_articles, novelty_score, resolve_path, save_json
from helper_llm import get_token, make_llm_call


def build_judge_prompt(incoming_text, best_match_text, similarity, novelty):
    return (
        "You are a football news publishing judge.\n"
        "Decide if the incoming article should be published or skipped.\n"
        "Rules:\n"
        "- publish when there is a material update.\n"
        "- treat semantic state progression as material (for example: rumor -> talks -> agreement -> official/finalized).\n"
        "- also treat changes in injury timeline, contract terms, transfer fee, involved clubs/players, or timing as material.\n"
        "- skip only when information is essentially unchanged.\n"
        "- reason must explain the criterion, not repeat article content.\n"
        "- keep reason concise (max 16 words).\n\n"
        f"Similarity score: {similarity:.4f}\n"
        f"Novelty score: {novelty:.4f}\n"
        f"Closest published article:\n{best_match_text}\n\n"
        f"Incoming article:\n{incoming_text}\n\n"
        "Return JSON only:\n"
        '{"decision":"publish or skip","reason":"one short criterion-based reason"}'
    )


def parse_judge_output(raw_text):
    text = str(raw_text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM did not return valid JSON: {text}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"LLM JSON output must be an object: {data}")

    decision = str(data.get("decision", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    return decision, reason


def llm_judge(incoming_text, best_match_text, similarity, novelty, llm_call):
    prompt = build_judge_prompt(incoming_text, best_match_text, similarity, novelty)
    raw = llm_call(prompt)
    decision, reason = parse_judge_output(raw)
    return decision, reason, raw


def main():
    parser = argparse.ArgumentParser(description="Task 2 Alternative using LLM as Judge")
    parser.add_argument("--news_csv", type=str, default=Path("./data/news.csv"))
    parser.add_argument("--incoming_json", type=str, default=Path("./data/incoming_news.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output/task2_alt_llm_results.json",
    )
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_token_env", type=str, default="OPENAI_API_KEY")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    news_path = resolve_path(args.news_csv, here)
    incoming_path = resolve_path(args.incoming_json, here)
    output_path = resolve_path(args.output, here)

    openai_token = get_token(args.openai_token_env)
    llm_call, selected_model = make_llm_call(args.openai_model, openai_token)

    references = load_news_articles(news_path)
    incoming = load_incoming_articles(incoming_path)

    decisions = []
    for incoming_index, incoming_text in enumerate(incoming):
        match = best_dense_match(references, incoming_text)
        best_match_index = int(match.get("best_match_index", -1))
        similarity = float(match.get("best_similarity_score", 0.0))
        best_match_text = str(match.get("best_match_text", ""))
        novelty = novelty_score(incoming_text, best_match_text)

        try:
            llm_decision, reason, llm_raw_output = llm_judge(
                incoming_text,
                best_match_text,
                similarity,
                novelty,
                llm_call,
            )
        except Exception as exc:
            raise RuntimeError(f"failed for {incoming_index}: {exc}") from exc

        llm_decision = str(llm_decision or "").strip().lower()
        reason = str(reason or "").strip()
        if llm_decision not in {"publish", "skip"}:
            raise RuntimeError(
                f"invalid decision: '{llm_decision}'. Raw output: {llm_raw_output}"
            )

        decisions.append(
            {
                "incoming_text": incoming_text,
                "best_match_index": best_match_index,
                "best_match_text": best_match_text,
                "llm_decision": llm_decision,
                "reason": reason,
                "llm_raw_output": llm_raw_output,
            }
        )

        if llm_decision == "publish":
            references.append(incoming_text)

    publish_count = sum(1 for row in decisions if row["llm_decision"] == "publish")
    skip_count = sum(1 for row in decisions if row["llm_decision"] == "skip")

    result = {
        "llm_model": selected_model,
        "publish_count": publish_count,
        "skip_count": skip_count,
        "results": decisions,
    }

    save_json(result, output_path)

    print("Decisions using LLM Judge")
    print("Publish:", publish_count)
    print("Skip:", skip_count)
    print("See full results at:", output_path)


if __name__ == "__main__":
    main()
