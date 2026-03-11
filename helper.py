
import json
from pathlib import Path
import pandas as pd
import numpy as np
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_news_articles(path):
    df = pd.read_csv(path)

    out = []
    values = df["text"].fillna("").astype(str).tolist()
    for value in values:
        out.append(value)
    return out


def clean_text(text):
    value = str(text or "")
    value = re.sub(r"\*+", "", value)
    value = re.sub(r"\[\d+\]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def tokenize(text):
    return re.findall(r"\b\w+\b", clean_text(text))


def load_incoming_articles(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        values = data.get("text_list", [])
    elif isinstance(data, list):
        values = data
    else:
        values = []

    out = []
    for value in values:
        if isinstance(value, dict):
            out.append(str(value.get("text", "")))
        else:
            out.append(str(value))
    return out

def informative_tokens(text):
    tokens = tokenize(text)
    out = []
    for token in tokens:
        if len(token) <= 2:
            continue
        if token.isnumeric():
            continue
        if token in ENGLISH_STOP_WORDS:
            continue
        out.append(token)
    return out


def novelty_score(new_article, reference_article):
    new_tokens = informative_tokens(new_article)
    if len(new_tokens) == 0:
        return 0.0

    reference_tokens = set(informative_tokens(reference_article))

    unseen = 0
    for token in new_tokens:
        if token not in reference_tokens:
            unseen += 1

    return float(unseen / len(new_tokens))


def best_dense_match(references, query):
    if len(references) == 0:
        return {
            "best_match_index": -1,
            "best_similarity_score": 0.0,
            "best_match_text": "",
            "scores": np.asarray([], dtype=float),
        }

    ret = Retriever(references)
    scores = np.asarray(ret.score_dense(query), dtype=float)
    best_match_index = int(np.argmax(scores))
    best_similarity_score = float(scores[best_match_index])
    best_similarity_score = max(0.0, min(1.0, best_similarity_score))
    best_match_text = str(references[best_match_index])

    return {
        "best_match_index": best_match_index,
        "best_similarity_score": best_similarity_score,
        "best_match_text": best_match_text,
        "scores": scores,
    }


class Retriever:
    # TF-IDF, BM25 and dense embeddings for docs

    #First clean and tokenize docs
    def __init__(self, docs, embedding_model="all-MiniLM-L12-v2"):
        self.docs = []
        for doc in docs:
            self.docs.append(str(doc))

        self.cleaned_docs = []
        self.tokenized_docs= []
        for doc in self.docs:
            clean_doc = clean_text(doc)
            self.cleaned_docs.append(clean_doc)
            self.tokenized_docs.append(tokenize(clean_doc))

        self.embedding_model = embedding_model


    #Now score (with chosen method) based on query and return top k
    def score_dense(self, query):

        self.embed_model = SentenceTransformer(self.embedding_model)
        
        self.embeddings = self.embed_model.encode(
            self.cleaned_docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = self.embed_model.encode(
            [clean_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return cosine_similarity(query_embedding, self.embeddings).flatten()

    def score_tfidf(self, query):

        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.cleaned_docs)

        query_vector = self.tfidf.transform([clean_text(query)])
        return cosine_similarity(query_vector, self.tfidf_matrix).flatten()

    def score_bm25(self, query):
        self.bm25 = BM25Okapi(self.tokenized_docs)
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)

    def top_k_rows(self, scores, k):
        limit = min(max(int(k), 0), len(self.docs))
        indices = np.argsort(scores)[::-1][:limit]
        rows= []
        for idx in indices:
            rows.append({"index": int(idx), "score": float(scores[idx]), "text": self.docs[idx]})
        return rows


def resolve_path(path, here):
    value = Path(path)
    if not value.is_absolute():
        value = (Path(here) / value).resolve()
    return value


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not append:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    if path.exists():
        try:
            existing = load_json(path)
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    if isinstance(existing, list):
        existing.append(obj)
    elif isinstance(existing, dict):
        existing = [existing, obj]
    else:
        existing = [obj]

    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
