from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi
import gzip
import json
import re


class BM25Retriever:

    def __init__(self, language="en", num_passages=50000):

        self.language = language
        self.num_passages = num_passages

        print(f"Loading MIRACL {language} corpus...")

        file_path = hf_hub_download(
            repo_id="miracl/miracl-corpus",
            filename=f"miracl-corpus-v1.0-{language}/docs-0.jsonl.gz",
            repo_type="dataset"
        )

        self.passages = []

        with gzip.open(file_path, "rt", encoding="utf-8") as file:

            for i, line in enumerate(file):

                item = json.loads(line)

                self.passages.append({
                    "docid": item["docid"],
                    "title": item["title"],
                    "text": item["text"]
                })

                if i + 1 >= num_passages:
                    break

        print(f"Loaded {len(self.passages)} passages.")

        print("Building BM25 index...")

        documents = [
            self.tokenize(item["text"])
            for item in self.passages
        ]

        self.bm25 = BM25Okapi(documents)

        print("BM25 index ready.")

    def tokenize(self, text):

        return re.findall(r"\w+", text.lower())

    def retrieve(self, claim, top_k=5):

        query_tokens = self.tokenize(claim)

        scores = self.bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for rank, index in enumerate(top_indices, start=1):

            item = self.passages[index]

            results.append({
                "rank": rank,
                "docid": item["docid"],
                "title": item["title"],
                "text": item["text"],
                "relevance_score": float(scores[index])
            })

        return results


if __name__ == "__main__":

    retriever = BM25Retriever(
        language="en",
        num_passages=50000
    )

    claim = "Alexander Fleming discovered penicillin in 1928."

    results = retriever.retrieve(
        claim,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("CLAIM")
    print("=" * 70)
    print(claim)

    print("\n" + "=" * 70)
    print("TOP 5 EVIDENCE")
    print("=" * 70)

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['relevance_score']:.4f}")
        print(f"Doc ID: {result['docid']}")
        print(f"Title: {result['title']}")
        print(f"Text: {result['text'][:500]}")
        print("-" * 70)