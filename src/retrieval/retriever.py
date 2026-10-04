from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import gzip
import json
import numpy as np


def retrieve_evidence(claim, language="en", num_samples=100):

    print("Loading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Embedding model loaded.")

    print(f"\nLoading MIRACL {language} corpus...")

    file_path = hf_hub_download(
        repo_id="miracl/miracl-corpus",
        filename=f"miracl-corpus-v1.0-{language}/docs-0.jsonl.gz",
        repo_type="dataset"
    )

    print("MIRACL file loaded.")

    passages = []

    with gzip.open(file_path, "rt", encoding="utf-8") as file:

        for i, line in enumerate(file):

            item = json.loads(line)

            passages.append({
                "docid": item["docid"],
                "title": item["title"],
                "text": item["text"]
            })

            if i + 1 >= num_samples:
                break

    print(f"Loaded {len(passages)} passages.")

    print("\nCreating claim embedding...")

    claim_embedding = model.encode(
        claim,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Claim embedding created.")

    print("\nCreating passage embeddings...")

    passage_texts = [item["text"] for item in passages]

    passage_embeddings = model.encode(
        passage_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Passage embeddings created.")

    similarities = np.dot(
        passage_embeddings,
        claim_embedding
    )

    top_indices = np.argsort(similarities)[::-1][:5]

    print("\n" + "=" * 70)
    print("CLAIM")
    print("=" * 70)

    print(claim)

    print("\n" + "=" * 70)
    print("TOP 5 EVIDENCE")
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):

        item = passages[index]

        print(f"\nRank: {rank}")
        print(f"Similarity Score: {similarities[index]:.4f}")
        print(f"Doc ID: {item['docid']}")
        print(f"Title: {item['title']}")
        print(f"Text: {item['text'][:500]}")

        print("-" * 70)


if __name__ == "__main__":

    claim = "Alexander Fleming discovered penicillin in 1928."

    retrieve_evidence(claim)