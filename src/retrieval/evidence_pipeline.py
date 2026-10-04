import json
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import EvidenceReranker


CLAIMS_FILE = Path("data/processed/claims.json")
EVIDENCE_FILE = Path("data/processed/evidence.json")


def load_claims():

    with open(CLAIMS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    claims = []

    for sample in data.get("samples", []):

        sample_id = sample.get("sample_id")
        language = sample.get("language", "en")

        for claim in sample.get("claims", []):

            claims.append({
                "sample_id": sample_id,
                "claim_id": claim.get("claim_id"),
                "claim": claim.get("claim"),
                "language": language
            })

    return claims


def save_evidence(evidence):

    EVIDENCE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(EVIDENCE_FILE, "w", encoding="utf-8") as file:

        json.dump(
            evidence,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():

    claims = load_claims()

    print(f"Total claims: {len(claims)}")

    print("\nLoading BM25 retriever...")

    retriever = BM25Retriever(
        language="en",
        num_passages=100000
    )

    print("\nLoading evidence reranker...")

    reranker = EvidenceReranker()

    print("\nStarting evidence retrieval...")

    evidence = []

    for i, item in enumerate(claims, start=1):

        claim = item["claim"]

        print("\n" + "=" * 70)
        print(f"Processing {i}/{len(claims)}")
        print(f"Sample ID: {item['sample_id']}")
        print(f"Claim ID: {item['claim_id']}")
        print(f"Claim: {claim}")

        candidates = retriever.retrieve(
            claim,
            top_k=20
        )

        results = reranker.rerank(
            claim,
            candidates,
            top_k=5
        )

        evidence.append({
            "sample_id": item["sample_id"],
            "claim_id": item["claim_id"],
            "claim": claim,
            "language": item["language"],
            "evidence": results
        })

        print("\nTop evidence:")

        for result in results:

            print(
                f"Rank {result['rank']} | "
                f"BM25: {result['relevance_score']:.4f} | "
                f"Semantic: {result['rerank_score']:.4f} | "
                f"{result['title']}"
            )

    save_evidence(evidence)

    print("\n" + "=" * 70)
    print("EVIDENCE RETRIEVAL COMPLETED")
    print("=" * 70)

    print(f"Total claims processed: {len(claims)}")
    print("BM25 candidates per claim: 20")
    print("Final evidence per claim: 5")
    print(f"Output file: {EVIDENCE_FILE}")


if __name__ == "__main__":
    main()