
import json
from pathlib import Path

from src.retrieval.vaarta_marathi_retriever import (
    VaartaMarathiRetriever,
)

CLAIMS_FILE = Path("data/processed/claims.json")
OUTPUT_FILE = Path("data/processed/marathi_evidence_test.json")


def main():
    with CLAIMS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    claims = []

    for sample in data.get("samples", []):
        if str(sample.get("language", "")).strip().lower() not in (
            "mr",
            "marathi",
        ):
            continue

        for claim in sample.get("claims", []):
            claim_text = claim.get("claim")

            if isinstance(claim_text, str) and claim_text.strip():
                claims.append({
                    "sample_id": sample.get("sample_id"),
                    "claim_id": claim.get("claim_id"),
                    "claim": claim_text.strip(),
                    "language": "mr",
                })

    print(f"Total Marathi claims: {len(claims)}")

    if not claims:
        print("No Marathi claims found.")
        return

    retriever = VaartaMarathiRetriever()
    results = []

    for index, item in enumerate(claims, start=1):
        print(
            f"\n[{index}/{len(claims)}] "
            f"Sample {item['sample_id']}, "
            f"Claim {item['claim_id']}"
        )
        print(f"Claim: {item['claim']}")

        evidence = retriever.retrieve(item["claim"])

        results.append({
            **item,
            "evidence": evidence,
            "retrieval_status": (
                "success" if evidence else "no_evidence_found"
            ),
        })

        print(f"Evidence passages: {len(evidence)}")

        if evidence:
            print(f"Top result: {evidence[0]['text'][:250]}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    successful = sum(
        1 for item in results if item["evidence"]
    )

    print("\n" + "=" * 60)
    print("MARATHI RETRIEVAL TEST COMPLETED")
    print("=" * 60)
    print(f"Total claims: {len(results)}")
    print(f"Claims with evidence: {successful}")
    print(f"Claims without evidence: {len(results) - successful}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
