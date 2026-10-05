import json
from pathlib import Path

from src.retrieval.miracl_english_shard_retriever import (
    MIRACLEnglishShardRetriever
)

from src.retrieval.miracl_shard_retriever import (
    MIRACLShardRetriever
)

from src.retrieval.reranker import EvidenceReranker


CLAIMS_FILE = Path(
    "data/processed/claims.json"
)

EVIDENCE_FILE = Path(
    "data/processed/evidence.json"
)


def load_claims():

    with open(
        CLAIMS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    claims = []

    for sample in data.get(
        "samples",
        []
    ):

        sample_id = sample.get(
            "sample_id"
        )

        metadata_language = sample.get(
            "language",
            "en"
        )

        for claim in sample.get(
            "claims",
            []
        ):

            claims.append({
                "sample_id": sample_id,
                "claim_id": claim.get(
                    "claim_id"
                ),
                "claim": claim.get(
                    "claim"
                ),
                "language": metadata_language
            })

    return claims


def detect_language(
    claim,
    metadata_language
):

    devanagari_count = sum(
        "\u0900" <= char <= "\u097F"
        for char in claim
    )

    if devanagari_count > 0:

        return "hi"

    return "en"


def save_evidence(
    evidence
):

    EVIDENCE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        EVIDENCE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evidence,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nEvidence saved to: "
        f"{EVIDENCE_FILE}"
    )


def main():

    claims = load_claims()

    print(
        f"Total claims: {len(claims)}"
    )

    print(
        "\nDetecting languages..."
    )

    for item in claims:

        item["detected_language"] = (
            detect_language(
                item["claim"],
                item["language"]
            )
        )

    english_items = [
        item
        for item in claims
        if item["detected_language"] == "en"
    ]

    hindi_items = [
        item
        for item in claims
        if item["detected_language"] == "hi"
    ]

    print(
        f"English claims: "
        f"{len(english_items)}"
    )

    print(
        f"Hindi claims: "
        f"{len(hindi_items)}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING SEMANTIC RERANKER"
    )

    print(
        "=" * 70
    )

    reranker = EvidenceReranker()

    all_evidence = {}

    # --------------------------------------------------
    # ENGLISH RETRIEVAL
    # --------------------------------------------------

    if english_items:

        print(
            "\n" + "=" * 70
        )

        print(
            "ENGLISH RETRIEVAL"
        )

        print(
            "=" * 70
        )

        english_retriever = (
            MIRACLEnglishShardRetriever()
        )

        english_claims = [
            item["claim"]
            for item in english_items
        ]

        print(
            f"\nRetrieving candidates for "
            f"{len(english_claims)} English claims..."
        )

        english_candidates = (
            english_retriever.retrieve_many(
                english_claims,
                top_k=200
            )
        )

        print(
            "\nEnglish BM25 retrieval completed."
        )

        for item, candidates in zip(
            english_items,
            english_candidates
        ):

            print(
                "\n" + "-" * 70
            )

            print(
                f"Processing English "
                f"Sample {item['sample_id']} "
                f"Claim {item['claim_id']}"
            )

            print(
                f"Claim: {item['claim']}"
            )

            print(
                "Reranking top 200 candidates..."
            )

            results = reranker.rerank(
                item["claim"],
                candidates,
                top_k=5
            )

            all_evidence[
                (
                    item["sample_id"],
                    item["claim_id"]
                )
            ] = {
                "sample_id": item[
                    "sample_id"
                ],
                "claim_id": item[
                    "claim_id"
                ],
                "claim": item[
                    "claim"
                ],
                "language": "en",
                "evidence": results
            }

    # --------------------------------------------------
    # HINDI RETRIEVAL
    # --------------------------------------------------

    if hindi_items:

        print(
            "\n" + "=" * 70
        )

        print(
            "HINDI RETRIEVAL"
        )

        print(
            "=" * 70
        )

        hindi_retriever = (
            MIRACLShardRetriever(
                language="hi"
            )
        )

        for item in hindi_items:

            print(
                "\n" + "-" * 70
            )

            print(
                f"Processing Hindi "
                f"Sample {item['sample_id']} "
                f"Claim {item['claim_id']}"
            )

            print(
                f"Claim: {item['claim']}"
            )

            print(
                "Retrieving BM25 candidates..."
            )

            candidates = (
                hindi_retriever.retrieve(
                    item["claim"],
                    top_k=200
                )
            )

            print(
                f"Candidates retrieved: "
                f"{len(candidates)}"
            )

            print(
                "Reranking top 200 candidates..."
            )

            results = reranker.rerank(
                item["claim"],
                candidates,
                top_k=5
            )

            all_evidence[
                (
                    item["sample_id"],
                    item["claim_id"]
                )
            ] = {
                "sample_id": item[
                    "sample_id"
                ],
                "claim_id": item[
                    "claim_id"
                ],
                "claim": item[
                    "claim"
                ],
                "language": "hi",
                "evidence": results
            }

    # --------------------------------------------------
    # RESTORE ORIGINAL CLAIM ORDER
    # --------------------------------------------------

    final_evidence = []

    for item in claims:

        key = (
            item["sample_id"],
            item["claim_id"]
        )

        if key in all_evidence:

            final_evidence.append(
                all_evidence[key]
            )

    save_evidence(
        final_evidence
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EVIDENCE RETRIEVAL COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Total claims processed: "
        f"{len(final_evidence)}"
    )

    print(
        f"English claims: "
        f"{len(english_items)}"
    )

    print(
        f"Hindi claims: "
        f"{len(hindi_items)}"
    )

    print(
        f"\nOutput: "
        f"{EVIDENCE_FILE}"
    )


if __name__ == "__main__":

    main()