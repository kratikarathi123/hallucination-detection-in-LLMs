
import json
from pathlib import Path

from src.retrieval.miracl_english_shard_retriever import (
    MIRACLEnglishShardRetriever,
)
from src.retrieval.miracl_hindi_shard_retriever import (
    MIRACLShardRetriever,
)
from src.retrieval.miracl_bengali_shard_retriever import (
    MIRACLBengaliShardRetriever,
)
from src.retrieval.vaarta_marathi_retriever import (
    VaartaMarathiRetriever,
)
from src.retrieval.reranker import EvidenceReranker


CLAIMS_FILE = Path("data/processed/claims.json")
EVIDENCE_FILE = Path("data/processed/evidence.json")
TEMP_FILE = Path("data/processed/evidence.tmp.json")

CANDIDATE_TOP_K = 200
FINAL_TOP_K = 5

SUPPORTED_LANGUAGES = {"en", "hi", "bn", "mr"}

LANGUAGE_NAMES = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "bangla": "bn",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
}


def load_claims():
    with CLAIMS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    claims = []

    for sample in data.get("samples", []):
        for claim in sample.get("claims", []):
            claim_text = claim.get("claim")

            if not isinstance(claim_text, str) or not claim_text.strip():
                continue

            claims.append({
                "sample_id": sample.get("sample_id"),
                "claim_id": claim.get("claim_id"),
                "claim": claim_text.strip(),
                "language": sample.get("language"),
            })

    return claims



def detect_language(claim, metadata_language):
    language = str(metadata_language or "").strip().lower()
    language = LANGUAGE_NAMES.get(language, language)

    # Detect the script used in the claim.
    has_devanagari = any(
        0x0900 <= ord(char) <= 0x097F
        for char in claim
    )

    has_bengali = any(
        0x0980 <= ord(char) <= 0x09FF
        for char in claim
    )

    has_tamil = any(
        0x0B80 <= ord(char) <= 0x0BFF
        for char in claim
    )

    has_telugu = any(
        0x0C00 <= ord(char) <= 0x0C7F
        for char in claim
    )

    # If metadata says English but the text uses a different script,
    # do not blindly trust the English label.
    if language == "en":
        if has_bengali:
            return "bn"
        if has_tamil:
            return "ta"
        if has_telugu:
            return "te"
        if has_devanagari:
            # Script alone cannot distinguish Hindi from Marathi.
            return "und"

        return "en"

    # Trust explicit supported metadata when it does not conflict
    # with an obvious script mismatch.
    if language in {"hi", "mr"}:
        return language

    if language == "bn" and has_bengali:
        return "bn"

    if language == "ta" and has_tamil:
        return "ta"

    if language == "te" and has_telugu:
        return "te"

    # Use script detection when metadata is missing or inconsistent.
    if has_bengali:
        return "bn"

    if has_tamil:
        return "ta"

    if has_telugu:
        return "te"

    if has_devanagari:
        return "und"

    if language in {"en", "hi", "mr", "bn", "ta", "te"}:
        return language

    return "en"



def claim_key(item):
    return item.get("sample_id"), item.get("claim_id")



def save_evidence(evidence, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(".tmp.json")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(evidence, file, ensure_ascii=False, indent=2)

    temp_file.replace(output_file)
    print(f"\nEvidence saved to: {output_file}")


def main(output_file=EVIDENCE_FILE):
    claims = load_claims()
    print(f"Total claims in claims.json: {len(claims)}")

    if not claims:
        print("No claims found. Nothing to retrieve.")
        return

    for item in claims:
        item["detected_language"] = detect_language(
            item["claim"],
            item["language"],
        )

    language_groups = {
        language: [
            item for item in claims
            if item["detected_language"] == language
        ]
        for language in ("en", "hi", "bn", "mr")
    }

    for language, items in language_groups.items():
        print(f"{language.upper()} claims: {len(items)}")

    unsupported_items = [
        item for item in claims
        if item["detected_language"] not in SUPPORTED_LANGUAGES
    ]

    print(f"Unsupported claims: {len(unsupported_items)}")

    existing_records = []

    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as file:
            existing_records = json.load(file)
            
        if not isinstance(existing_records, list):
            raise ValueError("Existing evidence.json must contain a JSON list.")

    existing_by_key = {}

    for record in existing_records:
        key = claim_key(record)

        if key in existing_by_key:
            raise ValueError(
                f"Duplicate claim ID in evidence.json: {key}"
            )

        existing_by_key[key] = record

    # Reuse records that already have evidence.
    # Only claims needing retrieval will be processed.
    pending_by_language = {}

    for language, items in language_groups.items():
        pending_by_language[language] = [
            item for item in items
            if not (
                existing_by_key.get(claim_key(item), {}).get("evidence")
            )
        ]

    pending_total = sum(
        len(items) for items in pending_by_language.values()
    )

    print(f"\nClaims needing retrieval: {pending_total}")
    print(
        "Existing records with evidence will be reused; "
        "their retrieval will not run again."
    )

    all_evidence = dict(existing_by_key)

    # Preserve unsupported claims without loading a retriever.
    for item in unsupported_items:
        key = claim_key(item)

        if key not in all_evidence:
            all_evidence[key] = {
                "sample_id": item["sample_id"],
                "claim_id": item["claim_id"],
                "claim": item["claim"],
                "language": item["detected_language"],
                "evidence": [],
                "retrieval_status": "language_not_supported",
            }

    retrievers = {}
    pending_en_hi_bn = any(
        pending_by_language[language]
        for language in ("en", "hi", "bn")
    )

    reranker = EvidenceReranker() if pending_en_hi_bn else None

    if pending_by_language["en"]:
        print("\nLoading English retriever...")
        retrievers["en"] = MIRACLEnglishShardRetriever()

    if pending_by_language["hi"]:
        print("\nLoading Hindi retriever...")
        retrievers["hi"] = MIRACLShardRetriever(language="hi")

    if pending_by_language["bn"]:
        print("\nLoading Bengali retriever...")
        retrievers["bn"] = MIRACLBengaliShardRetriever()

    if pending_by_language["mr"]:
        print("\nLoading Marathi retriever...")
        retrievers["mr"] = VaartaMarathiRetriever(
            candidate_top_k=CANDIDATE_TOP_K,
            final_top_k=FINAL_TOP_K,
        )

    for language in ("en", "hi", "bn", "mr"):
        items = pending_by_language[language]

        if not items:
            print(f"\n{language.upper()}: no new retrieval needed.")
            continue

        retriever = retrievers[language]

        print(f"\n{'=' * 70}")
        print(f"{language.upper()} RETRIEVAL")
        print(f"{'=' * 70}")

        if language == "en":
            candidate_batches = retriever.retrieve_many(
                [item["claim"] for item in items],
                top_k=CANDIDATE_TOP_K,
            )
        elif language == "mr":
            # Marathi retriever already performs semantic reranking.
            candidate_batches = [
                retriever.retrieve(
                    item["claim"],
                    candidate_top_k=CANDIDATE_TOP_K,
                    final_top_k=FINAL_TOP_K,
                )
                for item in items
            ]
        else:
            candidate_batches = [
                retriever.retrieve(
                    item["claim"],
                    top_k=CANDIDATE_TOP_K,
                )
                for item in items
            ]

        for item, candidates in zip(items, candidate_batches):
            key = claim_key(item)

            print(
                f"\nProcessing {language.upper()} "
                f"Sample {item['sample_id']} "
                f"Claim {item['claim_id']}"
            )
            print(f"Claim: {item['claim']}")
            print(f"Retrieved passages: {len(candidates)}")

            if language == "mr":
                results = candidates
            elif candidates:
                results = reranker.rerank(
                    item["claim"],
                    candidates,
                    top_k=FINAL_TOP_K,
                )
            else:
                results = []

            all_evidence[key] = {
                "sample_id": item["sample_id"],
                "claim_id": item["claim_id"],
                "claim": item["claim"],
                "language": language,
                "evidence": results,
                "retrieval_status": (
                    "success" if results else "no_evidence_found"
                ),
            }

    # Output claims in their original order, then preserve any
    # existing records that are not present in claims.json.
    final_evidence = []
    current_keys = set()

    for item in claims:
        key = claim_key(item)
        current_keys.add(key)

        if key in all_evidence:
            final_evidence.append(all_evidence[key])
        else:
            final_evidence.append({
                "sample_id": item["sample_id"],
                "claim_id": item["claim_id"],
                "claim": item["claim"],
                "language": item["detected_language"],
                "evidence": [],
                "retrieval_status": "retrieval_not_completed",
            })

    for record in existing_records:
        if claim_key(record) not in current_keys:
            final_evidence.append(record)

    save_evidence(final_evidence, output_file)

    print(f"\n{'=' * 70}")
    print("EVIDENCE RETRIEVAL COMPLETED")
    print(f"{'=' * 70}")
    print(f"Claims in claims.json: {len(claims)}")
    print(f"Existing records preserved: {len(existing_records)}")
    print(f"Claims needing retrieval: {pending_total}")
    print(f"Final evidence records: {len(final_evidence)}")
    print(f"English claims: {len(language_groups['en'])}")
    print(f"Hindi claims: {len(language_groups['hi'])}")
    print(f"Bengali claims: {len(language_groups['bn'])}")
    print(f"Marathi claims: {len(language_groups['mr'])}")
    print(f"Unsupported claims: {len(unsupported_items)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
