import json
from pathlib import Path


CLAIMS_FILE = Path("data/processed/claims.json")


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


if __name__ == "__main__":
    claims = load_claims()

    print(f"Total claims: {len(claims)}")
    print()

    for item in claims:
        print(
            f"Sample {item['sample_id']} | "
            f"Claim {item['claim_id']}: {item['claim']}"
        )