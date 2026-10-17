
import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_FILE = Path(
    "data/processed/sangraha_verified_corpus.json"
)


def main():
    print("Loading Vaarta CPT dataset...")

    dataset = load_dataset(
        "OmAlve/vaarta-cpt-dataset",
        split="train",
    )

    print("Total dataset rows:", len(dataset))
    print("Filtering verified Marathi documents...")

    filtered_dataset = dataset.filter(
        lambda row: row["source"] == "sangraha_verified"
        and row["language"] == "mr"
        and isinstance(row["text"], str)
        and bool(row["text"].strip())
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    documents = []
    seen_texts = set()
    duplicate_count = 0

    for row in filtered_dataset:
        text = row["text"].strip()

        if text in seen_texts:
            duplicate_count += 1
            continue

        seen_texts.add(text)

        documents.append({
            "docid": f"mr_sangraha_{len(documents) + 1:06d}",
            "title": "",
            "text": text,
            "source": "sangraha_verified",
            "language": "mr",
        })

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nCorpus extraction complete.")
    print("Documents saved:", len(documents))
    print("Exact duplicate texts skipped:", duplicate_count)
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
