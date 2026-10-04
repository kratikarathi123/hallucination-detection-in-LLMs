from sentence_transformers import SentenceTransformer
import numpy as np


class EvidenceReranker:

    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):

        print("Loading multilingual embedding model...")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded.")

    def rerank(self, claim, evidence, top_k=5):

        if not evidence:
            return []

        texts = [
            item["title"] + " " + item["text"]
            for item in evidence
        ]

        claim_embedding = self.model.encode(
            claim,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        evidence_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = np.dot(
            evidence_embeddings,
            claim_embedding
        )

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):

            item = evidence[index].copy()

            item["rerank_score"] = float(similarities[index])
            item["rank"] = rank

            results.append(item)

        return results


if __name__ == "__main__":

    reranker = EvidenceReranker()

    claim = "Alexander Fleming discovered penicillin in 1928."

    evidence = [
        {
            "rank": 1,
            "docid": "test1",
            "title": "Antibiotic",
            "text": "Alexander Fleming discovered penicillin in 1928.",
            "relevance_score": 10.5
        },
        {
            "rank": 2,
            "docid": "test2",
            "title": "History",
            "text": "Alexander Fleming was a Scottish physician and microbiologist.",
            "relevance_score": 9.5
        }
    ]

    results = reranker.rerank(
        claim,
        evidence,
        top_k=2
    )

    print("\n" + "=" * 70)
    print("RERANKED EVIDENCE")
    print("=" * 70)

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(f"BM25 Score: {result['relevance_score']:.4f}")
        print(f"Semantic Score: {result['rerank_score']:.4f}")
        print(f"Title: {result['title']}")
        print(f"Text: {result['text'][:300]}")