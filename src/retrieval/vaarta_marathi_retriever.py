
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval.reranker import EvidenceReranker


WIKIPEDIA_CORPUS = Path(
    "data/processed/marathi_wikipedia_corpus.json"
)
SANGRAHA_CORPUS = Path(
    "data/processed/sangraha_verified_corpus.json"
)


def tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class VaartaMarathiRetriever:

    def __init__(
        self,
        corpus_path=None,
        candidate_top_k=200,
        final_top_k=5,
        reranker=None,
    ):
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k
        self.reranker = reranker or EvidenceReranker()

        corpus_paths = (
            [Path(corpus_path)]
            if corpus_path is not None
            else [WIKIPEDIA_CORPUS, SANGRAHA_CORPUS]
        )

        self.documents = []
        self.indexes = []

        for path in corpus_paths:
            print(f"Loading Marathi corpus: {path}")

            with path.open("r", encoding="utf-8") as file:
                documents = json.load(file)

            if not isinstance(documents, list):
                raise ValueError(f"Corpus must be a JSON list: {path}")

            if not documents:
                raise ValueError(f"Marathi corpus is empty: {path}")

            valid_documents = [
                document
                for document in documents
                if isinstance(document.get("text"), str)
                and document["text"].strip()
            ]

            if not valid_documents:
                raise ValueError(f"No documents with text in: {path}")

            print(f"Loaded {len(valid_documents)} documents.")

            tokenized_documents = [
                tokenize(document["text"])
                for document in valid_documents
            ]

            bm25 = BM25Okapi(tokenized_documents)

            self.documents.append(valid_documents)
            self.indexes.append(bm25)

            print(f"BM25 index ready: {path}")

        print(
            f"Marathi retrieval ready with "
            f"{len(self.indexes)} corpus index(es)."
        )

    def retrieve(
        self,
        claim,
        candidate_top_k=None,
        final_top_k=None,
    ):
        candidate_top_k = (
            candidate_top_k
            if candidate_top_k is not None
            else self.candidate_top_k
        )
        final_top_k = (
            final_top_k
            if final_top_k is not None
            else self.final_top_k
        )

        query_tokens = tokenize(claim)

        if not query_tokens:
            return []

        candidates_by_id = {}

        for documents, bm25 in zip(self.documents, self.indexes):
            scores = bm25.get_scores(query_tokens)

            candidate_count = min(candidate_top_k, len(documents))

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )[:candidate_count]

            for index in ranked_indices:
                document = documents[index]
                docid = document.get("docid")

                candidate = {
                    "docid": docid,
                    "title": document.get("title", ""),
                    "text": document["text"],
                    "source": document.get("source", "unknown"),
                    "language": document.get("language", "mr"),
                    "relevance_score": float(scores[index]),
                }

                if docid not in candidates_by_id:
                    candidates_by_id[docid] = candidate
                elif (
                    candidate["relevance_score"]
                    > candidates_by_id[docid]["relevance_score"]
                ):
                    candidates_by_id[docid] = candidate

        candidates = list(candidates_by_id.values())

        if not candidates:
            return []

        return self.reranker.rerank(
            claim,
            candidates,
            top_k=final_top_k,
        )


if __name__ == "__main__":
    retriever = VaartaMarathiRetriever()

    claim = "ताजमहाल उत्तर प्रदेशातील आग्रा येथे आहे."
    results = retriever.retrieve(claim)

    print("\nMARATHI RETRIEVAL RESULTS")

    for result in results:
        print(f"\nRank: {result.get('rank')}")
        print(f"Document ID: {result.get('docid')}")
        print(f"Source: {result.get('source')}")
        print(f"Rerank Score: {result.get('rerank_score')}")
        print(f"Text: {result.get('text', '')[:400]}")
