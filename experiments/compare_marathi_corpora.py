import gc

from src.retrieval.vaarta_marathi_retriever import VaartaMarathiRetriever
from src.retrieval.reranker import EvidenceReranker

CORPORA = [
    ("MARATHI WIKIPEDIA", "data/processed/marathi_wikipedia_corpus.json"),
    ("SANGRAHA VERIFIED", "data/processed/sangraha_verified_corpus.json"),
]

QUERIES = [
    "The Taj Mahal is located in Agra.",
    "The Taj Mahal was commissioned by Shah Jahan.",
    "Construction of the Taj Mahal began in 1632.",
]

reranker = EvidenceReranker()

for corpus_name, corpus_path in CORPORA:
    print("\n" + "#" * 90)
    print("CORPUS:", corpus_name)
    print("#" * 90)

    retriever = VaartaMarathiRetriever(
        corpus_path=corpus_path,
        candidate_top_k=50,
        final_top_k=3,
        reranker=reranker,
    )

    for query in QUERIES:
        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        results = retriever.retrieve(query)

        for result in results:
            print(f"\nRank: {result.get('rank')}")
            print(f"Doc ID: {result.get('docid')}")
            print(f"Rerank Score: {result.get('rerank_score')}")
            print(f"Text: {result.get('text', '')[:500]}")

    del retriever
    gc.collect()

print("\nComparison test complete.")
