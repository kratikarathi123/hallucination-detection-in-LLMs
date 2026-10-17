from src.retrieval.vaarta_marathi_retriever import VaartaMarathiRetriever

CORPUS = "data/processed/sangraha_verified_corpus.json"

queries = [
    "The Taj Mahal is located in Agra.",
    "The Taj Mahal was commissioned by Shah Jahan.",
    "Construction of the Taj Mahal began in 1632.",
]

retriever = VaartaMarathiRetriever(
    corpus_path=CORPUS,
    candidate_top_k=50,
    final_top_k=3,
)

for query in queries:
    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = retriever.retrieve(query)

    if not results:
        print("No results returned.")
        continue

    for result in results:
        print(f"\nRank: {result.get('rank')}")
        print(f"Document ID: {result.get('docid')}")
        print(f"Rerank Score: {result.get('rerank_score')}")
        print(f"Text: {result.get('text', '')[:500]}")
