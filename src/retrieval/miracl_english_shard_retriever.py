import gzip
import json
import re

from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi


class MIRACLEnglishShardRetriever:

    def __init__(self):

        self.language = "en"
        self.num_shards = 66

    def tokenize(self, text):

        return re.findall(
            r"\w+",
            text.lower()
        )

    def load_shard(self, shard_number):

        filename = (
            f"miracl-corpus-v1.0-en/"
            f"docs-{shard_number}.jsonl.gz"
        )

        print(
            f"\nLoading English shard "
            f"{shard_number + 1}/{self.num_shards}"
        )

        file_path = hf_hub_download(
            repo_id="miracl/miracl-corpus",
            filename=filename,
            repo_type="dataset"
        )

        passages = []

        with gzip.open(
            file_path,
            "rt",
            encoding="utf-8"
        ) as file:

            for line in file:

                item = json.loads(line)

                passages.append({
                    "docid": item["docid"],
                    "title": item["title"],
                    "text": item["text"]
                })

        return passages

    def retrieve_many(
        self,
        claims,
        top_k=200
    ):

        query_tokens = [
            self.tokenize(claim)
            for claim in claims
        ]

        global_results = [
            [] for _ in claims
        ]

        for shard_number in range(
            self.num_shards
        ):

            passages = self.load_shard(
                shard_number
            )

            print(
                f"Building BM25 for shard "
                f"{shard_number + 1}..."
            )

            documents = [
                self.tokenize(
                    item["title"] + " " + item["text"]
                )
                for item in passages
            ]

            bm25 = BM25Okapi(documents)

            for claim_index, tokens in enumerate(
                query_tokens
            ):

                scores = bm25.get_scores(
                    tokens
                )

                shard_top_k = min(
                    top_k,
                    len(scores)
                )

                top_indices = scores.argsort()[
                    -shard_top_k:
                ][::-1]

                for index in top_indices:

                    score = float(
                        scores[index]
                    )

                    if score <= 0:
                        continue

                    item = passages[index].copy()

                    item["relevance_score"] = score

                    global_results[
                        claim_index
                    ].append(item)

                global_results[
                    claim_index
                ].sort(
                    key=lambda x: x[
                        "relevance_score"
                    ],
                    reverse=True
                )

                global_results[
                    claim_index
                ] = global_results[
                    claim_index
                ][:top_k]

            print(
                f"Processed shard "
                f"{shard_number + 1}/"
                f"{self.num_shards}"
            )

            del passages
            del documents
            del bm25

        final_results = []

        for results in global_results:

            ranked_results = []

            for rank, item in enumerate(
                results,
                start=1
            ):

                item["rank"] = rank

                ranked_results.append(item)

            final_results.append(
                ranked_results
            )

        return final_results

    def retrieve(
        self,
        claim,
        top_k=200
    ):

        results = self.retrieve_many(
            [claim],
            top_k=top_k
        )

        return results[0]