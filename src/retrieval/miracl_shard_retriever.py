import gzip
import json
import re

from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi


class MIRACLShardRetriever:

    def __init__(self, language="en"):

        self.language = language

        if language == "en":
            self.num_shards = 66
        elif language == "hi":
            self.num_shards = 2
        else:
            raise ValueError(
                f"Unsupported language: {language}"
            )

        self.shards = []

        self.build_indexes()

    def tokenize(self, text):

        return re.findall(
            r"\w+",
            text.lower(),
            flags=re.UNICODE
        )

    def load_shard(self, shard_number):

        filename = (
            f"miracl-corpus-v1.0-{self.language}/"
            f"docs-{shard_number}.jsonl.gz"
        )

        print(
            f"Loading MIRACL {self.language} "
            f"shard {shard_number + 1}/{self.num_shards}"
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

    def build_indexes(self):

        for shard_number in range(self.num_shards):

            passages = self.load_shard(shard_number)

            print(
                f"Building BM25 index for shard "
                f"{shard_number + 1}..."
            )

            documents = [
                self.tokenize(
                    item["title"] + " " + item["text"]
                )
                for item in passages
            ]

            bm25 = BM25Okapi(documents)

            self.shards.append({
                "passages": passages,
                "bm25": bm25
            })

            print(
                f"Shard {shard_number + 1} ready. "
                f"Passages: {len(passages)}"
            )

        print(
            f"\nMIRACL {self.language} retrieval index ready."
        )

    def retrieve(self, claim, top_k=200):

        query_tokens = self.tokenize(claim)

        best_results = []

        for shard in self.shards:

            passages = shard["passages"]
            bm25 = shard["bm25"]

            scores = bm25.get_scores(query_tokens)

            top_indices = scores.argsort()[-top_k:][::-1]

            for index in top_indices:

                score = float(scores[index])

                if score <= 0:
                    continue

                item = passages[index].copy()

                item["relevance_score"] = score

                best_results.append(item)

        best_results.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        best_results = best_results[:top_k]

        results = []

        for rank, item in enumerate(
            best_results,
            start=1
        ):

            item["rank"] = rank
            results.append(item)

        return results