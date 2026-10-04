from huggingface_hub import hf_hub_download


def load_miracl_files():

    topic_file = hf_hub_download(
        repo_id="miracl/miracl",
        filename="miracl-v1.0-en/topics/topics.miracl-v1.0-en-dev.tsv",
        repo_type="dataset"
    )

    qrel_file = hf_hub_download(
        repo_id="miracl/miracl",
        filename="miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-dev.tsv",
        repo_type="dataset"
    )

    return topic_file, qrel_file


def show_example():

    topic_file, qrel_file = load_miracl_files()

    print("TOPIC EXAMPLE")
    print("=" * 60)

    with open(topic_file, "r", encoding="utf-8") as file:

        for i, line in enumerate(file):

            print(line.strip())

            if i >= 3:
                break

    print("\nQREL EXAMPLE")
    print("=" * 60)

    with open(qrel_file, "r", encoding="utf-8") as file:

        for i, line in enumerate(file):

            print(line.strip())

            if i >= 10:
                break


if __name__ == "__main__":
    show_example()