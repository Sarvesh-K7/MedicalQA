from datasets import load_dataset
from models.tokenizer import get_tokenizer
from src.data.preprocess import create_messages

import numpy as np


def analyze():

    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    tokenizer = get_tokenizer()

    lengths = []

    for sample in dataset["train"]:

        messages = create_messages(sample)["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        token_len = len(tokenizer(text)["input_ids"])
        lengths.append(token_len)

    print("\n===== DATASET ANALYSIS =====\n")

    print("Samples:", len(lengths))
    print("Avg tokens:", np.mean(lengths))
    print("Max tokens:", np.max(lengths))
    print("95th percentile:", np.percentile(lengths, 95))
    print("99th percentile:", np.percentile(lengths, 99))

    print("\nAnswer distribution:")
    print(dataset["train"].to_pandas()["answer_idx"].value_counts())


if __name__ == "__main__":
    analyze()