from datasets import load_dataset
from configs.config import DATASET_NAME, SEED


def load_medqa():
    dataset = load_dataset(DATASET_NAME)

    split = dataset["train"].train_test_split(
        test_size=0.1,
        seed=SEED
    )

    return split["train"], split["test"], dataset["test"]