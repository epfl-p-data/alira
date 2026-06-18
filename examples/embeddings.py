"""Generate and cache embeddings in a file next to the dataset."""

from dotenv import load_dotenv

load_dotenv()

import numpy as np

from alira import ActiveLearner
from utils import load_data, resolve_embeddings_path


def make_embeddings(dataset: str) -> None:
    df = load_data(dataset)
    print(f"Loaded {len(df)} rows")

    learner = ActiveLearner(corpus=df["text"])
    embeddings = learner.get_embeddings()

    embeddings_path = resolve_embeddings_path(dataset)
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings ({embeddings.shape}) to {embeddings_path}")


if __name__ == "__main__":
    make_embeddings("aists_zefix_extracted")
