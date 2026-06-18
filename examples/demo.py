"""Active learning demo with sample datasets."""

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
from pathlib import Path

import numpy as np

from alira import ActiveLearner
from utils import load_data, resolve_embeddings_path, setup_logging


def run(dataset: str, query: str, text_column: str = "text") -> None:
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"results/{dataset}-{timestamp}")
    logger = setup_logging(output_dir)

    ################################################################

    # Load data and embeddings
    df = load_data(dataset)
    logger.info("Loaded %s rows", len(df))
    if text_column not in df.columns:
        raise ValueError(f"Dataset '{dataset}' has no column '{text_column}'")

    embedding_path = resolve_embeddings_path(dataset)
    if embedding_path.exists():
        logger.info("Loading cached embeddings...")
        embeddings = np.load(embedding_path)
        logger.info("Loaded embeddings with shape %s", embeddings.shape)
    else:
        logger.info("No cached embeddings found, will compute on demand.")
        embeddings = None

    ################################################################

    # Initialise Active Learner
    learner = ActiveLearner(corpus=df[text_column], embeddings=embeddings)
    logger.info("Embeddings shape: %s", learner.get_embeddings().shape)

    # Start training
    logger.info("Starting training for query: %s", query)
    learner.fit(query=query)

    # Get predictions
    df["score"] = learner.predict_proba()
    results_df = df[df["score"] >= 0.5].sort_values("score", ascending=False)

    # Save results
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved results to %s", results_path)

    logger.info("Done!")


if __name__ == "__main__":
    run(dataset="movies", query="superheroes")
