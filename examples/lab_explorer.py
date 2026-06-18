"""Lab Explorer Twin

Fetches publications from EPFL's data index and uses ALIRA to discover those 
related to a given research query via active learning with LLM validation.
"""

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
from pathlib import Path

import pandas as pd
import epfl_data_index as edi

from alira import ActiveLearner
from utils import setup_logging


def main(doc_type: str, query: str) -> None:
    ################################################################

    # Setup logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"results/lab-explorer-{timestamp}")
    logger = setup_logging(output_dir)

    ################################################################

    # Fetch data and embeddings
    logger.info("Fetching documents with type `%s`...", doc_type)
    response = edi.fetch_all(doc_type=doc_type)
    hits = response['hits']['hits']
    df = pd.DataFrame([hit["_source"] for hit in hits])[['id', 'text', 'embedding']]
    logger.info("Fetched %s documents with type %s", len(df), doc_type)

    ################################################################

    # Initialise Active Learner
    learner = ActiveLearner(corpus=df["text"], embeddings=df["embedding"])
    logger.info("Embeddings shape: %s", learner.get_embeddings().shape)

    # Start training
    logger.info("Starting training for query: %s", query)
    learner.fit(query=query)

    # Get predictions
    df["score"] = learner.predict_proba()
    results_df = df[df["score"] >= 0.5].sort_values("score", ascending=False)
    results_df = results_df.drop(columns=["embedding"])

    # Save results
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved results to %s", results_path)

    logger.info("Done!")


if __name__ == "__main__":
    main(doc_type="publication", query="discrete choice models")
