"""Shared utilities for example scripts."""

import logging
from pathlib import Path

import pandas as pd


def resolve_data_path(dataset: str) -> Path:
    """Return the path to a dataset, preferring CSV over Parquet."""
    base_path = Path(f"data/{dataset}")
    csv_path = base_path.with_suffix(".csv")
    parquet_path = base_path.with_suffix(".parquet")

    if csv_path.exists():
        return csv_path
    if parquet_path.exists():
        return parquet_path
    raise FileNotFoundError(f"No data file found at {csv_path} or {parquet_path}")


def load_data(dataset: str) -> pd.DataFrame:
    """Load a dataset by name, supporting CSV and Parquet."""
    path = resolve_data_path(dataset)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def resolve_embeddings_path(dataset: str) -> Path:
    """Return the path to pre-computed embeddings for a dataset."""
    return Path(f"data/{dataset}_embeddings.npy")


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both a timestamped file and the console."""
    output_dir.mkdir(exist_ok=True, parents=True)
    log_path = output_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)
