"""Run demo.py for every theme in aists_themes.json against the given dataset."""

import json
from pathlib import Path

from dotenv import load_dotenv

from demo import run
from utils import load_data, resolve_data_path


load_dotenv()


def prepare_publications() -> None:
    dataset = "aists_publications"
    data_path = resolve_data_path(dataset)
    df = load_data(dataset)
    output_path = data_path.with_name(f"{dataset}_processed.parquet")

    print(f"Loaded {len(df)} rows from {data_path}")

    df["text"] = (
        df["title"].fillna("").astype(str)
        + "\n\n"
        + df["abstract"].fillna("").astype(str)
    ).str.strip()

    df = df[["openalex_id", "doi", "text"]]

    df.to_parquet(output_path, index=False)
    print(f"Saved processed dataset ({len(df)} rows) to {output_path}")


def prepare_rc_vaud_full() -> None:
    dataset = "aists_rc_vaud_full"
    data_path = resolve_data_path(dataset)
    df = load_data(dataset)
    output_path = data_path

    print(f"Loaded {len(df)} rows from {data_path}")

    df["text"] = df["text"].fillna(df["name"])

    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset ({len(df)} rows) to {output_path}")


def prepare_zefix_extracted() -> None:
    dataset = "aists_zefix_extracted"
    data_path = resolve_data_path(dataset)
    df = load_data(dataset)
    output_path = data_path

    print(f"Loaded {len(df)} rows from {data_path}")

    df["text"] = df["text"].fillna(df["company_name"])

    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset ({len(df)} rows) to {output_path}")


def main(dataset: str) -> None:
    script_dir = Path(__file__).resolve().parent
    themes_path = script_dir / "data" / "aists_themes.json"

    with open(themes_path) as f:
        themes = json.load(f)["themes"]

    for theme in themes:
        theme_id = theme["id"]
        query = theme["optimized_translation"]
        print(f"--- Running: {theme_id} -> '{query}' ---")
        run(dataset, query)
        print(f"--- Done: {theme_id} ---\n")


if __name__ == "__main__":
    main("aists_zefix_extracted")
