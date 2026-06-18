"""
Compare two result sets to see how different they are.

Usage:
    python compare.py <folder1> <folder2>

The folders are resolved relative to examples/results/.
"""
import sys
from pathlib import Path

import pandas as pd


def main(name1: str, name2: str) -> None:
    results_dir = Path(__file__).parent / "results"
    result1 = pd.read_csv(results_dir / name1 / "results.csv")
    result2 = pd.read_csv(results_dir / name2 / "results.csv")

    print(f"Result 1 ({name1}): {len(result1)} rows")
    print(f"Result 2 ({name2}): {len(result2)} rows")

    # Compare common IDs
    ids1 = set(result1["id"])
    ids2 = set(result2["id"])

    common_ids = ids1 & ids2
    only_in_1 = ids1 - ids2
    only_in_2 = ids2 - ids1

    print(f"\nCommon IDs: {len(common_ids)}")
    print(f"Only in {name1}: {len(only_in_1)}")
    print(f"Only in {name2}: {len(only_in_2)}")

    if only_in_1:
        print(f"\nOnly in {name1}:")
        for id_ in only_in_1:
            name = result1[result1["id"] == id_]["name"].iloc[0]
            print(f"  - {id_}: {name}")

    if only_in_2:
        print(f"\nOnly in {name2}:")
        for id_ in only_in_2:
            name = result2[result2["id"] == id_]["name"].iloc[0]
            print(f"  - {id_}: {name}")

    # Compare scores and predictions for common items
    if common_ids:
        print("\nDetailed comparison for common items:")
        print(f"{'ID':<20} {'Name':<60} {'Score1(k)':<12} {'Score2(nk)':<12} {'Pred1':<8} {'Pred2':<8} {'Conf1':<8} {'Conf2':<8} {'Match'}")
        print("=" * 140)

        total_score_diff = 0
        score_diffs = []
        prediction_matches = 0

        for id_ in common_ids:
            row1 = result1[result1["id"] == id_].iloc[0]
            row2 = result2[result2["id"] == id_].iloc[0]

            score1 = float(row1["score"])
            score2 = float(row2["score"])
            diff = abs(score1 - score2)
            total_score_diff += diff
            score_diffs.append(diff)

            pred_match = row1["prediction_binary"] == row2["prediction_binary"]
            if pred_match:
                prediction_matches += 1

            name = str(row1["name"])[:55]
            match_marker = "✓" if pred_match and diff < 0.05 else "✗"

            print(f"{str(id_):<20} {name:<60} {score1:<12.6f} {score2:<12.6f} {str(row1['prediction_binary']):<8} {str(row2['prediction_binary']):<8} {float(row1['confidence']):<8.4f} {float(row2['confidence']):<8.4f} {match_marker}")

        print("=" * 140)
        print(f"\nSummary:")
        print(f"  Average score difference: {total_score_diff / len(common_ids):.6f}")
        print(f"  Max score difference: {max(score_diffs):.6f}")
        print(f"  Min score difference: {min(score_diffs):.6f}")
        print(f"  Median score difference: {sorted(score_diffs)[len(score_diffs)//2]:.6f}")
        print(f"  Prediction agreement: {prediction_matches}/{len(common_ids)} ({100*prediction_matches/len(common_ids):.1f}%)")

    # Show top 5 scores for each
    print(f"\n\nTop 5 scores ({name1}):")
    for _, row in result1.nlargest(5, "score").iterrows():
        print(f"  {row['score']:.4f} - {row['name'][:60]}")

    print(f"\nTop 5 scores ({name2}):")
    for _, row in result2.nlargest(5, "score").iterrows():
        print(f"  {row['score']:.4f} - {row['name'][:60]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {Path(__file__).name} <folder1> <folder2>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
