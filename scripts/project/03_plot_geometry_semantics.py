"""EXP-03.3: descriptive cosine plots and fixed-cutoff audit."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("experiment-results/exp02")
OUTPUT = Path("experiment-results/exp03")
CUTOFFS = [None, 0.00, 0.10, 0.20, 0.30, 0.40]
SEMANTIC_CRITERION = 0.70


def load(protocol):
    path = INPUT / f"protocol_{protocol.lower()}_atom_scores.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return np.array(
        [(float(r["geometry_cosine"]), float(r["auroc"])) for r in rows]
    )


def summarize(protocol, points):
    rows = []
    total = len(points)
    for cutoff in CUTOFFS:
        kept = points if cutoff is None else points[points[:, 0] >= cutoff]
        scores = kept[:, 1] if len(kept) else np.array([])
        rows.append(
            {
                "protocol": protocol,
                "minimum_cosine": "all" if cutoff is None else f"{cutoff:.2f}",
                "retained_atoms": len(kept),
                "retained_fraction": len(kept) / total,
                "auroc_q25": np.quantile(scores, 0.25) if len(scores) else "",
                "median_auroc": np.median(scores) if len(scores) else "",
                "mean_auroc": np.mean(scores) if len(scores) else "",
                "auroc_q75": np.quantile(scores, 0.75) if len(scores) else "",
                "fraction_auroc_at_least_0.70": (
                    np.mean(scores >= SEMANTIC_CRITERION) if len(scores) else ""
                ),
            }
        )
    return rows


def plot(protocol, points):
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(points[:, 0], points[:, 1], s=10, alpha=0.35)
    axis.axhline(SEMANTIC_CRITERION, color="tab:red", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Protocol-specific decoder–text cosine",
        ylabel="Held-out AUROC",
        title=f"Protocol {protocol}: geometry versus semantic detection",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / f"protocol_{protocol.lower()}_cosine_vs_auroc.png", dpi=200)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for protocol in ("A", "B"):
        points = load(protocol)
        summaries.extend(summarize(protocol, points))
        plot(protocol, points)

    path = OUTPUT / "cosine_cutoff_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {path} and two scatter plots.")


if __name__ == "__main__":
    main()
