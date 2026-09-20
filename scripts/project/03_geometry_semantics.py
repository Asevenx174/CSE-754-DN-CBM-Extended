"""EXP-03.1: geometry--semantics point estimates."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path("experiment-results/exp02")
OUT = Path("experiment-results/exp03")


def read(protocol):
    path = ROOT / f"protocol_{protocol.lower()}_atom_scores.csv"
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def rho(rows):
    x = [float(row["geometry_cosine"]) for row in rows]
    y = [float(row["auroc"]) for row in rows]
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def analyze(protocol):
    rows = read(protocol)
    atom_rho, atom_p = rho(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["mapped_sun_attribute"]].append(row)

    means = []
    within = []
    for concept, group in sorted(groups.items()):
        means.append(
            {
                "geometry_cosine": np.mean(
                    [float(row["geometry_cosine"]) for row in group]
                ),
                "auroc": np.mean([float(row["auroc"]) for row in group]),
            }
        )
        if len(group) >= 3:
            value, pvalue = rho(group)
            if np.isfinite(value):
                within.append(
                    {
                        "protocol": protocol,
                        "concept": concept,
                        "atoms": len(group),
                        "spearman_rho": value,
                        "pvalue": pvalue,
                    }
                )

    concept_rho, concept_p = rho(means)
    return {
        "atoms": len(rows),
        "concepts": len(groups),
        "atom_level_rho": atom_rho,
        "atom_level_pvalue": atom_p,
        "concept_balanced_rho": concept_rho,
        "concept_balanced_pvalue": concept_p,
        "within_concept_correlations": len(within),
    }, within


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    within = []
    for protocol in ("A", "B"):
        summary[protocol], values = analyze(protocol)
        within.extend(values)

    (OUT / "geometry_semantics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if within:
        with (OUT / "within_concept_correlations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=within[0].keys())
            writer.writeheader()
            writer.writerows(within)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
