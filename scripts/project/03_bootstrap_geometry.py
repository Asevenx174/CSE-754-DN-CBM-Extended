"""EXP-03.2: concept-aware bootstrap CIs for geometry--AUROC correlations."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


INPUT = Path("experiment-results/exp02")
OUTPUT = Path("experiment-results/exp03/geometry_bootstrap_ci.json")
REPEATS, SEED = 2000, 49


def load(protocol):
    path = INPUT / f"protocol_{protocol.lower()}_atom_scores.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    groups = defaultdict(list)
    for row in rows:
        groups[row["mapped_sun_attribute"]].append(
            (float(row["geometry_cosine"]), float(row["auroc"]))
        )
    return [np.asarray(group) for group in groups.values()]


def correlation(points):
    value = spearmanr(points[:, 0], points[:, 1]).statistic
    return float(value)


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "lower_95": float(low),
        "median": float(np.median(values)),
        "upper_95": float(high),
        "valid_replicates": int(len(values)),
    }


def bootstrap(protocol, rng):
    groups = load(protocol)
    atom_values, concept_values = [], []
    for _ in range(REPEATS):
        selected = rng.integers(0, len(groups), len(groups))
        atom_samples, concept_means = [], []
        for index in selected:
            group = groups[index]
            sample = group[rng.integers(0, len(group), len(group))]
            atom_samples.append(sample)
            concept_means.append(sample.mean(axis=0))
        atom_values.append(correlation(np.vstack(atom_samples)))
        concept_values.append(correlation(np.vstack(concept_means)))
    return {
        "bootstrap_replicates": REPEATS,
        "seed": SEED,
        "resampling": "concepts, then atoms within sampled concepts",
        "atom_level_rho": interval(atom_values),
        "concept_balanced_rho": interval(concept_values),
    }


def main():
    rng = np.random.default_rng(SEED)
    results = {protocol: bootstrap(protocol, rng) for protocol in ("A", "B")}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
