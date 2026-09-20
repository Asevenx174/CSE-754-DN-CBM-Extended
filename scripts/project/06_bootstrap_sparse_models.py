"""EXP-06.3: paired bootstrap for sparse aggregation versus both baselines."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score


EXP06 = Path("experiment-results/exp06")
PREDICTIONS = Path("data/exp06/sparse_test_predictions.pth")
REPEATS, SEED = 2000, 49


def metrics(y, single, maximum, sparse, index):
    labels = y[index]
    scores = [single[index], maximum[index], sparse[index]]
    auc = [roc_auc_score(labels, score) for score in scores]
    ap = [average_precision_score(labels, score) for score in scores]
    return auc[2] - auc[0], auc[2] - auc[1], ap[2] - ap[0], ap[2] - ap[1]


def bootstrap(key, values, seed):
    arrays = [values[name].numpy() for name in ("labels", "single", "raw_max", "sparse")]
    y, single, maximum, sparse = arrays
    observed = metrics(y, single, maximum, sparse, np.arange(len(y)))
    rng, samples = np.random.default_rng(seed), []
    while len(samples) < REPEATS:
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            samples.append(metrics(y, single, maximum, sparse, index))
    samples = np.asarray(samples)
    protocol, concept = key.split("|", 1)
    row = {"protocol": protocol, "concept": concept, "seed": seed}
    names = (
        "auroc_sparse_minus_single",
        "auroc_sparse_minus_raw_max",
        "ap_sparse_minus_single",
        "ap_sparse_minus_raw_max",
    )
    for column, name in enumerate(names):
        low, high = np.percentile(samples[:, column], [2.5, 97.5])
        row[f"delta_{name}"] = observed[column]
        row[f"{name}_ci_lower"] = low
        row[f"{name}_ci_upper"] = high
    return row


def main():
    cache = torch.load(PREDICTIONS, map_location="cpu", weights_only=True)
    items = sorted(cache["predictions"].items())
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(bootstrap)(key, values, SEED + index)
        for index, (key, values) in enumerate(items)
    )

    path = EXP06 / "sparse_paired_bootstrap.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    comparisons = (
        "auroc_sparse_minus_single",
        "auroc_sparse_minus_raw_max",
        "ap_sparse_minus_single",
        "ap_sparse_minus_raw_max",
    )
    for protocol in ("A", "B"):
        subset = [row for row in rows if row["protocol"] == protocol]
        summary[protocol] = {"concepts": len(subset)}
        for comparison in comparisons:
            lower = [row[f"{comparison}_ci_lower"] for row in subset]
            upper = [row[f"{comparison}_ci_upper"] for row in subset]
            summary[protocol][comparison] = {
                "ci_above_zero": int(sum(value > 0 for value in lower)),
                "ci_below_zero": int(sum(value < 0 for value in upper)),
                "ci_includes_zero": int(
                    sum(low <= 0 <= high for low, high in zip(lower, upper))
                ),
            }
    summary["resampling"] = "paired test-image bootstrap with replacement"
    summary["bootstrap_replicates"] = REPEATS
    (EXP06 / "sparse_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: value.item()),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
