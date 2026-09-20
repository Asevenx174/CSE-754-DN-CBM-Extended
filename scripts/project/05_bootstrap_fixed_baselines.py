"""EXP-05.3: paired bootstrap for raw maximum versus selected atom."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score


EXP05 = Path("experiment-results/exp05")
PREDICTIONS = Path("data/exp05/baseline_test_predictions.pth")
REPEATS, SEED = 2000, 49


def differences(y, single, maximum, index):
    labels = y[index]
    return (
        roc_auc_score(labels, maximum[index])
        - roc_auc_score(labels, single[index]),
        average_precision_score(labels, maximum[index])
        - average_precision_score(labels, single[index]),
    )


def bootstrap(key, values, seed):
    y = values["labels"].numpy()
    single = values["single"].numpy()
    maximum = values["raw_max"].numpy()
    rng = np.random.default_rng(seed)
    observed_auc, observed_ap = differences(
        y, single, maximum, np.arange(len(y))
    )
    samples = []
    while len(samples) < REPEATS:
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            samples.append(differences(y, single, maximum, index))
    samples = np.asarray(samples)
    auc_low, auc_high = np.percentile(samples[:, 0], [2.5, 97.5])
    ap_low, ap_high = np.percentile(samples[:, 1], [2.5, 97.5])
    protocol, concept = key.split("|", 1)
    return {
        "protocol": protocol,
        "concept": concept,
        "delta_auroc": observed_auc,
        "delta_auroc_ci_lower": auc_low,
        "delta_auroc_ci_upper": auc_high,
        "delta_ap": observed_ap,
        "delta_ap_ci_lower": ap_low,
        "delta_ap_ci_upper": ap_high,
        "bootstrap_replicates": REPEATS,
        "seed": seed,
    }


def main():
    cache = torch.load(PREDICTIONS, map_location="cpu", weights_only=True)
    items = sorted(cache["predictions"].items())
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(bootstrap)(key, values, SEED + index)
        for index, (key, values) in enumerate(items)
    )

    path = EXP05 / "fixed_baseline_paired_bootstrap.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for protocol in ("A", "B"):
        subset = [row for row in rows if row["protocol"] == protocol]
        summary[protocol] = {
            "concepts": len(subset),
            "auroc_ci_above_zero": sum(r["delta_auroc_ci_lower"] > 0 for r in subset),
            "auroc_ci_below_zero": sum(r["delta_auroc_ci_upper"] < 0 for r in subset),
            "auroc_ci_includes_zero": sum(
                r["delta_auroc_ci_lower"] <= 0 <= r["delta_auroc_ci_upper"]
                for r in subset
            ),
            "ap_ci_above_zero": sum(r["delta_ap_ci_lower"] > 0 for r in subset),
            "ap_ci_below_zero": sum(r["delta_ap_ci_upper"] < 0 for r in subset),
            "ap_ci_includes_zero": sum(
                r["delta_ap_ci_lower"] <= 0 <= r["delta_ap_ci_upper"]
                for r in subset
            ),
        }
    summary["resampling"] = "paired test-image bootstrap with replacement"
    summary["bootstrap_replicates"] = REPEATS
    (EXP05 / "fixed_baseline_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: value.item()), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
