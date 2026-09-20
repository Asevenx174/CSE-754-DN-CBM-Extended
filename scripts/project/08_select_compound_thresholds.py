"""EXP-08.3: select SUN compound-node thresholds on validation data."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve


EXP01 = Path("experiment-results/exp01")
EXP08 = Path("experiment-results/exp08")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
SCORES = Path("data/exp08/compound_scores.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_threshold(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    objective = tpr - fpr
    best = objective.max()
    if best <= 0:
        return np.nextafter(score.max(), np.inf), best, "no_positive_j_all_negative"
    tied = thresholds[np.isclose(objective, best)]
    finite = tied[np.isfinite(tied)]
    threshold = finite.max() if len(finite) else np.nextafter(score.max(), np.inf)
    return threshold, best, "optimized"


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    compound = torch.load(SCORES, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    val_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "validation"],
        dtype=torch.long,
    )
    if not len(val_ids):
        val_ids = torch.tensor(
            [int(row["sample_id"]) for row in splits if row["split"] == "val"],
            dtype=torch.long,
        )

    labels = cache["binary_labels"][val_ids].numpy()
    scores = compound["compound_scores"][val_ids].numpy()
    available = compound["available_attributes"].numpy()
    names = compound["attribute_names"]
    rows = []
    for index, name in enumerate(names):
        if not available[index]:
            continue
        y, score = labels[:, index], scores[:, index]
        threshold, best_j, status = select_threshold(y, score)
        rows.append(
            {
                "attribute_index": index,
                "sun_attribute": name,
                "validation_prevalence": float(y.mean()),
                "validation_positives": int(y.sum()),
                "threshold": float(threshold),
                "validation_youden_j": float(best_j),
                "threshold_status": status,
            }
        )

    path = EXP08 / "compound_validation_thresholds.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "documented implementation, not exact reproduction",
        "selection_split": "validation",
        "threshold_objective": "maximize Youden J (TPR - FPR)",
        "tie_rule": "highest threshold",
        "available_attributes": len(rows),
        "optimized_thresholds": sum(r["threshold_status"] == "optimized" for r in rows),
        "all_negative_fallbacks": sum(
            r["threshold_status"] != "optimized" for r in rows
        ),
        "test_metrics_computed": False,
    }
    (EXP08 / "compound_threshold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
