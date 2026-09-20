"""EXP-08.4: evaluate frozen SUN compound nodes on the test split."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


EXP01 = Path("experiment-results/exp01")
EXP08 = Path("experiment-results/exp08")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
SCORES = Path("data/exp08/compound_scores.pth")
PREDICTIONS = Path("data/exp08/sae_compound_test_predictions.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sample_jaccard(y, prediction):
    intersection = np.logical_and(y, prediction).sum(axis=1)
    union = np.logical_or(y, prediction).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0,
    )


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    compound = torch.load(SCORES, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    test_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "test"],
        dtype=torch.long,
    )
    labels = cache["binary_labels"][test_ids].numpy().astype(bool)
    scores = compound["compound_scores"][test_ids].numpy()
    available = compound["available_attributes"].numpy().astype(bool)
    names = compound["attribute_names"]
    thresholds = {
        row["sun_attribute"]: float(row["threshold"])
        for row in read(EXP08 / "compound_validation_thresholds.csv")
    }

    predictions = np.zeros_like(labels, dtype=bool)
    rows = []
    for index, name in enumerate(names):
        if not available[index]:
            continue
        prediction = scores[:, index] >= thresholds[name]
        predictions[:, index] = prediction
        y = labels[:, index]
        rows.append(
            {
                "attribute_index": index,
                "sun_attribute": name,
                "test_positives": int(y.sum()),
                "test_prevalence": float(y.mean()),
                "threshold": thresholds[name],
                "predicted_positive_fraction": float(prediction.mean()),
                "test_auroc": float(roc_auc_score(y, scores[:, index])),
                "test_ap": float(average_precision_score(y, scores[:, index])),
            }
        )

    available_jaccard = sample_jaccard(
        labels[:, available], predictions[:, available]
    )
    all_jaccard = sample_jaccard(labels, predictions)
    available_union = np.logical_or(
        labels[:, available], predictions[:, available]
    ).sum(axis=1)
    with (EXP08 / "sae_compound_test_attribute_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "documented implementation, not exact reproduction",
        "test_images": len(test_ids),
        "threshold_selection_split": "validation",
        "available_attributes": int(available.sum()),
        "unavailable_attributes": [
            name for name, keep in zip(names, available) if not keep
        ],
        "mean_image_jaccard_available_100": float(available_jaccard.mean()),
        "mean_image_jaccard_all_102_unavailable_negative": float(all_jaccard.mean()),
        "images_with_empty_union_available_100": int((available_union == 0).sum()),
        "macro_test_auroc_available_100": float(
            np.mean([row["test_auroc"] for row in rows])
        ),
        "macro_test_ap_available_100": float(
            np.mean([row["test_ap"] for row in rows])
        ),
        "published_sae_jaccard_reference": 0.183,
        "reference_is_not_a_reproduction_target": True,
        "prediction_cache": str(PREDICTIONS),
    }
    (EXP08 / "sae_compound_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sample_ids": test_ids,
            "labels": torch.from_numpy(labels),
            "predictions": torch.from_numpy(predictions),
            "available_attributes": torch.from_numpy(available),
            "attribute_names": names,
        },
        PREDICTIONS,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
