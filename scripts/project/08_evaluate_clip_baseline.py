"""EXP-08.5: matched direct-CLIP baseline for SUN attributes."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


EXP01 = Path("experiment-results/exp01")
EXP08 = Path("experiment-results/exp08")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
TEXT = Path("data/exp01/sun102_clip_RN50_embeddings.pth")
PREDICTIONS = Path("data/exp08/clip_test_predictions.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def threshold_for(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    objective = tpr - fpr
    best = objective.max()
    if best <= 0:
        return np.nextafter(score.max(), np.inf), best, "no_positive_j_all_negative"
    tied = thresholds[np.isclose(objective, best)]
    finite = tied[np.isfinite(tied)]
    return finite.max(), best, "optimized"


def sample_jaccard(y, prediction):
    intersection = np.logical_and(y, prediction).sum(axis=1)
    union = np.logical_or(y, prediction).sum(axis=1)
    return np.divide(
        intersection, union, out=np.zeros(len(y), dtype=float), where=union != 0
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    text = torch.load(TEXT, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    ids = {
        split: torch.tensor(
            [
                int(row["sample_id"])
                for row in splits
                if row["split"].strip().lower() in names
            ],
            dtype=torch.long,
        )
        for split, names in {
            "validation": {"val", "validation"},
            "test": {"test"},
        }.items()
    }
    features = cache["clip_features"].float()
    features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
    embeddings = text["embeddings"].float()
    scores = features @ embeddings.T
    labels = cache["binary_labels"].numpy().astype(bool)

    threshold_rows, thresholds = [], []
    for index, name in enumerate(text["concepts"]):
        y = labels[ids["validation"], index]
        score = scores[ids["validation"], index].numpy()
        threshold, best_j, status = threshold_for(y, score)
        thresholds.append(threshold)
        threshold_rows.append(
            {
                "attribute_index": index,
                "sun_attribute": name,
                "validation_positives": int(y.sum()),
                "threshold": float(threshold),
                "validation_youden_j": float(best_j),
                "threshold_status": status,
            }
        )

    test_scores = scores[ids["test"]].numpy()
    test_labels = labels[ids["test"]]
    test_predictions = test_scores >= np.asarray(thresholds)[None, :]
    metric_rows = []
    for index, name in enumerate(text["concepts"]):
        y, score = test_labels[:, index], test_scores[:, index]
        metric_rows.append(
            {
                "attribute_index": index,
                "sun_attribute": name,
                "test_positives": int(y.sum()),
                "test_prevalence": float(y.mean()),
                "threshold": float(thresholds[index]),
                "predicted_positive_fraction": float(test_predictions[:, index].mean()),
                "test_auroc": float(roc_auc_score(y, score)),
                "test_ap": float(average_precision_score(y, score)),
            }
        )

    write_csv(EXP08 / "clip_validation_thresholds.csv", threshold_rows)
    write_csv(EXP08 / "clip_test_attribute_metrics.csv", metric_rows)
    jaccard = sample_jaccard(test_labels, test_predictions)
    summary = {
        "status": "documented implementation, not exact reproduction",
        "test_images": len(ids["test"]),
        "sun_attributes": len(text["concepts"]),
        "score": "cosine(CLIP image embedding, CLIP text embedding)",
        "prompt_template": text["prompt_template"],
        "threshold_selection_split": "validation",
        "threshold_objective": "maximize Youden J (TPR - FPR)",
        "optimized_thresholds": sum(r["threshold_status"] == "optimized" for r in threshold_rows),
        "mean_image_jaccard_all_102": float(jaccard.mean()),
        "macro_test_auroc_all_102": float(np.mean([r["test_auroc"] for r in metric_rows])),
        "macro_test_ap_all_102": float(np.mean([r["test_ap"] for r in metric_rows])),
        "published_clip_jaccard_reference": 0.220,
        "reference_is_not_a_reproduction_target": True,
        "prediction_cache": str(PREDICTIONS),
    }
    (EXP08 / "clip_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "sample_ids": ids["test"],
            "labels": torch.from_numpy(test_labels),
            "predictions": torch.from_numpy(test_predictions),
        },
        PREDICTIONS,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
