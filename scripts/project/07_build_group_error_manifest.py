"""EXP-07.3: export recovered examples and errors for frozen group cases."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve


EXP01 = Path("experiment-results/exp01")
EXP06 = Path("experiment-results/exp06")
EXP07 = Path("experiment-results/exp07")
ACTIVATIONS = Path("data/exp02/sun_frozen_cache.pth")
PREDICTIONS = Path("data/exp06/sparse_test_predictions.pth")
PER_TYPE = 5


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def threshold(y, score):
    score = np.asarray(score, dtype=np.float32)
    above_maximum = np.nextafter(
        np.max(score), np.float32(np.inf)
    )

    if np.unique(score).size < 2:
        return (
            float(above_maximum),
            0.0,
            "constant_all_negative",
        )

    false_positive, true_positive, values = roc_curve(
        y, score
    )
    objective = true_positive - false_positive
    values = values.astype(np.float32, copy=True)
    values[~np.isfinite(values)] = above_maximum
    best = objective.max()

    if best <= 0:
        return (
            float(above_maximum),
            float(best),
            "no_positive_j_all_negative",
        )

    chosen = values[
        np.isclose(objective, best)
    ].max()

    return float(chosen), float(best), "optimized"


def main():
    cache = torch.load(ACTIVATIONS, map_location="cpu", weights_only=True)
    prediction_cache = torch.load(PREDICTIONS, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    paths = {int(row["sample_id"]): row["relative_path"] for row in splits}
    val_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "validation"],
        dtype=torch.long,
    )
    test_ids = prediction_cache["sample_ids"].long()
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    models = {
        (row["protocol"], row["concept"]): row
        for row in read(EXP06 / "selected_sparse_models.csv")
    }

    thresholds, manifest = [], []
    for case in read(EXP07 / "selected_group_cases.csv"):
        protocol, concept = case["protocol"], case["concept"]
        key = f"{protocol}|{concept}"
        model = models[(protocol, concept)]
        atoms = np.asarray(json.loads(model["candidate_atom_ids"]), dtype=int)
        mean = np.asarray(json.loads(model["training_mean"]), dtype=float)
        scale = np.asarray(json.loads(model["training_scale"]), dtype=float)
        coefficients = np.asarray(json.loads(model["coefficients"]), dtype=float)
        intercept = float(model["intercept"])

        val_x = cache["sae_activations"][val_ids][:, atoms].numpy()
        val_y = cache["binary_labels"][val_ids, concept_index[concept]].numpy()
        val_sparse = (
            ((val_x - mean) / scale) @ coefficients
            + intercept
        ).astype(np.float32)
        val_maximum = val_x.max(axis=1).astype(np.float32)

        (
            sparse_threshold,
            sparse_j,
            sparse_status,
        ) = threshold(val_y, val_sparse)

        (
            maximum_threshold,
            maximum_j,
            maximum_status,
        ) = threshold(val_y, val_maximum)

        test = prediction_cache["predictions"][key]
        y = test["labels"].numpy().astype(bool)
        sparse = test["sparse"].numpy().astype(np.float32)
        maximum = test["raw_max"].numpy().astype(np.float32)
        sparse_positive = sparse >= sparse_threshold
        maximum_positive = maximum >= maximum_threshold
        masks = {
            "positive_recovered_by_sparse": y & sparse_positive & ~maximum_positive,
            "positive_lost_by_sparse": y & ~sparse_positive & maximum_positive,
            "false_positive_corrected_by_sparse": ~y & ~sparse_positive & maximum_positive,
            "false_positive_introduced_by_sparse": ~y & sparse_positive & ~maximum_positive,
        }
        thresholds.append(
            {
                "protocol": protocol,
                "case_category": case["case_category"],
                "concept": concept,
                "sparse_validation_threshold": sparse_threshold,
                "raw_max_validation_threshold": maximum_threshold,
                "sparse_validation_youden_j": sparse_j,
                "raw_max_validation_youden_j": maximum_j,
                "sparse_threshold_status": sparse_status,
                "raw_max_threshold_status": maximum_status,
                **{
                    f"count_{name}": int(mask.sum())
                    for name, mask in masks.items()
                },
            }
        )

        for kind, mask in masks.items():
            indices = np.where(mask)[0]
            if "corrected" in kind or "lost" in kind:
                order = np.argsort(maximum[indices] - maximum_threshold)[::-1]
            else:
                order = np.argsort(sparse[indices] - sparse_threshold)[::-1]
            for rank, index in enumerate(indices[order[:PER_TYPE]], start=1):
                sample_id = int(test_ids[index])
                manifest.append(
                    {
                        "protocol": protocol,
                        "case_category": case["case_category"],
                        "concept": concept,
                        "example_type": kind,
                        "rank": rank,
                        "sample_id": sample_id,
                        "relative_path": paths[sample_id],
                        "binary_label": int(y[index]),
                        "worker_agreement": float(
                            cache["agreement"][sample_id, concept_index[concept]]
                        ),
                        "sparse_score": float(sparse[index]),
                        "raw_max_score": float(maximum[index]),
                    }
                )

    for filename, rows in (
        ("group_error_counts.csv", thresholds),
        ("group_error_manifest.csv", manifest),
    ):
        with (EXP07 / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Audited {len(thresholds)} groups and exported {len(manifest)} image records.")


if __name__ == "__main__":
    main()
