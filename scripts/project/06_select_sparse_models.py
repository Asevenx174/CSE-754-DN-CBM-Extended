"""EXP-06.1: train L1 readouts and select regularization on validation data."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


EXP01 = Path("experiment-results/exp01")
EXP05 = Path("experiment-results/exp05")
OUTPUT = Path("experiment-results/exp06")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
SEED = 49


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def split_ids(rows, name):
    aliases = {"train"} if name == "train" else {"val", "validation"}
    return torch.tensor(
        [int(row["sample_id"]) for row in rows if row["split"].lower() in aliases],
        dtype=torch.long,
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    train_ids, val_ids = split_ids(splits, "train"), split_ids(splits, "validation")
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    activations = cache["sae_activations"]
    labels = cache["binary_labels"]

    searches, selected = [], []
    for group in read(EXP05 / "validation_selected_atoms.csv"):
        protocol, concept = group["protocol"], group["concept"]
        atoms = [int(value) for value in group["candidate_atom_ids"].split(";")]
        x_train = activations[train_ids][:, atoms].numpy()
        x_val = activations[val_ids][:, atoms].numpy()
        y_train = labels[train_ids, concept_index[concept]].numpy()
        y_val = labels[val_ids, concept_index[concept]].numpy()
        mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
        scale[scale == 0] = 1.0
        x_train = (x_train - mean) / scale
        x_val = (x_val - mean) / scale

        candidates = []
        for regularization in C_GRID:
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=regularization,
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            ).fit(x_train, y_train)
            validation_auroc = roc_auc_score(y_val, model.decision_function(x_val))
            record = {
                "protocol": protocol,
                "concept": concept,
                "C": regularization,
                "validation_auroc": validation_auroc,
                "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
            }
            searches.append(record)
            candidates.append((validation_auroc, -regularization, model))

        validation_auroc, negative_c, model = max(candidates, key=lambda item: item[:2])
        selected.append(
            {
                "protocol": protocol,
                "concept": concept,
                "candidate_atom_ids": json.dumps(atoms),
                "training_mean": json.dumps(mean.tolist()),
                "training_scale": json.dumps(scale.tolist()),
                "coefficients": json.dumps(model.coef_[0].tolist()),
                "intercept": float(model.intercept_[0]),
                "selected_C": -negative_c,
                "validation_auroc": validation_auroc,
                "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
                "selection_rule": "highest validation AUROC, then smaller C",
            }
        )

    for filename, rows in (
        ("regularization_search.csv", searches),
        ("selected_sparse_models.csv", selected),
    ):
        with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    for protocol in ("A", "B"):
        rows = [row for row in selected if row["protocol"] == protocol]
        nonzero = np.array([row["nonzero_coefficients"] for row in rows])
        summary[protocol] = {
            "models": len(rows),
            "median_nonzero_coefficients": float(np.median(nonzero)),
            "zero_atom_models": int(np.sum(nonzero == 0)),
            "selected_C_counts": {
                str(value): sum(row["selected_C"] == value for row in rows)
                for value in C_GRID
            },
        }
    summary["test_metrics_computed"] = False
    (OUTPUT / "sparse_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
