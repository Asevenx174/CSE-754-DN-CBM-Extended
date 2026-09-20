"""EXP-10.3b: tune sparse readouts for frozen matched-random groups."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


EXP01 = Path("experiment-results/exp01")
EXP10 = Path("experiment-results/exp10")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
SEED = 49


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def fit_group(row, activations, labels, train_ids, val_ids, concept_index):
    atoms = json.loads(row["random_atom_ids"])
    index = concept_index[row["concept"]]
    x_train = activations[train_ids][:, atoms]
    x_val = activations[val_ids][:, atoms]
    y_train, y_val = labels[train_ids, index], labels[val_ids, index]
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train, x_val = (x_train - mean) / scale, (x_val - mean) / scale

    search, candidates = [], []
    for regularization in C_GRID:
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=regularization,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ).fit(x_train, y_train)
        auroc = roc_auc_score(y_val, model.decision_function(x_val))
        search.append(
            {
                "protocol": row["protocol"],
                "concept": row["concept"],
                "replicate": int(row["replicate"]),
                "C": regularization,
                "validation_auroc": auroc,
                "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
            }
        )
        candidates.append((auroc, -regularization, model))

    auroc, negative_c, model = max(candidates, key=lambda item: item[:2])
    selected = {
        "protocol": row["protocol"],
        "concept": row["concept"],
        "replicate": int(row["replicate"]),
        "candidate_atom_ids": json.dumps(atoms),
        "training_mean": json.dumps(mean.tolist()),
        "training_scale": json.dumps(scale.tolist()),
        "coefficients": json.dumps(model.coef_[0].tolist()),
        "intercept": float(model.intercept_[0]),
        "selected_C": -negative_c,
        "validation_auroc": auroc,
        "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
        "selection_rule": "highest validation AUROC, then smaller C",
    }
    return search, selected


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    split_ids = {
        name: np.array(
            [
                int(row["sample_id"])
                for row in splits
                if row["split"].strip().lower() in aliases
            ],
            dtype=int,
        )
        for name, aliases in {
            "train": {"train"},
            "validation": {"val", "validation"},
        }.items()
    }
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    groups = read(EXP10 / "matched_random_groups.csv")
    activations = cache["sae_activations"].numpy()
    labels = cache["binary_labels"].numpy()
    concept_index = {name: index for index, name in enumerate(concepts)}
    results = Parallel(n_jobs=args.n_jobs, prefer="threads", verbose=10)(
        delayed(fit_group)(
            row,
            activations,
            labels,
            split_ids["train"],
            split_ids["validation"],
            concept_index,
        )
        for row in groups
    )
    searches = [record for search, _ in results for record in search]
    selected = [model for _, model in results]
    write(EXP10 / "matched_random_regularization_search.csv", searches)
    write(EXP10 / "matched_random_selected_models.csv", selected)

    summary = {"test_metrics_computed": False, "C_grid": C_GRID}
    for protocol in ("A", "B"):
        rows = [row for row in selected if row["protocol"] == protocol]
        nonzero = np.array([row["nonzero_coefficients"] for row in rows])
        summary[protocol] = {
            "models": len(rows),
            "real_groups": len({row["concept"] for row in rows}),
            "replicates_per_group": 20,
            "median_nonzero_coefficients": float(np.median(nonzero)),
            "zero_atom_models": int((nonzero == 0).sum()),
            "selected_C_counts": {
                str(value): sum(row["selected_C"] == value for row in rows)
                for value in C_GRID
            },
        }
    (EXP10 / "matched_random_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
