"""EXP-10.4: small training-selected correlated-attribute shortcut audit."""

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


EXP01 = Path("experiment-results/exp01")
EXP06 = Path("experiment-results/exp06")
OUT = Path("experiment-results/exp10")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
PREDICTIONS = Path("data/exp06/sparse_test_predictions.pth")
PAIR_COUNT = 6
MIN_TRAIN_CELL = 30
MIN_TEST_CELL = 10


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def phi(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    predictions = torch.load(PREDICTIONS, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    train_ids = np.array(
        [int(row["sample_id"]) for row in splits if row["split"] == "train"],
        dtype=int,
    )
    names = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    valid = {
        line.strip()
        for line in (EXP01 / "valid_concepts.txt").read_text().splitlines()
        if line.strip()
    }
    available = {
        key.split("|", 1)[1]
        for key in predictions["predictions"]
        if key.startswith("B|")
    }
    selected_names = sorted(valid & available)
    index = {name: names.index(name) for name in selected_names}
    train = cache["binary_labels"][train_ids].numpy().astype(int)

    candidates = []
    for first, second in combinations(selected_names, 2):
        x, y = train[:, index[first]], train[:, index[second]]
        cells = [int(np.sum((x == a) & (y == b))) for a in (0, 1) for b in (0, 1)]
        if min(cells) < MIN_TRAIN_CELL:
            continue
        candidates.append(
            {
                "attribute_1": first,
                "attribute_2": second,
                "training_phi": phi(x, y),
                "absolute_training_phi": abs(phi(x, y)),
                "train_count_00": cells[0],
                "train_count_01": cells[1],
                "train_count_10": cells[2],
                "train_count_11": cells[3],
            }
        )
    pairs = sorted(
        candidates, key=lambda row: row["absolute_training_phi"], reverse=True
    )[:PAIR_COUNT]

    score_summary = {
        row["concept"]: float(row["sparse_test_auroc"])
        for row in read(EXP06 / "sparse_test_scores.csv")
        if row["protocol"] == "B"
    }
    rows = []
    for pair in pairs:
        for target, context in (
            (pair["attribute_1"], pair["attribute_2"]),
            (pair["attribute_2"], pair["attribute_1"]),
        ):
            record = predictions["predictions"][f"B|{target}"]
            y = record["labels"].numpy().astype(int)
            score = record["sparse"].numpy()
            context_y = cache["binary_labels"][
                predictions["sample_ids"], names.index(context)
            ].numpy().astype(int)
            result = {
                "target_attribute": target,
                "context_attribute": context,
                "training_phi": pair["training_phi"],
                "unconditional_test_auroc": score_summary[target],
            }
            supported = True
            conditional_auc = []
            for value in (0, 1):
                mask = context_y == value
                positives, negatives = int(y[mask].sum()), int((1 - y[mask]).sum())
                result[f"context_{value}_test_positives"] = positives
                result[f"context_{value}_test_negatives"] = negatives
                if min(positives, negatives) < MIN_TEST_CELL:
                    supported = False
                    result[f"context_{value}_auroc"] = ""
                    result[f"context_{value}_ap"] = ""
                else:
                    auc = float(roc_auc_score(y[mask], score[mask]))
                    conditional_auc.append(auc)
                    result[f"context_{value}_auroc"] = auc
                    result[f"context_{value}_ap"] = float(
                        average_precision_score(y[mask], score[mask])
                    )
            result["test_support_sufficient"] = supported
            result["minimum_conditional_auroc"] = (
                min(conditional_auc) if supported else ""
            )
            result["drop_to_minimum_conditional_auroc"] = (
                score_summary[target] - min(conditional_auc) if supported else ""
            )
            rows.append(result)

    OUT.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        ("correlated_attribute_pairs.csv", pairs),
        ("conditional_attribute_scores.csv", rows),
    ):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=values[0].keys())
            writer.writeheader()
            writer.writerows(values)
    supported_rows = [row for row in rows if row["test_support_sufficient"]]
    drops = np.array(
        [row["drop_to_minimum_conditional_auroc"] for row in supported_rows]
    )
    summary = {
        "pair_selection": "six largest absolute training phi correlations",
        "minimum_training_count_per_2x2_cell": MIN_TRAIN_CELL,
        "minimum_test_positives_and_negatives_per_stratum": MIN_TEST_CELL,
        "pairs_selected": len(pairs),
        "directional_analyses": len(rows),
        "supported_directional_analyses": len(supported_rows),
        "mean_drop_to_minimum_conditional_auroc": float(drops.mean()),
        "maximum_drop_to_minimum_conditional_auroc": float(drops.max()),
        "selection_used_test_outcomes": False,
        "interpretation": (
            "Conditional degradation can indicate contextual dependence, but does "
            "not by itself prove a shortcut or causal mechanism."
        ),
    }
    (OUT / "correlated_attribute_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
