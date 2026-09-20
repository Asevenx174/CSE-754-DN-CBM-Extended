"""EXP-10.2: individual-atom sensitivity to unanimous SUN annotations."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


EXP01 = Path("experiment-results/exp01")
EXP02 = Path("experiment-results/exp02")
OUT = Path("experiment-results/exp10")
CACHE = Path("data/exp02/sun_frozen_cache.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    test_ids = torch.tensor(
        [int(row["sample_id"]) for row in splits if row["split"] == "test"],
        dtype=torch.long,
    )
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    agreement = cache["agreement"][test_ids].numpy()
    activations = cache["sae_activations"][test_ids].numpy()

    rows = []
    for protocol in ("A", "B"):
        for original in read(EXP02 / f"protocol_{protocol.lower()}_atom_scores.csv"):
            concept = original["mapped_sun_attribute"]
            index = concept_index[concept]
            values = agreement[:, index]
            mask = np.isclose(values, 0.0) | np.isclose(values, 1.0)
            y = np.isclose(values[mask], 1.0).astype(int)
            score = activations[mask, int(original["atom_id"])]
            positives, negatives = int(y.sum()), int((1 - y).sum())
            if not positives or not negatives:
                continue
            unanimous_auc = roc_auc_score(y, score)
            unanimous_ap = average_precision_score(y, score)
            rows.append(
                {
                    "protocol": protocol,
                    "atom_id": int(original["atom_id"]),
                    "assigned_name": original["assigned_name"],
                    "mapped_sun_attribute": concept,
                    "unanimous_samples": len(y),
                    "unanimous_positives": positives,
                    "unanimous_negatives": negatives,
                    "original_test_auroc": float(original["auroc"]),
                    "unanimous_test_auroc": float(unanimous_auc),
                    "delta_auroc_unanimous_minus_original": float(unanimous_auc)
                    - float(original["auroc"]),
                    "original_test_ap": float(original["average_precision"]),
                    "unanimous_test_ap": float(unanimous_ap),
                    "delta_ap_unanimous_minus_original": float(unanimous_ap)
                    - float(original["average_precision"]),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "unanimous_atom_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "annotation_rule": "retain agreement exactly 0 or 1; exclude 1/3 and 2/3",
        "selection_changed": False,
        "split": "test",
    }
    for protocol in ("A", "B"):
        subset = [row for row in rows if row["protocol"] == protocol]
        delta_auc = np.array(
            [row["delta_auroc_unanimous_minus_original"] for row in subset]
        )
        delta_ap = np.array(
            [row["delta_ap_unanimous_minus_original"] for row in subset]
        )
        summary[protocol] = {
            "atoms": len(subset),
            "attributes": len({row["mapped_sun_attribute"] for row in subset}),
            "mean_original_auroc": float(
                np.mean([row["original_test_auroc"] for row in subset])
            ),
            "mean_unanimous_auroc": float(
                np.mean([row["unanimous_test_auroc"] for row in subset])
            ),
            "mean_delta_auroc": float(delta_auc.mean()),
            "median_delta_auroc": float(np.median(delta_auc)),
            "fraction_higher_unanimous_auroc": float((delta_auc > 0).mean()),
            "mean_delta_ap": float(delta_ap.mean()),
            "warning": "AP changes also reflect prevalence changes after filtering.",
        }
    (OUT / "worker_agreement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
