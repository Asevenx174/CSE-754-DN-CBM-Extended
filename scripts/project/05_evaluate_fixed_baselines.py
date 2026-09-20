"""EXP-05.2: evaluate frozen single-atom and raw-maximum baselines."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


EXP01 = Path("experiment-results/exp01")
EXP05 = Path("experiment-results/exp05")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
PREDICTIONS = Path("data/exp05/baseline_test_predictions.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def metric(y, score):
    return roc_auc_score(y, score), average_precision_score(y, score)


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    split = read(EXP01 / "sample_splits.csv")
    test_ids = torch.tensor(
        [int(row["sample_id"]) for row in split if row["split"] == "test"],
        dtype=torch.long,
    )
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    activations = cache["sae_activations"][test_ids]
    labels = cache["binary_labels"][test_ids]

    results, predictions = [], {}
    for row in read(EXP05 / "validation_selected_atoms.csv"):
        protocol, concept = row["protocol"], row["concept"]
        atoms = [int(value) for value in row["candidate_atom_ids"].split(";")]
        selected = int(row["selected_atom_id"])
        y = labels[:, concept_index[concept]].numpy()
        single = activations[:, selected].numpy()
        raw_max = activations[:, atoms].max(dim=1).values.numpy()
        single_auc, single_ap = metric(y, single)
        maximum_auc, maximum_ap = metric(y, raw_max)

        results.append(
            {
                "protocol": protocol,
                "concept": concept,
                "candidate_count": len(atoms),
                "selected_atom_id": selected,
                "validation_auroc": float(row["validation_auroc"]),
                "test_prevalence": float(y.mean()),
                "single_test_auroc": single_auc,
                "single_test_ap": single_ap,
                "raw_max_test_auroc": maximum_auc,
                "raw_max_test_ap": maximum_ap,
                "delta_auroc_raw_max_minus_single": maximum_auc - single_auc,
                "delta_ap_raw_max_minus_single": maximum_ap - single_ap,
            }
        )
        predictions[f"{protocol}|{concept}"] = {
            "labels": torch.from_numpy(y).to(torch.uint8),
            "single": torch.from_numpy(single),
            "raw_max": torch.from_numpy(raw_max),
        }

    path = EXP05 / "fixed_baseline_test_scores.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    for protocol in ("A", "B"):
        rows = [row for row in results if row["protocol"] == protocol]
        delta = np.array([row["delta_auroc_raw_max_minus_single"] for row in rows])
        summary[protocol] = {
            "concepts": len(rows),
            "mean_single_auroc": float(np.mean([r["single_test_auroc"] for r in rows])),
            "mean_raw_max_auroc": float(np.mean([r["raw_max_test_auroc"] for r in rows])),
            "mean_delta_auroc": float(delta.mean()),
            "median_delta_auroc": float(np.median(delta)),
            "raw_max_improved_fraction": float(np.mean(delta > 0)),
        }
    summary["prediction_cache"] = str(PREDICTIONS)
    (EXP05 / "fixed_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_ids": test_ids, "predictions": predictions}, PREDICTIONS)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
