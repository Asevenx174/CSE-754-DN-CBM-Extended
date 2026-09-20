"""EXP-06.2: evaluate frozen sparse aggregators on the test split."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


EXP06 = Path("experiment-results/exp06")
ACTIVATIONS = Path("data/exp02/sun_frozen_cache.pth")
BASELINES = Path("data/exp05/baseline_test_predictions.pth")
PREDICTIONS = Path("data/exp06/sparse_test_predictions.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def metrics(labels, scores):
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def main():
    activation_cache = torch.load(ACTIVATIONS, map_location="cpu", weights_only=True)
    baseline_cache = torch.load(BASELINES, map_location="cpu", weights_only=True)
    test_ids = baseline_cache["sample_ids"].long()
    activations = activation_cache["sae_activations"][test_ids].numpy()

    results, predictions = [], {}
    for row in read(EXP06 / "selected_sparse_models.csv"):
        protocol, concept = row["protocol"], row["concept"]
        key = f"{protocol}|{concept}"
        atoms = np.asarray(json.loads(row["candidate_atom_ids"]), dtype=int)
        mean = np.asarray(json.loads(row["training_mean"]), dtype=float)
        scale = np.asarray(json.loads(row["training_scale"]), dtype=float)
        coefficients = np.asarray(json.loads(row["coefficients"]), dtype=float)
        intercept = float(row["intercept"])
        sparse = ((activations[:, atoms] - mean) / scale) @ coefficients + intercept

        baseline = baseline_cache["predictions"][key]
        labels = baseline["labels"].numpy()
        single = baseline["single"].numpy()
        maximum = baseline["raw_max"].numpy()
        sparse_auc, sparse_ap = metrics(labels, sparse)
        single_auc, single_ap = metrics(labels, single)
        maximum_auc, maximum_ap = metrics(labels, maximum)

        results.append(
            {
                "protocol": protocol,
                "concept": concept,
                "candidate_count": len(atoms),
                "selected_C": float(row["selected_C"]),
                "nonzero_coefficients": int(np.count_nonzero(coefficients)),
                "positive_coefficients": int(np.sum(coefficients > 0)),
                "negative_coefficients": int(np.sum(coefficients < 0)),
                "sparse_test_auroc": sparse_auc,
                "sparse_test_ap": sparse_ap,
                "delta_auroc_sparse_minus_single": sparse_auc - single_auc,
                "delta_auroc_sparse_minus_raw_max": sparse_auc - maximum_auc,
                "delta_ap_sparse_minus_single": sparse_ap - single_ap,
                "delta_ap_sparse_minus_raw_max": sparse_ap - maximum_ap,
            }
        )
        predictions[key] = {
            "labels": baseline["labels"],
            "single": baseline["single"],
            "raw_max": baseline["raw_max"],
            "sparse": torch.from_numpy(sparse).float(),
        }

    score_path = EXP06 / "sparse_test_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    for protocol in ("A", "B"):
        rows = [row for row in results if row["protocol"] == protocol]
        summary[protocol] = {"concepts": len(rows)}
        for metric in ("auroc", "ap"):
            sparse = np.array([row[f"sparse_test_{metric}"] for row in rows])
            summary[protocol][f"mean_sparse_{metric}"] = float(sparse.mean())
            for baseline in ("single", "raw_max"):
                delta = np.array(
                    [row[f"delta_{metric}_sparse_minus_{baseline}"] for row in rows]
                )
                summary[protocol][f"mean_delta_{metric}_vs_{baseline}"] = float(delta.mean())
                summary[protocol][f"median_delta_{metric}_vs_{baseline}"] = float(
                    np.median(delta)
                )
                summary[protocol][f"improved_{metric}_vs_{baseline}_fraction"] = float(
                    np.mean(delta > 0)
                )
    summary["prediction_cache"] = str(PREDICTIONS)
    (EXP06 / "sparse_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_ids": test_ids, "predictions": predictions}, PREDICTIONS)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
