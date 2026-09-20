"""EXP-10.3c: test real same-name groups against matched-random groups."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


EXP01 = Path("experiment-results/exp01")
EXP06 = Path("experiment-results/exp06")
EXP10 = Path("experiment-results/exp10")
CACHE = Path("data/exp02/sun_frozen_cache.pth")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def evaluate(row, activations, labels, test_ids, concept_index):
    atoms = np.asarray(json.loads(row["candidate_atom_ids"]), dtype=int)
    mean = np.asarray(json.loads(row["training_mean"]), dtype=float)
    scale = np.asarray(json.loads(row["training_scale"]), dtype=float)
    coefficients = np.asarray(json.loads(row["coefficients"]), dtype=float)
    x = activations[test_ids][:, atoms]
    score = ((x - mean) / scale) @ coefficients + float(row["intercept"])
    y = labels[test_ids, concept_index[row["concept"]]]
    return {
        "protocol": row["protocol"],
        "concept": row["concept"],
        "replicate": int(row["replicate"]),
        "selected_C": float(row["selected_C"]),
        "nonzero_coefficients": int(row["nonzero_coefficients"]),
        "test_auroc": float(roc_auc_score(y, score)),
        "test_ap": float(average_precision_score(y, score)),
    }


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    splits = read(EXP01 / "sample_splits.csv")
    test_ids = np.array(
        [int(row["sample_id"]) for row in splits if row["split"] == "test"],
        dtype=int,
    )
    concepts = [
        line.strip()
        for line in (EXP01 / "sun102_vocabulary.txt").read_text().splitlines()
        if line.strip()
    ]
    concept_index = {name: index for index, name in enumerate(concepts)}
    activations = cache["sae_activations"].numpy()
    labels = cache["binary_labels"].numpy()
    models = read(EXP10 / "matched_random_selected_models.csv")
    random_scores = Parallel(n_jobs=args.n_jobs, prefer="threads", verbose=10)(
        delayed(evaluate)(
            row, activations, labels, test_ids, concept_index
        )
        for row in models
    )
    write(EXP10 / "matched_random_test_scores.csv", random_scores)

    real = {
        (row["protocol"], row["concept"]): row
        for row in read(EXP06 / "sparse_test_scores.csv")
    }
    comparisons = []
    for key, real_row in real.items():
        protocol, concept = key
        controls = [
            row
            for row in random_scores
            if row["protocol"] == protocol and row["concept"] == concept
        ]
        random_auc = np.array([row["test_auroc"] for row in controls])
        random_ap = np.array([row["test_ap"] for row in controls])
        real_auc = float(real_row["sparse_test_auroc"])
        real_ap = float(real_row["sparse_test_ap"])
        comparisons.append(
            {
                "protocol": protocol,
                "concept": concept,
                "random_replicates": len(controls),
                "real_sparse_auroc": real_auc,
                "random_mean_auroc": float(random_auc.mean()),
                "delta_auroc_real_minus_random_mean": real_auc
                - float(random_auc.mean()),
                "real_auroc_percentile_among_random": float(
                    100 * np.mean(random_auc <= real_auc)
                ),
                "empirical_p_auroc_random_at_least_real": float(
                    (1 + np.sum(random_auc >= real_auc)) / (len(random_auc) + 1)
                ),
                "real_sparse_ap": real_ap,
                "random_mean_ap": float(random_ap.mean()),
                "delta_ap_real_minus_random_mean": real_ap
                - float(random_ap.mean()),
                "empirical_p_ap_random_at_least_real": float(
                    (1 + np.sum(random_ap >= real_ap)) / (len(random_ap) + 1)
                ),
            }
        )
    write(EXP10 / "real_vs_matched_random_groups.csv", comparisons)

    summary = {
        "random_models_evaluated": len(random_scores),
        "random_replicates_per_real_group": 20,
        "empirical_pvalue_minimum": 1 / 21,
        "selected_model_convergence_warning": (
            "Some liblinear fits reached max_iter during validation tuning."
        ),
    }
    for protocol in ("A", "B"):
        rows = [row for row in comparisons if row["protocol"] == protocol]
        auc_delta = np.array(
            [row["delta_auroc_real_minus_random_mean"] for row in rows]
        )
        ap_delta = np.array(
            [row["delta_ap_real_minus_random_mean"] for row in rows]
        )
        test = wilcoxon(auc_delta, alternative="greater", zero_method="wilcox")
        summary[protocol] = {
            "groups": len(rows),
            "mean_real_sparse_auroc": float(
                np.mean([row["real_sparse_auroc"] for row in rows])
            ),
            "mean_random_control_auroc": float(
                np.mean([row["random_mean_auroc"] for row in rows])
            ),
            "mean_delta_auroc_real_minus_random": float(auc_delta.mean()),
            "groups_real_above_random_mean_auroc": int((auc_delta > 0).sum()),
            "groups_real_above_all_random_auroc": sum(
                row["empirical_p_auroc_random_at_least_real"] == 1 / 21
                for row in rows
            ),
            "wilcoxon_one_sided_pvalue_auroc": float(test.pvalue),
            "mean_delta_ap_real_minus_random": float(ap_delta.mean()),
            "groups_real_above_random_mean_ap": int((ap_delta > 0).sum()),
        }
    (EXP10 / "matched_random_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
