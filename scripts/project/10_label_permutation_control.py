"""EXP-10.1: concept-balanced label-permutation sanity control."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata


EXP01 = Path("experiment-results/exp01")
EXP02 = Path("experiment-results/exp02")
OUT = Path("experiment-results/exp10")
CACHE = Path("data/exp02/sun_frozen_cache.pth")
REPLICATES = 1000
SEED = 49


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def representatives(protocol):
    rows = read(EXP02 / f"protocol_{protocol.lower()}_atom_scores.csv")
    result = {}
    for row in rows:
        concept = row["mapped_sun_attribute"]
        atom = int(row["atom_id"])
        if concept not in result or atom < result[concept]:
            result[concept] = atom
    return result


def auc_vector(labels, ranks):
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    rank_sum = (labels * ranks).sum(axis=0)
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


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
    labels = cache["binary_labels"][test_ids].numpy().astype(float)
    activations = cache["sae_activations"][test_ids].numpy()
    rng = np.random.default_rng(SEED)
    permutations = np.array(
        [rng.permutation(len(test_ids)) for _ in range(REPLICATES)]
    )

    null_rows = [{"replicate": i} for i in range(REPLICATES)]
    representative_rows, summary = [], {}
    for protocol in ("A", "B"):
        selected = representatives(protocol)
        names = sorted(selected)
        atoms = [selected[name] for name in names]
        indices = [concept_index[name] for name in names]
        y = labels[:, indices]
        ranks = np.column_stack(
            [rankdata(activations[:, atom], method="average") for atom in atoms]
        )
        observed_auc = auc_vector(y, ranks)
        observed_mean = float(observed_auc.mean())
        null = np.empty(REPLICATES)
        for replicate, permutation in enumerate(permutations):
            null[replicate] = auc_vector(y[permutation], ranks).mean()
            null_rows[replicate][f"protocol_{protocol}_mean_auroc"] = null[replicate]
        for name, atom, auc in zip(names, atoms, observed_auc):
            representative_rows.append(
                {
                    "protocol": protocol,
                    "sun_attribute": name,
                    "representative_atom_id": atom,
                    "selection_rule": "smallest atom ID within frozen evaluated attribute",
                    "observed_test_auroc": float(auc),
                }
            )
        summary[protocol] = {
            "representative_atoms": len(atoms),
            "observed_concept_balanced_mean_auroc": observed_mean,
            "permutation_mean": float(null.mean()),
            "permutation_95_interval": [
                float(np.quantile(null, 0.025)),
                float(np.quantile(null, 0.975)),
            ],
            "one_sided_permutation_pvalue": float(
                (1 + np.sum(null >= observed_mean)) / (REPLICATES + 1)
            ),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("permutation_representatives.csv", representative_rows),
        ("label_permutation_null.csv", null_rows),
    ):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary.update(
        {
            "replicates": REPLICATES,
            "seed": SEED,
            "permutation": "joint test-image row permutation",
            "test_selection_used": False,
            "purpose": "evaluation-pipeline sanity check, not a semantic null model",
        }
    )
    (OUT / "label_permutation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
