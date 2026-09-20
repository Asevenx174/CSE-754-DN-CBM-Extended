"""EXP-08.6: matched SAE-versus-CLIP comparison with paired bootstrap."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


EXP08 = Path("experiment-results/exp08")
SAE = Path("data/exp08/sae_compound_test_predictions.pth")
CLIP = Path("data/exp08/clip_test_predictions.pth")
REPLICATES = 2000
SEED = 49


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sample_jaccard(y, prediction):
    intersection = np.logical_and(y, prediction).sum(axis=1)
    union = np.logical_or(y, prediction).sum(axis=1)
    return np.divide(
        intersection, union, out=np.zeros(len(y), dtype=float), where=union != 0
    )


def main():
    sae = torch.load(SAE, map_location="cpu", weights_only=True)
    clip = torch.load(CLIP, map_location="cpu", weights_only=True)
    if not torch.equal(sae["sample_ids"], clip["sample_ids"]):
        raise RuntimeError("SAE and CLIP test sample IDs differ.")
    if not torch.equal(sae["labels"], clip["labels"]):
        raise RuntimeError("SAE and CLIP test labels differ.")

    available = sae["available_attributes"].numpy().astype(bool)
    labels = sae["labels"].numpy().astype(bool)[:, available]
    sae_prediction = sae["predictions"].numpy().astype(bool)[:, available]
    clip_prediction = clip["predictions"].numpy().astype(bool)[:, available]
    sae_jaccard = sample_jaccard(labels, sae_prediction)
    clip_jaccard = sample_jaccard(labels, clip_prediction)
    paired_delta = clip_jaccard - sae_jaccard

    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(REPLICATES)
    for replicate in range(REPLICATES):
        ids = rng.integers(0, len(paired_delta), len(paired_delta))
        bootstrap[replicate] = paired_delta[ids].mean()

    sae_metrics = {
        row["sun_attribute"]: row
        for row in read(EXP08 / "sae_compound_test_attribute_metrics.csv")
    }
    clip_metrics = {
        row["sun_attribute"]: row
        for row in read(EXP08 / "clip_test_attribute_metrics.csv")
    }
    names = [name for name, keep in zip(sae["attribute_names"], available) if keep]
    rows = []
    for name in names:
        sae_row, clip_row = sae_metrics[name], clip_metrics[name]
        rows.append(
            {
                "sun_attribute": name,
                "sae_test_auroc": float(sae_row["test_auroc"]),
                "clip_test_auroc": float(clip_row["test_auroc"]),
                "delta_auroc_clip_minus_sae": float(clip_row["test_auroc"])
                - float(sae_row["test_auroc"]),
                "sae_test_ap": float(sae_row["test_ap"]),
                "clip_test_ap": float(clip_row["test_ap"]),
                "delta_ap_clip_minus_sae": float(clip_row["test_ap"])
                - float(sae_row["test_ap"]),
            }
        )
    with (EXP08 / "sae_clip_matched_attribute_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    auroc_delta = np.array([row["delta_auroc_clip_minus_sae"] for row in rows])
    ap_delta = np.array([row["delta_ap_clip_minus_sae"] for row in rows])
    summary = {
        "status": "documented implementation, not exact reproduction",
        "test_images": len(labels),
        "matched_attributes": len(names),
        "sae_mean_image_jaccard": float(sae_jaccard.mean()),
        "clip_mean_image_jaccard": float(clip_jaccard.mean()),
        "delta_jaccard_clip_minus_sae": float(paired_delta.mean()),
        "delta_jaccard_paired_bootstrap_95_ci": {
            "lower": float(np.quantile(bootstrap, 0.025)),
            "median": float(np.quantile(bootstrap, 0.5)),
            "upper": float(np.quantile(bootstrap, 0.975)),
        },
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
        "mean_delta_auroc_clip_minus_sae": float(auroc_delta.mean()),
        "attributes_clip_higher_auroc": int((auroc_delta > 0).sum()),
        "mean_delta_ap_clip_minus_sae": float(ap_delta.mean()),
        "attributes_clip_higher_ap": int((ap_delta > 0).sum()),
    }
    (EXP08 / "sae_clip_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
