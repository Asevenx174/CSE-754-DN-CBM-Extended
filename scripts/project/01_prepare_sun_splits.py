import argparse
import csv
import json
from pathlib import Path

import torch

from dncbm.data_utils.sun_attributes import SUNAttributes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation_dir",
        default=(
            "data/sunattributes/"
            "SUNAttributeDB/SUNAttributeDB"
        ),
    )

    parser.add_argument(
        "--image_root",
        default=(
            "data/sunattributes/"
            "SUNAttributeDB_Images/images"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="experiment-results/exp01",
    )

    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--minimum_test_support", type=int, default=30)

    return parser.parse_args()

def create_splits(number_of_images, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        number_of_images,
        generator=generator,
    )

    number_train = int(0.60 * number_of_images)
    number_validation = int(0.20 * number_of_images)

    train_end = number_train
    validation_end = number_train + number_validation

    return {
        "train": permutation[:train_end],
        "validation": permutation[
            train_end:validation_end
        ],
        "test": permutation[validation_end:],
    }

def calculate_support(dataset, splits, minimum_test_support):
    rows = []

    for concept_id, concept_name in enumerate(
        dataset.attribute_names
    ):
        row = {
            "concept_id": concept_id,
            "concept_name": concept_name,
        }

        for split_name, indices in splits.items():
            split_labels = dataset.binary_labels[
                indices, concept_id
            ]

            positives = int(split_labels.sum())
            negatives = int(len(indices) - positives)

            row[f"{split_name}_positive"] = positives
            row[f"{split_name}_negative"] = negatives

        row["eligible"] = (
            row["train_positive"] > 0
            and row["train_negative"] > 0
            and row["validation_positive"] > 0
            and row["validation_negative"] > 0
            and row["test_positive"] >= minimum_test_support
            and row["test_negative"] >= minimum_test_support
        )

        rows.append(row)

    return rows

def save_outputs(dataset, splits, support_rows, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = output_dir / "sample_splits.csv"

    with split_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sample_id",
            "split",
            "relative_path",
        ])

        for split_name, indices in splits.items():
            for sample_id in indices.tolist():
                writer.writerow([
                    sample_id,
                    split_name,
                    dataset.relative_paths[sample_id],
                ])

    support_path = output_dir / "concept_support.csv"

    with support_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=support_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(support_rows)

    valid_path = output_dir / "valid_concepts.txt"

    valid_names = [
        row["concept_name"]
        for row in support_rows
        if row["eligible"]
    ]

    valid_path.write_text(
        "\n".join(valid_names) + "\n"
    )

    summary = {
        "seed": args.seed,
        "label_rule": "agreement >= 2/3",
        "number_of_images": len(dataset),
        "number_of_attributes": len(dataset.attribute_names),
        "split_sizes": {
            name: len(indices)
            for name, indices in splits.items()
        },
        "minimum_test_support": args.minimum_test_support,
        "eligible_concepts": len(valid_names),
    }

    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary

def main():
    args = parse_args()

    dataset = SUNAttributes(
        annotation_dir=args.annotation_dir,
        image_root=args.image_root,
    )

    splits = create_splits(
        number_of_images=len(dataset),
        seed=args.seed,
    )

    combined = torch.cat(list(splits.values()))

    if len(torch.unique(combined)) != len(dataset):
        raise RuntimeError(
            "Split IDs overlap or some images are missing."
        )

    support_rows = calculate_support(
        dataset=dataset,
        splits=splits,
        minimum_test_support=args.minimum_test_support,
    )

    summary = save_outputs(
        dataset=dataset,
        splits=splits,
        support_rows=support_rows,
        args=args,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()