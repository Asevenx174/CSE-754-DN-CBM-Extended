"""EXP-07.4: render frozen group recoveries and errors for inspection."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


EXP07 = Path("experiment-results/exp07")
ORDER = (
    "positive_recovered_by_sparse",
    "positive_lost_by_sparse",
    "false_positive_corrected_by_sparse",
    "false_positive_introduced_by_sparse",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_root",
        default="data/sunattributes/SUNAttributeDB_Images/images",
    )
    return parser.parse_args()


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def safe(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def main():
    image_root = Path(arguments().image_root)
    output = EXP07 / "group_error_sheets"
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        (row["protocol"], row["concept"]): row
        for row in read(EXP07 / "group_error_counts.csv")
    }
    groups = defaultdict(list)
    for row in read(EXP07 / "group_error_manifest.csv"):
        groups[(row["protocol"], row["concept"])].append(row)

    for key, meta in sorted(counts.items()):
        rows = groups.get(key, [])
        figure, axes = plt.subplots(4, 5, figsize=(12, 10))
        for row_index, example_type in enumerate(ORDER):
            examples = sorted(
                [row for row in rows if row["example_type"] == example_type],
                key=lambda row: int(row["rank"]),
            )
            for column, axis in enumerate(axes[row_index]):
                axis.axis("off")
                if column >= len(examples):
                    continue
                row = examples[column]
                image = Image.open(image_root / row["relative_path"]).convert("RGB")
                axis.imshow(ImageOps.exif_transpose(image))
                axis.set_title(
                    f"a={float(row['worker_agreement']):.2f}\n"
                    f"s={float(row['sparse_score']):.2f}, "
                    f"m={float(row['raw_max_score']):.2f}",
                    fontsize=8,
                )
            axes[row_index, 0].set_ylabel(
                example_type.replace("_", "\n"), fontsize=8
            )

        sparse_j = float(meta["sparse_validation_youden_j"])
        maximum_j = float(meta["raw_max_validation_youden_j"])
        figure.suptitle(
            f"Protocol {meta['protocol']} | {meta['case_category']} | {meta['concept']}\n"
            f"validation J: sparse={sparse_j:.3f}, raw maximum={maximum_j:.3f}",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        filename = f"{meta['protocol']}_{safe(meta['concept'])}_{safe(meta['case_category'])}.png"
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)

    print(f"Rendered {len(counts)} frozen group-error sheets in {output}.")


if __name__ == "__main__":
    main()
