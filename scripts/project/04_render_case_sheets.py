"""EXP-04.3: render frozen diagnostic examples as one sheet per atom."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


EXP04 = Path("experiment-results/exp04")
ORDER = ["top_activation", "high_scoring_negative", "low_scoring_positive"]


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
    args = arguments()
    image_root = Path(args.image_root)
    output = EXP04 / "case_sheets"
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        (row["protocol"], row["atom_id"]): row
        for row in read(EXP04 / "selected_case_atoms.csv")
    }
    groups = defaultdict(list)
    for row in read(EXP04 / "inspection_manifest.csv"):
        groups[(row["protocol"], row["atom_id"])].append(row)

    for key, rows in sorted(groups.items()):
        meta = metadata[key]
        figure, axes = plt.subplots(3, 5, figsize=(12, 7.5))
        for row_index, example_type in enumerate(ORDER):
            examples = sorted(
                [row for row in rows if row["example_type"] == example_type],
                key=lambda row: int(row["rank"]),
            )
            for column, row in enumerate(examples):
                image = Image.open(image_root / row["relative_path"]).convert("RGB")
                axes[row_index, column].imshow(ImageOps.exif_transpose(image))
                axes[row_index, column].set_title(
                    f"z={float(row['activation']):.2f}  "
                    f"y={row['binary_label']}  a={float(row['worker_agreement']):.2f}",
                    fontsize=8,
                )
                axes[row_index, column].axis("off")
            axes[row_index, 0].set_ylabel(example_type.replace("_", "\n"), fontsize=9)

        figure.suptitle(
            f"Protocol {meta['protocol']} | atom {meta['atom_id']} | {meta['category']}\n"
            f"name: {meta['assigned_name']} | SUN: {meta['mapped_sun_attribute']} | "
            f"cos={float(meta['geometry_cosine']):.3f} | AUROC={float(meta['auroc']):.3f}",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        filename = f"{meta['protocol']}_atom_{meta['atom_id']}_{safe(meta['category'])}.png"
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)

    print(f"Rendered {len(groups)} frozen case sheets in {output}.")


if __name__ == "__main__":
    main()
