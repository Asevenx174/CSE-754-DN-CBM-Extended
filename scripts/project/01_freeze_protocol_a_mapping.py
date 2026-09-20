"""EXP-01A.3: freeze a conservative exact-only Protocol A mapping.

This primary mapping accepts strict original-name/SUN-name equality only. All
other names remain unevaluable under this policy and are recorded as unmatched,
not semantically incorrect. No activation, annotation outcome, AUROC, or test
result is read by this script.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review_path",
        default="experiment-results/exp01/protocol_a_sun_mapping_review.csv",
    )
    parser.add_argument(
        "--assignments_path",
        default="experiment-results/exp01/original_atom_assignments.csv",
    )
    parser.add_argument(
        "--valid_concepts_path",
        default="experiment-results/exp01/valid_concepts.txt",
    )
    parser.add_argument(
        "--mapping_output",
        default="experiment-results/exp01/protocol_a_sun_mapping.csv",
    )
    parser.add_argument(
        "--pairs_output",
        default="experiment-results/exp01/protocol_a_atom_concept_pairs.csv",
    )
    parser.add_argument(
        "--summary_output",
        default="experiment-results/exp01/protocol_a_mapping_summary.json",
    )
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_unique_lines(path):
    values = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entries found in {path}.")
    return values


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    review_rows = read_csv(args.review_path)
    assignment_rows = read_csv(args.assignments_path)
    valid_concepts = set(read_unique_lines(args.valid_concepts_path))

    if len(review_rows) != 3773:
        raise ValueError(f"Expected 3,773 unique names, found {len(review_rows)}.")
    if len(assignment_rows) != 8192:
        raise ValueError(f"Expected 8,192 atoms, found {len(assignment_rows)}.")
    if any(row["name_matches_reference"] != "True" for row in assignment_rows):
        raise ValueError("Original-name recovery contains reference mismatches.")

    mapping_rows = []
    mapping_by_name = {}

    for row in review_rows:
        original_name = row["original_name"]
        is_exact = row["mapping_category"] == "exact"

        if is_exact:
            mapped_attribute = row["mapped_sun_attribute"]
            if original_name != mapped_attribute:
                raise ValueError(
                    f"Exact row does not preserve the same string: {original_name!r}."
                )
            category = "exact"
            include_primary = True
            note = "accepted by conservative strict string equality"
        else:
            mapped_attribute = ""
            category = "unmatched"
            include_primary = False
            note = "not accepted by conservative exact-only primary policy"

        supported = bool(mapped_attribute and mapped_attribute in valid_concepts)
        frozen_row = {
            "original_name": original_name,
            "number_of_atoms": int(row["number_of_atoms"]),
            "original_cosine_min": float(row["original_cosine_min"]),
            "original_cosine_mean": float(row["original_cosine_mean"]),
            "original_cosine_max": float(row["original_cosine_max"]),
            "mapping_category": category,
            "mapped_sun_attribute": mapped_attribute,
            "include_in_primary": include_primary,
            "mapped_attribute_is_supported": supported,
            "mapping_policy": "conservative_exact_only_v1",
            "review_note": note,
        }
        mapping_rows.append(frozen_row)
        mapping_by_name[original_name] = frozen_row

    mapping_fields = list(mapping_rows[0].keys())
    write_csv(args.mapping_output, mapping_rows, mapping_fields)

    pair_rows = []
    for atom in assignment_rows:
        mapping = mapping_by_name[atom["recovered_name"]]
        if not mapping["include_in_primary"]:
            continue

        pair_rows.append(
            {
                "atom_id": int(atom["atom_id"]),
                "original_name": atom["recovered_name"],
                "original_cosine": float(atom["original_cosine"]),
                "mapping_category": mapping["mapping_category"],
                "mapped_sun_attribute": mapping["mapped_sun_attribute"],
                "mapped_attribute_is_supported": mapping[
                    "mapped_attribute_is_supported"
                ],
                "evaluation_eligible": mapping["mapped_attribute_is_supported"],
            }
        )

    pair_fields = list(pair_rows[0].keys())
    write_csv(args.pairs_output, pair_rows, pair_fields)

    exact_rows = [row for row in mapping_rows if row["mapping_category"] == "exact"]
    supported_exact_rows = [
        row for row in exact_rows if row["mapped_attribute_is_supported"]
    ]
    eligible_pairs = [row for row in pair_rows if row["evaluation_eligible"]]

    summary = {
        "mapping_policy": "conservative_exact_only_v1",
        "mapping_frozen_before_semantic_scoring": True,
        "unique_original_names": len(mapping_rows),
        "number_of_atoms": len(assignment_rows),
        "exact_name_matches": len(exact_rows),
        "predetermined_synonyms": 0,
        "ambiguous_names": 0,
        "unmatched_names": sum(
            row["mapping_category"] == "unmatched" for row in mapping_rows
        ),
        "mapped_atoms": len(pair_rows),
        "supported_mapped_attributes": len(supported_exact_rows),
        "evaluation_eligible_atoms": len(eligible_pairs),
        "mapping_output": args.mapping_output,
        "pairs_output": args.pairs_output,
        "mapping_sha256": sha256(args.mapping_output),
        "pairs_sha256": sha256(args.pairs_output),
        "interpretation": (
            "unmatched means unevaluable under the exact-only SUN policy, "
            "not semantically incorrect"
        ),
    }

    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
