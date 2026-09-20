"""EXP-01A.2: prepare a leakage-safe review table for 20k-to-SUN mapping.

Exact string matches are accepted automatically. For every other original name,
the script supplies one lexical suggestion only to help manual review. A
suggestion is never included in the primary audit unless its mapping category is
later changed to ``predetermined_synonym`` before semantic scores are inspected.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assignments_path",
        default="experiment-results/exp01/original_atom_assignments.csv",
    )
    parser.add_argument(
        "--sun_vocabulary_path",
        default="experiment-results/exp01/sun102_vocabulary.txt",
    )
    parser.add_argument(
        "--valid_concepts_path",
        default="experiment-results/exp01/valid_concepts.txt",
    )
    parser.add_argument(
        "--output_path",
        default="experiment-results/exp01/protocol_a_sun_mapping_review.csv",
    )
    parser.add_argument(
        "--summary_path",
        default="experiment-results/exp01/protocol_a_mapping_review_summary.json",
    )
    return parser.parse_args()


def normalize_for_suggestion(text):
    text = text.casefold()
    text = re.sub(r"[/_-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return " ".join(text.split())


def lexical_similarity(left, right):
    left_normalized = normalize_for_suggestion(left)
    right_normalized = normalize_for_suggestion(right)
    sequence_score = SequenceMatcher(
        None, left_normalized, right_normalized
    ).ratio()

    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0

    return max(sequence_score, token_score)


def load_assignments(path):
    grouped = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"atom_id", "recovered_name", "original_cosine"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Assignment CSV must contain {sorted(required)}."
            )

        for row in reader:
            grouped[row["recovered_name"]].append(float(row["original_cosine"]))

    if sum(len(scores) for scores in grouped.values()) != 8192:
        raise ValueError("Expected assignments for exactly 8,192 atoms.")
    return grouped


def load_lines(path):
    values = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entries found in {path}.")
    return values


def best_lexical_suggestion(original_name, sun_attributes):
    scored = [
        (lexical_similarity(original_name, attribute), attribute)
        for attribute in sun_attributes
    ]
    score, attribute = max(scored, key=lambda item: (item[0], item[1]))
    return attribute, score


def main():
    args = parse_args()
    grouped = load_assignments(args.assignments_path)
    sun_attributes = load_lines(args.sun_vocabulary_path)
    valid_concepts = set(load_lines(args.valid_concepts_path))

    if len(sun_attributes) != 102:
        raise ValueError(f"Expected 102 SUN attributes, found {len(sun_attributes)}.")
    if not valid_concepts.issubset(set(sun_attributes)):
        raise ValueError("valid_concepts contains names outside the SUN vocabulary.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for original_name in sorted(grouped, key=str.casefold):
        scores = grouped[original_name]
        is_exact = original_name in sun_attributes

        if is_exact:
            suggestion = original_name
            suggestion_score = 1.0
            category = "exact"
            mapped_attribute = original_name
            include_primary = True
            review_note = "automatic strict string match"
        else:
            suggestion, suggestion_score = best_lexical_suggestion(
                original_name, sun_attributes
            )
            category = "unreviewed"
            mapped_attribute = ""
            include_primary = False
            review_note = "lexical suggestion only; manual decision required"

        rows.append(
            {
                "original_name": original_name,
                "number_of_atoms": len(scores),
                "original_cosine_min": min(scores),
                "original_cosine_mean": sum(scores) / len(scores),
                "original_cosine_max": max(scores),
                "suggested_sun_attribute": suggestion,
                "suggestion_score": suggestion_score,
                "mapping_category": category,
                "mapped_sun_attribute": mapped_attribute,
                "include_in_primary": include_primary,
                "mapped_attribute_is_supported": (
                    mapped_attribute in valid_concepts if mapped_attribute else False
                ),
                "review_note": review_note,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    exact_rows = [row for row in rows if row["mapping_category"] == "exact"]
    supported_exact_rows = [
        row for row in exact_rows if row["mapped_attribute_is_supported"]
    ]
    summary = {
        "unique_original_names": len(rows),
        "number_of_atoms": sum(row["number_of_atoms"] for row in rows),
        "exact_name_matches": len(exact_rows),
        "atoms_in_exact_matches": sum(row["number_of_atoms"] for row in exact_rows),
        "supported_exact_name_matches": len(supported_exact_rows),
        "atoms_in_supported_exact_matches": sum(
            row["number_of_atoms"] for row in supported_exact_rows
        ),
        "unreviewed_names": sum(
            row["mapping_category"] == "unreviewed" for row in rows
        ),
        "output_path": str(output_path),
        "warning": (
            "suggested_sun_attribute is a lexical review aid, not an approved mapping"
        ),
    }

    Path(args.summary_path).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
