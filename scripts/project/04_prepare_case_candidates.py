"""EXP-04.1: freeze case-study categories and list eligible atoms."""

import csv
import json
from collections import Counter
from pathlib import Path


INPUT = Path("experiment-results/exp02")
OUTPUT = Path("experiment-results/exp04")
HIGH_G, LOW_G = 0.20, 0.10
HIGH_S, LOW_S = 0.70, 0.55


def category(geometry, semantics):
    geometry_level = "high" if geometry >= HIGH_G else "low" if geometry < LOW_G else None
    semantic_level = "high" if semantics >= HIGH_S else "low" if semantics <= LOW_S else None
    return f"{geometry_level}_geometry_{semantic_level}_detection" if geometry_level and semantic_level else None


def load(protocol):
    path = INPUT / f"protocol_{protocol.lower()}_atom_scores.csv"
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected, counts = [], {}
    for protocol in ("A", "B"):
        protocol_rows = []
        for row in load(protocol):
            geometry = float(row["geometry_cosine"])
            semantics = float(row["auroc"])
            label = category(geometry, semantics)
            if label:
                protocol_rows.append(
                    {
                        "protocol": protocol,
                        "category": label,
                        "atom_id": row["atom_id"],
                        "assigned_name": row["assigned_name"],
                        "mapped_sun_attribute": row["mapped_sun_attribute"],
                        "geometry_cosine": geometry,
                        "auroc": semantics,
                        "average_precision": row["average_precision"],
                        "test_positives": row["test_positives"],
                        "activation_frequency": row["activation_frequency"],
                    }
                )
        selected.extend(protocol_rows)
        counts[protocol] = dict(Counter(row["category"] for row in protocol_rows))

    path = OUTPUT / "case_candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "status": "post hoc exploratory",
        "rules": {
            "high_geometry": f">= {HIGH_G}",
            "low_geometry": f"< {LOW_G}",
            "high_detection": f">= {HIGH_S}",
            "low_detection": f"<= {LOW_S}",
            "intermediate_cases": "excluded",
        },
        "counts": counts,
        "candidate_table": str(path),
    }
    (OUTPUT / "case_category_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
