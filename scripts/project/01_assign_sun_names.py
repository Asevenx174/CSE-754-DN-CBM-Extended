import csv
import json
from collections import Counter
from pathlib import Path

import torch

CHECKPOINT_PATH = Path(
    "SAE/SAEImg/cc3m/clip_RN50/out/"
    "lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/"
    "sae_checkpoints/sparse_autoencoder_final.pt"
)

EMBEDDINGS_PATH = Path(
    "data/exp01/sun102_clip_RN50_embeddings.pth"
)

OUTPUT_DIR = Path("experiment-results/exp01")

def load_geometry():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )
    embedding_data = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu",
    )

    decoder = checkpoint["decoder._weight"].squeeze(0).float()
    text_embeddings = embedding_data["embeddings"].float()
    concepts = embedding_data["concepts"]

    if decoder.shape != (1024, 8192):
        raise ValueError(
            f"Expected decoder shape (1024, 8192), "
            f"received {tuple(decoder.shape)}."
        )

    if text_embeddings.shape != (102, 1024):
        raise ValueError(
            f"Expected embedding shape (102, 1024), "
            f"received {tuple(text_embeddings.shape)}."
        )

    decoder = decoder / decoder.norm(
        dim=0, keepdim=True
    ).clamp_min(1e-12)

    text_embeddings = text_embeddings / text_embeddings.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)

    return decoder, text_embeddings, concepts

def assign_names(decoder, text_embeddings):
    similarity = text_embeddings @ decoder

    best_scores, best_concept_indices = similarity.max(dim=0)

    return similarity, best_scores, best_concept_indices

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    decoder, text_embeddings, concepts = load_geometry()
    similarity, best_scores, best_indices = assign_names(
        decoder,
        text_embeddings,
    )

    assignment_path = OUTPUT_DIR / "sun_atom_assignments.csv"

    with assignment_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["atom_id", "sun_index", "sun_concept", "cosine"]
        )

        for atom_id in range(decoder.shape[1]):
            concept_index = int(best_indices[atom_id])
            writer.writerow(
                [
                    atom_id,
                    concept_index,
                    concepts[concept_index],
                    float(best_scores[atom_id]),
                ]
            )

    group_counts = Counter(
        concepts[int(index)] for index in best_indices
    )

    group_path = OUTPUT_DIR / "sun_group_sizes.csv"

    with group_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["sun_concept", "number_of_atoms"])

        for concept in concepts:
            writer.writerow([concept, group_counts.get(concept, 0)])

    summary = {
        "decoder_shape": list(decoder.shape),
        "text_embedding_shape": list(text_embeddings.shape),
        "similarity_shape": list(similarity.shape),
        "number_of_atoms": int(decoder.shape[1]),
        "number_of_sun_concepts": len(concepts),
        "concepts_receiving_atoms": len(group_counts),
        "minimum_cosine": float(best_scores.min()),
        "mean_cosine": float(best_scores.mean()),
        "maximum_cosine": float(best_scores.max()),
        "atom_assignments_path": str(assignment_path),
        "group_sizes_path": str(group_path),
    }

    summary_path = OUTPUT_DIR / "sun_naming_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()