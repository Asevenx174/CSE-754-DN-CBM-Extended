import argparse
import csv
import json
from pathlib import Path

import torch

CHECKPOINT_PATH = Path(
    "SAE/SAEImg/cc3m/clip_RN50/out/"
    "lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/"
    "sae_checkpoints/sparse_autoencoder_final.pt"
)

REFERENCE_NAMES_PATH = CHECKPOINT_PATH.parent.parent / "concept_names.csv"

VOCABULARY_PATH = Path(
    "data/vocab/clipdissect_20k.txt"
)

EMBEDDINGS_PATH = Path(
    "data/vocab/embeddings_clip_RN50_clipdissect_20k.pth"
)

OUTPUT_DIR = Path("experiment-results/exp01")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recover and verify DN-CBM's original 20k atom names "
            "for EXP-01A."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--atom_batch_size",
        type=int,
        default=512,
        help="Number of decoder columns scored per matrix-multiply batch.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
    )
    return parser.parse_args()


def load_resources():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    decoder = checkpoint["decoder._weight"].squeeze(0).float()

    text_embeddings = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu",
    ).float()

    vocabulary = [
        line.strip()
        for line in VOCABULARY_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    reference_names = []

    for line in REFERENCE_NAMES_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        atom_id, name = line.split(",", maxsplit=1)

        if int(atom_id) != len(reference_names):
            raise ValueError(
                f"Unexpected atom ID order at {atom_id}."
            )

        reference_names.append(name)

    if decoder.shape != (1024, 8192):
        raise ValueError(
            f"Unexpected decoder shape: {tuple(decoder.shape)}"
        )

    if text_embeddings.shape != (20000, 1024):
        raise ValueError(
            "Unexpected original embedding shape: "
            f"{tuple(text_embeddings.shape)}"
        )

    if len(vocabulary) != 20000:
        raise ValueError(
            f"Expected 20,000 names, found {len(vocabulary)}."
        )

    if len(reference_names) != 8192:
        raise ValueError(
            "Expected 8,192 original atom names, found "
            f"{len(reference_names)}."
        )

    return decoder, text_embeddings, vocabulary, reference_names

def recover_assignments(
    decoder,
    text_embeddings,
    vocabulary,
    device,
    atom_batch_size,
):
    decoder = decoder / decoder.norm(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-12)

    embedding_norms = text_embeddings.norm(dim=1)

    print(
        "Text embedding norm range:",
        float(embedding_norms.min()),
        float(embedding_norms.max()),
    )

    if not torch.allclose(
        embedding_norms,
        torch.ones_like(embedding_norms),
        atol=1e-3,
        rtol=1e-3,
    ):
        raise ValueError(
            "The original text embeddings are not unit-normalized."
        )

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if atom_batch_size <= 0:
        raise ValueError("atom_batch_size must be positive.")

    text_embeddings_device = text_embeddings.to(device)
    score_batches = []
    index_batches = []

    with torch.no_grad():
        for start in range(0, decoder.shape[1], atom_batch_size):
            stop = min(start + atom_batch_size, decoder.shape[1])
            similarity = (
                text_embeddings_device
                @ decoder[:, start:stop].to(device)
            )
            batch_scores, batch_indices = similarity.max(dim=0)
            score_batches.append(batch_scores.cpu())
            index_batches.append(batch_indices.cpu())

    best_scores = torch.cat(score_batches)
    best_indices = torch.cat(index_batches)

    recovered_names = [
        vocabulary[int(index)]
        for index in best_indices
    ]

    return best_scores, best_indices, recovered_names


def save_outputs(
    output_dir,
    best_scores,
    best_indices,
    recovered_names,
    reference_names,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (
        len(best_scores)
        == len(best_indices)
        == len(recovered_names)
        == len(reference_names)
        == 8192
    ):
        raise RuntimeError("Recovered assignment arrays do not contain 8,192 atoms.")

    assignment_path = output_dir / "original_atom_assignments.csv"
    mismatch_rows = []

    with assignment_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "atom_id",
                "vocabulary_index",
                "recovered_name",
                "original_cosine",
                "reference_name",
                "name_matches_reference",
            ]
        )

        for atom_id, (index, name, score, reference_name) in enumerate(
            zip(best_indices, recovered_names, best_scores, reference_names)
        ):
            matches = name == reference_name
            writer.writerow(
                [
                    atom_id,
                    int(index),
                    name,
                    float(score),
                    reference_name,
                    matches,
                ]
            )
            if not matches:
                mismatch_rows.append(
                    {
                        "atom_id": atom_id,
                        "recovered_name": name,
                        "reference_name": reference_name,
                    }
                )

    summary = {
        "number_of_atoms": len(recovered_names),
        "unique_recovered_names": len(set(recovered_names)),
        "reference_matches": len(recovered_names) - len(mismatch_rows),
        "reference_mismatches": len(mismatch_rows),
        "reference_match_rate": (
            (len(recovered_names) - len(mismatch_rows)) / len(recovered_names)
        ),
        "minimum_original_cosine": float(best_scores.min()),
        "mean_original_cosine": float(best_scores.mean()),
        "maximum_original_cosine": float(best_scores.max()),
        "assignment_path": str(assignment_path),
    }

    summary_path = output_dir / "original_naming_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if mismatch_rows:
        mismatch_path = output_dir / "original_name_mismatches.csv"
        with mismatch_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=mismatch_rows[0].keys())
            writer.writeheader()
            writer.writerows(mismatch_rows)
        summary["mismatch_path"] = str(mismatch_path)

    return summary


def main():
    args = parse_args()
    decoder, text_embeddings, vocabulary, reference_names = load_resources()
    best_scores, best_indices, recovered_names = recover_assignments(
        decoder=decoder,
        text_embeddings=text_embeddings,
        vocabulary=vocabulary,
        device=args.device,
        atom_batch_size=args.atom_batch_size,
    )
    summary = save_outputs(
        output_dir=args.output_dir,
        best_scores=best_scores,
        best_indices=best_indices,
        recovered_names=recovered_names,
        reference_names=reference_names,
    )
    print(json.dumps(summary, indent=2))

    if summary["reference_mismatches"]:
        raise RuntimeError(
            "Recovered names do not fully match the published concept_names.csv. "
            "Inspect original_name_mismatches.csv before continuing."
        )


if __name__ == "__main__":
    main()