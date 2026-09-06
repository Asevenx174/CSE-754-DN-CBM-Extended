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