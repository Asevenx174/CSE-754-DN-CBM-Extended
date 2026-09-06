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