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