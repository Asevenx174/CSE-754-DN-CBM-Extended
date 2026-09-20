"""EXP-02.1: cache frozen CLIP features and SAE activations for SUN."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import clip
from dncbm.data_utils.sun_attributes import SUNAttributes
from sparse_autoencoder import SparseAutoencoder


CHECKPOINT = Path(
    "SAE/SAEImg/cc3m/clip_RN50/out/"
    "lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200/"
    "sae_checkpoints/sparse_autoencoder_final.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--annotation_dir",
        default="data/sunattributes/SUNAttributeDB/SUNAttributeDB",
    )
    parser.add_argument(
        "--image_root",
        default="data/sunattributes/SUNAttributeDB_Images/images",
    )
    parser.add_argument(
        "--output",
        default="data/exp02/sun_frozen_cache.pth",
    )
    parser.add_argument(
        "--summary",
        default="experiment-results/exp02/cache_summary.json",
    )
    return parser.parse_args()


def load_models(device):
    clip_model, preprocess = clip.load("RN50", device=device)
    clip_model.eval()

    sae = SparseAutoencoder(
        n_input_features=1024,
        n_learned_features=8192,
        n_components=1,
    ).to(device)
    sae.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    sae.eval()
    return clip_model, preprocess, sae


def main():
    args = parse_args()
    device = torch.device(args.device)
    clip_model, preprocess, sae = load_models(device)
    dataset = SUNAttributes(
        args.annotation_dir, args.image_root, transform=preprocess
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    sample_ids, features, activations = [], [], []
    labels, agreements = [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Caching SUN", unit="batch"):
            image = batch["image"].to(device, non_blocking=True)
            feature = clip_model.encode_image(image).float()
            activation, _ = sae(feature)

            sample_ids.append(batch["sample_id"])
            features.append(feature.cpu())
            activations.append(activation.squeeze(1).cpu())
            labels.append(batch["labels"])
            agreements.append(batch["agreement"])

    cache = {
        "sample_ids": torch.cat(sample_ids),
        "clip_features": torch.cat(features).float(),
        "sae_activations": torch.cat(activations).float(),
        "binary_labels": torch.cat(labels).to(torch.uint8),
        "agreement": torch.cat(agreements).float(),
    }
    expected_ids = torch.arange(len(dataset))
    if not torch.equal(cache["sample_ids"], expected_ids):
        raise RuntimeError("Cache order does not match SUN sample IDs.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)

    summary = {
        "images": len(dataset),
        "clip_features_shape": list(cache["clip_features"].shape),
        "sae_activations_shape": list(cache["sae_activations"].shape),
        "binary_labels_shape": list(cache["binary_labels"].shape),
        "finite_features": bool(torch.isfinite(cache["clip_features"]).all()),
        "finite_activations": bool(torch.isfinite(cache["sae_activations"]).all()),
        "sample_order_verified": True,
        "output": str(output),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
