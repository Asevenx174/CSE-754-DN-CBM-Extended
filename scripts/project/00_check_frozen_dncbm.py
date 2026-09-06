"""EXP-00: verify the published frozen CLIP-RN50 DN-CBM SAE.

Run this from the Discover-then-Name repository root.  The input is a saved
CLIP feature tensor created by the existing repository, not raw images.  This
keeps EXP-00 focused on the frozen SAE and reuses the same loading code as
scripts/save_concept_strengths.py.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from dncbm import arg_parser
from dncbm.utils import common_init, get_sae_ckpt
from sparse_autoencoder import SparseAutoencoder


def parse_args():
    """Reuse DN-CBM's arguments, but set the published fixed-SAE defaults."""
    parser = arg_parser.get_common_parser()
    parser.set_defaults(
        sae_dataset="cc3m",
        img_enc_name="clip_RN50",
        hook_points=["out"],
        lr=5e-4,
        l1_coeff=3e-5,
        expansion_factor=8,
        resample_freq=10,
        train_sae_bs=4096,
        num_epochs=200,
    )
    parser.add_argument(
        "--features_path",
        required=True,
        help="Path to an existing [N, 1024] CLIP feature tensor saved with torch.save().",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=3)
    parser.add_argument(
        "--output_json",
        default="experiment-results/exp00/frozen_dncbm_sanity.json",
    )
    return parser.parse_args()


def load_frozen_sae(args):
    """Build the repository's SAE class and load its existing checkpoint."""
    input_dim = args.autoencoder_input_dim_dict[
        args.ae_input_dim_dict_key[args.modality]
    ]
    n_atoms = input_dim * args.expansion_factor
    sae = SparseAutoencoder(
        n_input_features=input_dim,
        n_learned_features=n_atoms,
        n_components=len(args.hook_points),
    ).to(args.device)
    sae = get_sae_ckpt(args, sae)
    sae.eval()
    return sae, input_dim, n_atoms


def load_features(path, expected_dim):
    """Accept the plain tensor format used by the repository feature scripts."""
    features = torch.load(path, map_location="cpu")
    if not isinstance(features, torch.Tensor):
        raise TypeError("features_path must contain one PyTorch tensor, not a dictionary.")
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"Expected [N, {expected_dim}] features, received {tuple(features.shape)}."
        )
    return features.float()


def diagnose(sae, features, args, n_atoms):
    """Run a few no-gradient batches and collect the EXP-00 diagnostics."""
    n_used = min(len(features), args.batch_size * args.max_batches)
    if n_used == 0:
        raise ValueError("The feature tensor is empty.")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    mse_sum = cosine_sum = active_sum = 0.0
    activation_hits = torch.zeros(n_atoms, dtype=torch.long)

    with torch.no_grad():
        for first in range(0, n_used, args.batch_size):
            original = features[first:first + args.batch_size].to(args.device)
            activations, reconstruction = sae(original)
            activations = activations.squeeze(1)
            reconstruction = reconstruction.squeeze(1)

            if activations.shape[1] != n_atoms:
                raise RuntimeError("Activation columns do not match the configured atom count.")

            mse_sum += torch.nn.functional.mse_loss(
                reconstruction, original, reduction="sum"
            ).item()
            cosine_sum += torch.nn.functional.cosine_similarity(
                reconstruction, original, dim=1
            ).sum().item()
            active = activations > 0
            active_sum += active.sum().item()
            activation_hits += active.sum(dim=0).cpu().long()

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = None

    elapsed = time.perf_counter() - start
    decoder = sae.decoder.weight.detach().cpu().squeeze()
    if decoder.shape[1] != n_atoms:
        raise RuntimeError("Decoder columns do not match the configured atom count.")
    decoder_norms = decoder.norm(dim=0)

    return {
        "n_images_checked": n_used,
        "feature_dimension": int(features.shape[1]),
        "n_sae_atoms": n_atoms,
        "reconstruction_mse_per_element": mse_sum / (n_used * features.shape[1]),
        "reconstruction_cosine_mean": cosine_sum / n_used,
        "mean_active_atoms_per_image": active_sum / n_used,
        "pilot_dead_atom_rate": (activation_hits == 0).float().mean().item(),
        "decoder_norm_min": decoder_norms.min().item(),
        "decoder_norm_mean": decoder_norms.mean().item(),
        "decoder_norm_max": decoder_norms.max().item(),
        "seconds": elapsed,
        "images_per_second": n_used / elapsed,
        "peak_gpu_memory_mb": peak_memory_mb,
        "atom_index_check": "passed: activation columns and decoder columns both equal n_sae_atoms",
    }


def main():
    args = parse_args()
    common_init(args, disable_make_dirs=True)
    sae, input_dim, n_atoms = load_frozen_sae(args)
    features = load_features(args.features_path, input_dim)
    report = diagnose(sae, features, args, n_atoms)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved EXP-00 report to: {output_path}")


if __name__ == "__main__":
    main()
